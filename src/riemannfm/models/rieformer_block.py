"""Single RieFormer transformer block.

Composes: Manifold Attention (A) → ATH-Norm (B) → Edge Self-Update (C)
→ Cross-Interaction (D) → [Text Injection (E)] → Feed-Forward.

Each sub-module is gated by ablation flags from the config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from riemannfm.models.attention.edge import (
    RiemannFMEdgeBias,
    RiemannFMEdgeSelfUpdate,
)
from riemannfm.models.attention.geodesic import RiemannFMGeodesicAttention
from riemannfm.models.heads import RiemannFMDualStreamCross
from riemannfm.models.normalization import RiemannFMATHNorm, RiemannFMPreNorm

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


class RiemannFMFeedForward(nn.Module):
    """Two-layer feed-forward with SiLU activation and residual.

    Args:
        dim: Input/output dimension.
        mult: Hidden dimension multiplier.
        dropout: Dropout rate.
    """

    def __init__(
        self, dim: int, mult: int = 4, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        result: Tensor = self.net(x)
        return result


class RiemannFMBlock(nn.Module):
    """Single RieFormer transformer block.

    Architecture per block:
      1. [A] Geodesic manifold attention (node self-attention with geo kernel)
      2. [B] ATH-Norm (or pre-norm fallback)
      3. [C] Edge self-update (factorized head/tail aggregation)
      4. [D] Cross-interaction (edge↔node bidirectional)
      5. [E] Text cross-attention (optional, every ``text_cross_attn_every`` layers)
      6. Feed-forward (node and edge)

    Args:
        node_dim: Node hidden dimension.
        edge_dim: Edge hidden dimension.
        num_heads: Number of attention heads.
        edge_heads: Number of edge factorization heads.
        manifold: Product manifold.
        time_dim: Time embedding dimension (for ATH-Norm).
        use_geodesic_kernel: Enable geodesic distance kernel in attention.
        use_ath_norm: Use ATH-Norm (True) or plain LayerNorm (False).
        use_edge_self_update: Enable edge self-update module.
        use_dual_stream_cross: Enable edge↔node cross-interaction.
        use_text_cross_attn: Enable text cross-attention in this block.
        text_dim: Text embedding dimension (0 to disable text injection).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_heads: int,
        edge_heads: int,
        manifold: RiemannFMProductManifold,
        time_dim: int,
        use_geodesic_kernel: bool = True,
        use_ath_norm: bool = True,
        use_edge_self_update: bool = True,
        use_dual_stream_cross: bool = True,
        use_text_cross_attn: bool = False,
        text_dim: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # [A] Manifold attention.
        self.attn = RiemannFMGeodesicAttention(
            node_dim, num_heads, manifold,
            use_geodesic_kernel=use_geodesic_kernel,
            dropout=dropout,
        )
        self.edge_bias = RiemannFMEdgeBias(edge_dim, num_heads)

        # [B] Normalization (pre-norm style).
        if use_ath_norm:
            self.norm1: nn.Module = RiemannFMATHNorm(node_dim, time_dim)
            self.norm2: nn.Module = RiemannFMATHNorm(node_dim, time_dim)
        else:
            self.norm1 = RiemannFMPreNorm(node_dim)
            self.norm2 = RiemannFMPreNorm(node_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)

        # [C] Edge self-update.
        self.use_edge_self_update = use_edge_self_update
        if use_edge_self_update:
            self.edge_update = RiemannFMEdgeSelfUpdate(
                node_dim, edge_dim, edge_heads, dropout=dropout,
            )

        # [D] Cross-interaction.
        self.use_dual_stream_cross = use_dual_stream_cross
        if use_dual_stream_cross:
            self.cross = RiemannFMDualStreamCross(
                node_dim, edge_dim, dropout=dropout,
            )

        # [E] Text cross-attention (optional).
        self.use_text_cross_attn = use_text_cross_attn and text_dim > 0
        if self.use_text_cross_attn:
            self.text_cross_attn = nn.MultiheadAttention(
                embed_dim=node_dim,
                num_heads=num_heads,
                kdim=text_dim,
                vdim=text_dim,
                dropout=dropout,
                batch_first=True,
            )
            self.text_norm = nn.LayerNorm(node_dim)

        # Feed-forward.
        self.ff_node = RiemannFMFeedForward(node_dim, dropout=dropout)
        self.ff_edge = RiemannFMFeedForward(edge_dim, dropout=dropout)

    def forward(
        self,
        h: Tensor,
        g: Tensor,
        x: Tensor,
        t_emb: Tensor,
        node_mask: Tensor,
        C_V: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass through one RieFormer block.

        Args:
            h: Node hidden states, shape ``(B, N, node_dim)``.
            g: Edge hidden states, shape ``(B, N, N, edge_dim)``.
            x: Manifold coordinates, shape ``(B, N, D)``.
            t_emb: Time embeddings, shape ``(B, time_dim)``.
            node_mask: Bool mask, shape ``(B, N)``.
            C_V: Node text embeddings for cross-attention,
                shape ``(B, N, text_dim)``.  None to skip.

        Returns:
            Updated (h, g) with same shapes.
        """
        # [A] Manifold attention with residual.
        bias = self.edge_bias(g)  # (B, H, N, N)
        h = h + self.attn(self.norm1(h, t_emb, node_mask), x, bias, node_mask)

        # [C] Edge self-update (already has residual connection inside).
        if self.use_edge_self_update:
            g = self.edge_norm(self.edge_update(h, g))

        # [D] Cross-interaction.
        if self.use_dual_stream_cross:
            h_cross, g_cross = self.cross(h, g, node_mask)
            h = h_cross
            g = g_cross

        # [E] Text cross-attention.
        if self.use_text_cross_attn and C_V is not None:
            h_normed = self.text_norm(h)
            # Key mask for text: same as node_mask.
            key_padding_mask = ~node_mask if node_mask is not None else None
            h_text, _ = self.text_cross_attn(
                h_normed, C_V, C_V,
                key_padding_mask=key_padding_mask,
            )
            h = h + h_text

        # Feed-forward with residual.
        h = h + self.ff_node(self.norm2(h, t_emb, node_mask))
        g = g + self.ff_edge(self.edge_norm(g))

        # Mask virtual node hidden states.
        if node_mask is not None:
            h = h * node_mask.unsqueeze(-1)

        return h, g
