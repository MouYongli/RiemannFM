"""RieFormer: stack of L RieFormer blocks.

Takes encoded node/edge hidden states and applies L transformer blocks,
each with manifold attention, edge updates, and optional text injection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from riemannfm.models.rieformer_block import RiemannFMBlock

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


class RiemannFMRieFormer(nn.Module):
    """Stack of L RieFormer transformer blocks.

    Args:
        num_layers: Number of transformer blocks L.
        node_dim: Node hidden dimension.
        edge_dim: Edge hidden dimension.
        num_heads: Number of attention heads.
        edge_heads: Number of edge factorization heads.
        manifold: Product manifold.
        time_dim: Time embedding dimension.
        text_dim: Text embedding dimension (0 to disable).
        text_cross_attn_every: Insert text cross-attention every N layers.
        use_geodesic_kernel: Enable geodesic kernel in attention.
        use_ath_norm: Use ATH-Norm or plain LayerNorm.
        use_edge_self_update: Enable edge self-update.
        use_dual_stream_cross: Enable edge↔node cross-interaction.
        use_text_condition: Enable text conditioning globally.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_layers: int,
        node_dim: int,
        edge_dim: int,
        num_heads: int,
        edge_heads: int,
        manifold: RiemannFMProductManifold,
        time_dim: int,
        text_dim: int = 0,
        text_cross_attn_every: int = 999,
        use_geodesic_kernel: bool = True,
        use_ath_norm: bool = True,
        use_edge_self_update: bool = True,
        use_dual_stream_cross: bool = True,
        use_text_condition: bool = True,
        cond_dim: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Text cross-attention is inserted every N layers.
            use_text = (
                use_text_condition
                and text_dim > 0
                and (i + 1) % text_cross_attn_every == 0
            )
            self.layers.append(
                RiemannFMBlock(
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    num_heads=num_heads,
                    edge_heads=edge_heads,
                    manifold=manifold,
                    time_dim=time_dim,
                    use_geodesic_kernel=use_geodesic_kernel,
                    use_ath_norm=use_ath_norm,
                    use_edge_self_update=use_edge_self_update,
                    use_dual_stream_cross=use_dual_stream_cross,
                    use_text_cross_attn=use_text,
                    text_dim=text_dim,
                    cond_dim=cond_dim,
                    dropout=dropout,
                ),
            )

        self.final_node_norm = nn.LayerNorm(node_dim)
        self.final_edge_norm = nn.LayerNorm(edge_dim)

    def forward(
        self,
        h: Tensor,
        g: Tensor,
        x: Tensor,
        t_emb: Tensor,
        node_mask: Tensor,
        C_V: Tensor | None = None,
        cond: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass through all L blocks.

        Args:
            h: Node hidden states, shape ``(B, N, node_dim)``.
            g: Edge hidden states, shape ``(B, N, N, edge_dim)``.
            x: Manifold coordinates, shape ``(B, N, D)``.
            t_emb: Time embeddings, shape ``(B, time_dim)``.
            node_mask: Bool mask, shape ``(B, N)``.
            C_V: Node text embeddings, shape ``(B, N, text_dim)``.
            cond: Auxiliary conditioning for ATH-Norm FiLM, shape
                ``(B, cond_dim)``.  Plumbed to each block.

        Returns:
            Final (h, g) after L blocks, with final layer norm.
        """
        for block in self.layers:
            h, g = block(h, g, x, t_emb, node_mask, C_V, cond=cond)

        h = self.final_node_norm(h)
        g = self.final_edge_norm(g)

        return h, g
