"""Single RED-Former block: one layer of the dual-stream transformer.

Each block consists of four stages:
1. Manifold-aware self-attention (node stream) with ATH-Norm
2. Edge self-attention (edge stream)
3. Node-edge dual-stream cross-interaction
4. Text condition injection (optional, every K layers)

Plus feed-forward networks for both streams.
"""

import torch.nn as nn
from torch import Tensor

from riemannfm.manifolds.product import RiemannFMProductManifold
from riemannfm.models.attention.edge import RiemannFMEdgeAttention
from riemannfm.models.attention.geodesic import RiemannFMGeodesicAttention
from riemannfm.models.attention.text_cross import RiemannFMTextCrossAttention
from riemannfm.models.dual_stream import RiemannFMDualStreamInteraction
from riemannfm.models.normalization import RiemannFMATHNorm


class RiemannFMRieFormerBlock(nn.Module):
    """Single RED-Former block with dual-stream processing.

    Args:
        node_dim: Node feature dimension.
        edge_dim: Edge feature dimension.
        num_heads: Number of attention heads for node attention.
        edge_heads: Number of attention heads for edge attention.
        time_embed_dim: Dimension of timestep embedding.
        text_dim: Text encoder dimension (0 to disable text cross-attention).
        dropout: Dropout rate.
        use_mrope: Whether to use Manifold RoPE in node attention.
        ff_mult: Feed-forward hidden dimension multiplier.
    """

    def __init__(
        self,
        node_dim: int = 768,
        edge_dim: int = 256,
        num_heads: int = 12,
        edge_heads: int = 4,
        time_embed_dim: int = 256,
        text_dim: int = 0,
        dropout: float = 0.1,
        use_mrope: bool = True,
        ff_mult: int = 4,
    ):
        super().__init__()

        # Stage 1: Node self-attention with geodesic kernel
        self.node_attn_norm = RiemannFMATHNorm(node_dim, time_embed_dim)
        self.node_attn = RiemannFMGeodesicAttention(
            hidden_dim=node_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_mrope=use_mrope,
        )

        # Stage 2: Edge self-attention
        self.edge_attn = RiemannFMEdgeAttention(
            edge_dim=edge_dim,
            num_heads=edge_heads,
            dropout=dropout,
        )

        # Stage 3: Node-edge cross-interaction
        self.cross_interaction = RiemannFMDualStreamInteraction(
            node_dim=node_dim,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        # Stage 4: Text cross-attention (optional)
        self.has_text_attn = text_dim > 0
        if self.has_text_attn:
            self.text_attn_norm = RiemannFMATHNorm(node_dim, time_embed_dim)
            self.text_attn = RiemannFMTextCrossAttention(
                node_dim=node_dim,
                text_dim=text_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

        # Feed-forward networks
        self.node_ff_norm = RiemannFMATHNorm(node_dim, time_embed_dim)
        self.node_ff = nn.Sequential(
            nn.Linear(node_dim, node_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim * ff_mult, node_dim),
            nn.Dropout(dropout),
        )

        self.edge_ff = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, edge_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim * ff_mult, edge_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h_v: Tensor,
        h_e: Tensor,
        manifold: RiemannFMProductManifold,
        positions: Tensor,
        t_embed: Tensor,
        depth: Tensor | None = None,
        text_embeds: Tensor | None = None,
        text_mask: Tensor | None = None,
        edge_mask: Tensor | None = None,
        node_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            h_v: Node features, shape (N, node_dim).
            h_e: Edge features, shape (N, N, edge_dim).
            manifold: Product manifold for geometry operations.
            positions: Node positions on manifold, shape (N, total_dim).
            t_embed: Timestep embedding, shape (1, time_embed_dim) or (N, time_embed_dim).
            depth: Hierarchy depth per node, shape (N,).
            text_embeds: Text token embeddings, shape (T, text_dim).
            text_mask: Text attention mask, shape (T,).
            edge_mask: Valid edge mask, shape (N, N).
            node_mask: Valid node mask, shape (N, N) for attention.

        Returns:
            (h_v_updated, h_e_updated).
        """
        # Stage 1: Node self-attention
        h_v_normed = self.node_attn_norm(h_v, t_embed, depth)
        h_v = h_v + self.node_attn(h_v_normed, manifold, positions, mask=node_mask)

        # Stage 2: Edge self-attention
        h_e = self.edge_attn(h_e, edge_mask=edge_mask)

        # Stage 3: Cross-interaction
        h_v, h_e = self.cross_interaction(h_v, h_e, edge_mask=edge_mask)

        # Stage 4: Text cross-attention (if enabled)
        if self.has_text_attn and text_embeds is not None:
            h_v_normed = self.text_attn_norm(h_v, t_embed, depth)
            h_v = self.text_attn(h_v_normed, text_embeds, text_mask)

        # Feed-forward
        h_v_normed = self.node_ff_norm(h_v, t_embed, depth)
        h_v = h_v + self.node_ff(h_v_normed)
        h_e = h_e + self.edge_ff(h_e)

        return h_v, h_e
