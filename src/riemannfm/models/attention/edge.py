"""Edge self-update and edge-to-attention bias (Def 5.10-5.11).

The edge self-update uses factorized head/tail aggregation to update
edge hidden states.  Edge bias projects edge features into per-head
scalars that bias the attention logits.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RiemannFMEdgeBias(nn.Module):
    """Project edge features into per-head attention bias (Def 5.6).

    Args:
        edge_dim: Edge hidden dimension.
        num_heads: Number of attention heads.
    """

    def __init__(self, edge_dim: int, num_heads: int) -> None:
        super().__init__()
        self.proj = nn.Linear(edge_dim, num_heads, bias=False)

    def forward(self, g: Tensor) -> Tensor:
        """Compute edge bias for attention.

        Args:
            g: Edge hidden states, shape ``(B, N, N, edge_dim)``.

        Returns:
            Per-head bias, shape ``(B, num_heads, N, N)``.
        """
        # (B, N, N, edge_dim) -> (B, N, N, H) -> (B, H, N, N)
        return self.proj(g).permute(0, 3, 1, 2)


class RiemannFMEdgeSelfUpdate(nn.Module):
    """Factorized edge self-update (Def 5.10-5.11).

    Updates edge features by aggregating head-node and tail-node
    information through outer-product style factorization:

      g'_{ij} = g_{ij} + MLP([head_agg_i || tail_agg_j || g_{ij}])

    Where head_agg and tail_agg are computed from node hidden states.

    Args:
        node_dim: Node hidden dimension.
        edge_dim: Edge hidden dimension.
        edge_heads: Number of factorization heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        edge_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.edge_heads = edge_heads
        head_dim = edge_dim // edge_heads

        # Node-to-edge projections for head/tail aggregation.
        self.proj_head = nn.Linear(node_dim, edge_heads * head_dim)
        self.proj_tail = nn.Linear(node_dim, edge_heads * head_dim)

        # MLP to combine: [head_agg || tail_agg || g_{ij}].
        self.mlp = nn.Sequential(
            nn.Linear(2 * edge_dim + edge_dim, edge_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim, edge_dim),
        )

    def forward(self, h: Tensor, g: Tensor) -> Tensor:
        """Update edge features.

        Args:
            h: Node hidden states, shape ``(B, N, node_dim)``.
            g: Edge hidden states, shape ``(B, N, N, edge_dim)``.

        Returns:
            Updated edge features, shape ``(B, N, N, edge_dim)``.
        """
        # Head/tail projections: (B, N, edge_dim).
        h_head = self.proj_head(h)  # (B, N, edge_dim) — sender
        h_tail = self.proj_tail(h)  # (B, N, edge_dim) — receiver

        # Outer-product factorization: broadcast to (B, N, N, edge_dim).
        h_head_ij = h_head.unsqueeze(2).expand_as(g)  # (B, N, 1, D) -> (B, N, N, D)
        h_tail_ij = h_tail.unsqueeze(1).expand_as(g)  # (B, 1, N, D) -> (B, N, N, D)

        # Combine and update with residual.
        combined = torch.cat([h_head_ij, h_tail_ij, g], dim=-1)
        return g + self.mlp(combined)
