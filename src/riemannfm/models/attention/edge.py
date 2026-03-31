"""Edge-stream self-attention (factorized line-graph attention).

Updates edge features by attending to other edges that share endpoints.
Edge (i,j)'s neighbors are edges (i,k) for k!=j and (k,j) for k!=i.

For efficiency, uses factorized attention:
    h'_ij = MLP(h_ij + sum_k gamma_{ik->ij} * h_ik + sum_k gamma_{kj->ij} * h_kj)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RiemannFMEdgeAttention(nn.Module):
    """Factorized edge self-attention on the conceptual line graph.

    Instead of full pairwise attention between all N^2 edges (which would be O(N^4)),
    we factorize into source-sharing edges and target-sharing edges.

    Args:
        edge_dim: Edge feature dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout rate.
    """

    def __init__(self, edge_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.head_dim = edge_dim // num_heads

        # Source-sharing attention (edges from same source node)
        self.q_src = nn.Linear(edge_dim, edge_dim, bias=False)
        self.k_src = nn.Linear(edge_dim, edge_dim, bias=False)
        self.v_src = nn.Linear(edge_dim, edge_dim, bias=False)

        # Target-sharing attention (edges to same target node)
        self.q_tgt = nn.Linear(edge_dim, edge_dim, bias=False)
        self.k_tgt = nn.Linear(edge_dim, edge_dim, bias=False)
        self.v_tgt = nn.Linear(edge_dim, edge_dim, bias=False)

        self.out_proj = nn.Linear(edge_dim, edge_dim)
        self.norm = nn.LayerNorm(edge_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_e: Tensor, edge_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            h_e: Edge features, shape (N, N, edge_dim).
            edge_mask: Boolean mask for valid edges, shape (N, N). True = valid edge.

        Returns:
            Updated edge features, shape (N, N, edge_dim).
        """
        N = h_e.shape[0]
        residual = h_e

        # Source-sharing attention: for edge (i,j), attend to edges (i,k) for all k
        # This is equivalent to: for each row i, do self-attention across columns
        # Reshape to (N, N, edge_dim) -> treat as N sequences of length N
        q_s = self.q_src(h_e)  # (N, N, D)
        k_s = self.k_src(h_e)  # (N, N, D)
        v_s = self.v_src(h_e)  # (N, N, D)

        # Attention per source node (row-wise)
        scale = self.head_dim**-0.5
        attn_src = torch.bmm(q_s, k_s.transpose(1, 2)) * scale  # (N, N, N)
        if edge_mask is not None:
            attn_src = attn_src.masked_fill(~edge_mask.unsqueeze(0).expand(N, -1, -1), float("-inf"))
        attn_src = F.softmax(attn_src, dim=-1)
        attn_src = self.dropout(attn_src)
        out_src = torch.bmm(attn_src, v_s)  # (N, N, D)

        # Target-sharing attention: for edge (i,j), attend to edges (k,j) for all k
        # Transpose, do row-wise attention, transpose back
        h_e_t = h_e.transpose(0, 1)  # (N, N, D) with target as batch dim
        q_t = self.q_tgt(h_e_t)
        k_t = self.k_tgt(h_e_t)
        v_t = self.v_tgt(h_e_t)

        attn_tgt = torch.bmm(q_t, k_t.transpose(1, 2)) * scale
        if edge_mask is not None:
            attn_tgt = attn_tgt.masked_fill(~edge_mask.T.unsqueeze(0).expand(N, -1, -1), float("-inf"))
        attn_tgt = F.softmax(attn_tgt, dim=-1)
        attn_tgt = self.dropout(attn_tgt)
        out_tgt = torch.bmm(attn_tgt, v_t).transpose(0, 1)  # (N, N, D)

        # Combine
        out = self.out_proj(out_src + out_tgt)
        result: Tensor = self.norm(residual + out)
        return result
