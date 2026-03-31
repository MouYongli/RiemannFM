"""Edge-Node Dual-Stream Cross-Interaction.

Bidirectional information exchange between node features and edge features:
- Edge -> Node: node aggregates information from its incident edges
- Node -> Edge: edge receives information from its endpoint nodes

This allows edge type predictions to inform node coordinate predictions
and vice versa, enabling joint coherent generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RiemannFMDualStreamInteraction(nn.Module):
    """Bidirectional cross-interaction between node and edge streams.

    Edge -> Node:
        h_i += MLP(sum_j alpha_{ij}^{E->V} * h_{ij}^E)

    Node -> Edge:
        h_{ij}^E += MLP(h_i^V || h_j^V || h_i^V * h_j^V)

    Args:
        node_dim: Node feature dimension.
        edge_dim: Edge feature dimension.
        dropout: Dropout rate.
    """

    def __init__(self, node_dim: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        # Edge -> Node: attention-weighted aggregation
        self.e2n_attn = nn.Linear(edge_dim, 1)
        self.e2n_value = nn.Linear(edge_dim, node_dim)
        self.e2n_proj = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.e2n_norm = nn.LayerNorm(node_dim)

        # Node -> Edge: endpoint feature fusion
        # Input: concat(h_i, h_j, h_i * h_j) = 3 * node_dim
        self.n2e_proj = nn.Sequential(
            nn.Linear(node_dim * 3, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.n2e_norm = nn.LayerNorm(edge_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_v: Tensor,
        h_e: Tensor,
        edge_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            h_v: Node features, shape (N, node_dim).
            h_e: Edge features, shape (N, N, edge_dim).
            edge_mask: Boolean mask for valid edges, shape (N, N).

        Returns:
            (h_v_updated, h_e_updated): Updated node and edge features.
        """
        N = h_v.shape[0]

        # === Edge -> Node ===
        # Compute attention weights for each node over its incident edges
        attn_logits = self.e2n_attn(h_e).squeeze(-1)  # (N, N)
        if edge_mask is not None:
            attn_logits = attn_logits.masked_fill(~edge_mask, float("-inf"))
        attn_weights = F.softmax(attn_logits, dim=1)  # (N, N), softmax over source j
        attn_weights = self.dropout(attn_weights)

        # Aggregate edge values weighted by attention
        edge_values = self.e2n_value(h_e)  # (N, N, node_dim)
        node_update = torch.einsum("ij,ijd->id", attn_weights, edge_values)  # (N, node_dim)
        h_v_new = self.e2n_norm(h_v + self.e2n_proj(node_update))

        # === Node -> Edge ===
        # For each edge (i,j), combine endpoint features
        h_i = h_v_new.unsqueeze(1).expand(-1, N, -1)  # (N, N, node_dim)
        h_j = h_v_new.unsqueeze(0).expand(N, -1, -1)  # (N, N, node_dim)
        h_ij = torch.cat([h_i, h_j, h_i * h_j], dim=-1)  # (N, N, 3*node_dim)
        edge_update = self.n2e_proj(h_ij)  # (N, N, edge_dim)
        h_e_new = self.n2e_norm(h_e + edge_update)

        return h_v_new, h_e_new
