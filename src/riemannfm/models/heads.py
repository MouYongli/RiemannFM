"""Prediction heads for RiemannFM (Def 5.16-5.19).

VF Head: predicts tangent vectors (velocity field) for continuous flow.
Edge Head: predicts per-relation edge probabilities for discrete flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


class RiemannFMVFHead(nn.Module):
    """Vector field prediction head (Def 5.16-5.17).

    Predicts tangent vectors V_hat_i at each node's manifold coordinate x_i.
    The output is projected onto the tangent space at x_i via proj_tangent.

    VF(h_i) = proj_{T_{x_i} M}( MLP(h_i) )

    Args:
        node_dim: Node hidden dimension.
        ambient_dim: Product manifold ambient dimension D.
        manifold: Product manifold for tangent projection.
    """

    def __init__(
        self,
        node_dim: int,
        ambient_dim: int,
        manifold: RiemannFMProductManifold,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, ambient_dim),
        )

    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        """Predict tangent vector field.

        Args:
            h: Node hidden states, shape ``(B, N, node_dim)``.
            x: Manifold coordinates, shape ``(B, N, D)``.

        Returns:
            Predicted tangent vectors V_hat, shape ``(B, N, D)``.
        """
        v_raw = self.mlp(h)  # (B, N, D)
        return self.manifold.proj_tangent(x, v_raw)


class RiemannFMEdgeHead(nn.Module):
    """Edge prediction head (Def 5.18-5.19).

    Predicts per-relation edge probabilities P_hat_{ij,k} from edge
    hidden states.  Uses a per-relation MLP followed by sigmoid.

    P_hat_{ij,k} = sigmoid( MLP_k(g_{ij}) )

    For memory efficiency, instead of K separate MLPs, we use a single
    shared MLP with output dimension K.

    Args:
        edge_dim: Edge hidden dimension.
        num_edge_types: Number of relation types K.
    """

    def __init__(
        self,
        edge_dim: int,
        num_edge_types: int,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, num_edge_types),
        )

    def forward(self, g: Tensor) -> Tensor:
        """Predict edge probabilities.

        Args:
            g: Edge hidden states, shape ``(B, N, N, edge_dim)``.

        Returns:
            Edge logits P_hat (pre-sigmoid), shape ``(B, N, N, K)``.
            Apply sigmoid for probabilities.
        """
        result: Tensor = self.mlp(g)  # (B, N, N, K)
        return result


class RiemannFMDualStreamCross(nn.Module):
    """Edge-Node cross-interaction (Def 5.12-5.13).

    Bidirectional information exchange between node and edge hidden states:
      - Edge-to-Node: aggregate edge info to update nodes.
      - Node-to-Edge: inject node info to update edges.

    Args:
        node_dim: Node hidden dimension.
        edge_dim: Edge hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Edge-to-Node: aggregate incoming edge features.
        self.edge_to_node = nn.Sequential(
            nn.Linear(edge_dim, node_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )
        # Node-to-Edge: combine head/tail node features to update edges.
        self.node_to_edge = nn.Sequential(
            nn.Linear(2 * node_dim, edge_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim, edge_dim),
        )

    def forward(
        self,
        h: Tensor,
        g: Tensor,
        node_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Bidirectional cross-interaction.

        Args:
            h: Node hidden states, shape ``(B, N, node_dim)``.
            g: Edge hidden states, shape ``(B, N, N, edge_dim)``.
            node_mask: Bool mask, shape ``(B, N)``.

        Returns:
            Tuple of updated (h, g), same shapes as inputs.
        """
        B, N, _ = h.shape

        # Edge-to-Node: mean aggregation of incoming edges.
        if node_mask is not None:
            # Mask edges from/to virtual nodes.
            edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)  # (B, N, N)
            g_masked = g * edge_mask.unsqueeze(-1)
            # Sum incoming edges and normalize by real node count.
            node_agg = g_masked.sum(dim=2)  # (B, N, edge_dim)
            count = edge_mask.sum(dim=2, keepdim=True).clamp(min=1)  # (B, N, 1)
            node_agg = node_agg / count
        else:
            node_agg = g.mean(dim=2)  # (B, N, edge_dim)

        h_update = h + self.edge_to_node(node_agg)  # (B, N, node_dim)

        # Node-to-Edge: concat head and tail node features.
        h_i = h.unsqueeze(2).expand(B, N, N, -1)  # (B, N, N, node_dim)
        h_j = h.unsqueeze(1).expand(B, N, N, -1)  # (B, N, N, node_dim)
        node_pair = torch.cat([h_i, h_j], dim=-1)  # (B, N, N, 2*node_dim)
        g_update = g + self.node_to_edge(node_pair)  # (B, N, N, edge_dim)

        return h_update, g_update
