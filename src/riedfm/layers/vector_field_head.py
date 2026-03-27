"""Output heads for flow matching: vector field prediction + edge type prediction.

RieDFMContinuousVFHead: predicts tangent vectors on the product manifold.
RieDFMDiscreteEdgeHead: predicts edge type probabilities.
"""

import torch
import torch.nn as nn
from torch import Tensor

from riedfm.manifolds.product import RieDFMProductManifold
from riedfm.utils.manifold_utils import lorentz_inner


class RieDFMContinuousVFHead(nn.Module):
    """Predicts tangent vectors (velocity field) on the product manifold.

    Takes node hidden states and produces a vector in the tangent space at
    the current position, split into H/S/E components with appropriate projections:
    - Hyperbolic: project to ensure Lorentz-orthogonality to x
    - Spherical: project to ensure orthogonality to x
    - Euclidean: no projection needed

    Args:
        hidden_dim: Input hidden dimension.
        manifold: Product manifold for tangent space projections.
    """

    def __init__(self, hidden_dim: int, manifold: RieDFMProductManifold):
        super().__init__()
        self.manifold = manifold
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, manifold.total_dim),
        )

    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        """
        Args:
            h: Node hidden states, shape (N, hidden_dim).
            x: Current node positions on manifold, shape (N, total_dim).

        Returns:
            Tangent vectors at x, shape (N, total_dim).
        """
        v_raw = self.mlp(h)  # (N, total_dim)

        # Split into sub-manifold components
        v_h, v_s, v_e = self.manifold.split(v_raw)
        x_h, x_s, _x_e = self.manifold.split(x)

        # Project hyperbolic component: v_h += <v_h, x_h>_L * x_h
        # This ensures <v_h, x_h>_L = 0 (tangent to hyperboloid)
        vx_inner = lorentz_inner(v_h, x_h).unsqueeze(-1)
        v_h = v_h + vx_inner * x_h

        # Project spherical component: v_s -= <v_s, x_s> * x_s
        # This ensures v_s is tangent to the sphere
        vs_inner = (v_s * x_s).sum(dim=-1, keepdim=True)
        v_s = v_s - vs_inner * x_s

        # Euclidean: no projection needed
        return self.manifold.combine(v_h, v_s, v_e)


class RieDFMDiscreteEdgeHead(nn.Module):
    """Predicts edge type probabilities for discrete flow matching.

    Takes edge hidden states and produces logits over K+1 edge types
    (including type 0 = no edge).

    Args:
        edge_dim: Edge feature dimension.
        num_edge_types: Total number of edge types (K+1).
    """

    def __init__(self, edge_dim: int, num_edge_types: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, num_edge_types),
        )

    def forward(self, h_e: Tensor) -> Tensor:
        """
        Args:
            h_e: Edge hidden states, shape (N, N, edge_dim) or (num_edges, edge_dim).

        Returns:
            Edge type logits, same spatial shape + (num_edge_types,).
        """
        result: Tensor = self.mlp(h_e)
        return result


class RieDFMCardinalityHead(nn.Module):
    """Predicts the number of tail entities for one-to-many relations.

    Given a head entity embedding and relation embedding, predicts
    a distribution over possible cardinalities.

    Args:
        hidden_dim: Input hidden dimension.
        max_cardinality: Maximum predicted cardinality.
    """

    def __init__(self, hidden_dim: int, max_cardinality: int = 64):
        super().__init__()
        self.max_cardinality = max_cardinality
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_cardinality),
        )

    def forward(self, h_head: Tensor, r_embed: Tensor) -> Tensor:
        """
        Args:
            h_head: Head entity embedding, shape (B, hidden_dim).
            r_embed: Relation embedding, shape (B, hidden_dim).

        Returns:
            Cardinality logits, shape (B, max_cardinality).
        """
        result: Tensor = self.mlp(torch.cat([h_head, r_embed], dim=-1))
        return result
