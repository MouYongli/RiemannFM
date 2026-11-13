"""Prediction heads for RiemannFM (spec §17).

  - ``RiemannFMVFHead``      : node tangent-vector head (def 17.1).
  - ``RiemannFMEdgeExHead``  : edge existence logit (def 17.2).
  - ``RiemannFMEdgeTypeHead``: edge type logits, bilinear w/ relation
                               embeddings (def 17.2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


class RiemannFMVFHead(nn.Module):
    """Vector field prediction head (spec def 17.1).

    ``V̂_i = Proj_{T_{x_i}M}( MLP(h_i) )``

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
        v_raw = self.mlp(h)
        return self.manifold.proj_tangent(x, v_raw)


class RiemannFMEdgeExHead(nn.Module):
    """Edge existence head (spec def 17.2, existence logit).

    ``ℓ̂^ex_{ij} = w_ex^T · h_E_{ij}^{(L)} + b_ex``

    Args:
        edge_dim: Edge hidden dimension.
    """

    def __init__(self, edge_dim: int) -> None:
        super().__init__()
        self.w_ex = nn.Linear(edge_dim, 1, bias=True)

    def forward(self, h_E: Tensor) -> Tensor:
        """Predict existence logits.

        Args:
            h_E: Edge hidden states, shape ``(B, N, N, edge_dim)``.

        Returns:
            Existence logits (pre-sigmoid), shape ``(B, N, N)``.
        """
        out: Tensor = self.w_ex(h_E).squeeze(-1)
        return out


class RiemannFMEdgeTypeHead(nn.Module):
    """Edge type head with bilinear relation matching (spec def 17.2).

    ``ℓ̂^(k)_{ij} = (h_E_{ij}^{(L)})^T · W_type · r_k + b_k``

    Uses the raw relation-embedding parameter ``R`` (not the evolving
    relation hidden ``H^R``), so the head stays inductive for new
    relations at inference time.

    Args:
        edge_dim: Edge hidden dimension ``d_b``.
        rel_emb_dim: Relation embedding dimension ``d_r``.
        num_edge_types: Number of relation types ``K``.
    """

    def __init__(
        self,
        edge_dim: int,
        rel_emb_dim: int,
        num_edge_types: int,
    ) -> None:
        super().__init__()
        self.W_type = nn.Parameter(torch.empty(edge_dim, rel_emb_dim))
        nn.init.xavier_uniform_(self.W_type)
        self.b_type = nn.Parameter(torch.zeros(num_edge_types))

    def forward(self, h_E: Tensor, R: Tensor) -> Tensor:
        """Predict per-relation type logits.

        Args:
            h_E: Edge hidden states, shape ``(B, N, N, edge_dim)``.
            R: Relation embedding parameter, shape ``(K, rel_emb_dim)``.

        Returns:
            Type logits (pre-sigmoid), shape ``(B, N, N, K)``.
        """
        return torch.einsum("bijd,dr,kr->bijk", h_E, self.W_type, R) + self.b_type
