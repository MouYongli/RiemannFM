"""Prediction heads for RiemannFM (Def 5.16-5.19).

VF Head: predicts tangent vectors (velocity field) for continuous flow.
Edge Head: predicts per-relation edge probabilities for discrete flow.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
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
    """Edge prediction head via bilinear matching (Def 5.18-5.19).

    Projects edge hidden states into a matching space and computes inner
    products with relation prototypes to predict per-relation logits.
    Supports multi-relational edges (independent sigmoid per relation).

    Args:
        edge_dim: Edge hidden dimension.
        num_edge_types: Number of relation types K.
        text_proj_dim: Relation text embedding dimension (0 to disable text).
    """

    def __init__(
        self,
        edge_dim: int,
        num_edge_types: int,
        text_proj_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_edge_types = num_edge_types
        self.text_proj_dim = text_proj_dim

        # Project edge hidden states to matching space.
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
        )

        # Relation prototypes: project from text or learn from scratch.
        if text_proj_dim > 0:
            self.rel_proto_proj = nn.Sequential(
                nn.Linear(text_proj_dim, edge_dim),
                nn.SiLU(),
                nn.Linear(edge_dim, edge_dim),
            )
        else:
            # Learnable relation prototypes when no text is available.
            self.rel_prototypes = nn.Parameter(
                nn.init.xavier_uniform_(Tensor(num_edge_types, edge_dim)),
            )

        # Per-relation bias for asymmetric thresholding.
        self.rel_bias = nn.Parameter(torch.zeros(num_edge_types))

    def forward(self, g: Tensor, C_R: Tensor | None = None) -> Tensor:
        """Predict edge logits via inner product matching.

        Args:
            g: Edge hidden states, shape ``(B, N, N, edge_dim)``.
            C_R: Relation text embeddings, shape ``(K, text_proj_dim)``.
                None if text conditioning is disabled.

        Returns:
            Edge logits P_hat (pre-sigmoid), shape ``(B, N, N, K)``.
        """
        # Project edges: (B, N, N, edge_dim) -> (B, N, N, edge_dim).
        g_proj = self.edge_proj(g)

        # Build relation prototypes: (K, edge_dim).
        rel_proto = (
            self.rel_proto_proj(C_R)
            if C_R is not None and self.text_proj_dim > 0
            else self.rel_prototypes
        )

        # Inner product: (B, N, N, edge_dim) @ (edge_dim, K) -> (B, N, N, K).
        logits: Tensor = torch.einsum("bijd,kd->bijk", g_proj, rel_proto) + self.rel_bias
        return logits


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
        # Edge-to-Node: QK attention aggregation (Def 5.12).
        self.W_q_e2n = nn.Linear(node_dim, node_dim, bias=False)
        self.W_k_e2n = nn.Linear(edge_dim, node_dim, bias=False)
        self.W_v_e2n = nn.Linear(edge_dim, node_dim, bias=False)
        self.scale_e2n = 1.0 / math.sqrt(node_dim)
        self.edge_to_node = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )
        # Node-to-Edge: [h_i || h_j || h_i ⊙ h_j] (Def 5.13).
        self.node_to_edge = nn.Sequential(
            nn.Linear(3 * node_dim, edge_dim),
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

        # Edge-to-Node: QK attention aggregation (Def 5.12).
        q = self.W_q_e2n(h)                              # (B, N, d_v)
        k = self.W_k_e2n(g)                              # (B, N, N, d_v)
        v = self.W_v_e2n(g)                              # (B, N, N, d_v)
        # attn[b,i,j] = q[b,i] · k[b,i,j] / sqrt(d_v)
        attn_scores = (q.unsqueeze(2) * k).sum(-1) * self.scale_e2n  # (B, N, N)
        if node_mask is not None:
            # Use a large finite negative bias instead of -inf so that fully
            # masked rows (virtual node queries) still produce a finite softmax
            # whose backward does not poison gradients with NaN.
            edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)  # (B, N, N)
            attn_scores = attn_scores.masked_fill(~edge_mask, -1e4)
        alpha = F.softmax(attn_scores, dim=2)             # softmax over j
        if node_mask is not None:
            # Zero out virtual query rows so they contribute nothing downstream.
            alpha = alpha * node_mask.unsqueeze(2).to(alpha.dtype)
        node_agg = (alpha.unsqueeze(-1) * v).sum(dim=2)   # (B, N, d_v)
        h_update = h + self.edge_to_node(node_agg)        # (B, N, node_dim)

        # Node-to-Edge: [h_i || h_j || h_i ⊙ h_j] (Def 5.13).
        # Use h_update (post edge-to-node) per Def 5.13, not the original h.
        h_i = h_update.unsqueeze(2).expand(B, N, N, -1)   # (B, N, N, node_dim)
        h_j = h_update.unsqueeze(1).expand(B, N, N, -1)   # (B, N, N, node_dim)
        h_hadamard = h_i * h_j                             # (B, N, N, node_dim)
        node_pair = torch.cat([h_i, h_j, h_hadamard], dim=-1)  # (B, N, N, 3*node_dim)
        g_update = g + self.node_to_edge(node_pair)        # (B, N, N, edge_dim)

        return h_update, g_update
