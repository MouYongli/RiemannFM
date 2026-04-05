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
    """Edge prediction head with relation Transformer (Def 5.18-5.19).

    For each edge (i,j) and relation k:
      1. Construct candidate: r_{ij}^(k) = MLP_proj([g_{ij} || c_{r_k}])
      2. Relation Transformer: self-attention over K candidates
      3. Classify: P_{ij}^(k) = sigmoid(w_cls^T r_tilde + b_cls)

    Args:
        edge_dim: Edge hidden dimension.
        num_edge_types: Number of relation types K.
        d_c: Relation text embedding dimension (0 to disable text injection).
        n_rel_layers: Number of relation Transformer layers.
        n_rel_heads: Number of attention heads in relation Transformer.
    """

    def __init__(
        self,
        edge_dim: int,
        num_edge_types: int,
        d_c: int = 0,
        n_rel_layers: int = 2,
        n_rel_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_edge_types = num_edge_types
        self.d_c = d_c

        # Step 1: per-relation candidate projection.
        proj_in = edge_dim + d_c if d_c > 0 else edge_dim
        self.rel_proj = nn.Sequential(
            nn.Linear(proj_in, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
        )

        # Step 2: relation Transformer (self-attention over K candidates).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=edge_dim,
            nhead=n_rel_heads,
            dim_feedforward=edge_dim * 2,
            batch_first=True,
            activation="gelu",
        )
        self.rel_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_rel_layers,
        )

        # Step 3: per-relation binary classifier.
        self.w_cls = nn.Linear(edge_dim, 1)

    def forward(self, g: Tensor, C_R: Tensor | None = None) -> Tensor:
        """Predict edge probabilities with relation interaction.

        Args:
            g: Edge hidden states, shape ``(B, N, N, edge_dim)``.
            C_R: Relation text embeddings, shape ``(K, d_c)``.
                None if text conditioning is disabled.

        Returns:
            Edge logits P_hat (pre-sigmoid), shape ``(B, N, N, K)``.
            Apply sigmoid for probabilities.
        """
        B, N, _, D = g.shape
        K = self.num_edge_types

        # Expand g to per-relation: (B, N, N, 1, D) -> (B, N, N, K, D).
        g_exp = g.unsqueeze(3).expand(-1, -1, -1, K, -1)

        if C_R is not None and self.d_c > 0:
            # Expand C_R: (K, d_c) -> (1, 1, 1, K, d_c) -> (B, N, N, K, d_c).
            c_r = C_R.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, N, -1, -1)
            proj_input = torch.cat([g_exp, c_r], dim=-1)  # (B, N, N, K, D + d_c)
        else:
            proj_input = g_exp

        # Step 1: per-relation candidates.
        r = self.rel_proj(proj_input)  # (B, N, N, K, D)

        # Step 2: relation Transformer — self-attention over K dim.
        r_flat = r.view(B * N * N, K, D)
        r_flat = self.rel_transformer(r_flat)  # (B*N*N, K, D)

        # Step 3: classify each relation independently.
        logits = self.w_cls(r_flat).squeeze(-1)  # (B*N*N, K)
        result: Tensor = logits.view(B, N, N, K)
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
            edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)  # (B, N, N)
            attn_scores = attn_scores.masked_fill(~edge_mask, float("-inf"))
        alpha = F.softmax(attn_scores, dim=2)             # softmax over j
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
