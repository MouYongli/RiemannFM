"""Input encoding for RiemannFM (Def 5.3-5.4).

Node encoder:  h_i^0 = MLP([pi(x_i) || c_i || m_i]) + t_emb
Edge encoder:  g_{ij}^0 = MLP([E_{ij,t} * W_rel || E_{ij,t} * C_R_mean])

Where pi(x_i) is the manifold coordinate projected into the hidden space,
c_i is the text embedding, and m_i is the node mask (real vs virtual).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RiemannFMNodeEncoder(nn.Module):
    """Encode node inputs into initial hidden representations (Def 5.3).

    Concatenates [manifold_coord, text_embed, mask_scalar] and projects
    through a two-layer MLP, then adds the time embedding.

    Args:
        ambient_dim: Product manifold ambient dimension D.
        text_dim: Text embedding dimension d_c (0 to disable text).
        node_dim: Output hidden dimension for nodes.
        time_dim: Time embedding dimension (added after MLP).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        ambient_dim: int,
        text_dim: int,
        node_dim: int,
        time_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Input: [x_i (D) || c_i (d_c) || m_i (1)]
        in_dim = ambient_dim + text_dim + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, node_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )
        # Project time embedding to node_dim for additive conditioning.
        self.time_proj = nn.Linear(time_dim, node_dim)

    def forward(
        self,
        x: Tensor,
        node_text: Tensor,
        node_mask: Tensor,
        t_emb: Tensor,
    ) -> Tensor:
        """Encode node inputs.

        Args:
            x: Manifold coordinates, shape ``(B, N, D)``.
            node_text: Text embeddings, shape ``(B, N, d_c)``.
            node_mask: Bool mask, shape ``(B, N)``.
            t_emb: Time embeddings, shape ``(B, time_dim)``.

        Returns:
            Node hidden states, shape ``(B, N, node_dim)``.
        """
        # Concatenate inputs: [x_i || c_i || m_i]
        mask_float = node_mask.unsqueeze(-1).float()  # (B, N, 1)
        features = [x, mask_float]
        if node_text.shape[-1] > 0:
            features.insert(1, node_text)
        cat = torch.cat(features, dim=-1)  # (B, N, D + d_c + 1)

        h: Tensor = self.mlp(cat)  # (B, N, node_dim)
        # Add time conditioning (broadcast over N).
        h = h + self.time_proj(t_emb).unsqueeze(1)  # (B, N, node_dim)
        return h


class RiemannFMEdgeEncoder(nn.Module):
    """Encode edge inputs into initial hidden representations (Def 5.4).

    For each edge (i,j), combines the multi-hot edge type vector with
    learnable relation embeddings and optional relation text embeddings.

    Args:
        num_edge_types: Number of relation types K.
        rel_emb_dim: Dimension of learnable relation embeddings.
        text_dim: Relation text embedding dimension d_c (0 to disable).
        edge_dim: Output hidden dimension for edges.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_edge_types: int,
        rel_emb_dim: int,
        text_dim: int,
        edge_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_edge_types = num_edge_types
        # Learnable per-relation embeddings: W_rel in R^{K x rel_emb_dim}.
        self.W_rel = nn.Parameter(
            nn.init.xavier_uniform_(
                Tensor(num_edge_types, rel_emb_dim),
            ),
        )
        # Input dim: weighted sum of W_rel (rel_emb_dim) + weighted C_R (text_dim)
        in_dim = rel_emb_dim + (text_dim if text_dim > 0 else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, edge_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim, edge_dim),
        )

    def forward(
        self,
        E_t: Tensor,
        C_R: Tensor | None = None,
    ) -> Tensor:
        """Encode edge inputs.

        Args:
            E_t: Interpolated edge types, shape ``(B, N, N, K)``.
            C_R: Relation text embeddings, shape ``(K, d_c)``.
                None if text conditioning is disabled.

        Returns:
            Edge hidden states, shape ``(B, N, N, edge_dim)``.
        """
        # Weighted sum of relation embeddings: E_t @ W_rel -> (B, N, N, rel_emb_dim)
        rel_feat = E_t @ self.W_rel  # (B,N,N,K) @ (K, rel_emb_dim)

        if C_R is not None and C_R.shape[-1] > 0:
            # Weighted sum of relation text embeddings.
            text_feat = E_t @ C_R  # (B,N,N,K) @ (K, d_c) -> (B,N,N,d_c)
            edge_input = torch.cat([rel_feat, text_feat], dim=-1)
        else:
            edge_input = rel_feat

        result: Tensor = self.mlp(edge_input)  # (B, N, N, edge_dim)
        return result
