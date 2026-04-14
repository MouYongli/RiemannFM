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

    The Lorentz time-like coordinate ``x_0 = sqrt(||x_{1:}||^2 + 1/|κ_h|)``
    is fully determined by the spatial coordinates and κ_h, and has
    near-zero batch variance in the small-tangent regime.  Feeding it
    through the encoder injects a shared DC mode that drives rank-1
    collapse downstream, so the H block's x_0 is sliced off before the
    MLP.  S and E blocks are passed through unchanged: sphere geometry
    is rotationally symmetric across its ambient coordinates and
    Euclidean has no constrained dimension.  Curvature information lost
    by dropping x_0 is recovered elsewhere (FiLM via ATH-Norm).

    Args:
        ambient_dim: Product manifold ambient dimension D.
        text_dim: Text embedding dimension d_c (0 to disable text).
        node_dim: Output hidden dimension for nodes.
        time_dim: Time embedding dimension (added after MLP).
        pe_dim: Random-walk positional encoding dimension (0 to disable).
        dim_h_ambient: Lorentz ambient dim (d_h + 1), 0 if H disabled.
        dim_s_ambient: Sphere ambient dim (d_s + 1), 0 if S disabled.
        dim_e: Euclidean dim, 0 if E disabled.  The three must sum to
            ``ambient_dim`` and are assumed in canonical H→S→E order.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        ambient_dim: int,
        text_dim: int,
        node_dim: int,
        time_dim: int,
        pe_dim: int = 0,
        dim_h_ambient: int = 0,
        dim_s_ambient: int = 0,
        dim_e: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pe_dim = pe_dim

        if dim_h_ambient + dim_s_ambient + dim_e != ambient_dim:
            msg = (
                f"dim_h_ambient ({dim_h_ambient}) + dim_s_ambient "
                f"({dim_s_ambient}) + dim_e ({dim_e}) must equal "
                f"ambient_dim ({ambient_dim})"
            )
            raise ValueError(msg)
        self.dim_h_ambient = dim_h_ambient
        self.dim_s_ambient = dim_s_ambient
        self.dim_e = dim_e

        # Encoded manifold dim: drop x_0 from H block (if present).
        drop_h = 1 if dim_h_ambient > 0 else 0
        encoded_manifold_dim = ambient_dim - drop_h
        in_dim = encoded_manifold_dim + text_dim + pe_dim + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, node_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )
        # Project time embedding to node_dim for additive conditioning.
        # NOTE: Deviation from Def 5.3 — adds time conditioning in the encoder
        # in addition to ATH-Norm per-layer injection (double time injection).
        self.time_proj = nn.Linear(time_dim, node_dim)

    def forward(
        self,
        x: Tensor,
        node_text: Tensor,
        node_mask: Tensor,
        t_emb: Tensor,
        node_pe: Tensor | None = None,
    ) -> Tensor:
        """Encode node inputs.

        Args:
            x: Manifold coordinates, shape ``(B, N, D)``.
            node_text: Text embeddings, shape ``(B, N, d_c)``.
            node_mask: Bool mask, shape ``(B, N)``.
            t_emb: Time embeddings.  Shape ``(B, time_dim)`` for a
                batch-scalar schedule, or ``(B, N, time_dim)`` when the
                collator assigned per-node mask labels (M_x=0 / M_c=1).
            node_pe: Random-walk positional encoding, shape ``(B, N, pe_dim)``.
                Required iff ``pe_dim > 0``.

        Returns:
            Node hidden states, shape ``(B, N, node_dim)``.
        """
        # Drop Lorentz time-like coord x_0 before feeding to MLP.
        # Order is H (ambient d_h+1), S (ambient d_s+1), E (d_e).
        if self.dim_h_ambient > 0:
            x_h_spatial = x[..., 1 : self.dim_h_ambient]
            x_rest = x[..., self.dim_h_ambient :]
            x_encoded = torch.cat([x_h_spatial, x_rest], dim=-1)
        else:
            x_encoded = x

        # Concatenate inputs: [x_encoded || c_i || pe_i || m_i]
        mask_float = node_mask.unsqueeze(-1).float()  # (B, N, 1)
        features: list[Tensor] = [x_encoded]
        if node_text.shape[-1] > 0:
            features.append(node_text)
        if self.pe_dim > 0:
            assert node_pe is not None, "pe_dim>0 requires node_pe"
            features.append(node_pe.to(x.dtype))
        features.append(mask_float)
        cat = torch.cat(features, dim=-1)  # (B, N, D + d_c + pe_dim + 1)

        h: Tensor = self.mlp(cat)  # (B, N, node_dim)
        # Add time conditioning.  Broadcast along N when scalar per batch;
        # use per-node projection when the embedding already carries N.
        t_proj = self.time_proj(t_emb)
        if t_proj.dim() == 2:
            t_proj = t_proj.unsqueeze(1)  # (B, 1, node_dim)
        h = h + t_proj
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
