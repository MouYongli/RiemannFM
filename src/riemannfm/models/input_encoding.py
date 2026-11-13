"""Input encoding for RiemannFM (Def 5.3-5.4).

Node encoder:  h_i^0 = MLP([pi(x_i) || c_i || m_i]) + t_emb
Edge encoder:  g_{ij}^0 = MLP([E_{ij,t} * W_rel || E_{ij,t} * relation_text_mean])

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
    MLP.  The S block keeps all d_s+1 ambient coords (sphere is
    rotationally symmetric across them — hard-dropping the first would
    break O(d_s+1) symmetry), but is passed through a LayerNorm to
    remove the transient DC mode caused by points clustering near the
    anchor pole at init (s_0 ≈ 1/√κ_s with low batch variance).  The E
    block has no constrained dimension; a LayerNorm is applied for
    scale consistency with S.  Curvature information stripped by the
    block-wise normalization is recovered elsewhere (FiLM via ATH-Norm).

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

        # Block-wise input normalization: LN on S (kill s_0 DC, keep
        # O(d_s+1) symmetry) and E (scale consistency).  H block is
        # already handled by x_0 slicing; LN-ing x_{1:} would distort
        # Lorentz geometry.
        self.ln_s = nn.LayerNorm(dim_s_ambient) if dim_s_ambient > 0 else None
        self.ln_e = nn.LayerNorm(dim_e) if dim_e > 0 else None

        # Encoded manifold dim: drop x_0 from H block (if present).
        drop_h = 1 if dim_h_ambient > 0 else 0
        encoded_manifold_dim = ambient_dim - drop_h
        # Three per-node scalar bits (spec def 10.4):
        #   - node_mask     : real (1) vs virtual (0)
        #   - m_text        : text kept (1) vs replaced by mask_emb (0)
        #   - m_coord       : coord from data (1) vs held at prior (0)
        in_dim = encoded_manifold_dim + text_dim + pe_dim + 3
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
        m_text: Tensor | None = None,
        m_coord: Tensor | None = None,
        node_pe: Tensor | None = None,
    ) -> Tensor:
        """Encode node inputs (spec def 10.4).

        Args:
            x: Manifold coordinates, shape ``(B, N, D)``.
            node_text: Text embeddings (already text-masked by caller when
                applicable), shape ``(B, N, d_c)``.
            node_mask: Real-vs-virtual bool mask, shape ``(B, N)``.
            t_emb: Time embeddings.  Shape ``(B, time_dim)`` for a
                batch-scalar schedule, or ``(B, N, time_dim)`` when a
                per-node time is used.
            m_text: Text-visibility bit, shape ``(B, N)``. ``None`` is
                treated as all-ones.
            m_coord: Coord-visibility bit, shape ``(B, N)``. ``None`` is
                treated as all-ones.
            node_pe: Random-walk positional encoding,
                shape ``(B, N, pe_dim)``. Required iff ``pe_dim > 0``.

        Returns:
            Node hidden states, shape ``(B, N, node_dim)``.
        """
        # Drop Lorentz time-like coord x_0, then apply block-wise LN to
        # S and E.  Order is H (ambient d_h+1), S (ambient d_s+1), E (d_e).
        h_end = self.dim_h_ambient
        s_end = h_end + self.dim_s_ambient
        parts: list[Tensor] = []
        if self.dim_h_ambient > 0:
            parts.append(x[..., 1:h_end])
        if self.dim_s_ambient > 0:
            assert self.ln_s is not None
            parts.append(self.ln_s(x[..., h_end:s_end]))
        if self.dim_e > 0:
            assert self.ln_e is not None
            parts.append(self.ln_e(x[..., s_end:]))
        x_encoded = torch.cat(parts, dim=-1)

        B, N = node_mask.shape
        mask_float = node_mask.unsqueeze(-1).to(x.dtype)  # (B, N, 1)
        if m_text is None:
            m_text_bit = torch.ones(B, N, 1, dtype=x.dtype, device=x.device)
        else:
            m_text_bit = m_text.unsqueeze(-1).to(x.dtype)
        if m_coord is None:
            m_coord_bit = torch.ones(B, N, 1, dtype=x.dtype, device=x.device)
        else:
            m_coord_bit = m_coord.unsqueeze(-1).to(x.dtype)

        features: list[Tensor] = [x_encoded]
        if node_text.shape[-1] > 0:
            features.append(node_text)
        if self.pe_dim > 0:
            assert node_pe is not None, "pe_dim>0 requires node_pe"
            features.append(node_pe.to(x.dtype))
        features.append(mask_float)
        features.append(m_text_bit)
        features.append(m_coord_bit)
        cat = torch.cat(features, dim=-1)

        h: Tensor = self.mlp(cat)
        t_proj = self.time_proj(t_emb)
        if t_proj.dim() == 2:
            t_proj = t_proj.unsqueeze(1)
        h = h + t_proj
        return h


class RiemannFMEdgeEncoder(nn.Module):
    """Encode edge inputs into initial hidden representations (spec def 10.6).

    ``h_{ij}^{E,(0)} = MLP_edge([E_t·R ‖ E_t·relation_text ‖ μ_{t,ij}])``

    The relation embedding ``R`` is owned at the top level (shared with
    the edge type head, see ``RiemannFM.rel_emb``) and passed at forward
    time so there is a single global matrix.

    Args:
        num_edge_types: Number of relation types K.
        rel_emb_dim: Dimension of learnable relation embeddings.
        text_dim: Relation text embedding dimension d_c (0 disables).
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
        self.rel_emb_dim = rel_emb_dim
        # +1 for μ_{t,ij} bit (spec def 10.6).
        in_dim = rel_emb_dim + (text_dim if text_dim > 0 else 0) + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, edge_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim, edge_dim),
        )

    def forward(
        self,
        E_t: Tensor,
        R: Tensor,
        mu_t: Tensor,
        relation_text: Tensor | None = None,
    ) -> Tensor:
        """Encode edge inputs.

        Args:
            E_t: Interpolated edge types, shape ``(B, N, N, K)``.
            R: Global relation embedding, shape ``(K, rel_emb_dim)``.
            mu_t: Mask indicator (1 = masked), shape ``(B, N, N)``.
            relation_text: Relation text embeddings, shape ``(K, d_c)``. ``None``
                when text conditioning is disabled.

        Returns:
            Edge hidden states, shape ``(B, N, N, edge_dim)``.
        """
        rel_feat = E_t @ R  # (B, N, N, rel_emb_dim)

        features: list[Tensor] = [rel_feat]
        if relation_text is not None and relation_text.shape[-1] > 0:
            features.append(E_t @ relation_text)  # (B, N, N, d_c)
        features.append(mu_t.to(rel_feat.dtype).unsqueeze(-1))  # (B, N, N, 1)
        edge_input = torch.cat(features, dim=-1)

        out: Tensor = self.mlp(edge_input)
        return out


class RiemannFMRelationEncoder(nn.Module):
    """Encode relation inputs into initial hidden representations (spec def 10.5).

    ``h_k^{R,(0)} = MLP_rel([r_k ‖ c_{r_k}]) + W_tp^R · t_emb``

    Args:
        rel_emb_dim: Dimension of learnable relation embeddings d_r.
        text_dim: Relation text embedding dimension d_c (0 disables).
        rel_dim: Output relation-hidden dimension (= d_r by default).
        time_dim: Time embedding dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        rel_emb_dim: int,
        text_dim: int,
        rel_dim: int,
        time_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rel_emb_dim = rel_emb_dim
        in_dim = rel_emb_dim + (text_dim if text_dim > 0 else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, rel_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(rel_dim, rel_dim),
        )
        self.time_proj = nn.Linear(time_dim, rel_dim)

    def forward(
        self,
        R: Tensor,
        t_emb: Tensor,
        relation_text: Tensor | None = None,
    ) -> Tensor:
        """Encode relation inputs.

        Args:
            R: Relation embedding parameter, shape ``(K, rel_emb_dim)``.
            t_emb: Time embedding, shape ``(B, time_dim)``.
            relation_text: Relation text embeddings, shape ``(K, d_c)``. ``None``
                when text conditioning is disabled.

        Returns:
            Relation hidden states, shape ``(B, K, rel_dim)``.
        """
        parts: list[Tensor] = [R]
        if relation_text is not None and relation_text.shape[-1] > 0:
            parts.append(relation_text.to(R.dtype))
        rel_in = torch.cat(parts, dim=-1)  # (K, in_dim)

        rel_hidden: Tensor = self.mlp(rel_in)  # (K, rel_dim)
        # Broadcast to batch: (B, K, rel_dim).
        B = t_emb.shape[0]
        rel_hidden = rel_hidden.unsqueeze(0).expand(B, -1, -1)

        t_proj: Tensor = self.time_proj(t_emb).unsqueeze(1)  # (B, 1, rel_dim)
        out: Tensor = rel_hidden + t_proj
        return out
