"""Top-level RiemannFM model (spec §10, §11, §17).

Wires together:
  - time embedding
  - node / relation / edge encoders (spec §10)
  - RieFormer backbone with A_V, A_R, C, D_VR, D_VE, E_V, E_R, FFN_V/R/E
    (spec §11-16)
  - prediction heads: VF head + split K+1 edge heads (spec §17)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from riemannfm.models.heads import (
    RiemannFMEdgeExHead,
    RiemannFMEdgeTypeHead,
    RiemannFMVFHead,
)
from riemannfm.models.input_encoding import (
    RiemannFMEdgeEncoder,
    RiemannFMNodeEncoder,
    RiemannFMRelationEncoder,
)
from riemannfm.models.positional import RiemannFMTimeEmbedding
from riemannfm.models.rieformer import RiemannFMRieFormer

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


class RiemannFM(nn.Module):
    """Top-level RiemannFM model.

    Args:
        manifold: Product manifold M = H × S × E.
        num_layers: Number of RieFormer blocks L.
        node_dim: d_v.
        rel_dim: d_r (relation hidden dim).
        edge_dim: d_b.
        num_heads_V / num_heads_R: Head counts per stream.
        num_edge_types: K.
        input_text_dim: Raw text embedding dim (0 disables text).
        text_proj_dim: Projected text dim used by E_V / E_R.
        rel_emb_dim: Learnable relation embedding ``R`` dim.
        pe_dim: Random-walk PE dim (0 disables).
        use_a_r / use_c / use_d_vr / use_d_ve / use_e_v / use_e_r:
            Per-module ablation toggles.
        use_geodesic_kernel: Toggle geodesic kernel in A_V.
        use_relation_similarity_bias: Toggle cosine-sim bias in A_R.
        ff_mult: FFN hidden multiplier.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        manifold: RiemannFMProductManifold,
        num_layers: int = 6,
        node_dim: int = 384,
        rel_dim: int = 128,
        edge_dim: int = 128,
        num_heads_V: int = 6,
        num_heads_R: int = 4,
        num_edge_types: int = 10,
        input_text_dim: int = 0,
        text_proj_dim: int = 256,
        pe_dim: int = 0,
        rel_emb_dim: int = 32,
        use_a_r: bool = True,
        use_c: bool = True,
        use_d_vr: bool = True,
        use_d_ve: bool = True,
        use_e_v: bool = True,
        use_e_r: bool = True,
        use_geodesic_kernel: bool = True,
        use_relation_similarity_bias: bool = True,
        ff_mult: int = 4,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.node_dim = node_dim
        self.rel_dim = rel_dim
        self.edge_dim = edge_dim
        self.num_edge_types = num_edge_types
        self.rel_emb_dim = rel_emb_dim
        self.text_proj_dim = text_proj_dim if input_text_dim > 0 else 0
        self.pe_dim = pe_dim

        ambient_dim = manifold.ambient_dim
        time_dim = node_dim

        dim_h_ambient = (
            manifold.hyperbolic.ambient_dim if manifold.hyperbolic is not None else 0
        )
        dim_s_ambient = (
            manifold.spherical.ambient_dim if manifold.spherical is not None else 0
        )
        dim_e = manifold.euclidean.ambient_dim if manifold.euclidean is not None else 0

        self._has_h_cond = manifold.hyperbolic is not None
        self._has_s_cond = manifold.spherical is not None
        cond_dim = int(self._has_h_cond) + int(self._has_s_cond)

        # Global relation embedding R (spec §5).
        self.rel_emb = nn.Parameter(torch.empty(num_edge_types, rel_emb_dim))
        nn.init.xavier_uniform_(self.rel_emb)

        # Text projection to the internal text_proj_dim. Two parallel
        # projections so that entity-text (long, descriptive) and
        # relation-text (short predicate labels) do not compete for the
        # same low-dim subspace. Each path is followed by LayerNorm to
        # stabilize the scale fed into downstream cross-attention.
        self.entity_text_proj: nn.Sequential | None
        self.relation_text_proj: nn.Sequential | None
        if input_text_dim > 0 and self.text_proj_dim > 0:
            self.entity_text_proj = nn.Sequential(
                nn.Linear(input_text_dim, self.text_proj_dim),
                nn.LayerNorm(self.text_proj_dim),
            )
            self.relation_text_proj = nn.Sequential(
                nn.Linear(input_text_dim, self.text_proj_dim),
                nn.LayerNorm(self.text_proj_dim),
            )
        else:
            self.entity_text_proj = None
            self.relation_text_proj = None

        self.time_emb = RiemannFMTimeEmbedding(time_dim)

        self.node_encoder = RiemannFMNodeEncoder(
            ambient_dim=ambient_dim,
            text_dim=self.text_proj_dim,
            node_dim=node_dim,
            time_dim=time_dim,
            pe_dim=pe_dim,
            dim_h_ambient=dim_h_ambient,
            dim_s_ambient=dim_s_ambient,
            dim_e=dim_e,
            dropout=dropout,
        )
        self.rel_encoder = RiemannFMRelationEncoder(
            rel_emb_dim=rel_emb_dim,
            text_dim=self.text_proj_dim,
            rel_dim=rel_dim,
            time_dim=time_dim,
            dropout=dropout,
        )
        self.edge_encoder = RiemannFMEdgeEncoder(
            num_edge_types=num_edge_types,
            rel_emb_dim=rel_emb_dim,
            text_dim=self.text_proj_dim,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        self.backbone = RiemannFMRieFormer(
            num_layers=num_layers,
            manifold=manifold,
            node_dim=node_dim,
            rel_dim=rel_dim,
            edge_dim=edge_dim,
            num_edge_types=num_edge_types,
            rel_emb_dim=rel_emb_dim,
            num_heads_V=num_heads_V,
            num_heads_R=num_heads_R,
            time_dim=time_dim,
            cond_dim=cond_dim,
            text_dim=self.text_proj_dim,
            dim_h_ambient=dim_h_ambient,
            dim_s_ambient=dim_s_ambient,
            dim_e=dim_e,
            use_a_r=use_a_r,
            use_c=use_c,
            use_d_vr=use_d_vr,
            use_d_ve=use_d_ve,
            use_e_v=use_e_v,
            use_e_r=use_e_r,
            use_geodesic_kernel=use_geodesic_kernel,
            use_relation_similarity_bias=use_relation_similarity_bias,
            ff_mult=ff_mult,
            dropout=dropout,
        )
        self._cond_dim = cond_dim

        # Prediction heads (spec §17).
        self.vf_head = RiemannFMVFHead(node_dim, ambient_dim, manifold)
        self.edge_ex_head = RiemannFMEdgeExHead(edge_dim)
        self.edge_type_head = RiemannFMEdgeTypeHead(
            edge_dim=edge_dim,
            rel_emb_dim=rel_emb_dim,
            num_edge_types=num_edge_types,
        )

    def _project_entity_text(self, x: Tensor) -> Tensor:
        """Project raw entity (node) text embeddings to ``text_proj_dim``."""
        if self.entity_text_proj is not None and x.shape[-1] > 0:
            out: Tensor = self.entity_text_proj(x)
            return out
        return torch.zeros(
            *x.shape[:-1], self.text_proj_dim, device=x.device, dtype=x.dtype,
        )

    def _project_relation_text(self, x: Tensor) -> Tensor:
        """Project raw relation text embeddings to ``text_proj_dim``."""
        if self.relation_text_proj is not None and x.shape[-1] > 0:
            out: Tensor = self.relation_text_proj(x)
            return out
        return torch.zeros(
            *x.shape[:-1], self.text_proj_dim, device=x.device, dtype=x.dtype,
        )

    def _build_curvature_cond(
        self, batch_size: int, device: torch.device, dtype: torch.dtype,
    ) -> Tensor | None:
        """Per-batch curvature conditioning for A_V's ATH-Norm FiLM (§12.4)."""
        if self._cond_dim == 0:
            return None
        parts: list[Tensor] = []
        if self._has_h_cond:
            parts.append(self.manifold.hyperbolic.curvature.reshape(1))  # type: ignore[union-attr]
        if self._has_s_cond:
            parts.append(self.manifold.spherical.curvature.reshape(1))  # type: ignore[union-attr]
        cond = torch.cat(parts, dim=-1).to(device=device, dtype=dtype)
        return cond.unsqueeze(0).expand(batch_size, -1)

    def forward(
        self,
        x_t: Tensor,
        E_t: Tensor,
        mu_t: Tensor,
        t: Tensor,
        node_text: Tensor,
        node_mask: Tensor,
        relation_text: Tensor | None = None,
        node_pe: Tensor | None = None,
        m_text: Tensor | None = None,
        m_coord: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full forward pass (spec §18.1).

        Args:
            x_t: Interpolated manifold coordinates, shape ``(B, N, D)``.
            E_t: Interpolated edge types, shape ``(B, N, N, K)``.
            mu_t: Edge mask indicator, shape ``(B, N, N)``.
            t: Time steps, shape ``(B,)``.
            node_text: Node text embeddings (already text-masked),
                shape ``(B, N, input_text_dim)``.
            node_mask: Real-vs-virtual bool mask, shape ``(B, N)``.
            relation_text: Relation text embeddings (spec C_R),
                shape ``(K, input_text_dim)``.
            node_pe: Random-walk PE, shape ``(B, N, pe_dim)``.
            m_text / m_coord: Per-node modality-mask bits (spec §9.3).

        Returns:
            ``(V_hat, ell_ex, ell_type, h_V)``.
        """
        node_text_proj = self._project_entity_text(node_text)
        relation_text_proj = (
            self._project_relation_text(relation_text)
            if relation_text is not None else None
        )

        t_emb = self.time_emb(t)

        h_V = self.node_encoder(
            x_t, node_text_proj, node_mask, t_emb,
            m_text=m_text, m_coord=m_coord, node_pe=node_pe,
        )
        h_R = self.rel_encoder(self.rel_emb, t_emb, relation_text_proj)
        h_E = self.edge_encoder(E_t, self.rel_emb, mu_t, relation_text_proj)

        C_V = node_text_proj if self.text_proj_dim > 0 else None
        cond = self._build_curvature_cond(x_t.shape[0], x_t.device, x_t.dtype)
        h_V, h_R, h_E = self.backbone(
            h_V, h_R, h_E, x_t, self.rel_emb, t_emb, node_mask,
            C_V=C_V, relation_text=relation_text_proj, cond=cond,
        )

        V_hat = self.vf_head(h_V, x_t)
        ell_ex = self.edge_ex_head(h_E)
        ell_type = self.edge_type_head(h_E, self.rel_emb)

        return V_hat, ell_ex, ell_type, h_V
