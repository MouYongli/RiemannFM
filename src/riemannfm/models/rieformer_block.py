"""Single RieFormer transformer block (spec §11).

Eight-stage composition per spec §11.1 (Pre-Norm residuals throughout):

  1. [A_V]  Node manifold self-attention (spec §12)
  2. [A_R]  Relation self-attention (spec §13)
  3. [C]    Edge self-update, MLP on [h_i ‖ h_j ‖ π^T(log_xi xj) ‖ h_E]
            (spec §14)
  4. [D_VR] Bidirectional node ↔ relation cross-attention (spec §15.1)
  5. [D_VE] Node ← edge cross-attention + edge ← node MLP (spec §15.2)
  6. [E_V]  Node-text cross-attention (spec §16.1)
  7. [E_R]  Relation-text cross-attention (spec §16.2)
  8. [FFN]  Independent feed-forwards for V, R, E (spec §16.4)

Only A_V's ATH-Norm receives the curvature conditioning vector (§12.4);
the rest use ATH-Norm without the curvature channel (§13.2, §15, §16).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from riemannfm.models.attention.cross import (
    RiemannFMNodeEdgeCross,
    RiemannFMNodeRelationCross,
)
from riemannfm.models.attention.edge import (
    RiemannFMEdgeBias,
    RiemannFMEdgeSelfUpdate,
)
from riemannfm.models.attention.geodesic import RiemannFMGeodesicAttention
from riemannfm.models.attention.relation import RiemannFMRelationSelfAttention
from riemannfm.models.normalization import RiemannFMATHNorm
from riemannfm.models.text_condition import (
    RiemannFMNodeTextCross,
    RiemannFMRelationTextCross,
)

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


class RiemannFMFeedForward(nn.Module):
    """Two-layer FFN with SiLU (spec def 11.2).

    Args:
        dim: Input / output dimension.
        mult: Hidden-dim multiplier (default 4).
        dropout: Dropout rate.
    """

    def __init__(
        self, dim: int, mult: int = 4, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.net(x)
        return out


class RiemannFMBlock(nn.Module):
    """Single RieFormer transformer block (spec §11.1).

    Args:
        manifold: Product manifold.
        node_dim: d_v.
        rel_dim: d_r (relation hidden dim).
        edge_dim: d_b.
        num_edge_types: K.
        rel_emb_dim: Raw R dim (= d_r by default).
        num_heads_V: Heads for A_V, D_VR (V-side), D_VE (V-side), E_V.
        num_heads_R: Heads for A_R, D_VR (R-side), E_R.
        time_dim: ATH-Norm time embedding dim.
        cond_dim: Curvature-conditioning dim for A_V's ATH-Norm only.
        text_dim: Projected text dim (0 disables E_V / E_R in this block).
        dim_h_ambient / dim_s_ambient / dim_e: Product-manifold ambient
            dims per component, used by C's π^T.
        use_a_r / use_c / use_d_vr / use_d_ve / use_e_v / use_e_r:
            Per-module ablation toggles (all default True).
        use_geodesic_kernel: Forwarded to A_V.
        use_relation_similarity_bias: Forwarded to A_R.
        ff_mult: FFN hidden multiplier.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        manifold: RiemannFMProductManifold,
        node_dim: int,
        rel_dim: int,
        edge_dim: int,
        num_edge_types: int,
        rel_emb_dim: int,
        num_heads_V: int,
        num_heads_R: int,
        time_dim: int,
        cond_dim: int = 0,
        text_dim: int = 0,
        dim_h_ambient: int = 0,
        dim_s_ambient: int = 0,
        dim_e: int = 0,
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
    ) -> None:
        super().__init__()
        self.use_a_r = use_a_r
        self.use_c = use_c
        self.use_d_vr = use_d_vr
        self.use_d_ve = use_d_ve
        self.use_e_v = use_e_v and text_dim > 0
        self.use_e_r = use_e_r and text_dim > 0

        # [A_V] Node manifold self-attention (curvature-conditioned ATH-Norm).
        self.norm_a_v = RiemannFMATHNorm(node_dim, time_dim, cond_dim=cond_dim)
        self.a_v = RiemannFMGeodesicAttention(
            node_dim, num_heads_V, manifold,
            use_geodesic_kernel=use_geodesic_kernel,
            dropout=dropout,
        )
        self.edge_bias = RiemannFMEdgeBias(edge_dim, num_heads_V)

        # [A_R]
        if self.use_a_r:
            self.norm_a_r = RiemannFMATHNorm(rel_dim, time_dim, cond_dim=0)
            self.a_r = RiemannFMRelationSelfAttention(
                rel_dim=rel_dim,
                num_heads=num_heads_R,
                rel_emb_dim=rel_emb_dim,
                use_similarity_bias=use_relation_similarity_bias,
                dropout=dropout,
            )

        # [C]
        if self.use_c:
            self.norm_c = RiemannFMATHNorm(edge_dim, time_dim, cond_dim=0)
            self.edge_update = RiemannFMEdgeSelfUpdate(
                manifold=manifold,
                node_dim=node_dim,
                edge_dim=edge_dim,
                dim_h_ambient=dim_h_ambient,
                dim_s_ambient=dim_s_ambient,
                dim_e=dim_e,
                dropout=dropout,
            )

        # [D_VR]
        if self.use_d_vr:
            self.norm_d_vr_v = RiemannFMATHNorm(node_dim, time_dim, cond_dim=0)
            self.norm_d_vr_r = RiemannFMATHNorm(rel_dim, time_dim, cond_dim=0)
            self.d_vr = RiemannFMNodeRelationCross(
                node_dim=node_dim,
                rel_dim=rel_dim,
                num_heads_V=num_heads_V,
                num_heads_R=num_heads_R,
                dropout=dropout,
            )

        # [D_VE]
        if self.use_d_ve:
            self.norm_d_ve_v = RiemannFMATHNorm(node_dim, time_dim, cond_dim=0)
            self.norm_d_ve_e = RiemannFMATHNorm(edge_dim, time_dim, cond_dim=0)
            self.d_ve = RiemannFMNodeEdgeCross(
                node_dim=node_dim,
                edge_dim=edge_dim,
                num_heads=num_heads_V,
                dropout=dropout,
            )

        # [E_V]
        if self.use_e_v:
            self.norm_e_v = RiemannFMATHNorm(node_dim, time_dim, cond_dim=0)
            self.e_v = RiemannFMNodeTextCross(
                node_dim=node_dim,
                text_dim=text_dim,
                num_heads=num_heads_V,
                dropout=dropout,
            )

        # [E_R]
        if self.use_e_r:
            self.norm_e_r = RiemannFMATHNorm(rel_dim, time_dim, cond_dim=0)
            self.e_r = RiemannFMRelationTextCross(
                rel_dim=rel_dim,
                text_dim=text_dim,
                num_heads=num_heads_R,
                dropout=dropout,
            )

        # [FFN_V / FFN_R / FFN_E]
        self.norm_ff_v = RiemannFMATHNorm(node_dim, time_dim, cond_dim=0)
        self.norm_ff_r = RiemannFMATHNorm(rel_dim, time_dim, cond_dim=0)
        self.norm_ff_e = RiemannFMATHNorm(edge_dim, time_dim, cond_dim=0)
        self.ff_v = RiemannFMFeedForward(node_dim, mult=ff_mult, dropout=dropout)
        self.ff_r = RiemannFMFeedForward(rel_dim, mult=ff_mult, dropout=dropout)
        self.ff_e = RiemannFMFeedForward(edge_dim, mult=ff_mult, dropout=dropout)

    def forward(
        self,
        h_V: Tensor,
        h_R: Tensor,
        h_E: Tensor,
        x: Tensor,
        R: Tensor,
        t_emb: Tensor,
        node_mask: Tensor,
        C_V: Tensor | None = None,
        relation_text: Tensor | None = None,
        cond: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass through one RieFormer block (spec §11.1).

        Args:
            h_V: Node hidden, shape ``(B, N, d_v)``.
            h_R: Relation hidden, shape ``(B, K, d_r)``.
            h_E: Edge hidden, shape ``(B, N, N, d_b)``.
            x: Manifold coordinates, shape ``(B, N, D)``.
            R: Raw relation embedding parameter ``(K, rel_emb_dim)`` (used
                by A_R's cosine-similarity bias).
            t_emb: Time embedding, shape ``(B, time_dim)``.
            node_mask: Bool mask, shape ``(B, N)``.
            C_V: Projected node text (already text-masked),
                shape ``(B, N, text_dim)`` or ``None``.
            relation_text: Projected relation text, shape ``(K, text_dim)`` or
                ``(B, K, text_dim)`` or ``None``.
            cond: ATH-Norm curvature conditioning ``(B, cond_dim)`` for A_V.

        Returns:
            Updated ``(h_V, h_R, h_E)``.
        """
        # [A_V]: node self-attention with M-RoPE + geodesic kernel + edge bias.
        bias = self.edge_bias(h_E)
        h_V_bar = self.norm_a_v(h_V, t_emb, cond=cond)
        h_V = h_V + self.a_v(h_V_bar, x, bias, node_mask)

        # [A_R]: relation self-attention.
        if self.use_a_r:
            h_R_bar = self.norm_a_r(h_R, t_emb)
            h_R = h_R + self.a_r(h_R_bar, R)

        # [C]: edge self-update uses the **post-A_V** node hidden.
        if self.use_c:
            h_E_bar = self.norm_c(h_E, t_emb)
            h_E = h_E + self.edge_update(h_E_bar, h_V, x)

        # [D_VR]: bidirectional node ↔ relation cross.
        if self.use_d_vr:
            v_bar = self.norm_d_vr_v(h_V, t_emb)
            r_bar = self.norm_d_vr_r(h_R, t_emb)
            dv, dr = self.d_vr(v_bar, r_bar, node_mask)
            h_V = h_V + dv
            h_R = h_R + dr

        # [D_VE]: node ← edge cross-attn + edge ← node MLP.
        if self.use_d_ve:
            v_bar = self.norm_d_ve_v(h_V, t_emb)
            e_bar = self.norm_d_ve_e(h_E, t_emb)
            dv, de = self.d_ve(v_bar, e_bar, h_V, node_mask)
            h_V = h_V + dv
            h_E = h_E + de

        # [E_V]: node text conditioning.
        if self.use_e_v and C_V is not None:
            v_bar = self.norm_e_v(h_V, t_emb)
            h_V = h_V + self.e_v(v_bar, C_V, node_mask)

        # [E_R]: relation text conditioning.
        if self.use_e_r and relation_text is not None:
            r_bar = self.norm_e_r(h_R, t_emb)
            h_R = h_R + self.e_r(r_bar, relation_text)

        # [FFN] per-stream.
        h_V = h_V + self.ff_v(self.norm_ff_v(h_V, t_emb))
        h_R = h_R + self.ff_r(self.norm_ff_r(h_R, t_emb))
        h_E = h_E + self.ff_e(self.norm_ff_e(h_E, t_emb))

        # Mask virtual-node rows so they don't contribute downstream.
        h_V = h_V * node_mask.unsqueeze(-1)

        return h_V, h_R, h_E
