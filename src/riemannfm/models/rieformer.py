"""RieFormer: stack of L RieFormer blocks (spec §11–16).

Threads the three evolving streams — node ``H^V``, relation ``H^R``, and
edge ``H^E`` — through ``num_layers`` identical blocks, then applies a
final per-stream LayerNorm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from riemannfm.models.rieformer_block import RiemannFMBlock

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


class RiemannFMRieFormer(nn.Module):
    """Stack of L RieFormer transformer blocks (spec §11).

    Args:
        num_layers: L.
        manifold: Product manifold.
        node_dim / rel_dim / edge_dim: Hidden dims d_v / d_r / d_b.
        num_edge_types / rel_emb_dim: Config for A_R's bias matrix.
        num_heads_V / num_heads_R: Head counts for V- and R-side
            attentions.
        time_dim: Time embedding dim.
        cond_dim: Curvature-conditioning dim for A_V's ATH-Norm.
        text_dim: Projected text dim (0 disables E_V / E_R globally).
        dim_h_ambient / dim_s_ambient / dim_e: Product-manifold ambient
            dims per component, forwarded to edge self-update's π^T.
        use_a_r / use_c / use_d_vr / use_d_ve / use_e_v / use_e_r:
            Module-level ablation toggles.
        use_geodesic_kernel / use_relation_similarity_bias: Attention
            ablation toggles.
        ff_mult: FFN hidden multiplier.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_layers: int,
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

        self.layers = nn.ModuleList([
            RiemannFMBlock(
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
                text_dim=text_dim,
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
            for _ in range(num_layers)
        ])

        self.final_node_norm = nn.LayerNorm(node_dim)
        self.final_rel_norm = nn.LayerNorm(rel_dim)
        self.final_edge_norm = nn.LayerNorm(edge_dim)

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
        C_R: Tensor | None = None,
        cond: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass through all L blocks (spec §18.1).

        Returns:
            Final ``(h_V, h_R, h_E)`` after the L blocks + final
            per-stream LayerNorm.
        """
        for block in self.layers:
            h_V, h_R, h_E = block(
                h_V, h_R, h_E, x, R, t_emb, node_mask,
                C_V=C_V, C_R=C_R, cond=cond,
            )

        h_V = self.final_node_norm(h_V)
        h_R = self.final_rel_norm(h_R)
        h_E = self.final_edge_norm(h_E)
        return h_V, h_R, h_E
