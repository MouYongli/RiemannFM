"""Combined pretraining loss (spec §19).

``L = λ_X · L_X + λ_ex · L_ex + λ_ty · L_ty + λ_align^R · L_align^R``

  - L_X       : node continuous flow matching (def 7.2).
  - L_ex      : edge existence BCE on masked positions (def 8.2).
  - L_ty      : edge type BCE on masked-and-positive positions (def 8.3).
  - L_align^R : optional relation-text InfoNCE (def 19.1).

Node-text alignment and text-/geometry-masked reconstruction losses
were removed per spec §9.7 — modality masking data augmentation covers
the same role as the old L_align / L_mask_c / L_mask_x.

Parameter-free orchestrator: projections for the optional
``L_align^R`` live in ``riemannfm.models.pretrain_heads``; this module
consumes pre-projected ``z_R, z_C`` tensors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from riemannfm.losses.flow_matching_loss import (
    continuous_flow_loss,
    edge_existence_loss,
    edge_type_loss,
)
from riemannfm.losses.relation_align import relation_align_infonce

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


class RiemannFMCombinedLoss(nn.Module):
    """Combined pretraining loss (spec §19).

    Args:
        manifold: Product manifold for the tangent norm in L_X.
        lambda_X: Weight for node flow matching (default 1.0).
        lambda_ex: Weight for edge existence (default 1.0).
        lambda_ty: Weight for edge type (default 0.5).
        lambda_align_R: Weight for relation-text alignment (default 0.02).
        align_tau: Temperature for L_align^R (default 0.1).
    """

    def __init__(
        self,
        manifold: RiemannFMProductManifold,
        lambda_X: float = 1.0,
        lambda_ex: float = 1.0,
        lambda_ty: float = 0.5,
        lambda_align_R: float = 0.0,
        align_tau: float = 0.1,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.lambda_X = lambda_X
        self.lambda_ex = lambda_ex
        self.lambda_ty = lambda_ty
        self.lambda_align_R = lambda_align_R
        self.align_tau = align_tau

    def forward(
        self,
        V_hat: Tensor,
        u_t: Tensor,
        x_t: Tensor,
        ell_ex: Tensor,
        ell_type: Tensor,
        E_1: Tensor,
        mu_t: Tensor,
        node_mask: Tensor,
        m_coord: Tensor | None = None,
        z_R: Tensor | None = None,
        z_C: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the combined loss.

        Args:
            V_hat: Predicted vector field, shape ``(B, N, D)``.
            u_t: Target vector field, shape ``(B, N, D)``.
            x_t: Interpolated coordinates, shape ``(B, N, D)``.
            ell_ex: Edge existence logits, shape ``(B, N, N)``.
            ell_type: Edge type logits, shape ``(B, N, N, K)``.
            E_1: Target edge types, shape ``(B, N, N, K)``.
            mu_t: Mask indicator, shape ``(B, N, N)``.
            node_mask: Real-vs-virtual bool mask, shape ``(B, N)``.
            m_coord: Per-node coordinate-mask gate for L_X, shape
                ``(B, N)``. ``None`` uses ``node_mask`` (all real nodes
                participate). Stage 2 wires this from modality masking.
            z_R: Pre-projected relation embeddings, shape ``(K, d_p)``.
                Omitted when ``lambda_align_R == 0``.
            z_C: Pre-projected relation text embeddings, shape
                ``(K, d_p)``.

        Returns:
            ``(total_loss, metrics)``.
        """
        l_x_mask = node_mask if m_coord is None else (node_mask & m_coord)
        l_X = continuous_flow_loss(self.manifold, V_hat, u_t, x_t, l_x_mask)
        l_ex = edge_existence_loss(ell_ex, E_1, node_mask, mu_t)
        l_ty = edge_type_loss(ell_type, E_1, node_mask, mu_t)

        if self.lambda_align_R > 0 and z_R is not None and z_C is not None:
            l_align_R = relation_align_infonce(z_R, z_C, tau=self.align_tau)
        else:
            l_align_R = torch.tensor(
                0.0, device=V_hat.device, dtype=V_hat.dtype,
            )

        total = (
            self.lambda_X * l_X
            + self.lambda_ex * l_ex
            + self.lambda_ty * l_ty
            + self.lambda_align_R * l_align_R
        )

        metrics = {
            "loss/total": total.detach(),
            "loss/X": l_X.detach(),
            "loss/ex": l_ex.detach(),
            "loss/ty": l_ty.detach(),
            "loss/align_R": l_align_R.detach(),
        }
        return total, metrics
