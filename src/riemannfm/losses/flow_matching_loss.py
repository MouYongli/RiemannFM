"""Flow matching losses (spec §7.4, §8.5).

  - L_X   : continuous node flow matching, Riemannian MSE of the tangent
            residual (def 7.2).
  - L_ex  : edge existence BCE on masked positions (def 8.2).
  - L_ty  : edge type BCE on masked-and-positive positions (def 8.3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


def continuous_flow_loss(
    manifold: RiemannFMProductManifold,
    V_hat: Tensor,
    u_t: Tensor,
    x_t: Tensor,
    node_mask: Tensor,
) -> Tensor:
    """Node continuous flow matching loss L_X (spec def 7.2).

    ``L_X = (1/Z) · Σ_i m_i · ‖V̂_i − u_{t,i}‖²_{T_{x_t}M}``

    The squared norm is computed via :meth:`tangent_norm_sq` (no
    intermediate ``sqrt`` — avoids NaN gradients when residuals vanish).

    Args:
        manifold: Product manifold for tangent norm computation.
        V_hat: Predicted vector field, shape ``(B, N, D)``.
        u_t: Target vector field, shape ``(B, N, D)``.
        x_t: Base points, shape ``(B, N, D)``.
        node_mask: Bool/float gating mask, shape ``(B, N)``.

    Returns:
        Scalar loss (mean over gated nodes).
    """
    residual = V_hat - u_t  # (B, N, D)

    norm_sq = manifold.tangent_norm_sq(
        x_t.float(), residual.float(),
    )  # (B, N)

    mask_float = node_mask.to(norm_sq.dtype)
    denom = mask_float.sum().clamp(min=1)
    return (norm_sq * mask_float).sum() / denom


def _pair_mask(node_mask: Tensor) -> Tensor:
    """Valid node-pair mask m_i · m_j, shape ``(B, N, N)``."""
    m = node_mask.to(torch.float32)
    return m.unsqueeze(2) * m.unsqueeze(1)


def edge_existence_loss(
    ell_ex: Tensor,
    E_1: Tensor,
    node_mask: Tensor,
    mu_t: Tensor,
) -> Tensor:
    """Edge existence BCE on masked positions (spec def 8.2).

    ``L_ex = (1/Z_ex) · Σ_{ij} m_i m_j · μ_{t,ij} · BCE(ℓ̂^ex_{ij}, e_{1,ij})``

    Only masked positions (``μ_{t,ij} = 1``) contribute — decoded
    positions are conditional inputs, not prediction targets.

    Args:
        ell_ex: Existence logits, shape ``(B, N, N)``.
        E_1: Data edge types, shape ``(B, N, N, K)``.
        node_mask: Bool mask, shape ``(B, N)``.
        mu_t: Mask indicator, shape ``(B, N, N)`` (1 = masked).

    Returns:
        Scalar loss.
    """
    e_true = (E_1.sum(dim=-1) > 0.5).to(ell_ex.dtype)  # (B, N, N)
    weight = _pair_mask(node_mask).to(ell_ex.dtype) * mu_t.to(ell_ex.dtype)

    if weight.sum() < 0.5:
        return torch.tensor(0.0, device=ell_ex.device, dtype=ell_ex.dtype)

    loss_elem = F.binary_cross_entropy_with_logits(
        ell_ex, e_true, reduction="none",
    )
    return (loss_elem * weight).sum() / weight.sum()


def edge_type_loss(
    ell_type: Tensor,
    E_1: Tensor,
    node_mask: Tensor,
    mu_t: Tensor,
) -> Tensor:
    """Edge type BCE on masked-and-positive positions (spec def 8.3).

    ``L_ty = (1/Z_ty) · Σ_{ij} m_i m_j · μ_{t,ij} · e_{1,ij}
              · Σ_k BCE(ℓ̂^(k)_{ij}, E_{1,ij}^(k))``

    Double-gated: only masked positions where the truth has at least one
    active relation contribute. Type learning signal is not diluted by
    the overwhelming negative pairs.

    Args:
        ell_type: Per-relation type logits, shape ``(B, N, N, K)``.
        E_1: Data edge types, shape ``(B, N, N, K)``.
        node_mask: Bool mask, shape ``(B, N)``.
        mu_t: Mask indicator, shape ``(B, N, N)``.

    Returns:
        Scalar loss.
    """
    has_edge = (E_1.sum(dim=-1) > 0.5).to(ell_type.dtype)  # (B, N, N)
    pair_gate = _pair_mask(node_mask).to(ell_type.dtype)
    position_weight = (
        pair_gate * mu_t.to(ell_type.dtype) * has_edge
    )  # (B, N, N)

    if position_weight.sum() < 0.5:
        return torch.tensor(0.0, device=ell_type.device, dtype=ell_type.dtype)

    # Per-channel BCE over K relations, reduced by mean over K so the
    # scale is comparable to L_ex.
    loss_elem = F.binary_cross_entropy_with_logits(
        ell_type, E_1.to(ell_type.dtype), reduction="none",
    )  # (B, N, N, K)
    loss_per_pos = loss_elem.mean(dim=-1)  # (B, N, N)
    return (loss_per_pos * position_weight).sum() / position_weight.sum()
