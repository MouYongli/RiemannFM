"""Flow matching losses: continuous (Def 6.6) and discrete (Def 6.8).

L_cont: Riemannian norm MSE of vector field prediction error.
L_disc: Weighted binary cross-entropy for edge prediction.
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
    """Continuous flow matching loss L_cont (Def 6.6).

    L_cont = (1/|V_real|) * sum_{i: m_i=1} ||V_hat_i - u_t_i||^2_{T_{x_t} M}

    The squared norm is computed via :meth:`tangent_norm_sq`, which avoids
    the ``sqrt`` of :meth:`tangent_norm`.  Going through ``sqrt`` then
    ``pow(2)`` is mathematically equivalent in the forward pass but produces
    ``NaN`` gradients whenever the residual vanishes at any token (the
    backward of ``sqrt`` at zero is ``inf``, multiplied by the mask zero).

    Args:
        manifold: Product manifold for tangent norm computation.
        V_hat: Predicted vector field, shape ``(B, N, D)``.
        u_t: Target vector field, shape ``(B, N, D)``.
        x_t: Base points for tangent norm, shape ``(B, N, D)``.
        node_mask: Bool mask, shape ``(B, N)``. True = real node.

    Returns:
        Scalar loss (mean over real nodes).
    """
    # Residual in tangent space.
    residual = V_hat - u_t  # (B, N, D)

    # Riemannian norm squared per node, computed without an intermediate sqrt.
    # Force float32 for Lorentz inner product stability under AMP.
    norm_sq = manifold.tangent_norm_sq(
        x_t.float(), residual.float(),
    )  # (B, N)

    # Mask virtual nodes and average over real nodes.
    mask_float = node_mask.float()  # (B, N)
    num_real = mask_float.sum().clamp(min=1)
    loss = (norm_sq * mask_float).sum() / num_real

    return loss


def discrete_flow_loss(
    P_hat: Tensor,
    E_1: Tensor,
    node_mask: Tensor,
    rho_k: Tensor | None = None,
    avg_edge_density: float = 0.05,
    w_max: float = 10.0,
) -> Tensor:
    """Discrete flow matching loss L_disc (Def 6.8).

    Weighted binary cross-entropy per relation:
        L_disc = sum_k BCE(P_hat_k, E_1_k, weight=w_k)

    Per-relation positive class weight:
        w_k+ = min( (1 - rho_k) / rho_k, w_max )

    Args:
        P_hat: Predicted edge logits (pre-sigmoid), shape ``(B, N, N, K)``.
        E_1: Target edge types, shape ``(B, N, N, K)``.
        node_mask: Bool mask, shape ``(B, N)``.
        rho_k: Per-relation edge density, shape ``(K,)``.
            When provided, computes per-relation weights (Def 6.8).
            Falls back to uniform ``avg_edge_density`` when None.
        avg_edge_density: Fallback uniform edge density when ``rho_k`` is None.
        w_max: Maximum positive class weight cap.

    Returns:
        Scalar loss (mean over real node pairs and relations).
    """
    K = E_1.shape[-1]

    # Per-relation positive class weights (Def 6.8).
    if rho_k is not None:
        rho = rho_k.to(device=E_1.device, dtype=E_1.dtype).clamp(min=1e-6)
        w_pos = ((1.0 - rho) / rho).clamp(max=w_max)  # (K,)
    else:
        rho_scalar = max(avg_edge_density, 1e-6)
        w_pos = torch.full(
            (K,), min((1.0 - rho_scalar) / rho_scalar, w_max),
            device=E_1.device, dtype=E_1.dtype,
        )

    # Mask: only real node pairs.
    pair_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)  # (B, N, N)

    # Per-element BCE with logits (numerically stable).
    bce = F.binary_cross_entropy_with_logits(
        P_hat, E_1, reduction="none",
    )  # (B, N, N, K)

    # Apply per-relation positive class weighting: scale losses where E_1=1.
    # w_pos shape (K,) broadcasts over (B, N, N, K).
    weight = torch.where(E_1 > 0.5, w_pos, torch.ones_like(w_pos))
    bce_weighted = bce * weight  # (B, N, N, K)

    # Mask virtual nodes and average.
    mask_expanded = pair_mask.unsqueeze(-1)  # (B, N, N, 1)
    num_elements = mask_expanded.sum().clamp(min=1) * K
    loss = (bce_weighted * mask_expanded).sum() / num_elements

    return loss
