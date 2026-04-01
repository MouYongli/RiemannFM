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

    The Riemannian norm is computed via the product manifold's tangent_norm.

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

    # Riemannian norm squared per node.
    # Force float32 for Lorentz inner product stability under AMP.
    norm_sq = manifold.tangent_norm(
        x_t.float(), residual.float(),
    ).pow(2)  # (B, N)

    # Mask virtual nodes and average over real nodes.
    mask_float = node_mask.float()  # (B, N)
    num_real = mask_float.sum().clamp(min=1)
    loss = (norm_sq * mask_float).sum() / num_real

    return loss


def discrete_flow_loss(
    P_hat: Tensor,
    E_1: Tensor,
    node_mask: Tensor,
    avg_edge_density: float = 0.05,
    w_max: float = 10.0,
) -> Tensor:
    """Discrete flow matching loss L_disc (Def 6.8).

    Weighted binary cross-entropy per relation:
        L_disc = sum_k BCE(P_hat_k, E_1_k, weight=w_k)

    Positive class weight for relation k:
        w_k+ = min( (1 - rho_k) / rho_k, w_max )

    For the MVP, we use a uniform rho = avg_edge_density.

    Args:
        P_hat: Predicted edge logits (pre-sigmoid), shape ``(B, N, N, K)``.
        E_1: Target edge types, shape ``(B, N, N, K)``.
        node_mask: Bool mask, shape ``(B, N)``.
        avg_edge_density: Edge density rho for class weight computation.
        w_max: Maximum positive class weight cap.

    Returns:
        Scalar loss (mean over real node pairs and relations).
    """
    # Positive class weight.
    rho = max(avg_edge_density, 1e-6)
    w_pos = min((1.0 - rho) / rho, w_max)

    # Mask: only real node pairs.
    pair_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)  # (B, N, N)

    # Per-element BCE with logits (numerically stable).
    bce = F.binary_cross_entropy_with_logits(
        P_hat, E_1, reduction="none",
    )  # (B, N, N, K)

    # Apply positive class weighting: scale losses where E_1=1.
    weight = torch.where(E_1 > 0.5, w_pos, 1.0)
    bce_weighted = bce * weight  # (B, N, N, K)

    # Mask virtual nodes and average.
    mask_expanded = pair_mask.unsqueeze(-1)  # (B, N, N, 1)
    num_elements = mask_expanded.sum().clamp(min=1) * E_1.shape[-1]
    loss = (bce_weighted * mask_expanded).sum() / num_elements

    return loss
