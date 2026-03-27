"""Riemannian optimizer setup and gradient clipping utilities.

Sets up separate parameter groups for:
1. Euclidean parameters (standard Adam)
2. Manifold parameters (Riemannian Adam via geoopt)
3. Curvature parameters (separate, lower learning rate)
"""

import torch
import torch.nn as nn
from torch.optim import AdamW


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    curvature_lr: float = 1e-5,
    weight_decay: float = 0.01,
    use_riemannian: bool = True,
) -> torch.optim.Optimizer:
    """Build optimizer with separate parameter groups.

    Args:
        model: The model to optimize.
        lr: Base learning rate for most parameters.
        curvature_lr: Learning rate for curvature parameters.
        weight_decay: Weight decay for AdamW.
        use_riemannian: Whether to use RiemannianAdam for manifold params.

    Returns:
        Configured optimizer.
    """
    curvature_params = []
    regular_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "log_abs_curv" in name or "log_curv" in name:
            curvature_params.append(param)
        else:
            regular_params.append(param)

    param_groups = [
        {"params": regular_params, "lr": lr, "weight_decay": weight_decay},
        {"params": curvature_params, "lr": curvature_lr, "weight_decay": 0.0},
    ]

    if use_riemannian:
        try:
            from geoopt.optim import RiemannianAdam

            return RiemannianAdam(param_groups)  # type: ignore[no-any-return]
        except ImportError:
            pass

    return AdamW(param_groups)


def clip_riemannian_grad(
    model: nn.Module,
    max_norm: float = 1.0,
) -> float:
    """Clip gradients by global norm, respecting Riemannian metrics.

    For simplicity, we use the Euclidean norm of gradients as a proxy.
    A full Riemannian gradient norm would require the metric tensor,
    which is expensive to compute for all parameters.

    Args:
        model: Model whose gradients to clip.
        max_norm: Maximum gradient norm.

    Returns:
        The total gradient norm before clipping.
    """
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
