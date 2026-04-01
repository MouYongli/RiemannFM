"""Riemannian optimizer setup (Algorithm 1, step 10-11).

Wraps geoopt.optim.RiemannianAdam with dual parameter groups:
  - Model parameters: standard learning rate + weight decay.
  - Curvature parameters: separate (lower) learning rate, no weight decay.

Includes a curvature projection callback to enforce sign constraints
after each optimizer step (Algo 1, step 11):
  - kappa_h <- min(kappa_h, -eps)
  - kappa_s <- max(kappa_s, eps)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from riemannfm.manifolds.utils import CURVATURE_EPS

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


def build_optimizer(
    model: nn.Module,
    manifold: RiemannFMProductManifold,
    lr: float = 1e-4,
    curvature_lr: float = 1e-5,
    weight_decay: float = 0.01,
    use_riemannian_optim: bool = True,
) -> torch.optim.Optimizer:
    """Build optimizer with dual parameter groups.

    Group 1: All non-curvature parameters (standard LR + weight decay).
    Group 2: Curvature parameters (lower LR, no weight decay).

    Args:
        model: Full model (includes manifold as a sub-module).
        manifold: Product manifold to identify curvature params.
        lr: Base learning rate for model parameters.
        curvature_lr: Learning rate for curvature parameters.
        weight_decay: Weight decay for model parameters.
        use_riemannian_optim: Use geoopt RiemannianAdam (True) or
            standard AdamW (False).

    Returns:
        Configured optimizer.
    """
    # Identify curvature parameter IDs.
    curvature_params = set()
    for name, param in manifold.named_parameters():
        if "curvature" in name.lower() or "kappa" in name.lower():
            curvature_params.add(id(param))

    # Split into two groups.
    model_params = []
    curv_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in curvature_params:
            curv_params.append(param)
        else:
            model_params.append(param)

    param_groups = [
        {"params": model_params, "lr": lr, "weight_decay": weight_decay},
        {"params": curv_params, "lr": curvature_lr, "weight_decay": 0.0},
    ]

    if use_riemannian_optim:
        import geoopt

        return geoopt.optim.RiemannianAdam(param_groups)

    return torch.optim.AdamW(param_groups)


def project_curvatures(manifold: RiemannFMProductManifold) -> None:
    """Project curvatures to valid ranges (Algo 1, step 11).

    Enforces sign constraints after optimizer step:
      - Hyperbolic: kappa_h <= -CURVATURE_EPS
      - Spherical:  kappa_s >= +CURVATURE_EPS

    Args:
        manifold: Product manifold with learnable curvatures.
    """
    with torch.no_grad():
        if manifold.hyperbolic is not None:
            kappa = manifold.hyperbolic._curvature
            kappa.clamp_(max=-CURVATURE_EPS)
        if manifold.spherical is not None:
            kappa = manifold.spherical._curvature
            kappa.clamp_(min=CURVATURE_EPS)
