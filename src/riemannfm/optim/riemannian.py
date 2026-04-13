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
    align_lr: float | None = None,
    weight_decay: float = 0.01,
    use_riemannian_optim: bool = True,
) -> torch.optim.Optimizer:
    """Build optimizer with three parameter groups.

    Group 1: Main model parameters (standard LR + weight decay).
    Group 2: Curvature parameters (lower LR, no weight decay).
    Group 3: InfoNCE projection heads (align_lr, no weight decay).

    The InfoNCE projection heads (proj_g, proj_c, proj_mask_c) are excluded
    from weight decay because their gradients are small relative to the
    main model and global gradient clipping reduces them further; weight
    decay dominates and shrinks them toward zero, preventing the
    contrastive losses from learning.

    Args:
        model: Full model (includes manifold as a sub-module).
        manifold: Product manifold to identify curvature params.
        lr: Base learning rate for model parameters.
        curvature_lr: Learning rate for curvature parameters.
        align_lr: Learning rate for InfoNCE projections (proj_g, proj_c,
            proj_mask_c).  Defaults to ``lr`` when None.
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

    # Identify InfoNCE projection head IDs.
    # All InfoNCE heads (proj_g / proj_c for L_align, proj_mask_c for
    # L_mask_c) share identical optimization needs: warmup-free, wd=0,
    # higher LR.  Missing any silently routes it to the backbone group
    # and stalls its learning (seen with proj_mask pre-2026-04-12).
    align_head_prefixes = ("proj_g.", "proj_c.", "proj_mask_c.")
    align_proj_params = set()
    for name, param in model.named_parameters():
        if any(p in name for p in align_head_prefixes):
            align_proj_params.add(id(param))

    # Split into three groups.
    model_params = []
    curv_params = []
    proj_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in curvature_params:
            curv_params.append(param)
        elif id(param) in align_proj_params:
            proj_params.append(param)
        else:
            model_params.append(param)

    # Regression guard: every ``proj_*`` head under ``loss_fn`` must be
    # routed to the alignment group.  Catches the case where a new
    # InfoNCE head is added but ``align_head_prefixes`` is not updated.
    for name, param in model.named_parameters():
        if (
            name.startswith("loss_fn.proj_")
            and param.requires_grad
            and id(param) not in align_proj_params
        ):
            raise RuntimeError(
                f"InfoNCE head parameter '{name}' is not routed to the "
                f"alignment group; update `align_head_prefixes` in "
                f"riemannfm.optim.riemannian.build_optimizer."
            )

    _align_lr = align_lr if align_lr is not None else lr
    param_groups = [
        {"params": model_params, "lr": lr, "weight_decay": weight_decay},
        {"params": curv_params, "lr": curvature_lr, "weight_decay": 0.0},
        {"params": proj_params, "lr": _align_lr, "weight_decay": 0.0},
    ]

    if use_riemannian_optim:
        import geoopt

        optimizer: torch.optim.Optimizer = geoopt.optim.RiemannianAdam(param_groups)
        return optimizer

    return torch.optim.AdamW(param_groups)


def project_curvatures(manifold: RiemannFMProductManifold) -> None:
    """Project curvatures to valid ranges (Algo 1, step 11).

    Enforces sign constraints after optimizer step:
      - Hyperbolic: kappa_h <= -CURVATURE_EPS
      - Spherical:  kappa_s >= +CURVATURE_EPS

    ``Tensor.clamp_`` does not replace ``NaN`` values, so we explicitly
    sanitize first to recover from any upstream poisoning (e.g. a NaN
    optimizer step) instead of leaving the manifold in an unrecoverable
    state.

    Args:
        manifold: Product manifold with learnable curvatures.
    """
    with torch.no_grad():
        if manifold.hyperbolic is not None:
            kappa = manifold.hyperbolic._curvature
            kappa.nan_to_num_(nan=-1.0, posinf=-CURVATURE_EPS, neginf=-1.0)
            kappa.clamp_(max=-CURVATURE_EPS)
        if manifold.spherical is not None:
            kappa = manifold.spherical._curvature
            kappa.nan_to_num_(nan=1.0, posinf=1.0, neginf=CURVATURE_EPS)
            kappa.clamp_(min=CURVATURE_EPS)
