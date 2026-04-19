"""Optimizer construction (spec §20).

Three parameter groups:
  1. ``main``     : RieFormer backbone, input/time/encoding MLPs,
                    prediction heads, ``entity_emb``, ``mask_emb``.
                    Standard LR + weight decay.
  2. ``curvature``: Learnable curvatures κ_h, κ_s. Lower LR, no WD.
  3. ``relation`` : Global relation embedding ``rel_emb`` and the
                    optional ``W_p / W_p_c`` projections for
                    ``L_align^R``. Lower LR (spec §20.2), no WD.

Curvature sign constraints (κ_h < 0, κ_s > 0) are re-enforced in
``project_curvatures`` after each optimizer step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from riemannfm.manifolds.utils import CURVATURE_EPS

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


_RELATION_PARAM_MARKERS: tuple[str, ...] = (
    ".rel_emb",
    "pretrain_heads.W_p.",
    "pretrain_heads.W_p_c.",
)


def _is_relation_param(name: str) -> bool:
    """Match parameters that belong to the relation group."""
    if name.endswith("rel_emb") or ".rel_emb" in name:
        return True
    return any(marker in name for marker in _RELATION_PARAM_MARKERS)


def build_optimizer(
    model: nn.Module,
    manifold: RiemannFMProductManifold,
    lr: float = 1e-4,
    curvature_lr: float = 1e-5,
    relation_lr: float | None = None,
    weight_decay: float = 0.01,
    use_riemannian_optim: bool = True,
) -> torch.optim.Optimizer:
    """Build optimizer with (main, curvature, relation) parameter groups.

    Args:
        model: Full LightningModule exposing the backbone and
            ``pretrain_heads``.
        manifold: Product manifold (source of curvature parameters).
        lr: Base learning rate for the main group.
        curvature_lr: Learning rate for curvature parameters.
        relation_lr: Learning rate for the relation group. Defaults to
            ``lr / 3`` (spec §20.2).
        weight_decay: Weight decay for the main group. Other groups
            have ``weight_decay = 0``.
        use_riemannian_optim: Use ``geoopt.optim.RiemannianAdam``
            (manifold-aware) vs standard ``AdamW``.

    Returns:
        Configured optimizer with three param groups in the order
        ``[main, curvature, relation]``.
    """
    curvature_ids: set[int] = set()
    for name, param in manifold.named_parameters():
        if "curvature" in name.lower() or "kappa" in name.lower():
            curvature_ids.add(id(param))

    relation_ids: set[int] = set()
    for name, param in model.named_parameters():
        if _is_relation_param(name):
            relation_ids.add(id(param))

    main_params = []
    curv_params = []
    rel_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in curvature_ids:
            curv_params.append(param)
        elif id(param) in relation_ids:
            rel_params.append(param)
        else:
            main_params.append(param)

    _rel_lr = relation_lr if relation_lr is not None else lr / 3.0
    param_groups = [
        {"params": main_params, "lr": lr, "weight_decay": weight_decay},
        {"params": curv_params, "lr": curvature_lr, "weight_decay": 0.0},
        {"params": rel_params, "lr": _rel_lr, "weight_decay": 0.0},
    ]

    if use_riemannian_optim:
        import geoopt

        optimizer: torch.optim.Optimizer = geoopt.optim.RiemannianAdam(param_groups)
        return optimizer

    return torch.optim.AdamW(param_groups)


def project_curvatures(manifold: RiemannFMProductManifold) -> None:
    """Project curvatures to valid ranges (spec Algo 1).

    Enforces κ_h ≤ -ε, κ_s ≥ +ε after the optimizer step. Sanitises
    ``NaN`` / ``Inf`` entries that could otherwise leave the manifold
    in an unrecoverable state.
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
