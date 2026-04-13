"""Tests for optimizer construction and parameter-group routing.

Regression guard: every InfoNCE projection head under ``loss_fn``
(proj_g, proj_c, proj_mask_c, and any future head) must be routed to
the alignment parameter group so it shares warmup-free scheduling,
higher LR, and wd=0.  A previous filter only matched
``proj_g.``/``proj_c.``, silently dumping ``proj_mask`` into the
backbone group and stalling L_mask at ln(M) for 8000+ steps; this
test exists to prevent the same class of bug when new heads are added.
"""

from __future__ import annotations

import pytest
from torch import nn

from riemannfm.losses.combined_loss import RiemannFMCombinedLoss
from riemannfm.manifolds.product import RiemannFMProductManifold
from riemannfm.optim.riemannian import build_optimizer


class _ToyModel(nn.Module):
    """Minimal model exposing a ``loss_fn`` submodule with InfoNCE heads."""

    def __init__(self, manifold: RiemannFMProductManifold) -> None:
        super().__init__()
        self.manifold = manifold
        self.backbone = nn.Linear(16, 16)
        self.loss_fn = RiemannFMCombinedLoss(
            manifold,
            mu_align=1.0,
            nu_mask_c=1.0,
            nu_mask_x=1.0,
            input_text_dim=32,
            node_dim=16,
        )


def _manifold() -> RiemannFMProductManifold:
    return RiemannFMProductManifold(dim_hyperbolic=4, dim_spherical=4, dim_euclidean=4)


class TestProjHeadRouting:
    """All InfoNCE projection heads must land in the alignment group."""

    def test_proj_heads_in_align_group(self) -> None:
        manifold = _manifold()
        model = _ToyModel(manifold)
        opt = build_optimizer(
            model=model,
            manifold=manifold,
            lr=3e-6,
            curvature_lr=1e-6,
            align_lr=1e-4,
            weight_decay=1e-3,
            use_riemannian_optim=False,
        )
        # pg2 is the alignment group (model, curv, align).
        align_ids = {id(p) for p in opt.param_groups[2]["params"]}

        for name, param in model.loss_fn.named_parameters():
            if name.startswith(("proj_g.", "proj_c.", "proj_mask_c.")):
                assert id(param) in align_ids, (
                    f"{name} missing from align group; "
                    f"routing filter is stale."
                )

    def test_align_group_has_align_lr_and_no_wd(self) -> None:
        manifold = _manifold()
        model = _ToyModel(manifold)
        opt = build_optimizer(
            model=model, manifold=manifold,
            lr=3e-6, curvature_lr=1e-6, align_lr=1e-4, weight_decay=1e-3,
            use_riemannian_optim=False,
        )
        assert opt.param_groups[2]["lr"] == 1e-4
        assert opt.param_groups[2]["weight_decay"] == 0.0

    def test_align_lr_defaults_to_base_lr(self) -> None:
        manifold = _manifold()
        model = _ToyModel(manifold)
        opt = build_optimizer(
            model=model, manifold=manifold,
            lr=5e-5, curvature_lr=1e-6, align_lr=None, weight_decay=1e-3,
            use_riemannian_optim=False,
        )
        assert opt.param_groups[2]["lr"] == 5e-5

    def test_missing_head_raises(self) -> None:
        """Adding a new ``proj_xxx`` head without updating the filter
        should fail loudly at optimizer construction."""
        manifold = _manifold()
        model = _ToyModel(manifold)
        # Sneak in an extra projection head that the filter does not match.
        model.loss_fn.add_module("proj_rogue", nn.Linear(16, 16, bias=False))
        with pytest.raises(RuntimeError, match="not routed to the alignment group"):
            build_optimizer(
                model=model, manifold=manifold,
                lr=3e-6, curvature_lr=1e-6, align_lr=1e-4, weight_decay=1e-3,
                use_riemannian_optim=False,
            )

    def test_backbone_params_not_in_align_group(self) -> None:
        manifold = _manifold()
        model = _ToyModel(manifold)
        opt = build_optimizer(
            model=model, manifold=manifold,
            lr=3e-6, curvature_lr=1e-6, align_lr=1e-4, weight_decay=1e-3,
            use_riemannian_optim=False,
        )
        align_ids = {id(p) for p in opt.param_groups[2]["params"]}
        for name, param in model.backbone.named_parameters():
            assert id(param) not in align_ids, f"backbone.{name} leaked into align group"


class TestCurvatureRouting:
    def test_curvature_in_own_group(self) -> None:
        manifold = _manifold()
        model = _ToyModel(manifold)
        opt = build_optimizer(
            model=model, manifold=manifold,
            lr=3e-6, curvature_lr=1e-6, align_lr=1e-4, weight_decay=1e-3,
            use_riemannian_optim=False,
        )
        curv_ids = {id(p) for p in opt.param_groups[1]["params"]}
        for name, param in manifold.named_parameters():
            if "curvature" in name.lower() or "kappa" in name.lower():
                assert id(param) in curv_ids, f"curvature param {name} misrouted"
        assert opt.param_groups[1]["lr"] == 1e-6
        assert opt.param_groups[1]["weight_decay"] == 0.0
