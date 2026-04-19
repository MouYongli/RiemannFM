"""Tests for optimizer construction and parameter-group routing (spec §20).

Three groups: main, curvature, relation.
  - main     : backbone + input/encoder MLPs + entity_emb + mask_emb.
  - curvature: learnable κ_h, κ_s only.
  - relation : global rel_emb (owned by the model) + optional align
               projections W_p, W_p_c (under pretrain_heads).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from riemannfm.manifolds.product import RiemannFMProductManifold
from riemannfm.models.pretrain_heads import RiemannFMPretrainHeads
from riemannfm.optim.riemannian import build_optimizer


class _ToyModel(nn.Module):
    """Minimal module exposing a ``rel_emb`` and ``pretrain_heads``."""

    def __init__(
        self,
        manifold: RiemannFMProductManifold,
        lambda_align_R: float = 1.0,
        rel_emb_dim: int = 16,
        num_edge_types: int = 4,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.backbone = nn.Linear(16, 16)
        self.rel_emb = nn.Parameter(torch.randn(num_edge_types, rel_emb_dim))
        self.pretrain_heads = RiemannFMPretrainHeads(
            manifold=manifold,
            num_entities=8,
            input_text_dim=32,
            lambda_align_R=lambda_align_R,
            d_p=24,
            rel_emb_dim=rel_emb_dim,
        )


def _manifold() -> RiemannFMProductManifold:
    return RiemannFMProductManifold(dim_hyperbolic=4, dim_spherical=4, dim_euclidean=4)


class TestRelationRouting:
    def test_rel_emb_in_relation_group(self) -> None:
        m = _manifold()
        model = _ToyModel(m)
        opt = build_optimizer(
            model=model, manifold=m,
            lr=1e-4, curvature_lr=1e-5, relation_lr=3e-5, weight_decay=1e-3,
            use_riemannian_optim=False,
        )
        rel_ids = {id(p) for p in opt.param_groups[2]["params"]}
        assert id(model.rel_emb) in rel_ids

    def test_align_projections_in_relation_group(self) -> None:
        m = _manifold()
        model = _ToyModel(m, lambda_align_R=0.05)
        opt = build_optimizer(
            model=model, manifold=m,
            lr=1e-4, curvature_lr=1e-5, relation_lr=3e-5, weight_decay=1e-3,
            use_riemannian_optim=False,
        )
        rel_ids = {id(p) for p in opt.param_groups[2]["params"]}
        assert model.pretrain_heads.W_p is not None
        for p in model.pretrain_heads.W_p.parameters():
            assert id(p) in rel_ids
        assert model.pretrain_heads.W_p_c is not None
        for p in model.pretrain_heads.W_p_c.parameters():
            assert id(p) in rel_ids

    def test_relation_group_lr_and_no_wd(self) -> None:
        m = _manifold()
        model = _ToyModel(m)
        opt = build_optimizer(
            model=model, manifold=m,
            lr=1e-4, curvature_lr=1e-5, relation_lr=3.3e-5, weight_decay=1e-3,
            use_riemannian_optim=False,
        )
        assert opt.param_groups[2]["lr"] == pytest.approx(3.3e-5)
        assert opt.param_groups[2]["weight_decay"] == 0.0

    def test_relation_lr_defaults_to_lr_over_three(self) -> None:
        m = _manifold()
        model = _ToyModel(m)
        opt = build_optimizer(
            model=model, manifold=m,
            lr=3e-5, curvature_lr=1e-5, relation_lr=None, weight_decay=1e-3,
            use_riemannian_optim=False,
        )
        assert opt.param_groups[2]["lr"] == pytest.approx(1e-5)

    def test_entity_and_mask_emb_not_in_relation_group(self) -> None:
        m = _manifold()
        model = _ToyModel(m)
        opt = build_optimizer(
            model=model, manifold=m,
            lr=1e-4, curvature_lr=1e-5, relation_lr=3e-5, weight_decay=1e-3,
            use_riemannian_optim=False,
        )
        rel_ids = {id(p) for p in opt.param_groups[2]["params"]}
        assert id(model.pretrain_heads.entity_emb.weight) not in rel_ids
        assert id(model.pretrain_heads.mask_emb) not in rel_ids

    def test_backbone_params_not_in_relation_group(self) -> None:
        m = _manifold()
        model = _ToyModel(m)
        opt = build_optimizer(
            model=model, manifold=m,
            lr=1e-4, curvature_lr=1e-5, relation_lr=3e-5, weight_decay=1e-3,
            use_riemannian_optim=False,
        )
        rel_ids = {id(p) for p in opt.param_groups[2]["params"]}
        for name, p in model.backbone.named_parameters():
            assert id(p) not in rel_ids, f"backbone.{name} leaked into relation group"


class TestCurvatureRouting:
    def test_curvature_in_own_group(self) -> None:
        m = _manifold()
        model = _ToyModel(m)
        opt = build_optimizer(
            model=model, manifold=m,
            lr=1e-4, curvature_lr=1e-6, relation_lr=3e-5, weight_decay=1e-3,
            use_riemannian_optim=False,
        )
        curv_ids = {id(p) for p in opt.param_groups[1]["params"]}
        for name, p in m.named_parameters():
            if "curvature" in name.lower() or "kappa" in name.lower():
                assert id(p) in curv_ids, f"curvature param {name} misrouted"
        assert opt.param_groups[1]["lr"] == 1e-6
        assert opt.param_groups[1]["weight_decay"] == 0.0
