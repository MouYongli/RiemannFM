"""Tests for RiemannFMPretrainHeads (entity_emb + mask_emb + L_align^R heads)."""

from __future__ import annotations

import pytest
import torch

from riemannfm.manifolds.product import RiemannFMProductManifold
from riemannfm.models.pretrain_heads import RiemannFMPretrainHeads


def _manifold() -> RiemannFMProductManifold:
    return RiemannFMProductManifold(dim_hyperbolic=4, dim_spherical=4, dim_euclidean=4)


class TestConstruction:
    def test_align_heads_gated_off_without_text(self) -> None:
        heads = RiemannFMPretrainHeads(
            manifold=_manifold(),
            num_entities=8,
            input_text_dim=0,
            lambda_align_R=1.0,
        )
        assert heads.W_p is None
        assert heads.W_p_c is None

    def test_align_heads_gated_off_when_weight_zero(self) -> None:
        heads = RiemannFMPretrainHeads(
            manifold=_manifold(),
            num_entities=8,
            input_text_dim=32,
            lambda_align_R=0.0,
        )
        assert heads.W_p is None
        assert heads.W_p_c is None

    def test_align_heads_present_when_enabled(self) -> None:
        heads = RiemannFMPretrainHeads(
            manifold=_manifold(),
            num_entities=8,
            input_text_dim=32,
            lambda_align_R=0.05,
            d_p=24,
            rel_emb_dim=16,
        )
        assert heads.W_p is not None
        assert heads.W_p_c is not None


class TestProjectRelationAlign:
    def test_shapes(self) -> None:
        K, d_r, d_c, d_p = 7, 16, 32, 24
        heads = RiemannFMPretrainHeads(
            manifold=_manifold(),
            num_entities=8,
            input_text_dim=d_c,
            lambda_align_R=0.05,
            d_p=d_p,
            rel_emb_dim=d_r,
        )
        R = torch.randn(K, d_r)
        relation_text = torch.randn(K, d_c)
        z_R, z_C = heads.project_relation_align(R, relation_text)
        assert z_R.shape == (K, d_p)
        assert z_C.shape == (K, d_p)

    def test_asserts_when_disabled(self) -> None:
        heads = RiemannFMPretrainHeads(
            manifold=_manifold(),
            num_entities=8,
            input_text_dim=0,
            lambda_align_R=0.05,
        )
        with pytest.raises(AssertionError):
            heads.project_relation_align(torch.randn(4, 16), torch.randn(4, 0))


class TestEntityAndMaskEmb:
    def test_entity_emb_seeded_around_origin(self) -> None:
        manifold = _manifold()
        heads = RiemannFMPretrainHeads(
            manifold=manifold,
            num_entities=32,
            input_text_dim=16,
        )
        origin = manifold.origin(device=heads.entity_emb.weight.device)
        mean = heads.entity_emb.weight.mean(dim=0)
        assert torch.allclose(mean, origin, atol=0.05)

    def test_mask_emb_shape(self) -> None:
        heads = RiemannFMPretrainHeads(
            manifold=_manifold(),
            num_entities=8,
            input_text_dim=32,
        )
        assert heads.mask_emb.shape == (32,)

    def test_mask_emb_placeholder_when_no_text(self) -> None:
        heads = RiemannFMPretrainHeads(
            manifold=_manifold(),
            num_entities=8,
            input_text_dim=0,
        )
        assert heads.mask_emb.shape == (1,)
