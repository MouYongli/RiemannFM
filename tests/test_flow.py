"""Tests for RiemannFM flow matching (spec §7, §8).

Noise sampling, geodesic interpolation endpoints, vector-field target
direction, and the joint flow orchestrator.  Masked discrete flow is
covered in ``test_discrete_flow.py``.
"""

from __future__ import annotations

import pytest
import torch

from riemannfm.flow.continuous_flow import (
    geodesic_interpolation,
    sample_time,
    vector_field_target,
)
from riemannfm.flow.joint_flow import FlowMatchingSample, RiemannFMJointFlow
from riemannfm.flow.noise import sample_continuous_noise
from riemannfm.manifolds.product import RiemannFMProductManifold

B = 8
N = 6
K = 5
DIM_H = 4
DIM_S = 4
DIM_E = 4
ATOL = 5e-3


@pytest.fixture()
def manifold() -> RiemannFMProductManifold:
    return RiemannFMProductManifold(DIM_H, DIM_S, DIM_E)


class TestContinuousNoise:
    def test_shape(self, manifold: RiemannFMProductManifold) -> None:
        x = sample_continuous_noise(manifold, B, N)
        assert x.shape == (B, N, manifold.ambient_dim)

    def test_on_manifold(self, manifold: RiemannFMProductManifold) -> None:
        x = sample_continuous_noise(manifold, B, N)
        x_proj = manifold.proj_manifold(x)
        assert torch.allclose(x, x_proj, atol=1e-4)


class TestGeodesicInterpolation:
    def test_t0_gives_noise(self, manifold: RiemannFMProductManifold) -> None:
        x_0 = manifold.sample_noise(B, N)
        x_1 = manifold.sample_noise(B, N)
        t = torch.zeros(B)
        x_t = geodesic_interpolation(manifold, x_0, x_1, t)
        assert torch.allclose(x_t, x_0, atol=ATOL)

    def test_t1_gives_data(self, manifold: RiemannFMProductManifold) -> None:
        x_0 = manifold.sample_noise(B, N, radius_h=1.0, sigma_e=0.5)
        x_1 = manifold.sample_noise(B, N, radius_h=1.0, sigma_e=0.5)
        t = torch.ones(B)
        x_t = geodesic_interpolation(manifold, x_0, x_1, t)
        d = manifold.dist(x_t, x_1)
        assert d.max() < 0.1, f"max dist at t=1: {d.max():.4f}"

    def test_on_manifold(self, manifold: RiemannFMProductManifold) -> None:
        x_0 = manifold.sample_noise(B, N)
        x_1 = manifold.sample_noise(B, N)
        t = torch.rand(B)
        x_t = geodesic_interpolation(manifold, x_0, x_1, t)
        x_t_proj = manifold.proj_manifold(x_t)
        assert torch.allclose(x_t, x_t_proj, atol=1e-3)

    def test_monotonic_distance(self, manifold: RiemannFMProductManifold) -> None:
        x_0 = manifold.sample_noise(B, N, radius_h=1.0, sigma_e=0.5)
        x_1 = manifold.sample_noise(B, N, radius_h=1.0, sigma_e=0.5)
        d_t03 = manifold.dist(
            geodesic_interpolation(manifold, x_0, x_1, torch.full((B,), 0.3)),
            x_1,
        )
        d_t07 = manifold.dist(
            geodesic_interpolation(manifold, x_0, x_1, torch.full((B,), 0.7)),
            x_1,
        )
        assert (d_t07 <= d_t03 + 1e-4).all()


class TestVectorFieldTarget:
    def test_direction_toward_data(self, manifold: RiemannFMProductManifold) -> None:
        x_0 = manifold.sample_noise(B, N)
        x_1 = manifold.sample_noise(B, N)
        t = torch.full((B,), 0.5)
        x_t = geodesic_interpolation(manifold, x_0, x_1, t)
        u_t = vector_field_target(manifold, x_t, x_1, t)
        eps = 0.01
        x_step = manifold.exp_map(x_t, eps * u_t)
        d_before = manifold.dist(x_t, x_1)
        d_after = manifold.dist(x_step, x_1)
        assert (d_after <= d_before + 1e-3).all()

    def test_clamping_near_t1(self, manifold: RiemannFMProductManifold) -> None:
        x_0 = manifold.sample_noise(B, N)
        x_1 = manifold.sample_noise(B, N)
        t = torch.full((B,), 0.999)
        x_t = geodesic_interpolation(manifold, x_0, x_1, t)
        u_t = vector_field_target(manifold, x_t, x_1, t, t_max=0.999)
        assert torch.isfinite(u_t).all()

    def test_shape(self, manifold: RiemannFMProductManifold) -> None:
        x_t = manifold.sample_noise(B, N)
        x_1 = manifold.sample_noise(B, N)
        t = torch.rand(B)
        u_t = vector_field_target(manifold, x_t, x_1, t)
        assert u_t.shape == x_t.shape


class TestSampleTime:
    def test_shape_uniform(self) -> None:
        assert sample_time(B, distribution="uniform").shape == (B,)

    def test_range_uniform(self) -> None:
        t = sample_time(1000, distribution="uniform")
        assert (t >= 0).all()
        assert (t < 1).all()

    def test_logit_normal_in_range(self) -> None:
        t = sample_time(1000, distribution="logit_normal")
        assert (t > 0).all()
        assert (t < 1).all()

    def test_beta_biased_toward_one(self) -> None:
        # Beta(5, 1) is heavily right-biased (spec §9.6 text-mask mode).
        t = sample_time(4096, distribution="beta", beta_a=5.0, beta_b=1.0)
        assert (t > 0).all()
        assert (t < 1).all()
        assert t.mean().item() > 0.6


class TestJointFlow:
    def test_sample_output_type(self, manifold: RiemannFMProductManifold) -> None:
        flow = RiemannFMJointFlow(manifold)
        x_1 = manifold.sample_noise(B, N)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)
        sample = flow.sample(x_1, E_1, mask)
        assert isinstance(sample, FlowMatchingSample)

    def test_sample_shapes(self, manifold: RiemannFMProductManifold) -> None:
        D = manifold.ambient_dim
        flow = RiemannFMJointFlow(manifold)
        x_1 = manifold.sample_noise(B, N)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)
        s = flow.sample(x_1, E_1, mask)
        assert s.x_t.shape == (B, N, D)
        assert s.u_t.shape == (B, N, D)
        assert s.E_t.shape == (B, N, N, K)
        assert s.mu_t.shape == (B, N, N)
        assert s.E_1.shape == (B, N, N, K)
        assert s.t.shape == (B,)
        assert s.node_mask.shape == (B, N)

    def test_disable_continuous(self, manifold: RiemannFMProductManifold) -> None:
        flow = RiemannFMJointFlow(manifold, disable_continuous=True)
        x_1 = manifold.sample_noise(B, N)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)
        s = flow.sample(x_1, E_1, mask)
        assert torch.allclose(s.x_t, x_1)
        assert (s.u_t == 0).all()

    def test_disable_discrete(self, manifold: RiemannFMProductManifold) -> None:
        flow = RiemannFMJointFlow(manifold, disable_discrete=True)
        x_1 = manifold.sample_noise(B, N)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)
        s = flow.sample(x_1, E_1, mask)
        assert torch.allclose(s.E_t, E_1)
        assert (s.mu_t == 0).all()

    def test_finite_outputs(self, manifold: RiemannFMProductManifold) -> None:
        flow = RiemannFMJointFlow(manifold)
        x_1 = manifold.sample_noise(B, N)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)
        s = flow.sample(x_1, E_1, mask)
        assert torch.isfinite(s.x_t).all()
        assert torch.isfinite(s.u_t).all()
        assert torch.isfinite(s.E_t).all()
        assert torch.isfinite(s.mu_t).all()
