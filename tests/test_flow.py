"""Tests for RiemannFM flow matching (Phase 2: Stream A).

Covers noise sampling, geodesic interpolation endpoints,
vector field target direction, discrete interpolation with shared mask,
and the joint flow orchestrator.
"""

from __future__ import annotations

import pytest
import torch

from riemannfm.flow.continuous_flow import (
    geodesic_interpolation,
    sample_time,
    vector_field_target,
)
from riemannfm.flow.discrete_flow import discrete_interpolation
from riemannfm.flow.joint_flow import FlowMatchingSample, RiemannFMJointFlow
from riemannfm.flow.noise import (
    compute_edge_density,
    sample_continuous_noise,
    sample_discrete_noise,
)
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


@pytest.fixture()
def ambient_dim(manifold: RiemannFMProductManifold) -> int:
    return manifold.ambient_dim


class TestContinuousNoise:
    def test_shape(self, manifold: RiemannFMProductManifold) -> None:
        x = sample_continuous_noise(manifold, B, N)
        assert x.shape == (B, N, manifold.ambient_dim)

    def test_on_manifold(self, manifold: RiemannFMProductManifold) -> None:
        x = sample_continuous_noise(manifold, B, N)
        # Re-projecting should be nearly identity.
        x_proj = manifold.proj_manifold(x)
        assert torch.allclose(x, x_proj, atol=1e-4)


class TestDiscreteNoise:
    def test_shape(self) -> None:
        E_1 = torch.zeros(B, N, N, K)
        E_0 = sample_discrete_noise(E_1, avg_edge_density=0.05)
        assert E_0.shape == (B, N, N, K)

    def test_binary(self) -> None:
        E_1 = torch.zeros(B, N, N, K)
        E_0 = sample_discrete_noise(E_1, avg_edge_density=0.5)
        assert ((E_0 == 0) | (E_0 == 1)).all()

    def test_density_roughly_correct(self) -> None:
        rho = 0.3
        E_1 = torch.zeros(1000, 4, 4, 3)
        E_0 = sample_discrete_noise(E_1, avg_edge_density=rho)
        actual = E_0.mean().item()
        assert abs(actual - rho) < 0.05, f"expected ~{rho}, got {actual}"


class TestEdgeDensity:
    def test_compute(self) -> None:
        E_1 = torch.zeros(B, N, N, K)
        E_1[:, :3, :3, 0] = 1.0  # relation 0 dense in first 3 nodes
        mask = torch.ones(B, N, dtype=torch.bool)
        rho = compute_edge_density(E_1, mask)
        assert rho.shape == (K,)
        assert rho[0] > rho[1]  # relation 0 should have higher density


class TestGeodesicInterpolation:
    def test_t0_gives_noise(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        x_0 = manifold.sample_noise(B, N)
        x_1 = manifold.sample_noise(B, N)
        t = torch.zeros(B)
        x_t = geodesic_interpolation(manifold, x_0, x_1, t)
        assert torch.allclose(x_t, x_0, atol=ATOL)

    def test_t1_gives_data(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        # Use small radius to keep points close for numerical precision.
        x_0 = manifold.sample_noise(B, N, radius_h=1.0, sigma_e=0.5)
        x_1 = manifold.sample_noise(B, N, radius_h=1.0, sigma_e=0.5)
        t = torch.ones(B)
        x_t = geodesic_interpolation(manifold, x_0, x_1, t)
        # Check geodesic distance is near zero at t=1.
        d = manifold.dist(x_t, x_1)
        assert d.max() < 0.1, f"max dist at t=1: {d.max():.4f}"

    def test_on_manifold(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        x_0 = manifold.sample_noise(B, N)
        x_1 = manifold.sample_noise(B, N)
        t = torch.rand(B)
        x_t = geodesic_interpolation(manifold, x_0, x_1, t)
        x_t_proj = manifold.proj_manifold(x_t)
        assert torch.allclose(x_t, x_t_proj, atol=1e-3)

    def test_monotonic_distance(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        # Use small radius to keep points close enough for monotonic interpolation
        # (avoids wrapping issues on the sphere when points are > pi apart).
        x_0 = manifold.sample_noise(B, N, radius_h=1.0, sigma_e=0.5)
        x_1 = manifold.sample_noise(B, N, radius_h=1.0, sigma_e=0.5)
        # Distance to x_1 should decrease as t increases.
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
    def test_direction_toward_data(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        x_0 = manifold.sample_noise(B, N)
        x_1 = manifold.sample_noise(B, N)
        t = torch.full((B,), 0.5)
        x_t = geodesic_interpolation(manifold, x_0, x_1, t)
        u_t = vector_field_target(manifold, x_t, x_1, t)
        # The VF should point from x_t toward x_1.
        # Moving a small step along u_t should decrease distance to x_1.
        eps = 0.01
        x_step = manifold.exp_map(x_t, eps * u_t)
        d_before = manifold.dist(x_t, x_1)
        d_after = manifold.dist(x_step, x_1)
        assert (d_after <= d_before + 1e-3).all()

    def test_clamping_near_t1(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        x_0 = manifold.sample_noise(B, N)
        x_1 = manifold.sample_noise(B, N)
        t = torch.full((B,), 0.999)
        x_t = geodesic_interpolation(manifold, x_0, x_1, t)
        u_t = vector_field_target(manifold, x_t, x_1, t, t_max=0.999)
        # Should not produce NaN/Inf.
        assert torch.isfinite(u_t).all()

    def test_shape(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        x_t = manifold.sample_noise(B, N)
        x_1 = manifold.sample_noise(B, N)
        t = torch.rand(B)
        u_t = vector_field_target(manifold, x_t, x_1, t)
        assert u_t.shape == x_t.shape


class TestDiscreteInterpolation:
    def test_t0_gives_noise(self) -> None:
        E_0 = torch.zeros(B, N, N, K)
        E_1 = torch.ones(B, N, N, K)
        t = torch.zeros(B)
        E_t = discrete_interpolation(E_0, E_1, t)
        # At t=0, z_ij ~ Bernoulli(0) = 0, so E_t = E_0.
        assert torch.allclose(E_t, E_0)

    def test_t1_gives_data(self) -> None:
        E_0 = torch.zeros(B, N, N, K)
        E_1 = torch.ones(B, N, N, K)
        t = torch.ones(B)
        E_t = discrete_interpolation(E_0, E_1, t)
        # At t=1, z_ij ~ Bernoulli(1) = 1, so E_t = E_1.
        assert torch.allclose(E_t, E_1)

    def test_shared_mask_across_relations(self) -> None:
        # When E_0=0 and E_1 varies, z_ij should be the same for all k.
        E_0 = torch.zeros(B, N, N, K)
        E_1 = torch.ones(B, N, N, K)
        t = torch.full((B,), 0.5)
        E_t = discrete_interpolation(E_0, E_1, t)
        # All relations for a given (i,j) should be the same (shared z_ij).
        # E_t[b,i,j,:] should be all 0s or all 1s.
        for b in range(B):
            for i in range(N):
                for j in range(N):
                    vals = E_t[b, i, j, :]
                    assert vals.std() == 0.0, f"Not shared at ({b},{i},{j})"

    def test_binary_output(self) -> None:
        E_0 = torch.zeros(B, N, N, K)
        E_1 = torch.ones(B, N, N, K)
        t = torch.rand(B)
        E_t = discrete_interpolation(E_0, E_1, t)
        assert ((E_t == 0) | (E_t == 1)).all()


class TestSampleTime:
    def test_shape(self) -> None:
        t = sample_time(B)
        assert t.shape == (B,)

    def test_range(self) -> None:
        t = sample_time(1000)
        assert (t >= 0).all()
        assert (t < 1).all()


class TestJointFlow:
    def test_sample_output_type(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        flow = RiemannFMJointFlow(manifold)
        x_1 = manifold.sample_noise(B, N)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)
        sample = flow.sample(x_1, E_1, mask)
        assert isinstance(sample, FlowMatchingSample)

    def test_sample_shapes(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        D = manifold.ambient_dim
        flow = RiemannFMJointFlow(manifold)
        x_1 = manifold.sample_noise(B, N)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)
        s = flow.sample(x_1, E_1, mask)
        assert s.x_t.shape == (B, N, D)
        assert s.u_t.shape == (B, N, D)
        assert s.E_t.shape == (B, N, N, K)
        assert s.E_1.shape == (B, N, N, K)
        assert s.t.shape == (B,)
        assert s.node_mask.shape == (B, N)

    def test_disable_continuous(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        flow = RiemannFMJointFlow(manifold, disable_continuous=True)
        x_1 = manifold.sample_noise(B, N)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)
        s = flow.sample(x_1, E_1, mask)
        assert torch.allclose(s.x_t, x_1)
        assert (s.u_t == 0).all()

    def test_disable_discrete(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        flow = RiemannFMJointFlow(manifold, disable_discrete=True)
        x_1 = manifold.sample_noise(B, N)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)
        s = flow.sample(x_1, E_1, mask)
        assert torch.allclose(s.E_t, E_1)

    def test_finite_outputs(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        flow = RiemannFMJointFlow(manifold)
        x_1 = manifold.sample_noise(B, N)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)
        s = flow.sample(x_1, E_1, mask)
        assert torch.isfinite(s.x_t).all()
        assert torch.isfinite(s.u_t).all()
        assert torch.isfinite(s.E_t).all()
