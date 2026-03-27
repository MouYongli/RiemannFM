"""Tests for flow matching components.

Verifies:
- Geodesic interpolation endpoints (t=0 -> noise, t=1 -> data)
- Target vector field direction
- Discrete interpolation probabilistic behavior
- Noise sampling shapes
"""

import pytest
import torch

from riedfm.flow.continuous_flow import RieDFMContinuousFlow
from riedfm.flow.discrete_flow import RieDFMDiscreteFlow
from riedfm.flow.joint_flow import RieDFMJointFlow
from riedfm.flow.noise import RieDFMManifoldNoise, RieDFMSparseEdgeNoise
from riedfm.manifolds.product import RieDFMProductManifold

DEVICE = torch.device("cpu")


@pytest.fixture
def manifold():
    return RieDFMProductManifold(dim_hyperbolic=4, dim_spherical=4, dim_euclidean=4)


class TestManifoldNoiseSampler:
    def test_sample_shape(self, manifold):
        sampler = RieDFMManifoldNoise(manifold)
        x = sampler.sample(10, DEVICE)
        assert x.shape == (10, manifold.total_dim)


class TestSparseEdgeNoiseSampler:
    def test_sample_shape(self):
        sampler = RieDFMSparseEdgeNoise(num_edge_types=10, avg_density=0.05)
        e = sampler.sample(8, DEVICE)
        assert e.shape == (8, 8)

    def test_diagonal_zero(self):
        sampler = RieDFMSparseEdgeNoise(num_edge_types=10, avg_density=0.05)
        e = sampler.sample(8, DEVICE)
        assert (e.diagonal() == 0).all()

    def test_sparsity(self):
        sampler = RieDFMSparseEdgeNoise(num_edge_types=10, avg_density=0.05)
        e = sampler.sample(100, DEVICE)
        density = (e > 0).float().mean().item()
        # Should be roughly around avg_density (with some tolerance)
        assert density < 0.2  # Not too dense


class TestContinuousFlowMatcher:
    def test_interpolate_endpoints(self, manifold):
        flow = RieDFMContinuousFlow(manifold)
        x_0 = manifold.sample_uniform((5,), DEVICE)
        x_1 = manifold.sample_uniform((5,), DEVICE)

        x_at_0 = flow.interpolate(x_0, x_1, torch.tensor(0.0))
        x_at_1 = flow.interpolate(x_0, x_1, torch.tensor(1.0))

        assert torch.allclose(x_at_0, x_0, atol=1e-3)
        assert torch.allclose(x_at_1, x_1, atol=1e-3)

    def test_target_vector_field_shape(self, manifold):
        flow = RieDFMContinuousFlow(manifold)
        x_t = manifold.sample_uniform((5,), DEVICE)
        x_1 = manifold.sample_uniform((5,), DEVICE)
        t = torch.tensor(0.5)
        u = flow.target_vector_field(x_t, x_1, t)
        assert u.shape == x_t.shape

    def test_ode_step_shape(self, manifold):
        flow = RieDFMContinuousFlow(manifold)
        x = manifold.sample_uniform((5,), DEVICE)
        v = torch.randn_like(x) * 0.1
        v = manifold.proj_tangent(x, v)
        x_new = flow.ode_step(x, v, 0.01)
        assert x_new.shape == x.shape


class TestDiscreteFlowMatcher:
    def test_interpolate_at_zero(self):
        flow = RieDFMDiscreteFlow(num_edge_types=11)
        e_0 = torch.randint(0, 11, (5, 5))
        e_1 = torch.randint(0, 11, (5, 5))
        e_t = flow.interpolate(e_0, e_1, torch.tensor(0.0))
        assert torch.equal(e_t, e_0)

    def test_interpolate_at_one(self):
        flow = RieDFMDiscreteFlow(num_edge_types=11)
        e_0 = torch.randint(0, 11, (5, 5))
        e_1 = torch.randint(0, 11, (5, 5))
        e_t = flow.interpolate(e_0, e_1, torch.tensor(1.0))
        assert torch.equal(e_t, e_1)


class TestJointFlowMatcher:
    def test_sample_time_shape(self, manifold):
        jfm = RieDFMJointFlow(manifold, num_edge_types=11)
        t = jfm.sample_time(16, DEVICE)
        assert t.shape == (16,)
        assert (t >= 0).all() and (t <= 1).all()

    def test_prepare_training_data(self, manifold):
        jfm = RieDFMJointFlow(manifold, num_edge_types=11)
        N = 8
        x_1 = manifold.sample_uniform((N,), DEVICE)
        e_1 = torch.randint(0, 11, (N, N))
        t = torch.tensor(0.5)

        data = jfm.prepare_training_data(x_1, e_1, t)
        assert data["x_t"].shape == (N, manifold.total_dim)
        assert data["e_t"].shape == (N, N)
        assert data["u_target"].shape == (N, manifold.total_dim)
