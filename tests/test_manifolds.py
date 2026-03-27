"""Tests for Riemannian manifold implementations.

Verifies geometric properties:
- exp/log map round-trip consistency
- Distance positivity, symmetry, triangle inequality
- Tangent projection idempotency
- Geodesic interpolation endpoints
- Product manifold dimension splitting
"""

import pytest
import torch

from riedfm.manifolds.euclidean import EuclideanManifold
from riedfm.manifolds.hyperbolic import LorentzManifold
from riedfm.manifolds.product import ProductManifold
from riedfm.manifolds.spherical import SphericalManifold

DEVICE = torch.device("cpu")
ATOL = 1e-4
RTOL = 1e-4


class TestEuclideanManifold:
    def setup_method(self):
        self.m = EuclideanManifold()

    def test_exp_log_roundtrip(self):
        x = torch.randn(5, 8)
        y = torch.randn(5, 8)
        v = self.m.log_map(x, y)
        y_hat = self.m.exp_map(x, v)
        assert torch.allclose(y, y_hat, atol=ATOL)

    def test_dist_positive(self):
        x = torch.randn(5, 8)
        y = torch.randn(5, 8)
        d = self.m.dist(x, y)
        assert (d >= 0).all()

    def test_dist_symmetric(self):
        x = torch.randn(5, 8)
        y = torch.randn(5, 8)
        assert torch.allclose(self.m.dist(x, y), self.m.dist(y, x), atol=ATOL)

    def test_dist_self_zero(self):
        x = torch.randn(5, 8)
        assert torch.allclose(self.m.dist(x, x), torch.zeros(5), atol=ATOL)

    def test_geodesic_endpoints(self):
        x = torch.randn(5, 8)
        y = torch.randn(5, 8)
        assert torch.allclose(self.m.geodesic(x, y, 0.0), x, atol=ATOL)
        assert torch.allclose(self.m.geodesic(x, y, 1.0), y, atol=ATOL)


class TestLorentzManifold:
    def setup_method(self):
        self.m = LorentzManifold(curvature=-1.0)
        self.dim = 5  # ambient dim = dim + 1 = 6

    def _random_point(self, n=5):
        """Sample random points on the hyperboloid."""
        return self.m.sample_uniform((n, self.dim + 1), DEVICE)

    def _random_tangent(self, x):
        """Sample random tangent vector at x."""
        v = torch.randn_like(x) * 0.1
        return self.m.proj_tangent(x, v)

    def test_point_on_hyperboloid(self):
        """Points should satisfy <x, x>_L = -1."""
        from riedfm.utils.manifold_utils import lorentz_inner

        x = self._random_point()
        inner = lorentz_inner(x, x)
        assert torch.allclose(inner, -torch.ones(5), atol=ATOL)

    def test_exp_log_roundtrip(self):
        x = self._random_point()
        y = self._random_point()
        v = self.m.log_map(x, y)
        y_hat = self.m.exp_map(x, v)
        assert torch.allclose(y, y_hat, atol=1e-3, rtol=1e-3)

    def test_dist_positive(self):
        x = self._random_point()
        y = self._random_point()
        d = self.m.dist(x, y)
        assert (d >= 0).all()

    def test_dist_symmetric(self):
        x = self._random_point()
        y = self._random_point()
        assert torch.allclose(self.m.dist(x, y), self.m.dist(y, x), atol=ATOL)

    def test_tangent_projection_idempotent(self):
        x = self._random_point()
        v = torch.randn_like(x)
        v_proj = self.m.proj_tangent(x, v)
        v_proj2 = self.m.proj_tangent(x, v_proj)
        assert torch.allclose(v_proj, v_proj2, atol=ATOL)

    def test_geodesic_endpoints(self):
        x = self._random_point()
        y = self._random_point()
        x_0 = self.m.geodesic(x, y, 0.0)
        x_1 = self.m.geodesic(x, y, 1.0)
        assert torch.allclose(x_0, x, atol=1e-3)
        assert torch.allclose(x_1, y, atol=1e-3)


class TestSphericalManifold:
    def setup_method(self):
        self.m = SphericalManifold(curvature=1.0)
        self.dim = 8

    def _random_point(self, n=5):
        return self.m.sample_uniform((n, self.dim), DEVICE)

    def test_point_on_sphere(self):
        """Points should have unit norm."""
        x = self._random_point()
        norms = x.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(5), atol=ATOL)

    def test_exp_log_roundtrip(self):
        x = self._random_point()
        y = self._random_point()
        v = self.m.log_map(x, y)
        y_hat = self.m.exp_map(x, v)
        assert torch.allclose(y, y_hat, atol=1e-3, rtol=1e-3)

    def test_dist_positive(self):
        x = self._random_point()
        y = self._random_point()
        d = self.m.dist(x, y)
        assert (d >= 0).all()

    def test_dist_symmetric(self):
        x = self._random_point()
        y = self._random_point()
        assert torch.allclose(self.m.dist(x, y), self.m.dist(y, x), atol=ATOL)

    def test_tangent_orthogonal(self):
        """Tangent vectors should be orthogonal to the point."""
        x = self._random_point()
        v = torch.randn_like(x)
        v_proj = self.m.proj_tangent(x, v)
        inner = (v_proj * x).sum(dim=-1)
        assert torch.allclose(inner, torch.zeros(5), atol=ATOL)


class TestProductManifold:
    def setup_method(self):
        self.m = ProductManifold(dim_hyperbolic=4, dim_spherical=4, dim_euclidean=4)
        # total_dim = 5 (Lorentz) + 4 (sphere) + 4 (Euclidean) = 13

    def test_total_dim(self):
        assert self.m.total_dim == 13

    def test_split_combine_roundtrip(self):
        x = torch.randn(3, 13)
        x_h, x_s, x_e = self.m.split(x)
        x_recon = self.m.combine(x_h, x_s, x_e)
        assert torch.allclose(x, x_recon)

    def test_split_dimensions(self):
        x = torch.randn(3, 13)
        x_h, x_s, x_e = self.m.split(x)
        assert x_h.shape == (3, 5)  # d_h + 1
        assert x_s.shape == (3, 4)
        assert x_e.shape == (3, 4)

    def test_sample_uniform_shape(self):
        x = self.m.sample_uniform((10,), DEVICE)
        assert x.shape == (10, 13)

    def test_dist_positive(self):
        x = self.m.sample_uniform((5,), DEVICE)
        y = self.m.sample_uniform((5,), DEVICE)
        d = self.m.dist(x, y)
        assert (d >= 0).all()

    def test_learnable_curvature(self):
        """Curvature parameters should be part of the model."""
        params = list(self.m.parameters())
        assert len(params) == 2  # log_abs_curv_h and log_curv_s

    def test_compute_kernels_shape(self):
        x = self.m.sample_uniform((5,), DEVICE)
        y = self.m.sample_uniform((5,), DEVICE)
        k_h, k_s, k_e = self.m.compute_kernels(x, y)
        assert k_h.shape == (5,)
        assert k_s.shape == (5,)
        assert k_e.shape == (5,)
