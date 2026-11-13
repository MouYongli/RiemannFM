"""Tests for Riemannian manifold operations (Layer 0).

Verifies geometric properties: exp/log roundtrip, distance axioms,
tangent orthogonality, curvature gradients, and numerical stability.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from riemannfm.manifolds.euclidean import EuclideanManifold
from riemannfm.manifolds.lorentz import LorentzManifold
from riemannfm.manifolds.product import RiemannFMProductManifold
from riemannfm.manifolds.spherical import SphericalManifold
from riemannfm.manifolds.utils import lorentz_inner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ATOL = 1e-4  # float32 tolerance for exact comparisons
ATOL_ROUNDTRIP = 5e-3  # looser tolerance for exp/log roundtrips at high curvature
DIM = 8
B = 16  # batch size for statistical tests


def _random_lorentz_point(
    manifold: LorentzManifold, batch: int = B,
) -> Tensor:
    """Generate random points on the hyperboloid via noise sampling."""
    return manifold.sample_noise(batch, radius=1.0)


def _random_sphere_point(
    manifold: SphericalManifold, batch: int = B,
) -> Tensor:
    """Generate random points on the sphere."""
    return manifold.sample_noise(batch)


def _random_tangent_lorentz(
    manifold: LorentzManifold, x: Tensor, scale: float = 0.5,
) -> Tensor:
    """Random tangent vector at x on the hyperboloid."""
    v = scale * torch.randn_like(x)
    return manifold.proj_tangent(x, v)


def _random_tangent_sphere(
    manifold: SphericalManifold, x: Tensor, scale: float = 0.5,
) -> Tensor:
    """Random tangent vector at x on the sphere."""
    v = scale * torch.randn_like(x)
    return manifold.proj_tangent(x, v)


# ===================================================================
# Lorentz Manifold Tests
# ===================================================================


class TestLorentzManifold:
    """Tests for the Lorentz (hyperboloid) model."""

    @pytest.fixture(params=[-1.0, -0.5, -2.0], ids=["k=-1", "k=-0.5", "k=-2"])
    def manifold(self, request: pytest.FixtureRequest) -> LorentzManifold:
        return LorentzManifold(DIM, curvature=request.param, learnable=False)

    def test_origin_on_manifold(self, manifold: LorentzManifold) -> None:
        o = manifold.origin(B)
        kh = manifold.curvature
        inner = lorentz_inner(o, o)
        expected = 1.0 / kh
        assert torch.allclose(inner, expected.expand_as(inner), atol=ATOL)

    def test_exp_log_roundtrip(self, manifold: LorentzManifold) -> None:
        """exp_x(log_x(y)) should recover y."""
        x = _random_lorentz_point(manifold)
        y = _random_lorentz_point(manifold)
        v = manifold.log_map(x, y)
        y_rec = manifold.exp_map(x, v)
        assert torch.allclose(y_rec, y, atol=ATOL_ROUNDTRIP)

    def test_log_exp_roundtrip(self, manifold: LorentzManifold) -> None:
        """log_x(exp_x(v)) should recover v for small v."""
        x = _random_lorentz_point(manifold)
        v = _random_tangent_lorentz(manifold, x, scale=0.3)
        y = manifold.exp_map(x, v)
        v_rec = manifold.log_map(x, y)
        assert torch.allclose(v_rec, v, atol=ATOL_ROUNDTRIP)

    def test_proj_tangent_orthogonal(self, manifold: LorentzManifold) -> None:
        """Projected vector must be Lorentz-orthogonal to the base point."""
        x = _random_lorentz_point(manifold)
        v = torch.randn_like(x)
        proj = manifold.proj_tangent(x, v)
        inner = lorentz_inner(proj, x)
        assert torch.allclose(inner, torch.zeros_like(inner), atol=1e-3)

    def test_proj_tangent_idempotent(self, manifold: LorentzManifold) -> None:
        x = _random_lorentz_point(manifold)
        v = torch.randn_like(x)
        proj1 = manifold.proj_tangent(x, v)
        proj2 = manifold.proj_tangent(x, proj1)
        assert torch.allclose(proj1, proj2, atol=ATOL)

    def test_dist_symmetry(self, manifold: LorentzManifold) -> None:
        x = _random_lorentz_point(manifold)
        y = _random_lorentz_point(manifold)
        assert torch.allclose(manifold.dist(x, y), manifold.dist(y, x), atol=ATOL)

    def test_dist_non_negative(self, manifold: LorentzManifold) -> None:
        x = _random_lorentz_point(manifold)
        y = _random_lorentz_point(manifold)
        assert (manifold.dist(x, y) >= -ATOL).all()

    def test_dist_identity(self, manifold: LorentzManifold) -> None:
        x = _random_lorentz_point(manifold)
        d = manifold.dist(x, x)
        # safe_arccosh clamps to 1+eps, so dist(x,x) is small but not exactly 0
        assert torch.allclose(d, torch.zeros_like(d), atol=2e-3)

    def test_dist_triangle_inequality(self, manifold: LorentzManifold) -> None:
        x = _random_lorentz_point(manifold)
        y = _random_lorentz_point(manifold)
        z = _random_lorentz_point(manifold)
        dxy = manifold.dist(x, y)
        dyz = manifold.dist(y, z)
        dxz = manifold.dist(x, z)
        assert (dxz <= dxy + dyz + ATOL).all()

    def test_proj_manifold(self, manifold: LorentzManifold) -> None:
        x = _random_lorentz_point(manifold)
        perturbed = x + 0.01 * torch.randn_like(x)
        proj = manifold.proj_manifold(perturbed)
        kh = manifold.curvature
        inner = lorentz_inner(proj, proj)
        assert torch.allclose(inner, (1.0 / kh).expand_as(inner), atol=ATOL)

    def test_noise_on_manifold(self, manifold: LorentzManifold) -> None:
        samples = manifold.sample_noise(64, radius=3.0)
        kh = manifold.curvature
        inner = lorentz_inner(samples, samples)
        assert torch.allclose(inner, (1.0 / kh).expand_as(inner), atol=ATOL)


# ===================================================================
# Spherical Manifold Tests
# ===================================================================


class TestSphericalManifold:
    """Tests for the spherical manifold."""

    @pytest.fixture(params=[1.0, 0.5, 2.0], ids=["k=1", "k=0.5", "k=2"])
    def manifold(self, request: pytest.FixtureRequest) -> SphericalManifold:
        return SphericalManifold(DIM, curvature=request.param, learnable=False)

    def _check_on_sphere(self, manifold: SphericalManifold, x: Tensor) -> None:
        ks = manifold.curvature
        norm_sq = (x * x).sum(dim=-1)
        expected = 1.0 / ks
        assert torch.allclose(norm_sq, expected.expand_as(norm_sq), atol=ATOL)

    def test_origin_on_manifold(self, manifold: SphericalManifold) -> None:
        o = manifold.origin(B)
        self._check_on_sphere(manifold, o)

    def test_exp_log_roundtrip(self, manifold: SphericalManifold) -> None:
        x = _random_sphere_point(manifold)
        y = _random_sphere_point(manifold)
        v = manifold.log_map(x, y)
        y_rec = manifold.exp_map(x, v)
        assert torch.allclose(y_rec, y, atol=ATOL_ROUNDTRIP)

    def test_log_exp_roundtrip(self, manifold: SphericalManifold) -> None:
        x = _random_sphere_point(manifold)
        v = _random_tangent_sphere(manifold, x, scale=0.3)
        y = manifold.exp_map(x, v)
        v_rec = manifold.log_map(x, y)
        assert torch.allclose(v_rec, v, atol=ATOL_ROUNDTRIP)

    def test_proj_tangent_orthogonal(self, manifold: SphericalManifold) -> None:
        x = _random_sphere_point(manifold)
        v = torch.randn_like(x)
        proj = manifold.proj_tangent(x, v)
        dot = (proj * x).sum(dim=-1)
        assert torch.allclose(dot, torch.zeros_like(dot), atol=ATOL)

    def test_proj_tangent_idempotent(self, manifold: SphericalManifold) -> None:
        x = _random_sphere_point(manifold)
        v = torch.randn_like(x)
        proj1 = manifold.proj_tangent(x, v)
        proj2 = manifold.proj_tangent(x, proj1)
        assert torch.allclose(proj1, proj2, atol=ATOL)

    def test_dist_symmetry(self, manifold: SphericalManifold) -> None:
        x = _random_sphere_point(manifold)
        y = _random_sphere_point(manifold)
        assert torch.allclose(manifold.dist(x, y), manifold.dist(y, x), atol=ATOL)

    def test_dist_non_negative(self, manifold: SphericalManifold) -> None:
        x = _random_sphere_point(manifold)
        y = _random_sphere_point(manifold)
        assert (manifold.dist(x, y) >= -ATOL).all()

    def test_dist_identity(self, manifold: SphericalManifold) -> None:
        x = _random_sphere_point(manifold)
        d = manifold.dist(x, x)
        # safe_arccos clamps to 1-eps; dist scales by 1/sqrt(ks), so for
        # small ks the floor is larger
        assert torch.allclose(d, torch.zeros_like(d), atol=3e-3)

    def test_dist_triangle_inequality(self, manifold: SphericalManifold) -> None:
        x = _random_sphere_point(manifold)
        y = _random_sphere_point(manifold)
        z = _random_sphere_point(manifold)
        dxy = manifold.dist(x, y)
        dyz = manifold.dist(y, z)
        dxz = manifold.dist(x, z)
        assert (dxz <= dxy + dyz + ATOL).all()

    def test_proj_manifold(self, manifold: SphericalManifold) -> None:
        x = _random_sphere_point(manifold)
        perturbed = x + 0.01 * torch.randn_like(x)
        proj = manifold.proj_manifold(perturbed)
        self._check_on_sphere(manifold, proj)

    def test_noise_on_manifold(self, manifold: SphericalManifold) -> None:
        samples = manifold.sample_noise(64)
        self._check_on_sphere(manifold, samples)


# ===================================================================
# Euclidean Manifold Tests
# ===================================================================


class TestEuclideanManifold:
    @pytest.fixture
    def manifold(self) -> EuclideanManifold:
        return EuclideanManifold(DIM)

    def test_exp_log_roundtrip(self, manifold: EuclideanManifold) -> None:
        x = torch.randn(B, DIM)
        y = torch.randn(B, DIM)
        v = manifold.log_map(x, y)
        y_rec = manifold.exp_map(x, v)
        assert torch.allclose(y_rec, y, atol=ATOL)

    def test_dist(self, manifold: EuclideanManifold) -> None:
        x = torch.randn(B, DIM)
        y = torch.randn(B, DIM)
        d = manifold.dist(x, y)
        expected = torch.norm(y - x, dim=-1)
        assert torch.allclose(d, expected, atol=ATOL)

    def test_origin(self, manifold: EuclideanManifold) -> None:
        o = manifold.origin(B)
        assert torch.allclose(o, torch.zeros(B, DIM), atol=ATOL)


# ===================================================================
# Product Manifold Tests
# ===================================================================


class TestProductManifold:
    @pytest.fixture
    def manifold(self) -> RiemannFMProductManifold:
        return RiemannFMProductManifold(
            dim_hyperbolic=DIM, dim_spherical=DIM, dim_euclidean=DIM,
        )

    def test_ambient_dim(self, manifold: RiemannFMProductManifold) -> None:
        # H: DIM+1, S: DIM+1, E: DIM
        assert manifold.ambient_dim == DIM + 1 + DIM + 1 + DIM

    def test_split_combine_roundtrip(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        x = torch.randn(B, manifold.ambient_dim)
        parts = manifold.split(x)
        x_rec = manifold.combine(parts)
        assert torch.allclose(x_rec, x, atol=ATOL)

    def test_origin_components_on_manifold(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        o = manifold.origin(B)
        parts = manifold.split(o)

        h = manifold.hyperbolic
        assert h is not None
        kh = h.curvature
        inner_h = lorentz_inner(parts["hyperbolic"], parts["hyperbolic"])
        assert torch.allclose(inner_h, (1.0 / kh).expand_as(inner_h), atol=ATOL)

        s = manifold.spherical
        assert s is not None
        ks = s.curvature
        norm_sq_s = (parts["spherical"] ** 2).sum(dim=-1)
        assert torch.allclose(norm_sq_s, (1.0 / ks).expand_as(norm_sq_s), atol=ATOL)

        assert torch.allclose(
            parts["euclidean"], torch.zeros_like(parts["euclidean"]), atol=ATOL,
        )

    def test_exp_log_roundtrip(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        x = manifold.sample_noise(B, radius_h=2.0, sigma_e=1.0)
        y = manifold.sample_noise(B, radius_h=2.0, sigma_e=1.0)
        v = manifold.log_map(x, y)
        y_rec = manifold.exp_map(x, v)
        assert torch.allclose(y_rec, y, atol=ATOL_ROUNDTRIP)

    def test_dist_decomposition(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        """Product distance^2 == sum of component distances^2."""
        x = manifold.sample_noise(B, radius_h=2.0)
        y = manifold.sample_noise(B, radius_h=2.0)
        d_product = manifold.dist(x, y)
        d_components = manifold.component_dists(x, y)
        d_sq_sum = sum(d.pow(2) for d in d_components.values())
        assert torch.allclose(d_product.pow(2), d_sq_sum, atol=ATOL)

    def test_tangent_norm(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        x = manifold.sample_noise(B, radius_h=2.0)
        v = manifold.proj_tangent(x, torch.randn(B, manifold.ambient_dim))
        norm = manifold.tangent_norm(x, v)
        assert (norm >= -ATOL).all()
        assert norm.shape == (B,)

    def test_noise_on_manifold(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        samples = manifold.sample_noise(64, radius_h=3.0)
        parts = manifold.split(samples)

        h = manifold.hyperbolic
        assert h is not None
        kh = h.curvature
        inner_h = lorentz_inner(parts["hyperbolic"], parts["hyperbolic"])
        assert torch.allclose(inner_h, (1.0 / kh).expand_as(inner_h), atol=ATOL)

        s = manifold.spherical
        assert s is not None
        ks = s.curvature
        norm_sq_s = (parts["spherical"] ** 2).sum(dim=-1)
        assert torch.allclose(norm_sq_s, (1.0 / ks).expand_as(norm_sq_s), atol=ATOL)

    def test_ablation_h_only(self) -> None:
        m = RiemannFMProductManifold(
            dim_hyperbolic=DIM, dim_spherical=0, dim_euclidean=0,
        )
        assert m.ambient_dim == DIM + 1
        assert m.hyperbolic is not None
        assert m.spherical is None
        assert m.euclidean is None
        with torch.no_grad():
            o = m.origin(B)
        assert o.shape == (B, DIM + 1)

    def test_ablation_e_only(self) -> None:
        m = RiemannFMProductManifold(
            dim_hyperbolic=0, dim_spherical=0, dim_euclidean=DIM,
        )
        assert m.ambient_dim == DIM
        assert m.hyperbolic is None
        assert m.euclidean is not None
        with torch.no_grad():
            o = m.origin(4)
        assert torch.allclose(o, torch.zeros(4, DIM), atol=ATOL)

    def test_from_config(self) -> None:
        cfg = SimpleNamespace(
            dim_hyperbolic=32, dim_spherical=32, dim_euclidean=32,
            init_curvature_h=-1.0, init_curvature_s=1.0, learn_curvature=True,
        )
        m = RiemannFMProductManifold.from_config(cfg)
        assert m.ambient_dim == 33 + 33 + 32


# ===================================================================
# Curvature Gradient Tests
# ===================================================================


class TestCurvatureGradient:
    def test_gradient_flows_through_curvature(self) -> None:
        m = LorentzManifold(DIM, curvature=-1.0, learnable=True)
        x = m.sample_noise(B, radius=2.0)
        y = m.sample_noise(B, radius=2.0)
        loss = m.dist(x, y).sum()
        loss.backward()
        assert m._curvature.grad is not None
        assert m._curvature.grad.isfinite().all()

    def test_fixed_curvature_no_grad(self) -> None:
        m = LorentzManifold(DIM, curvature=-1.0, learnable=False)
        # Detach noise so that the graph traces through dist, not sampling
        x = m.sample_noise(B, radius=2.0).detach().requires_grad_(True)
        y = m.sample_noise(B, radius=2.0).detach().requires_grad_(True)
        loss = m.dist(x, y).sum()
        loss.backward()
        assert m._curvature.grad is None

    def test_curvature_stays_negative(self) -> None:
        m = LorentzManifold(DIM, curvature=-0.01, learnable=True)
        optimizer = torch.optim.SGD([m._curvature], lr=100.0)
        # Push curvature toward positive (should be clamped)
        for _ in range(5):
            optimizer.zero_grad()
            x = m.sample_noise(B, radius=1.0)
            y = m.sample_noise(B, radius=1.0)
            loss = -m.dist(x, y).sum()  # negative loss pushes curvature up
            loss.backward()
            optimizer.step()
        # The property clamps, raw parameter might have drifted
        assert m.curvature.item() < 0

    def test_spherical_curvature_gradient(self) -> None:
        m = SphericalManifold(DIM, curvature=1.0, learnable=True)
        x = m.sample_noise(B)
        y = m.sample_noise(B)
        loss = m.dist(x, y).sum()
        loss.backward()
        assert m._curvature.grad is not None
        assert m._curvature.grad.isfinite().all()


# ===================================================================
# Numerical Stability Tests
# ===================================================================


class TestNumericalStability:
    def test_log_map_identical_points_lorentz(self) -> None:
        m = LorentzManifold(DIM, curvature=-1.0, learnable=False)
        x = m.sample_noise(B, radius=2.0)
        v = m.log_map(x, x)
        assert v.isfinite().all()
        # The edge-case guard returns exactly zero when x ≈ y
        assert torch.allclose(v, torch.zeros_like(v), atol=ATOL)

    def test_log_map_identical_points_sphere(self) -> None:
        m = SphericalManifold(DIM, curvature=1.0, learnable=False)
        x = m.sample_noise(B)
        v = m.log_map(x, x)
        assert v.isfinite().all()
        assert torch.allclose(v, torch.zeros_like(v), atol=ATOL)

    def test_exp_map_zero_tangent_lorentz(self) -> None:
        m = LorentzManifold(DIM, curvature=-1.0, learnable=False)
        x = m.sample_noise(B, radius=2.0)
        v = torch.zeros_like(x)
        y = m.exp_map(x, v)
        assert y.isfinite().all()
        assert torch.allclose(y, x, atol=ATOL)

    def test_exp_map_zero_tangent_sphere(self) -> None:
        m = SphericalManifold(DIM, curvature=1.0, learnable=False)
        x = m.sample_noise(B)
        v = torch.zeros_like(x)
        y = m.exp_map(x, v)
        assert y.isfinite().all()
        assert torch.allclose(y, x, atol=ATOL)

    def test_dist_nearby_points(self) -> None:
        m = LorentzManifold(DIM, curvature=-1.0, learnable=False)
        x = m.sample_noise(B, radius=2.0)
        v = _random_tangent_lorentz(m, x, scale=1e-6)
        y = m.exp_map(x, v)
        d = m.dist(x, y)
        assert d.isfinite().all()
        assert (d >= 0).all()

    def test_large_curvature(self) -> None:
        for kh in [-50.0, -100.0]:
            m = LorentzManifold(DIM, curvature=kh, learnable=False)
            x = m.sample_noise(B, radius=1.0)
            y = m.sample_noise(B, radius=1.0)
            d = m.dist(x, y)
            assert d.isfinite().all()

        for ks in [50.0, 100.0]:
            m = SphericalManifold(DIM, curvature=ks, learnable=False)
            x = m.sample_noise(B)
            y = m.sample_noise(B)
            d = m.dist(x, y)
            assert d.isfinite().all()
