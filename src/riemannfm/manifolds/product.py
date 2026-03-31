"""Product manifold: H^d1 x S^d2 x R^d3 with learnable curvatures.

This is the core embedding space for RiemannFM. Knowledge graph entities are
embedded in a product of three spaces with complementary geometric properties:
- Hyperbolic (Lorentz model): encodes hierarchical relations (trees)
- Spherical: encodes symmetric/cyclic relations
- Euclidean: encodes flat numerical attributes

The curvature parameters kappa_H < 0 and kappa_S > 0 are learnable.
"""

import torch
import torch.nn as nn
from torch import Tensor

from riemannfm.manifolds.euclidean import EuclideanManifold
from riemannfm.manifolds.lorentz import LorentzManifold
from riemannfm.manifolds.spherical import SphericalManifold


class RiemannFMProductManifold(nn.Module):
    """Product manifold M = H^{d_h} x S^{d_s} x R^{d_e} with learnable curvatures.

    Points in this space are represented as concatenated vectors:
        x = [x_H (dim d_h+1) | x_S (dim d_s) | x_E (dim d_e)]

    Note: Hyperbolic ambient dimension is d_h+1 (Lorentz model adds a time dimension).

    Args:
        dim_hyperbolic: Intrinsic dimension of hyperbolic space (ambient = dim+1).
        dim_spherical: Dimension of spherical space.
        dim_euclidean: Dimension of Euclidean space.
        init_curvature_h: Initial hyperbolic curvature (negative).
        init_curvature_s: Initial spherical curvature (positive).
        learn_curvature: Whether to make curvatures learnable parameters.
    """

    def __init__(
        self,
        dim_hyperbolic: int = 16,
        dim_spherical: int = 16,
        dim_euclidean: int = 16,
        init_curvature_h: float = -1.0,
        init_curvature_s: float = 1.0,
        learn_curvature: bool = True,
    ):
        super().__init__()
        self.dim_h = dim_hyperbolic
        self.dim_s = dim_spherical
        self.dim_e = dim_euclidean

        # Ambient dimensions (Lorentz model needs +1 for time coordinate)
        self.ambient_h = dim_hyperbolic + 1
        self.ambient_s = dim_spherical
        self.ambient_e = dim_euclidean
        self.total_dim = self.ambient_h + self.ambient_s + self.ambient_e

        # Learnable curvature parameters
        # We store log(|curvature|) to ensure correct sign after exp
        if learn_curvature:
            self.log_abs_curv_h = nn.Parameter(torch.tensor(abs(init_curvature_h)).log())
            self.log_curv_s = nn.Parameter(torch.tensor(init_curvature_s).log())
        else:
            self.register_buffer("log_abs_curv_h", torch.tensor(abs(init_curvature_h)).log())
            self.register_buffer("log_curv_s", torch.tensor(init_curvature_s).log())

    @property
    def curvature_h(self) -> float:
        """Current hyperbolic curvature (always negative)."""
        return float(-self.log_abs_curv_h.exp().item())

    @property
    def curvature_s(self) -> float:
        """Current spherical curvature (always positive)."""
        return float(self.log_curv_s.exp().item())

    def _get_manifolds(self) -> tuple[LorentzManifold, SphericalManifold, EuclideanManifold]:
        """Create sub-manifold instances with current curvatures."""
        return (
            LorentzManifold(curvature=self.curvature_h),
            SphericalManifold(curvature=self.curvature_s),
            EuclideanManifold(),
        )

    def split(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Split a product manifold point into sub-manifold components.

        Args:
            x: Concatenated point, shape (..., total_dim).

        Returns:
            (x_H, x_S, x_E) with shapes (..., ambient_h), (..., ambient_s), (..., ambient_e).
        """
        x_h = x[..., : self.ambient_h]
        x_s = x[..., self.ambient_h : self.ambient_h + self.ambient_s]
        x_e = x[..., self.ambient_h + self.ambient_s :]
        return x_h, x_s, x_e

    def combine(self, x_h: Tensor, x_s: Tensor, x_e: Tensor) -> Tensor:
        """Combine sub-manifold components into a product manifold point.

        Args:
            x_h: Hyperbolic component, shape (..., ambient_h).
            x_s: Spherical component, shape (..., ambient_s).
            x_e: Euclidean component, shape (..., ambient_e).

        Returns:
            Concatenated point, shape (..., total_dim).
        """
        return torch.cat([x_h, x_s, x_e], dim=-1)

    def exp_map(self, x: Tensor, v: Tensor) -> Tensor:
        """Product exponential map: apply exp_map independently on each sub-manifold."""
        mH, mS, mE = self._get_manifolds()
        x_h, x_s, x_e = self.split(x)
        v_h, v_s, v_e = self.split(v)
        return self.combine(
            mH.exp_map(x_h, v_h),
            mS.exp_map(x_s, v_s),
            mE.exp_map(x_e, v_e),
        )

    def log_map(self, x: Tensor, y: Tensor) -> Tensor:
        """Product logarithmic map: apply log_map independently on each sub-manifold."""
        mH, mS, mE = self._get_manifolds()
        x_h, x_s, x_e = self.split(x)
        y_h, y_s, y_e = self.split(y)
        return self.combine(
            mH.log_map(x_h, y_h),
            mS.log_map(x_s, y_s),
            mE.log_map(x_e, y_e),
        )

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        """Product distance: sqrt(d_H^2 + d_S^2 + d_E^2) (Pythagorean theorem)."""
        mH, mS, mE = self._get_manifolds()
        x_h, x_s, x_e = self.split(x)
        y_h, y_s, y_e = self.split(y)
        d_h = mH.dist(x_h, y_h)
        d_s = mS.dist(x_s, y_s)
        d_e = mE.dist(x_e, y_e)
        return (d_h**2 + d_s**2 + d_e**2).clamp(min=0).sqrt()

    def proj_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        """Product tangent projection: project independently on each sub-manifold."""
        mH, mS, mE = self._get_manifolds()
        x_h, x_s, x_e = self.split(x)
        v_h, v_s, v_e = self.split(v)
        return self.combine(
            mH.proj_tangent(x_h, v_h),
            mS.proj_tangent(x_s, v_s),
            mE.proj_tangent(x_e, v_e),
        )

    def geodesic(self, x: Tensor, y: Tensor, t: float | Tensor) -> Tensor:
        """Product geodesic: interpolate independently on each sub-manifold."""
        mH, mS, mE = self._get_manifolds()
        x_h, x_s, x_e = self.split(x)
        y_h, y_s, y_e = self.split(y)
        return self.combine(
            mH.geodesic(x_h, y_h, t),
            mS.geodesic(x_s, y_s, t),
            mE.geodesic(x_e, y_e, t),
        )

    def sample_uniform(self, batch_shape: tuple[int, ...], device: torch.device) -> Tensor:
        """Sample uniformly from the product manifold."""
        mH, mS, mE = self._get_manifolds()
        x_h = mH.sample_uniform((*batch_shape, self.ambient_h), device)
        x_s = mS.sample_uniform((*batch_shape, self.ambient_s), device)
        x_e = mE.sample_uniform((*batch_shape, self.ambient_e), device)
        return self.combine(x_h, x_s, x_e)

    def origin(self, device: torch.device) -> Tensor:
        """Origin point on the product manifold."""
        mH, mS, mE = self._get_manifolds()
        return self.combine(
            mH.origin(self.ambient_h, device),
            mS.origin(self.ambient_s, device),
            mE.origin(self.ambient_e, device),
        )

    def compute_kernels(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute sub-manifold kernel values for geodesic kernel attention.

        Returns:
            (kappa_H, kappa_S, kappa_E): kernel values for each sub-space, all shape (...).
            - Hyperbolic kernel: -arccosh(-<x,y>_L) (negative distance, closer = larger)
            - Spherical kernel: <x,y> (cosine similarity)
            - Euclidean kernel: -||x-y||^2 (negative squared distance)
        """
        mH, _mS, _mE = self._get_manifolds()
        x_h, x_s, x_e = self.split(x)
        y_h, y_s, y_e = self.split(y)

        # Hyperbolic kernel: negative geodesic distance
        k_h = -mH.dist(x_h, y_h)
        # Spherical kernel: cosine similarity (inner product on unit sphere)
        k_s = (x_s * y_s).sum(dim=-1)
        # Euclidean kernel: negative squared distance
        k_e = -((x_e - y_e) ** 2).sum(dim=-1)

        return k_h, k_s, k_e
