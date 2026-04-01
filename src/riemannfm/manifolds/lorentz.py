"""Lorentz (hyperboloid) model of hyperbolic space H^{d_h}_{kappa_h}.

Implements Definitions 3.1, 3.2, 3.7, 4.1, 4.3-4.6 from the paper.
The hyperboloid is defined as:

    H = { z in R^{d_h+1} | <z, z>_L = 1/kappa_h,  z_0 > 0 }

where kappa_h < 0 and <.,.>_L is the Lorentz inner product.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from riemannfm.manifolds.base import RiemannFMManifold
from riemannfm.manifolds.utils import (
    CURVATURE_EPS,
    clamp_norm,
    lorentz_inner,
    lorentz_norm,
    safe_arccosh,
)


class LorentzManifold(RiemannFMManifold):
    """Hyperbolic space in the Lorentz (hyperboloid) model.

    Points satisfy ``<x, x>_L = 1/kappa_h`` with ``kappa_h < 0``, so the
    Lorentz quadratic form evaluates to a negative constant.  The time
    component ``x[..., 0]`` is always positive.

    Args:
        dim: Intrinsic dimension ``d_h``.
        curvature: Initial sectional curvature ``kappa_h`` (must be negative).
        learnable: Whether the curvature is a trainable parameter.
    """

    def __init__(
        self,
        dim: int,
        curvature: float = -1.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if curvature >= 0:
            msg = f"Hyperbolic curvature must be negative, got {curvature}"
            raise ValueError(msg)
        self._dim = dim
        self._curvature = nn.Parameter(
            torch.tensor(curvature, dtype=torch.float32),
            requires_grad=learnable,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def ambient_dim(self) -> int:
        return self._dim + 1

    @property
    def curvature(self) -> Tensor:
        """Clamped curvature, always <= -CURVATURE_EPS."""
        return self._curvature.clamp(max=-CURVATURE_EPS)

    # ------------------------------------------------------------------
    # Manifold operations (Def 4.1 -- 4.6)
    # ------------------------------------------------------------------

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        """Geodesic distance (Def 4.1).

        d_H(x, y) = (1 / sqrt(|kh|)) * arccosh(-kh * <x, y>_L)

        Args:
            x: shape ``(..., d_h+1)``.
            y: shape ``(..., d_h+1)``.

        Returns:
            shape ``(...)``.
        """
        kh = self.curvature
        inner = lorentz_inner(x, y)  # (...)
        # kh < 0 and <x,y>_L <= 1/kh < 0, so kh * <x,y>_L >= 1
        return safe_arccosh(kh * inner) / kh.abs().sqrt()

    def log_map(self, x: Tensor, y: Tensor) -> Tensor:
        """Logarithmic map (Def 4.4).

        u = y - kh * <x, y>_L * x   (tangent projection of y at x)
        log_x(y) = (d(x,y) / ||u||_L) * u

        Args:
            x: Base point, shape ``(..., d_h+1)``.
            y: Target point, shape ``(..., d_h+1)``.

        Returns:
            Tangent vector at *x*, shape ``(..., d_h+1)``.
        """
        kh = self.curvature
        alpha = lorentz_inner(x, y, keepdim=True)  # (..., 1)
        u = y - kh * alpha * x  # tangent component
        u_norm = lorentz_norm(u, keepdim=True)  # (..., 1)
        d = self.dist(x, y).unsqueeze(-1)  # (..., 1)
        # When x ≈ y, u_norm → 0 but d has an artificial floor from arccosh
        # clamping.  Guard on u_norm to return the zero tangent vector.
        safe = u_norm > 1e-6
        scale = torch.where(safe, d / clamp_norm(u_norm), torch.zeros_like(d))
        return scale * u

    def exp_map(self, x: Tensor, v: Tensor) -> Tensor:
        """Exponential map (Def 4.5).

        exp_x(v) = cosh(sqrt(|kh|)*||v||_L)*x
                  + sinh(sqrt(|kh|)*||v||_L) / (sqrt(|kh|)*||v||_L) * v

        Args:
            x: Base point on H, shape ``(..., d_h+1)``.
            v: Tangent vector at *x*, shape ``(..., d_h+1)``.

        Returns:
            Point on H, shape ``(..., d_h+1)``.
        """
        kh = self.curvature
        sqrt_abs_k = kh.abs().sqrt()
        v_norm = clamp_norm(lorentz_norm(v, keepdim=True))  # (..., 1)
        t = sqrt_abs_k * v_norm  # (..., 1)
        result = torch.cosh(t) * x + (torch.sinh(t) / t) * v
        return self.proj_manifold(result)

    def proj_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        """Project ambient vector onto tangent space at *x* (Def 5.16).

        proj(v) = v - kh * <v, x>_L * x

        This ensures <proj(v), x>_L = 0.

        Args:
            x: Base point on H, shape ``(..., d_h+1)``.
            v: Arbitrary ambient vector, shape ``(..., d_h+1)``.

        Returns:
            Tangent vector at *x*, shape ``(..., d_h+1)``.
        """
        kh = self.curvature
        alpha = lorentz_inner(x, v, keepdim=True)
        return v - kh * alpha * x

    def proj_manifold(self, x: Tensor) -> Tensor:
        """Re-project onto the hyperboloid.

        Keeps the spatial components fixed and recomputes ``x_0`` from the
        constraint ``<x, x>_L = 1/kh``:

            x_0 = sqrt(||x_spatial||^2 + 1/|kh|)

        Args:
            x: Approximate hyperboloid point, shape ``(..., d_h+1)``.

        Returns:
            Exact hyperboloid point, shape ``(..., d_h+1)``.
        """
        kh = self.curvature
        spatial = x[..., 1:]
        spatial_sq = (spatial * spatial).sum(dim=-1, keepdim=True)
        # <x,x>_L = -x0^2 + ||spatial||^2 = 1/kh
        # x0^2 = ||spatial||^2 - 1/kh = ||spatial||^2 + 1/|kh|
        x0 = torch.sqrt(spatial_sq + 1.0 / kh.abs()).clamp(min=1e-8)
        return torch.cat([x0, spatial], dim=-1)

    def inner(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """Lorentz inner product of tangent vectors (Def 4.3).

        Args:
            x: Base point (unused, metric is constant in ambient coords).
            u: Tangent vector, shape ``(..., d_h+1)``.
            v: Tangent vector, shape ``(..., d_h+1)``.

        Returns:
            shape ``(...)``.
        """
        return lorentz_inner(u, v)

    def tangent_norm(self, x: Tensor, v: Tensor) -> Tensor:
        """Riemannian norm under the Lorentz metric (Def 4.6, H component).

        Args:
            x: Base point (unused).
            v: Tangent vector, shape ``(..., d_h+1)``.

        Returns:
            shape ``(...)``.
        """
        return lorentz_norm(v)

    def origin(
        self,
        *batch_shape: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Canonical origin on the hyperboloid (Def 3.7).

        x_origin = (1/sqrt(|kh|), 0, ..., 0)

        Args:
            *batch_shape: Leading batch dimensions.
            device: Target device.
            dtype: Target dtype.

        Returns:
            shape ``(*batch_shape, d_h+1)``.
        """
        kh = self.curvature
        o = torch.zeros(*batch_shape, self.ambient_dim, device=device, dtype=dtype)
        o[..., 0] = (1.0 / kh.abs().sqrt()).to(device=device, dtype=dtype)
        return o

    def sample_noise(
        self,
        *batch_shape: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
        radius: float = 5.0,
    ) -> Tensor:
        """Sample noise via wrapped normal at the origin (Def 6.1, H component).

        Generates random tangent vectors at the origin and maps them onto the
        hyperboloid with ``exp_map``.  The tangent vector norms are scaled so
        that the resulting points lie within a geodesic ball of the given
        *radius* (in expectation).

        Args:
            *batch_shape: Leading batch dimensions.
            device: Target device.
            dtype: Target dtype.
            generator: Optional RNG.
            radius: Approximate geodesic radius of the noise ball.

        Returns:
            shape ``(*batch_shape, d_h+1)``.
        """
        o = self.origin(*batch_shape, device=device, dtype=dtype)  # on hyperboloid
        # Tangent space at origin: v_0 = 0, v_spatial ~ N(0, I)
        v = torch.zeros(*batch_shape, self.ambient_dim, device=device, dtype=dtype)
        v[..., 1:] = torch.randn(
            *batch_shape, self._dim, device=device, dtype=dtype, generator=generator
        )
        # Scale to target radius (adjust norm so average geodesic dist ~ radius)
        v_norm = clamp_norm(lorentz_norm(v, keepdim=True))
        # Sample radii uniformly in [0, radius] via sqrt for area-proportional
        u = torch.rand(
            *batch_shape, 1, device=device, dtype=dtype, generator=generator
        )
        target_norm = radius * u.sqrt()
        v = v * (target_norm / v_norm)
        return self.exp_map(o, v)
