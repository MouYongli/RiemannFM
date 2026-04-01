"""Spherical manifold S^{d_s}_{kappa_s} with learnable curvature.

Implements Definitions 3.2, 3.7, 4.1, 4.3-4.6 from the paper.
The sphere is defined as:

    S = { z in R^{d_s+1} | ||z||^2 = 1/kappa_s }

where kappa_s > 0.  Radius = 1 / sqrt(kappa_s).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from riemannfm.manifolds.base import RiemannFMManifold
from riemannfm.manifolds.utils import CURVATURE_EPS, clamp_norm, safe_arccos


class SphericalManifold(RiemannFMManifold):
    """Sphere of radius ``1 / sqrt(kappa_s)`` embedded in R^{d_s+1}.

    Args:
        dim: Intrinsic dimension ``d_s``.
        curvature: Initial curvature ``kappa_s`` (must be positive).
        learnable: Whether the curvature is a trainable parameter.
    """

    def __init__(
        self,
        dim: int,
        curvature: float = 1.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if curvature <= 0:
            msg = f"Spherical curvature must be positive, got {curvature}"
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
        """Clamped curvature, always >= CURVATURE_EPS."""
        return self._curvature.clamp(min=CURVATURE_EPS)

    @property
    def radius(self) -> Tensor:
        """Sphere radius ``1 / sqrt(kappa_s)``."""
        return 1.0 / self.curvature.sqrt()

    # ------------------------------------------------------------------
    # Manifold operations (Def 4.1 -- 4.6)
    # ------------------------------------------------------------------

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        """Geodesic distance (Def 4.1).

        d_S(x, y) = (1 / sqrt(ks)) * arccos(ks * x^T y)

        Args:
            x: shape ``(..., d_s+1)``.
            y: shape ``(..., d_s+1)``.

        Returns:
            shape ``(...)``.
        """
        ks = self.curvature
        dot = (x * y).sum(dim=-1)
        return safe_arccos(ks * dot) / ks.sqrt()

    def log_map(self, x: Tensor, y: Tensor) -> Tensor:
        """Logarithmic map (Def 4.4).

        w = y - ks * (x^T y) * x   (tangent projection of y at x)
        log_x(y) = (d(x,y) / ||w||) * w

        Args:
            x: Base point on S, shape ``(..., d_s+1)``.
            y: Target point on S, shape ``(..., d_s+1)``.

        Returns:
            Tangent vector at *x*, shape ``(..., d_s+1)``.
        """
        ks = self.curvature
        dot = (x * y).sum(dim=-1, keepdim=True)  # (..., 1)
        w = y - ks * dot * x
        w_norm = torch.linalg.norm(w, dim=-1, keepdim=True)
        d = self.dist(x, y).unsqueeze(-1)
        # When x ≈ y, w_norm → 0 but d has an artificial floor from arccos
        # clamping.  Guard on w_norm to return the zero tangent vector.
        safe = w_norm > 1e-6
        scale = torch.where(safe, d / clamp_norm(w_norm), torch.zeros_like(d))
        return scale * w

    def exp_map(self, x: Tensor, v: Tensor) -> Tensor:
        """Exponential map (Def 4.5).

        exp_x(v) = cos(sqrt(ks)*||v||)*x
                  + sin(sqrt(ks)*||v||) / (sqrt(ks)*||v||) * v

        Args:
            x: Base point on S, shape ``(..., d_s+1)``.
            v: Tangent vector at *x*, shape ``(..., d_s+1)``.

        Returns:
            Point on S, shape ``(..., d_s+1)``.
        """
        ks = self.curvature
        sqrt_ks = ks.sqrt()
        v_norm = clamp_norm(torch.linalg.norm(v, dim=-1, keepdim=True))
        t = sqrt_ks * v_norm  # (..., 1)
        result = torch.cos(t) * x + (torch.sin(t) / t) * v
        return self.proj_manifold(result)

    def proj_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        """Project ambient vector onto tangent space at *x* (Def 5.16).

        proj(v) = v - ks * (x^T v) * x

        This ensures x^T proj(v) = 0.

        Args:
            x: Base point on S, shape ``(..., d_s+1)``.
            v: Arbitrary ambient vector, shape ``(..., d_s+1)``.

        Returns:
            Tangent vector at *x*, shape ``(..., d_s+1)``.
        """
        ks = self.curvature
        dot = (x * v).sum(dim=-1, keepdim=True)
        return v - ks * dot * x

    def proj_manifold(self, x: Tensor) -> Tensor:
        """Re-project onto the sphere of radius ``1/sqrt(ks)``.

        Args:
            x: Approximate sphere point, shape ``(..., d_s+1)``.

        Returns:
            Exact sphere point, shape ``(..., d_s+1)``.
        """
        r = self.radius
        return x * (r / clamp_norm(torch.linalg.norm(x, dim=-1, keepdim=True)))

    def inner(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """Standard Euclidean inner product of tangent vectors (Def 4.3).

        Args:
            x: Base point (unused for sphere).
            u: Tangent vector, shape ``(..., d_s+1)``.
            v: Tangent vector, shape ``(..., d_s+1)``.

        Returns:
            shape ``(...)``.
        """
        return (u * v).sum(dim=-1)

    def tangent_norm(self, x: Tensor, v: Tensor) -> Tensor:
        """Euclidean norm of tangent vector (Def 4.6, S component).

        Args:
            x: Base point (unused).
            v: Tangent vector, shape ``(..., d_s+1)``.

        Returns:
            shape ``(...)``.
        """
        n: Tensor = torch.linalg.norm(v, dim=-1)
        return n

    def origin(
        self,
        *batch_shape: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Canonical origin on the sphere (Def 3.7).

        x_origin = (1/sqrt(ks), 0, ..., 0)

        Args:
            *batch_shape: Leading batch dimensions.
            device: Target device.
            dtype: Target dtype.

        Returns:
            shape ``(*batch_shape, d_s+1)``.
        """
        ks = self.curvature
        o = torch.zeros(*batch_shape, self.ambient_dim, device=device, dtype=dtype)
        o[..., 0] = (1.0 / ks.sqrt()).to(device=device, dtype=dtype)
        return o

    def sample_noise(
        self,
        *batch_shape: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample uniformly on the sphere (Def 6.1, S component).

        Normalises Gaussian samples to lie on the sphere of radius
        ``1/sqrt(ks)``.

        Args:
            *batch_shape: Leading batch dimensions.
            device: Target device.
            dtype: Target dtype.
            generator: Optional RNG.

        Returns:
            shape ``(*batch_shape, d_s+1)``.
        """
        z = torch.randn(
            *batch_shape, self.ambient_dim, device=device, dtype=dtype,
            generator=generator,
        )
        z = z / clamp_norm(torch.linalg.norm(z, dim=-1, keepdim=True))
        return z * self.radius
