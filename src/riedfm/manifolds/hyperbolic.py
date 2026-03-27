"""Hyperbolic manifold using the Lorentz (hyperboloid) model.

The Lorentz model represents the d-dimensional hyperbolic space as the upper sheet
of a two-sheeted hyperboloid in (d+1)-dimensional Minkowski space:

    H^d = {x in R^{d+1} : <x, x>_L = -1, x_0 > 0}

where <x, y>_L = -x_0*y_0 + x_1*y_1 + ... + x_d*y_d is the Lorentz inner product.

This model is chosen over the Poincare ball for better numerical stability at large
distances, which is critical for encoding deep hierarchies in knowledge graphs.
"""

import torch
from torch import Tensor

from riedfm.manifolds.base import Manifold
from riedfm.utils.manifold_utils import (
    EPS,
    clamp_norm,
    lorentz_inner,
    project_to_lorentz,
    safe_arccosh,
    safe_sqrt,
)


class LorentzManifold(Manifold):
    """Lorentz model of hyperbolic space H^d with curvature kappa < 0.

    Points are (d+1)-dimensional vectors on the hyperboloid <x,x>_L = -1/|kappa|.
    For simplicity, we work with curvature = -1 (unit hyperboloid) and scale
    distances by 1/sqrt(|kappa|) when computing distances.

    Attributes:
        curvature: Negative curvature parameter (default -1.0). Can be made learnable
                   by wrapping in nn.Parameter at the ProductManifold level.
    """

    def __init__(self, curvature: float = -1.0):
        assert curvature < 0, f"Hyperbolic curvature must be negative, got {curvature}"
        self.curvature = curvature

    @property
    def abs_c(self) -> float:
        """Absolute value of curvature."""
        return abs(self.curvature)

    @property
    def sqrt_abs_c(self) -> float:
        """Square root of absolute curvature, used as scaling factor."""
        return self.abs_c**0.5

    def exp_map(self, x: Tensor, v: Tensor) -> Tensor:
        """Exponential map on the Lorentz hyperboloid.

        exp_x(v) = cosh(||v||_L) * x + sinh(||v||_L) * v / ||v||_L

        where ||v||_L = sqrt(<v, v>_L) is the Lorentz norm of the tangent vector.
        """
        v_norm = self._tangent_norm(x, v).unsqueeze(-1).clamp(min=EPS)
        result = torch.cosh(v_norm) * x + torch.sinh(v_norm) * v / v_norm
        return project_to_lorentz(result)  # Numerical re-projection

    def log_map(self, x: Tensor, y: Tensor) -> Tensor:
        """Logarithmic map on the Lorentz hyperboloid.

        log_x(y) = d(x,y) * (y + <x,y>_L * x) / ||(y + <x,y>_L * x)||_L
        """
        xy_inner = lorentz_inner(x, y).unsqueeze(-1)  # <x, y>_L, should be <= -1
        # y + <x,y>_L * x gives the unnormalized direction in the tangent space
        direction = y + xy_inner * x
        direction_norm = self._tangent_norm(x, direction).unsqueeze(-1).clamp(min=EPS)
        d = safe_arccosh(-xy_inner.squeeze(-1)).unsqueeze(-1) / self.sqrt_abs_c
        return d * direction / direction_norm

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        """Geodesic distance on the hyperboloid.

        d(x, y) = (1/sqrt(|kappa|)) * arccosh(-<x, y>_L)
        """
        inner = lorentz_inner(x, y)
        return safe_arccosh(-inner) / self.sqrt_abs_c

    def proj_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        """Project v onto tangent space at x by enforcing <v, x>_L = 0.

        v_tangent = v + <v, x>_L * x
        """
        vx_inner = lorentz_inner(v, x).unsqueeze(-1)
        return v + vx_inner * x

    def sample_uniform(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        """Sample approximately uniformly by projecting Gaussian samples onto hyperboloid.

        The spatial coordinates are sampled from N(0, 1) and the time coordinate
        is computed to satisfy the hyperboloid constraint.
        """
        # shape[-1] is the ambient dimension d+1, so spatial dim is shape[-1]-1
        spatial_shape = shape[:-1] + (shape[-1] - 1,)
        spatial = torch.randn(spatial_shape, device=device)
        return project_to_lorentz(
            torch.cat([torch.zeros(*shape[:-1], 1, device=device), spatial], dim=-1)
        )

    def origin(self, dim: int, device: torch.device) -> Tensor:
        """Origin of the hyperboloid: (1, 0, 0, ..., 0) in Lorentz coordinates."""
        o = torch.zeros(dim, device=device)
        o[0] = 1.0
        return o

    def inner(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """Riemannian inner product on the hyperboloid equals the Lorentz inner product
        restricted to the tangent space (where it is positive-definite)."""
        return lorentz_inner(u, v)

    def _tangent_norm(self, x: Tensor, v: Tensor) -> Tensor:
        """Compute the norm of a tangent vector at x using the Lorentz metric.

        For tangent vectors, <v,v>_L >= 0, but due to numerics we clamp.
        """
        return safe_sqrt(lorentz_inner(v, v).clamp(min=0))
