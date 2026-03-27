"""Spherical manifold for encoding symmetric/cyclic relations.

The d-dimensional sphere S^d is embedded in R^{d+1} as:
    S^d = {x in R^{d+1} : ||x|| = 1/sqrt(kappa)}

For unit sphere (kappa=1): S^d = {x : ||x|| = 1}.
"""

import math

import torch
from torch import Tensor

from riedfm.manifolds.base import Manifold
from riedfm.utils.manifold_utils import EPS, project_to_sphere, safe_arccos, safe_sqrt


class SphericalManifold(Manifold):
    """Spherical manifold S^d with curvature kappa > 0.

    Points are unit vectors in R^{d+1} (for the unit sphere, kappa=1).
    Distances are scaled by 1/sqrt(kappa).

    Attributes:
        curvature: Positive curvature parameter (default 1.0).
    """

    def __init__(self, curvature: float = 1.0):
        assert curvature > 0, f"Spherical curvature must be positive, got {curvature}"
        self.curvature = curvature

    @property
    def sqrt_c(self) -> float:
        return self.curvature**0.5

    def exp_map(self, x: Tensor, v: Tensor) -> Tensor:
        """Exponential map on the sphere.

        exp_x(v) = cos(||v||) * x + sin(||v||) * v / ||v||
        """
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=EPS)
        result = torch.cos(v_norm) * x + torch.sin(v_norm) * v / v_norm
        return project_to_sphere(result)  # Numerical re-projection

    def log_map(self, x: Tensor, y: Tensor) -> Tensor:
        """Logarithmic map on the sphere.

        log_x(y) = theta * (y - <x,y> * x) / ||y - <x,y> * x||
        where theta = arccos(<x, y>)
        """
        xy_inner = (x * y).sum(dim=-1, keepdim=True)
        direction = y - xy_inner * x
        direction_norm = direction.norm(dim=-1, keepdim=True).clamp(min=EPS)
        theta = safe_arccos(xy_inner) / self.sqrt_c
        return theta * direction / direction_norm

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        """Geodesic (great-circle) distance on the sphere.

        d(x, y) = (1/sqrt(kappa)) * arccos(<x, y>)
        """
        inner = (x * y).sum(dim=-1)
        return safe_arccos(inner) / self.sqrt_c

    def proj_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        """Project v onto tangent space at x by removing the normal component.

        v_tangent = v - <v, x> * x
        """
        vx_inner = (v * x).sum(dim=-1, keepdim=True)
        return v - vx_inner * x

    def sample_uniform(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        """Sample uniformly on the sphere by normalizing Gaussian vectors."""
        x = torch.randn(shape, device=device)
        return project_to_sphere(x)

    def origin(self, dim: int, device: torch.device) -> Tensor:
        """North pole: (1, 0, 0, ..., 0)."""
        o = torch.zeros(dim, device=device)
        o[0] = 1.0
        return o

    def inner(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """On the sphere, the Riemannian metric is the induced Euclidean metric."""
        return (u * v).sum(dim=-1)
