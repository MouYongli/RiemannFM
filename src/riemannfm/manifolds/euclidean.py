"""Euclidean manifold (flat space) for encoding numerical attributes."""

import torch
from torch import Tensor

from riemannfm.manifolds.base import Manifold


class EuclideanManifold(Manifold):
    """Flat Euclidean space R^d.

    In Euclidean space, all geometric operations reduce to standard linear algebra:
    - Exponential map is vector addition
    - Logarithmic map is vector subtraction
    - Geodesics are straight lines
    - Distance is L2 norm
    """

    def exp_map(self, x: Tensor, v: Tensor) -> Tensor:
        return x + v

    def log_map(self, x: Tensor, y: Tensor) -> Tensor:
        return y - x

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        result: Tensor = (x - y).norm(dim=-1)
        return result

    def proj_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        # In Euclidean space, any vector is tangent everywhere
        return v

    def sample_uniform(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        return torch.randn(shape, device=device)

    def origin(self, dim: int, device: torch.device) -> Tensor:
        return torch.zeros(dim, device=device)

    def inner(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        return (u * v).sum(dim=-1)
