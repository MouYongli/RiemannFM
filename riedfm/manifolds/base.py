"""Abstract base class for Riemannian manifolds."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Manifold(ABC):
    """Abstract Riemannian manifold with core geometric operations.

    All concrete manifolds must implement exp/log maps, geodesic distance,
    tangent space projection, uniform sampling, and an origin point.
    """

    @abstractmethod
    def exp_map(self, x: Tensor, v: Tensor) -> Tensor:
        """Exponential map: move from point x along tangent vector v.

        Args:
            x: Base point on the manifold, shape (..., dim).
            v: Tangent vector at x, shape (..., dim).

        Returns:
            Point on the manifold, shape (..., dim).
        """

    @abstractmethod
    def log_map(self, x: Tensor, y: Tensor) -> Tensor:
        """Logarithmic map: tangent vector at x pointing toward y.

        Args:
            x: Base point on the manifold, shape (..., dim).
            y: Target point on the manifold, shape (..., dim).

        Returns:
            Tangent vector at x, shape (..., dim).
        """

    @abstractmethod
    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        """Geodesic distance between x and y.

        Args:
            x: Point on the manifold, shape (..., dim).
            y: Point on the manifold, shape (..., dim).

        Returns:
            Scalar distances, shape (...).
        """

    @abstractmethod
    def proj_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        """Project vector v onto the tangent space at x.

        Args:
            x: Base point on the manifold, shape (..., dim).
            v: Arbitrary vector, shape (..., dim).

        Returns:
            Tangent vector at x, shape (..., dim).
        """

    @abstractmethod
    def sample_uniform(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        """Sample points uniformly from the manifold.

        Args:
            shape: Output shape, last dim is the manifold dimension.
            device: Torch device.

        Returns:
            Points on the manifold, shape (*shape).
        """

    @abstractmethod
    def origin(self, dim: int, device: torch.device) -> Tensor:
        """Return the canonical origin point on the manifold.

        Args:
            dim: Manifold dimension.
            device: Torch device.

        Returns:
            Origin point, shape (dim,).
        """

    def geodesic(self, x: Tensor, y: Tensor, t: float | Tensor) -> Tensor:
        """Interpolate along the geodesic from x to y at time t in [0, 1].

        Args:
            x: Start point, shape (..., dim).
            y: End point, shape (..., dim).
            t: Interpolation parameter in [0, 1], scalar or broadcastable tensor.

        Returns:
            Interpolated point, shape (..., dim).
        """
        v = self.log_map(x, y)
        if isinstance(t, (int, float)):
            return self.exp_map(x, t * v)
        # Ensure t is broadcastable with v
        while t.dim() < v.dim():
            t = t.unsqueeze(-1)
        return self.exp_map(x, t * v)

    def inner(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """Riemannian inner product of tangent vectors u, v at point x.

        Default implementation uses Euclidean inner product.
        Override for non-Euclidean manifolds.

        Args:
            x: Base point, shape (..., dim).
            u: Tangent vector at x, shape (..., dim).
            v: Tangent vector at x, shape (..., dim).

        Returns:
            Inner product values, shape (...).
        """
        return (u * v).sum(dim=-1)

    def norm(self, x: Tensor, v: Tensor) -> Tensor:
        """Riemannian norm of tangent vector v at point x.

        Args:
            x: Base point, shape (..., dim).
            v: Tangent vector at x, shape (..., dim).

        Returns:
            Norm values, shape (...).
        """
        return self.inner(x, v, v).clamp(min=0).sqrt()
