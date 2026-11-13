"""Euclidean manifold R^{d_e}."""

from __future__ import annotations

import torch
from torch import Tensor

from riemannfm.manifolds.base import RiemannFMManifold


class EuclideanManifold(RiemannFMManifold):
    """Flat Euclidean space R^{d_e}.

    All operations are trivial (exp = add, log = subtract, dist = L2 norm).
    Included for interface uniformity with the hyperbolic and spherical
    components inside the product manifold.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self._dim = dim

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def ambient_dim(self) -> int:
        return self._dim

    @property
    def curvature(self) -> Tensor:
        return torch.tensor(0.0)

    # ------------------------------------------------------------------
    # Manifold operations
    # ------------------------------------------------------------------

    def exp_map(self, x: Tensor, v: Tensor) -> Tensor:
        return x + v

    def log_map(self, x: Tensor, y: Tensor) -> Tensor:
        return y - x

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        d: Tensor = torch.linalg.norm(y - x, dim=-1)
        return d

    def proj_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        return v

    def proj_manifold(self, x: Tensor) -> Tensor:
        return x

    def inner(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        return (u * v).sum(dim=-1)

    def tangent_norm(self, x: Tensor, v: Tensor) -> Tensor:
        n: Tensor = torch.linalg.norm(v, dim=-1)
        return n

    def origin(
        self,
        *batch_shape: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        return torch.zeros(*batch_shape, self._dim, device=device, dtype=dtype)

    def sample_noise(
        self,
        *batch_shape: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
        sigma: float = 1.0,
    ) -> Tensor:
        """Sample Gaussian noise N(0, sigma^2 I) (Def 6.1, Euclidean component).

        Args:
            *batch_shape: Leading batch dimensions.
            device: Target device.
            dtype: Target dtype.
            generator: Optional RNG.
            sigma: Standard deviation of the Gaussian.

        Returns:
            shape ``(*batch_shape, d_e)``.
        """
        return sigma * torch.randn(
            *batch_shape, self._dim, device=device, dtype=dtype, generator=generator
        )
