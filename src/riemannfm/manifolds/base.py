"""Abstract base class for Riemannian manifold components."""

from __future__ import annotations

import abc

import torch
from torch import Tensor, nn


class RiemannFMManifold(nn.Module, abc.ABC):
    """Base interface for a single Riemannian manifold component.

    Each concrete subclass (Lorentz, Spherical, Euclidean) implements the
    manifold operations used throughout RiemannFM: exponential / logarithmic
    maps, geodesic distance, tangent-space projection, and noise sampling.

    All methods operate on the **last** tensor dimension (ambient coordinates)
    and support arbitrary leading batch dimensions ``(...)``.
    """

    # ------------------------------------------------------------------
    # Abstract properties
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Intrinsic dimension of the manifold (``d_h``, ``d_s``, or ``d_e``)."""

    @property
    @abc.abstractmethod
    def ambient_dim(self) -> int:
        """Ambient (embedding) dimension (``d_h+1``, ``d_s+1``, or ``d_e``)."""

    @property
    @abc.abstractmethod
    def curvature(self) -> Tensor:
        """Current curvature scalar (may be a clamped ``nn.Parameter``)."""

    # ------------------------------------------------------------------
    # Abstract methods -- manifold operations
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def exp_map(self, x: Tensor, v: Tensor) -> Tensor:
        """Exponential map: move from *x* along tangent vector *v*.

        Args:
            x: Base point on the manifold, shape ``(..., A)``.
            v: Tangent vector at *x*, shape ``(..., A)``.

        Returns:
            Point on the manifold, shape ``(..., A)``.
        """

    @abc.abstractmethod
    def log_map(self, x: Tensor, y: Tensor) -> Tensor:
        """Logarithmic map: tangent vector at *x* pointing toward *y*.

        Args:
            x: Base point, shape ``(..., A)``.
            y: Target point, shape ``(..., A)``.

        Returns:
            Tangent vector at *x*, shape ``(..., A)``.
        """

    @abc.abstractmethod
    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        """Geodesic distance between two points.

        Args:
            x: shape ``(..., A)``.
            y: shape ``(..., A)``.

        Returns:
            Non-negative scalar distance, shape ``(...)``.
        """

    @abc.abstractmethod
    def proj_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        """Project an ambient vector onto the tangent space at *x*.

        Args:
            x: Base point on the manifold, shape ``(..., A)``.
            v: Arbitrary ambient vector, shape ``(..., A)``.

        Returns:
            Tangent vector at *x*, shape ``(..., A)``.
        """

    @abc.abstractmethod
    def proj_manifold(self, x: Tensor) -> Tensor:
        """Re-project a point back onto the manifold (numerical drift fix).

        Args:
            x: Approximate manifold point, shape ``(..., A)``.

        Returns:
            Exact manifold point, shape ``(..., A)``.
        """

    @abc.abstractmethod
    def inner(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """Riemannian inner product of two tangent vectors at *x*.

        Args:
            x: Base point, shape ``(..., A)``.
            u: Tangent vector, shape ``(..., A)``.
            v: Tangent vector, shape ``(..., A)``.

        Returns:
            Inner product, shape ``(...)``.
        """

    @abc.abstractmethod
    def tangent_norm(self, x: Tensor, v: Tensor) -> Tensor:
        """Riemannian norm of a tangent vector at *x*.

        Args:
            x: Base point, shape ``(..., A)``.
            v: Tangent vector, shape ``(..., A)``.

        Returns:
            Non-negative norm, shape ``(...)``.
        """

    def tangent_norm_sq(self, x: Tensor, v: Tensor) -> Tensor:
        """Squared Riemannian norm of a tangent vector at *x*.

        Avoids the ``sqrt`` in :meth:`tangent_norm`, whose backward at zero is
        ``inf`` and produces ``NaN`` gradients when chained with masking.
        Subclasses should override if a more numerically stable implementation
        is available; the default delegates to :meth:`inner` and clamps to
        non-negative.

        Args:
            x: Base point, shape ``(..., A)``.
            v: Tangent vector, shape ``(..., A)``.

        Returns:
            Squared norm, shape ``(...)``.
        """
        return self.inner(x, v, v).clamp(min=0.0)

    @abc.abstractmethod
    def origin(
        self,
        *batch_shape: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Canonical origin point on the manifold (Def 3.7).

        Args:
            *batch_shape: Leading batch dimensions.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Origin point, shape ``(*batch_shape, A)``.
        """

    @abc.abstractmethod
    def sample_noise(
        self,
        *batch_shape: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample from the noise prior (Def 6.1).

        Args:
            *batch_shape: Leading batch dimensions.
            device: Target device.
            dtype: Target dtype.
            generator: Optional RNG for reproducibility.

        Returns:
            Noisy point on the manifold, shape ``(*batch_shape, A)``.
        """
