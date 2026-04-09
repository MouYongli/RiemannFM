"""Product manifold H^{d_h}_{kh} x S^{d_s}_{ks} x R^{d_e}.

Implements Definitions 3.2, 3.7, 4.2, 4.6, 5.2 from the paper.
Supports ablation configs where one or two components have dimension zero.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from riemannfm.manifolds.euclidean import EuclideanManifold
from riemannfm.manifolds.lorentz import LorentzManifold
from riemannfm.manifolds.spherical import SphericalManifold

if TYPE_CHECKING:
    from riemannfm.manifolds.base import RiemannFMManifold

# Canonical ordering of components.
_COMPONENT_ORDER = ("hyperbolic", "spherical", "euclidean")


class RiemannFMProductManifold(nn.Module):
    """Product Riemannian manifold M = H x S x E.

    Composes up to three component manifolds and dispatches all geometric
    operations component-wise.  Components whose intrinsic dimension is zero
    are omitted, enabling the ablation configurations in ``configs/manifold/``.

    Args:
        dim_hyperbolic: Intrinsic dimension of the hyperbolic component.
        dim_spherical: Intrinsic dimension of the spherical component.
        dim_euclidean: Dimension of the Euclidean component.
        init_curvature_h: Initial curvature kappa_h (negative).
        init_curvature_s: Initial curvature kappa_s (positive).
        learn_curvature: Whether curvatures are trainable parameters.
    """

    def __init__(
        self,
        dim_hyperbolic: int = 32,
        dim_spherical: int = 32,
        dim_euclidean: int = 32,
        init_curvature_h: float = -1.0,
        init_curvature_s: float = 1.0,
        learn_curvature: bool = True,
    ) -> None:
        super().__init__()

        self.components = nn.ModuleDict()
        self._split_sizes: list[int] = []
        self._component_names: list[str] = []

        if dim_hyperbolic > 0:
            self.components["hyperbolic"] = LorentzManifold(
                dim_hyperbolic, init_curvature_h, learn_curvature,
            )
            self._split_sizes.append(dim_hyperbolic + 1)
            self._component_names.append("hyperbolic")

        if dim_spherical > 0:
            self.components["spherical"] = SphericalManifold(
                dim_spherical, init_curvature_s, learn_curvature,
            )
            self._split_sizes.append(dim_spherical + 1)
            self._component_names.append("spherical")

        if dim_euclidean > 0:
            self.components["euclidean"] = EuclideanManifold(dim_euclidean)
            self._split_sizes.append(dim_euclidean)
            self._component_names.append("euclidean")

        if not self._component_names:
            msg = "At least one manifold component must have dim > 0"
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def ambient_dim(self) -> int:
        """Total ambient dimension D = (d_h+1) + (d_s+1) + d_e."""
        return sum(self._split_sizes)

    @property
    def num_components(self) -> int:
        return len(self._component_names)

    @property
    def hyperbolic(self) -> LorentzManifold | None:
        if "hyperbolic" in self.components:
            return self.components["hyperbolic"]  # type: ignore[return-value]
        return None

    @property
    def spherical(self) -> SphericalManifold | None:
        if "spherical" in self.components:
            return self.components["spherical"]  # type: ignore[return-value]
        return None

    @property
    def euclidean(self) -> EuclideanManifold | None:
        if "euclidean" in self.components:
            return self.components["euclidean"]  # type: ignore[return-value]
        return None

    def _get(self, name: str) -> RiemannFMManifold:
        """Return a typed component manifold."""
        return self.components[name]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Split / combine
    # ------------------------------------------------------------------

    def split(self, x: Tensor) -> dict[str, Tensor]:
        """Split concatenated coordinates into per-component tensors.

        Args:
            x: shape ``(..., D)`` where ``D = self.ambient_dim``.

        Returns:
            Dict mapping component name to tensor of shape ``(..., A_c)``.
        """
        parts = x.split(self._split_sizes, dim=-1)
        return dict(zip(self._component_names, parts))

    def combine(self, parts: dict[str, Tensor]) -> Tensor:
        """Concatenate per-component tensors into a single product vector.

        Args:
            parts: Dict mapping component name to tensor.

        Returns:
            shape ``(..., D)``.
        """
        return torch.cat(
            [parts[name] for name in self._component_names], dim=-1,
        )

    # ------------------------------------------------------------------
    # Component-wise dispatch helpers
    # ------------------------------------------------------------------

    def _apply_unary(
        self, method: str, x: Tensor, **kwargs: Any,
    ) -> Tensor:
        """Apply a unary manifold method component-wise and recombine."""
        x_parts = self.split(x)
        result = {
            name: getattr(self._get(name), method)(x_parts[name], **kwargs)
            for name in self._component_names
        }
        return self.combine(result)

    def _apply_binary(
        self, method: str, x: Tensor, y: Tensor, **kwargs: Any,
    ) -> Tensor:
        """Apply a binary manifold method component-wise and recombine."""
        x_parts = self.split(x)
        y_parts = self.split(y)
        result = {
            name: getattr(self._get(name), method)(
                x_parts[name], y_parts[name], **kwargs,
            )
            for name in self._component_names
        }
        return self.combine(result)

    # ------------------------------------------------------------------
    # Manifold operations
    # ------------------------------------------------------------------

    def exp_map(self, x: Tensor, v: Tensor) -> Tensor:
        """Component-wise exponential map (Def 4.5).

        Args:
            x: Base point, shape ``(..., D)``.
            v: Tangent vector, shape ``(..., D)``.

        Returns:
            shape ``(..., D)``.
        """
        return self._apply_binary("exp_map", x, v)

    def log_map(self, x: Tensor, y: Tensor) -> Tensor:
        """Component-wise logarithmic map (Def 4.4).

        Args:
            x: Base point, shape ``(..., D)``.
            y: Target point, shape ``(..., D)``.

        Returns:
            Tangent vector at *x*, shape ``(..., D)``.
        """
        return self._apply_binary("log_map", x, y)

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        """Product geodesic distance (Def 4.2).

        d_M(x, y) = sqrt( d_H^2 + d_S^2 + d_R^2 )

        Args:
            x: shape ``(..., D)``.
            y: shape ``(..., D)``.

        Returns:
            shape ``(...)``.
        """
        x_parts = self.split(x)
        y_parts = self.split(y)
        d_sq = torch.zeros(
            x.shape[:-1], device=x.device, dtype=x.dtype,
        )
        for name in self._component_names:
            d_sq = d_sq + self._get(name).dist(
                x_parts[name], y_parts[name],
            ).pow(2)
        return d_sq.sqrt()

    def proj_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        """Component-wise tangent projection (Def 5.16).

        Args:
            x: Base point, shape ``(..., D)``.
            v: Ambient vector, shape ``(..., D)``.

        Returns:
            Tangent vector at *x*, shape ``(..., D)``.
        """
        return self._apply_binary("proj_tangent", x, v)

    def proj_manifold(self, x: Tensor) -> Tensor:
        """Component-wise manifold re-projection.

        Args:
            x: Approximate product point, shape ``(..., D)``.

        Returns:
            Exact product point, shape ``(..., D)``.
        """
        return self._apply_unary("proj_manifold", x)

    def inner(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """Product Riemannian inner product.

        Args:
            x: Base point, shape ``(..., D)``.
            u: Tangent vector, shape ``(..., D)``.
            v: Tangent vector, shape ``(..., D)``.

        Returns:
            shape ``(...)``.
        """
        x_parts = self.split(x)
        u_parts = self.split(u)
        v_parts = self.split(v)
        total = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        for name in self._component_names:
            total = total + self._get(name).inner(
                x_parts[name], u_parts[name], v_parts[name],
            )
        return total

    def tangent_norm(self, x: Tensor, v: Tensor) -> Tensor:
        """Product tangent norm (Def 4.6).

        ||v||_{T_x M} = sqrt( ||v^H||_L^2 + ||v^S||^2 + ||v^R||^2 )

        Args:
            x: Base point, shape ``(..., D)``.
            v: Tangent vector, shape ``(..., D)``.

        Returns:
            shape ``(...)``.
        """
        return self.tangent_norm_sq(x, v).sqrt()

    def tangent_norm_sq(self, x: Tensor, v: Tensor) -> Tensor:
        """Squared product tangent norm without ``sqrt`` (Def 4.6).

        ||v||^2_{T_x M} = ||v^H||_L^2 + ||v^S||^2 + ||v^R||^2

        Dispatches to each component's :meth:`tangent_norm_sq` so that
        component-specific safety (e.g. Lorentz clamping) is preserved.

        Args:
            x: Base point, shape ``(..., D)``.
            v: Tangent vector, shape ``(..., D)``.

        Returns:
            shape ``(...)``.
        """
        x_parts = self.split(x)
        v_parts = self.split(v)
        norm_sq = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        for name in self._component_names:
            norm_sq = norm_sq + self._get(name).tangent_norm_sq(
                x_parts[name], v_parts[name],
            )
        return norm_sq

    def origin(
        self,
        *batch_shape: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Concatenated component origins (Def 3.7).

        Args:
            *batch_shape: Leading batch dimensions.
            device: Target device.
            dtype: Target dtype.

        Returns:
            shape ``(*batch_shape, D)``.
        """
        parts = {
            name: self._get(name).origin(
                *batch_shape, device=device, dtype=dtype,
            )
            for name in self._component_names
        }
        return self.combine(parts)

    def sample_noise(
        self,
        *batch_shape: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
        radius_h: float = 5.0,
        sigma_e: float = 1.0,
    ) -> Tensor:
        """Sample from the product noise prior (Def 6.1).

        Args:
            *batch_shape: Leading batch dimensions.
            device: Target device.
            dtype: Target dtype.
            generator: Optional RNG.
            radius_h: Geodesic radius for the hyperbolic noise ball.
            sigma_e: Std for the Euclidean Gaussian noise.

        Returns:
            shape ``(*batch_shape, D)``.
        """
        parts: dict[str, Tensor] = {}
        for name in self._component_names:
            m = self.components[name]
            if isinstance(m, LorentzManifold):
                parts[name] = m.sample_noise(
                    *batch_shape, device=device, dtype=dtype,
                    generator=generator, radius=radius_h,
                )
            elif isinstance(m, EuclideanManifold):
                parts[name] = m.sample_noise(
                    *batch_shape, device=device, dtype=dtype,
                    generator=generator, sigma=sigma_e,
                )
            elif isinstance(m, SphericalManifold):
                parts[name] = m.sample_noise(
                    *batch_shape, device=device, dtype=dtype,
                    generator=generator,
                )
        return self.combine(parts)

    def component_dists(self, x: Tensor, y: Tensor) -> dict[str, Tensor]:
        """Per-component geodesic distances.

        Useful for the geodesic kernel in manifold attention (Def 5.7).

        Args:
            x: shape ``(..., D)``.
            y: shape ``(..., D)``.

        Returns:
            Dict mapping component name to distance tensor of shape ``(...)``.
        """
        x_parts = self.split(x)
        y_parts = self.split(y)
        return {
            name: self._get(name).dist(x_parts[name], y_parts[name])
            for name in self._component_names
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Any) -> RiemannFMProductManifold:
        """Construct from a Hydra / OmegaConf config object.

        Expected keys: ``dim_hyperbolic``, ``dim_spherical``,
        ``dim_euclidean``, ``init_curvature_h``, ``init_curvature_s``,
        ``learn_curvature``.
        """
        return cls(
            dim_hyperbolic=cfg.dim_hyperbolic,
            dim_spherical=cfg.dim_spherical,
            dim_euclidean=cfg.dim_euclidean,
            init_curvature_h=cfg.init_curvature_h,
            init_curvature_s=cfg.init_curvature_s,
            learn_curvature=cfg.learn_curvature,
        )
