"""Noise prior for continuous flow matching (spec §7.1).

The discrete edge flow is masked (spec §8), not Bernoulli-prior: the
forward process is defined directly by a Bernoulli mask indicator at
each position (see ``discrete_flow.discrete_interpolation``), so no
edge noise sampler is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor

    from riemannfm.manifolds.product import RiemannFMProductManifold


def sample_continuous_noise(
    manifold: RiemannFMProductManifold,
    *batch_shape: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
    radius_h: float = 5.0,
    sigma_e: float = 1.0,
) -> Tensor:
    """Sample continuous noise from the product manifold prior (spec §7.1).

    Delegates to ``manifold.sample_noise`` which samples:
      - Hyperbolic: wrapped normal in the hyperbolic ball of radius ``radius_h``
      - Spherical: uniform on the sphere
      - Euclidean: Gaussian with std ``sigma_e``
    """
    return manifold.sample_noise(
        *batch_shape,
        device=device,
        dtype=dtype,
        generator=generator,
        radius_h=radius_h,
        sigma_e=sigma_e,
    )
