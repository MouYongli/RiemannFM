"""Noise priors for flow matching (Def 6.1-6.2).

Continuous noise: sample from the product manifold noise prior.
Discrete noise: per-relation Bernoulli(rho_k) for edge types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
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
    """Sample continuous noise from the product manifold prior (Def 6.1).

    Delegates to ``manifold.sample_noise`` which samples:
      - Hyperbolic: wrapped normal in the hyperbolic ball of radius ``radius_h``
      - Spherical: uniform on the sphere
      - Euclidean: Gaussian with std ``sigma_e``

    Args:
        manifold: Product manifold M = H x S x E.
        *batch_shape: Leading batch dimensions (e.g. B, N).
        device: Target device.
        dtype: Target dtype.
        generator: Optional RNG for reproducibility.
        radius_h: Geodesic radius for hyperbolic noise ball.
        sigma_e: Standard deviation for Euclidean Gaussian noise.

    Returns:
        Noise samples on the manifold, shape ``(*batch_shape, D)``.
    """
    return manifold.sample_noise(
        *batch_shape,
        device=device,
        dtype=dtype,
        generator=generator,
        radius_h=radius_h,
        sigma_e=sigma_e,
    )


def sample_discrete_noise(
    E_1: Tensor,
    avg_edge_density: float = 0.05,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample discrete noise from per-relation Bernoulli priors (Def 6.2).

    For each relation k, the noise prior is Bernoulli(rho_k) where rho_k
    is estimated as the average edge density (or per-relation density).

    For the MVP, we use a single ``avg_edge_density`` for all relations.

    Args:
        E_1: Target edge types (data), shape ``(B, N, N, K)``.
            Used only for shape; values are ignored.
        avg_edge_density: Expected edge density rho (probability of edge=1).
        generator: Optional RNG.

    Returns:
        Noise edge types E_0, shape ``(B, N, N, K)``, binary {0, 1}.
    """
    return torch.bernoulli(
        torch.full_like(E_1, avg_edge_density),
        generator=generator,
    )


def compute_edge_density(E_1: Tensor, node_mask: Tensor) -> Tensor:
    """Compute per-relation edge density rho_k from a batch (Def 6.2).

    Args:
        E_1: Target edge types, shape ``(B, N, N, K)``.
        node_mask: Bool mask, shape ``(B, N)``.

    Returns:
        Per-relation density, shape ``(K,)``.
    """
    # Only count real node pairs.
    pair_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)  # (B, N, N)
    num_pairs = pair_mask.sum().clamp(min=1)
    # Sum edges per relation, divide by number of valid pairs.
    rho = (E_1 * pair_mask.unsqueeze(-1)).sum(dim=(0, 1, 2)) / num_pairs
    return rho
