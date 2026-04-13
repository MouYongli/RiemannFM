"""Continuous flow matching on Riemannian manifolds (Def 6.3, 6.5).

Geodesic interpolation between noise x_0 and data x_1, and the
conditional vector field target u_t for training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


def geodesic_interpolation(
    manifold: RiemannFMProductManifold,
    x_0: Tensor,
    x_1: Tensor,
    t: Tensor,
) -> Tensor:
    """Geodesic interpolation from noise to data (Def 6.3).

    x_t = exp_{x_0}( t * log_{x_0}(x_1) )

    This traces the geodesic from x_0 (noise) toward x_1 (data),
    reaching x_1 at t=1.

    Args:
        manifold: Product manifold.
        x_0: Noise samples, shape ``(B, N, D)``.
        x_1: Data samples, shape ``(B, N, D)``.
        t: Time steps.  Accepts:
            - ``(B,)``            — scalar per batch (classic flow).
            - ``(B, N)``          — per-node time (used for M_x/M_c mask labels).
            - ``(B, 1)`` / ``(B, 1, 1)`` — manual broadcast.

    Returns:
        Interpolated points on the manifold, shape ``(B, N, D)``.
    """
    # Broadcast t to (B, *, 1) against (B, N, D).
    # - (B,)   -> (B, 1, 1) broadcasts identically across N.
    # - (B, N) -> (B, N, 1) gives per-node time.
    if t.dim() == 1:
        t = t.unsqueeze(-1).unsqueeze(-1)
    elif t.dim() == 2:
        t = t.unsqueeze(-1)

    # Log map: direction from x_0 to x_1 in tangent space at x_0.
    v = manifold.log_map(x_0, x_1)  # (B, N, D)
    # Scale by t and exponentiate.
    x_t = manifold.exp_map(x_0, t * v)  # (B, N, D)
    return x_t


def vector_field_target(
    manifold: RiemannFMProductManifold,
    x_t: Tensor,
    x_1: Tensor,
    t: Tensor,
    t_max: float = 0.999,
) -> Tensor:
    """Conditional vector field target (Def 6.5).

    u_t(x_t | x_1) = (1 / (1 - t)) * log_{x_t}(x_1)

    This is the tangent vector at x_t pointing toward x_1, scaled by
    1/(1-t) to ensure the flow reaches x_1 at t=1.

    Args:
        manifold: Product manifold.
        x_t: Interpolated points, shape ``(B, N, D)``.
        x_1: Data points, shape ``(B, N, D)``.
        t: Time steps.  Accepts ``(B,)`` (scalar per batch) or ``(B, N)``
            (per-node; e.g. M_x nodes forced to t=0).
        t_max: Maximum time value to clamp at (avoids 1/(1-1) singularity).

    Returns:
        Vector field target (tangent vectors at x_t), shape ``(B, N, D)``.
    """
    # Broadcast t to (B, *, 1) against (B, N, D) — supports per-node time.
    if t.dim() == 1:
        t = t.unsqueeze(-1).unsqueeze(-1)
    elif t.dim() == 2:
        t = t.unsqueeze(-1)

    # Clamp t to avoid singularity at t=1.
    t_clamped = t.clamp(max=t_max)

    # Log map from x_t to x_1.
    v = manifold.log_map(x_t, x_1)  # (B, N, D)

    # Scale by 1/(1-t).
    scale = 1.0 / (1.0 - t_clamped)
    u_t = scale * v

    return u_t


def sample_time(
    batch_size: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
    distribution: str = "uniform",
    logit_normal_mu: float = 0.0,
    logit_normal_sigma: float = 1.0,
) -> Tensor:
    """Sample time steps from the specified distribution.

    Args:
        batch_size: Number of time steps to sample.
        device: Target device.
        dtype: Target dtype.
        generator: Optional RNG.
        distribution: ``"uniform"`` or ``"logit_normal"``.
        logit_normal_mu: Mean of the normal in logit space (logit_normal only).
        logit_normal_sigma: Std of the normal in logit space (logit_normal only).

    Returns:
        Time steps, shape ``(B,)``.
    """
    if distribution == "logit_normal":
        z = torch.randn(
            batch_size, device=device, dtype=dtype, generator=generator,
        )
        return torch.sigmoid(logit_normal_mu + logit_normal_sigma * z)
    # Default: uniform [0, 1).
    return torch.rand(
        batch_size, device=device, dtype=dtype, generator=generator,
    )
