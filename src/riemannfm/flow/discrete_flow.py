"""Discrete flow matching for edge types (Def 6.4).

Interpolation between noise edge types E_0 and data edge types E_1
using a shared Bernoulli mask z_ij that is shared across all K relation
types for each node pair (i, j).
"""

from __future__ import annotations

import torch
from torch import Tensor


def discrete_interpolation(
    E_0: Tensor,
    E_1: Tensor,
    t: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Discrete interpolation with shared mask (Def 6.4).

    For each node pair (i, j), sample a single Bernoulli variable:
        z_ij ~ Bernoulli(t)

    Then for ALL relation types k:
        E_t[i,j,k] = z_ij * E_1[i,j,k] + (1 - z_ij) * E_0[i,j,k]

    The key insight is that z_ij is shared across K, ensuring that
    edges switch from noise to data atomically per node pair.

    Args:
        E_0: Noise edge types, shape ``(B, N, N, K)``.
        E_1: Data edge types, shape ``(B, N, N, K)``.
        t: Time steps, shape ``(B,)`` or ``(B, 1)`` or ``(B, 1, 1)``.
        generator: Optional RNG.

    Returns:
        Interpolated edge types E_t, shape ``(B, N, N, K)``.
    """
    B, N, _, _K = E_1.shape

    # Ensure t has shape (B, 1, 1) for z_ij sampling.
    if t.dim() == 1:
        t = t.unsqueeze(-1).unsqueeze(-1)
    elif t.dim() == 2:
        t = t.unsqueeze(-1)

    # Sample z_ij ~ Bernoulli(t), shared across K.
    z = torch.bernoulli(
        t.expand(B, N, N),
        generator=generator,
    )  # (B, N, N), binary

    # Expand z to match edge dimensions: (B, N, N, 1) for broadcast over K.
    z = z.unsqueeze(-1)  # (B, N, N, 1)

    # Interpolate: when z=1 use data, when z=0 use noise.
    E_t = z * E_1 + (1.0 - z) * E_0

    return E_t
