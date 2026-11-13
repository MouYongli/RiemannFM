"""Masked discrete flow matching for edges (spec §8).

Per §8.1-8.3 of math_CN.v1.0.1, the edge state space is augmented with an
explicit mask indicator μ_{t,ij} ∈ {0, 1}. The forward process samples
μ_{t,ij} ~ Bernoulli(1 - α(t)) independently at each (i, j), where α(t)
is a monotonically increasing schedule with α(0)=0 and α(1)=1. At masked
positions the edge value tensor is set to 0_K; at unmasked positions it
equals the data value E_{1,ij}.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def schedule_alpha(t: Tensor, kind: str = "cosine") -> Tensor:
    """Edge schedule α(t) (Def 8.1).

    α(t) is the expected decoded coverage Pr[μ_{t,ij} = 0]. Satisfies
    α(0) = 0, α(1) = 1 and is monotonically increasing.

    Args:
        t: Time, any shape; values in [0, 1].
        kind: One of ``"linear"``, ``"cosine"``, ``"concave"``.

    Returns:
        α(t), same shape as ``t``.
    """
    if kind == "linear":
        return t
    if kind == "cosine":
        return 1.0 - torch.cos(math.pi * t / 2.0)
    if kind == "concave":
        return 1.0 - (1.0 - t) ** 2
    raise ValueError(f"Unknown schedule kind: {kind!r}")


def discrete_interpolation(
    E_1: Tensor,
    t: Tensor,
    schedule: str = "cosine",
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Masked-flow forward process for edges (spec §8.3).

    Given the data edge tensor E_1 and time t, sample the mask indicator
    μ_t ~ Bernoulli(1 - α(t)) per position (shared across K relations,
    since the mask covers a (i, j) slot as a whole), and zero out masked
    positions in the edge value tensor.

    Args:
        E_1: Data edge types, shape ``(B, N, N, K)``.
        t: Time, shape ``(B,)`` or ``(B, 1)`` or ``(B, 1, 1)``.
        schedule: Schedule kind for α(t).
        generator: Optional RNG.

    Returns:
        ``(E_t, mu_t)`` where ``E_t`` has shape ``(B, N, N, K)`` (data
        values at unmasked positions, zeros at masked) and ``mu_t`` has
        shape ``(B, N, N)`` (1 = masked, 0 = decoded).
    """
    B, N, _, _K = E_1.shape

    if t.dim() == 1:
        t = t.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
    elif t.dim() == 2:
        t = t.unsqueeze(-1)

    alpha = schedule_alpha(t, kind=schedule)  # same broadcast shape as t
    mask_prob = (1.0 - alpha).expand(B, N, N)

    mu_t = torch.bernoulli(mask_prob, generator=generator).to(E_1.dtype)  # (B, N, N)
    keep = (1.0 - mu_t).unsqueeze(-1)  # (B, N, N, 1), 1 at decoded positions
    E_t = keep * E_1

    return E_t, mu_t
