"""Numerical stability utilities for Riemannian manifold operations."""

from __future__ import annotations

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_NORM: float = 1e-15
"""Floor for tangent-vector norms to avoid division by zero."""

CLAMP_EPS: float = 1e-6
"""Margin inside the domain of arccosh / arccos to avoid boundary singularities."""

CURVATURE_EPS: float = 1e-5
"""Minimum absolute curvature to prevent degenerate geometry."""


# ---------------------------------------------------------------------------
# Safe inverse trigonometric / hyperbolic functions
# ---------------------------------------------------------------------------


def safe_arccosh(x: Tensor) -> Tensor:
    """Numerically stable ``arccosh``.

    Clamps input to ``[1 + eps, inf)`` so that both the forward value and
    the backward gradient (``1 / sqrt(x^2 - 1)``) remain finite.

    Args:
        x: Input tensor, shape ``(...)``.

    Returns:
        ``arccosh(x)`` with the same shape.
    """
    return torch.arccosh(x.clamp(min=1.0 + CLAMP_EPS))


def safe_arccos(x: Tensor) -> Tensor:
    """Numerically stable ``arccos``.

    Clamps input to ``[-1 + eps, 1 - eps]`` to avoid NaN at the boundaries.

    Args:
        x: Input tensor, shape ``(...)``.

    Returns:
        ``arccos(x)`` with the same shape.
    """
    return torch.arccos(x.clamp(min=-1.0 + CLAMP_EPS, max=1.0 - CLAMP_EPS))


# ---------------------------------------------------------------------------
# Norm helpers
# ---------------------------------------------------------------------------


def clamp_norm(v: Tensor, min_norm: float = MIN_NORM) -> Tensor:
    """Clamp a non-negative scalar tensor away from zero.

    Args:
        v: Non-negative scalar (norm) tensor, shape ``(...)``.
        min_norm: Lower bound.

    Returns:
        ``max(v, min_norm)`` element-wise.
    """
    return v.clamp(min=min_norm)


# ---------------------------------------------------------------------------
# Lorentz (Minkowski) helpers
# ---------------------------------------------------------------------------


def lorentz_inner(
    a: Tensor, b: Tensor, *, keepdim: bool = False
) -> Tensor:
    """Minkowski (Lorentz) inner product ``<a, b>_L``.

    .. math::
        \\langle a, b \\rangle_L = -a_0 b_0 + \\sum_{l=1}^{d} a_l b_l

    Args:
        a: shape ``(..., d+1)``.
        b: shape ``(..., d+1)``.
        keepdim: If ``True``, the last dimension is retained as size 1.

    Returns:
        shape ``(...)`` or ``(..., 1)``.
    """
    time = -a[..., :1] * b[..., :1]  # (..., 1)
    space = (a[..., 1:] * b[..., 1:]).sum(dim=-1, keepdim=True)  # (..., 1)
    result = time + space  # (..., 1)
    if not keepdim:
        result = result.squeeze(-1)
    return result


def lorentz_norm(v: Tensor, *, keepdim: bool = False) -> Tensor:
    """Riemannian norm of a tangent vector under the Lorentz metric.

    For a tangent vector on the hyperboloid, ``<v, v>_L >= 0``.  At ``v = 0``
    the naive ``sqrt(<v, v>_L)`` has an infinite backward gradient that
    PyTorch returns as ``NaN`` — unlike :func:`torch.linalg.norm`, which
    picks subgradient zero at the origin.  We mirror that behaviour with the
    double-``where`` trick: substitute a safe constant before ``sqrt`` and
    mask the result back to zero on the unsafe branch, so the backward never
    touches ``sqrt(0)``.

    Args:
        v: Tangent vector, shape ``(..., d+1)``.
        keepdim: If ``True``, the last dimension is retained as size 1.

    Returns:
        shape ``(...)`` or ``(..., 1)``.
    """
    sq = lorentz_inner(v, v, keepdim=keepdim).clamp(min=0.0)
    safe = sq > MIN_NORM
    safe_sq = torch.where(safe, sq, torch.ones_like(sq))
    return torch.where(safe, safe_sq.sqrt(), torch.zeros_like(sq))
