"""Numerical stability helpers for manifold operations."""

import torch
from torch import Tensor

# Small epsilon for numerical stability
EPS = 1e-7
# Maximum norm for clamping to avoid overflow in arccosh/arccos
MAX_NORM = 1e6


def safe_arccosh(x: Tensor) -> Tensor:
    """Numerically stable arccosh: clamps input to [1+eps, inf)."""
    return torch.acosh(x.clamp(min=1.0 + EPS))


def safe_arccos(x: Tensor) -> Tensor:
    """Numerically stable arccos: clamps input to [-1+eps, 1-eps]."""
    return torch.acos(x.clamp(min=-1.0 + EPS, max=1.0 - EPS))


def safe_sqrt(x: Tensor) -> Tensor:
    """Numerically stable sqrt: clamps input to [eps, inf)."""
    return torch.sqrt(x.clamp(min=EPS))


def safe_tanh(x: Tensor) -> Tensor:
    """Numerically stable tanh: clamps output to avoid exactly +-1."""
    return torch.tanh(x).clamp(min=-1.0 + EPS, max=1.0 - EPS)


def clamp_norm(v: Tensor, max_norm: float = MAX_NORM) -> Tensor:
    """Clamp the L2 norm of a vector to avoid overflow.

    Args:
        v: Input tensor, shape (..., dim).
        max_norm: Maximum allowed norm.

    Returns:
        Clamped tensor with same shape.
    """
    norms = v.norm(dim=-1, keepdim=True).clamp(min=EPS)
    scale = torch.where(norms > max_norm, max_norm / norms, torch.ones_like(norms))
    return v * scale


def lorentz_inner(u: Tensor, v: Tensor) -> Tensor:
    """Lorentz (Minkowski) inner product: -u0*v0 + u1*v1 + ... + un*vn.

    Args:
        u: Tensor of shape (..., d+1) in Lorentz coordinates.
        v: Tensor of shape (..., d+1) in Lorentz coordinates.

    Returns:
        Inner product values, shape (...).
    """
    # Time component (index 0) gets negative sign
    return -u[..., 0] * v[..., 0] + (u[..., 1:] * v[..., 1:]).sum(dim=-1)


def project_to_lorentz(x: Tensor) -> Tensor:
    """Project a point onto the Lorentz hyperboloid by fixing x0 = sqrt(1 + ||x_spatial||^2).

    Args:
        x: Tensor of shape (..., d+1). The spatial components x[..., 1:] are kept,
           and x[..., 0] is recomputed.

    Returns:
        Point on the hyperboloid, shape (..., d+1).
    """
    spatial = x[..., 1:]
    spatial_sq = (spatial * spatial).sum(dim=-1, keepdim=True)
    time = safe_sqrt(1.0 + spatial_sq)
    return torch.cat([time, spatial], dim=-1)


def project_to_sphere(x: Tensor) -> Tensor:
    """Project a point onto the unit sphere by normalizing.

    Args:
        x: Tensor of shape (..., d).

    Returns:
        Point on the unit sphere, shape (..., d).
    """
    return x / x.norm(dim=-1, keepdim=True).clamp(min=EPS)
