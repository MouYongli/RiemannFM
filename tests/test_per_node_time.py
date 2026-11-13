"""Per-node time broadcasting in continuous flow.

The collator writes ``t_node`` with M_x positions at 0 and M_c at 1, so
``geodesic_interpolation`` and ``vector_field_target`` must accept
``t`` of shape ``(B, N)``.  These regression tests confirm:

  1. Per-node ``t`` matches scalar ``t`` when the per-node tensor is
     filled uniformly with the same batch value.
  2. A per-node ``t=0`` column really pins ``x_t = x_0`` at that node.
"""

from __future__ import annotations

import torch

from riemannfm.flow.continuous_flow import (
    geodesic_interpolation,
    vector_field_target,
)
from riemannfm.manifolds.product import RiemannFMProductManifold

B = 3
N = 5
DIM_H = 4
DIM_S = 4
DIM_E = 4


def _manifold() -> RiemannFMProductManifold:
    return RiemannFMProductManifold(DIM_H, DIM_S, DIM_E)


def test_per_node_t_matches_scalar_when_uniform() -> None:
    manifold = _manifold()
    x_0 = manifold.sample_noise(B, N, radius_h=1.0)
    x_1 = manifold.sample_noise(B, N, radius_h=1.0)

    t_scalar = torch.rand(B)
    t_node = t_scalar.unsqueeze(-1).expand(B, N).contiguous()

    x_t_scalar = geodesic_interpolation(manifold, x_0, x_1, t_scalar)
    x_t_pernode = geodesic_interpolation(manifold, x_0, x_1, t_node)
    assert torch.allclose(x_t_scalar, x_t_pernode, atol=1e-5)

    u_scalar = vector_field_target(manifold, x_t_scalar, x_1, t_scalar)
    u_pernode = vector_field_target(manifold, x_t_pernode, x_1, t_node)
    assert torch.allclose(u_scalar, u_pernode, atol=1e-5)


def test_per_node_t_zero_pins_x_t_to_x_0() -> None:
    manifold = _manifold()
    x_0 = manifold.sample_noise(B, N, radius_h=1.0)
    x_1 = manifold.sample_noise(B, N, radius_h=1.0)

    t_node = torch.full((B, N), 0.5)
    t_node[:, 0] = 0.0  # first node pinned to noise

    x_t = geodesic_interpolation(manifold, x_0, x_1, t_node)
    # Pinned node: x_t must equal x_0 exactly.
    assert torch.allclose(x_t[:, 0], x_0[:, 0], atol=1e-5)
    # Other nodes: x_t should generally differ from x_0.
    diff = (x_t[:, 1:] - x_0[:, 1:]).abs().sum()
    assert diff > 1e-3
