"""Virtual (absorbing) node utilities for handling variable-size graphs.

Virtual nodes act as padding to ensure fixed-size tensor operations.
During generation, nodes that converge to the absorbing state are
removed in post-processing.
"""

import torch
from torch import Tensor

from riedfm.manifolds.product import RieDFMProductManifold


def add_virtual_nodes(
    x: Tensor,
    edge_types: Tensor,
    target_size: int,
    manifold: RieDFMProductManifold,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pad a graph with virtual nodes to reach target_size.

    Virtual nodes are placed at the manifold origin with no edges.

    Args:
        x: Node coordinates, shape (N, D).
        edge_types: Edge type matrix, shape (N, N).
        target_size: Desired total number of nodes.
        manifold: Product manifold for origin computation.

    Returns:
        (x_padded, edge_padded, virtual_mask):
            x_padded: shape (target_size, D)
            edge_padded: shape (target_size, target_size)
            virtual_mask: shape (target_size,), True for virtual nodes
    """
    N = x.shape[0]
    assert target_size >= N, f"target_size ({target_size}) < N ({N})"

    if target_size == N:
        return x, edge_types, torch.zeros(N, dtype=torch.bool, device=x.device)

    device = x.device
    num_virtual = target_size - N

    # Virtual nodes at manifold origin
    origin = manifold.origin(device)
    virtual_x = origin.unsqueeze(0).expand(num_virtual, -1)
    x_padded = torch.cat([x, virtual_x], dim=0)

    # Padded edge types (virtual edges = 0)
    edge_padded = torch.zeros(target_size, target_size, dtype=edge_types.dtype, device=device)
    edge_padded[:N, :N] = edge_types

    # Virtual mask
    virtual_mask = torch.zeros(target_size, dtype=torch.bool, device=device)
    virtual_mask[N:] = True

    return x_padded, edge_padded, virtual_mask


def remove_virtual_nodes(
    x: Tensor,
    edge_types: Tensor,
    manifold: RieDFMProductManifold,
    threshold: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Remove virtual nodes from a generated graph.

    Virtual nodes are identified by their proximity to the manifold origin
    (absorbing state). Nodes within `threshold` distance of the origin
    are considered virtual and removed.

    Args:
        x: Node coordinates, shape (N, D).
        edge_types: Edge type matrix, shape (N, N).
        manifold: Product manifold.
        threshold: Distance threshold for virtual node detection.

    Returns:
        (x_real, edge_real): Coordinates and edge types for real nodes only.
    """
    origin = manifold.origin(x.device).unsqueeze(0)  # (1, D)
    dists = manifold.dist(x, origin.expand(x.shape[0], -1))  # (N,)

    real_mask = dists > threshold
    real_indices = real_mask.nonzero(as_tuple=True)[0]
    N_real = real_indices.shape[0]

    if N_real == 0:
        return x[:1], edge_types[:1, :1]

    x_real = x[real_indices]
    edge_real = edge_types[real_indices][:, real_indices]

    return x_real, edge_real
