"""Noise sampling for flow matching: manifold uniform + sparse edge prior."""

import torch
import torch.nn as nn
from torch import Tensor

from riedfm.manifolds.product import ProductManifold


class ManifoldNoiseSampler(nn.Module):
    """Sample noise points uniformly on the product manifold.

    For continuous node coordinates, the base distribution is:
    - Hyperbolic: wrapped normal (project Gaussian onto hyperboloid)
    - Spherical: uniform on the sphere (normalize Gaussian)
    - Euclidean: standard Gaussian
    """

    def __init__(self, manifold: ProductManifold):
        super().__init__()
        self.manifold = manifold

    def sample(self, num_nodes: int, device: torch.device) -> Tensor:
        """Sample noise coordinates for num_nodes nodes.

        Args:
            num_nodes: Number of nodes to sample.
            device: Torch device.

        Returns:
            Noise points on product manifold, shape (num_nodes, total_dim).
        """
        return self.manifold.sample_uniform((num_nodes,), device)

    def sample_batch(self, batch_sizes: list[int], device: torch.device) -> Tensor:
        """Sample noise for a batch of graphs with different sizes.

        Args:
            batch_sizes: List of node counts per graph.
            device: Torch device.

        Returns:
            Noise points, shape (sum(batch_sizes), total_dim).
        """
        total = sum(batch_sizes)
        return self.manifold.sample_uniform((total,), device)


class SparseEdgeNoiseSampler(nn.Module):
    """Sparse-aware noise prior for discrete edge types.

    Instead of uniform noise over all K+1 edge types (which produces
    unrealistically dense graphs), we bias toward type 0 (no-edge) to match
    the sparsity of real knowledge graphs.

    The noise distribution is:
        P(e=0) = 1 - avg_density
        P(e=k) = avg_density / K  for k = 1, ..., K
    """

    def __init__(self, num_edge_types: int, avg_density: float = 0.05):
        """
        Args:
            num_edge_types: Number of non-zero edge types K.
            avg_density: Expected edge density in real graphs (fraction of non-zero edges).
        """
        super().__init__()
        self.num_edge_types = num_edge_types
        self.avg_density = avg_density

        # Build categorical probabilities: [P(no-edge), P(type1), ..., P(typeK)]
        probs = torch.zeros(num_edge_types + 1)
        probs[0] = 1.0 - avg_density
        if num_edge_types > 0:
            probs[1:] = avg_density / num_edge_types
        self.register_buffer("probs", probs)

    def sample(self, num_nodes: int, device: torch.device) -> Tensor:
        """Sample noisy edge type matrix.

        Args:
            num_nodes: Number of nodes N.
            device: Torch device.

        Returns:
            Edge type matrix, shape (N, N), values in {0, 1, ..., K}.
        """
        probs = self.probs.to(device)
        # Sample from categorical for each edge slot
        flat = torch.multinomial(probs.expand(num_nodes * num_nodes, -1), 1).squeeze(-1)
        E = flat.reshape(num_nodes, num_nodes)
        # Zero out diagonal (no self-loops)
        E.fill_diagonal_(0)
        return E
