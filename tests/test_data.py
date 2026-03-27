"""Tests for data loading and collation."""

import pytest
import torch

from riedfm.data.collator import GraphCollator
from riedfm.data.graph_data import GraphData
from riedfm.manifolds.product import ProductManifold

DEVICE = torch.device("cpu")


@pytest.fixture
def manifold():
    return ProductManifold(dim_hyperbolic=4, dim_spherical=4, dim_euclidean=4)


class TestGraphData:
    def test_basic_creation(self, manifold):
        N = 5
        x = manifold.sample_uniform((N,), DEVICE)
        e = torch.randint(0, 10, (N, N))
        data = GraphData(x=x, edge_types=e, num_nodes=N)
        assert data.num_nodes == N
        assert data.x.shape == (N, manifold.total_dim)

    def test_num_edges(self, manifold):
        N = 5
        x = manifold.sample_uniform((N,), DEVICE)
        e = torch.zeros(N, N, dtype=torch.long)
        e[0, 1] = 1
        e[1, 2] = 2
        data = GraphData(x=x, edge_types=e, num_nodes=N)
        assert data.num_edges == 2

    def test_to_device(self, manifold):
        N = 3
        x = manifold.sample_uniform((N,), DEVICE)
        e = torch.randint(0, 5, (N, N))
        data = GraphData(x=x, edge_types=e, num_nodes=N)
        data_cpu = data.to(torch.device("cpu"))
        assert data_cpu.x.device == torch.device("cpu")


class TestGraphCollator:
    def test_collation_shape(self, manifold):
        collator = GraphCollator(manifold, max_nodes=8)

        # Create 3 graphs of different sizes
        graphs = []
        for n in [3, 5, 4]:
            x = manifold.sample_uniform((n,), DEVICE)
            e = torch.randint(0, 10, (n, n))
            graphs.append(GraphData(x=x, edge_types=e, num_nodes=n))

        batch = collator(graphs)
        assert batch["x"].shape == (3, 8, manifold.total_dim)
        assert batch["edge_types"].shape == (3, 8, 8)
        assert batch["node_mask"].shape == (3, 8)
        assert batch["batch_size"] == 3

    def test_node_mask_correctness(self, manifold):
        collator = GraphCollator(manifold, max_nodes=6)

        x = manifold.sample_uniform((3,), DEVICE)
        e = torch.randint(0, 5, (3, 3))
        graphs = [GraphData(x=x, edge_types=e, num_nodes=3)]

        batch = collator(graphs)
        mask = batch["node_mask"][0]
        assert mask[:3].all()  # First 3 nodes are real
        assert not mask[3:].any()  # Rest are virtual

    def test_auto_max_nodes(self, manifold):
        """Without max_nodes, should pad to largest in batch."""
        collator = GraphCollator(manifold)

        graphs = []
        for n in [3, 7, 5]:
            x = manifold.sample_uniform((n,), DEVICE)
            e = torch.randint(0, 5, (n, n))
            graphs.append(GraphData(x=x, edge_types=e, num_nodes=n))

        batch = collator(graphs)
        assert batch["x"].shape[1] == 7  # Padded to max(3, 7, 5)
