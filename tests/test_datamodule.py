"""Tests for the pretraining data pipeline: GraphData → Dataset → Collator → DataModule.

Validates that the data pipeline produces batches matching the math spec:
- edge_types: (B, N_max, N_max, K) multi-hot binary      [Def 3.4]
- node_text:  (B, N_max, d_c) float                       [Def 3.6]
- node_mask:  (B, N_max) bool                              [Def 3.8]
- node_ids:   (B, N_max) long                              [Def 3.7]
"""

from pathlib import Path
from typing import ClassVar

import pytest
import torch

from riemannfm.data.collator import RiemannFMGraphCollator
from riemannfm.data.graph import RiemannFMGraphData
from riemannfm.data.sampler import RiemannFMSubgraphSampler

# ─── Path to mini dataset (skip tests if not available) ─────────────────────

MINI_DATA_DIR = Path("data/wikidata_5m_mini")
HAS_MINI_DATA = (MINI_DATA_DIR / "processed" / "train_triples.pt").exists()
requires_mini_data = pytest.mark.skipif(not HAS_MINI_DATA, reason="wikidata_5m_mini not found")


# ─── RiemannFMGraphData ────────────────────────────────────────────────────

class TestGraphData:
    def test_creation(self):
        N, K, d_c = 5, 10, 768
        data = RiemannFMGraphData(
            edge_types=torch.zeros(N, N, K),
            node_text=torch.randn(N, d_c),
            node_mask=torch.ones(N, dtype=torch.bool),
            num_nodes=N,
            node_ids=torch.arange(N),
            num_edge_types=K,
        )
        assert data.total_nodes == N
        assert data.n_relations == K
        assert data.num_real_nodes == N
        assert data.num_edges == 0

    def test_multi_hot_edges(self):
        N, K, d_c = 3, 5, 16
        edge_types = torch.zeros(N, N, K)
        # Node 0 → Node 1 has relations 0 and 2
        edge_types[0, 1, 0] = 1.0
        edge_types[0, 1, 2] = 1.0
        # Node 1 → Node 2 has relation 4
        edge_types[1, 2, 4] = 1.0

        data = RiemannFMGraphData(
            edge_types=edge_types,
            node_text=torch.zeros(N, d_c),
            node_mask=torch.ones(N, dtype=torch.bool),
            num_nodes=N,
            node_ids=torch.arange(N),
            num_edge_types=K,
        )
        assert data.num_edges == 2  # 2 (i,j) pairs with edges
        assert edge_types[0, 1].sum() == 2  # multi-relational

    def test_to_device(self):
        N, K, d_c = 3, 5, 16
        data = RiemannFMGraphData(
            edge_types=torch.zeros(N, N, K),
            node_text=torch.randn(N, d_c),
            node_mask=torch.ones(N, dtype=torch.bool),
            num_nodes=N,
            node_ids=torch.arange(N),
            num_edge_types=K,
        )
        moved = data.to(torch.device("cpu"))
        assert moved.edge_types.device == torch.device("cpu")


# ─── RiemannFMSubgraphSampler ─────────────────────────────────────────────

class TestSubgraphSampler:
    @pytest.fixture
    def sampler(self):
        # Small synthetic KG: 10 entities, 3 relations
        triples = torch.tensor([
            [0, 0, 1], [1, 1, 2], [2, 0, 3], [3, 2, 4],
            [4, 1, 5], [5, 0, 6], [6, 2, 7], [7, 1, 8],
            [0, 2, 2], [1, 0, 3],  # multi-relational: 0→2 has rel 0, 1→3 has rel 0
        ])
        return RiemannFMSubgraphSampler(
            triples=triples, num_edge_types=3, max_nodes=6, max_hops=2,
        )

    def test_sample_returns_multi_hot(self, sampler):
        node_ids, edge_types = sampler.sample()
        N = len(node_ids)
        assert edge_types.shape == (N, N, 3)
        assert edge_types.dtype == torch.float32
        # Values should be 0 or 1
        assert ((edge_types == 0) | (edge_types == 1)).all()

    def test_sample_respects_max_nodes(self, sampler):
        node_ids, edge_types = sampler.sample()
        assert len(node_ids) <= 6
        assert edge_types.shape[0] <= 6

    def test_multi_relational_edges(self):
        # Two relations between same pair
        triples = torch.tensor([
            [0, 0, 1],  # entity 0 → entity 1 via relation 0
            [0, 1, 1],  # entity 0 → entity 1 via relation 1
        ])
        sampler = RiemannFMSubgraphSampler(
            triples=triples, num_edge_types=2, max_nodes=4, max_hops=1,
        )
        node_ids, edge_types = sampler.sample_from_seed(0)
        # Both nodes should be in subgraph
        assert set(node_ids) == {0, 1}
        local_0 = node_ids.index(0)
        local_1 = node_ids.index(1)
        # Both relations should be set
        assert edge_types[local_0, local_1, 0] == 1.0
        assert edge_types[local_0, local_1, 1] == 1.0


# ─── RiemannFMGraphCollator ───────────────────────────────────────────────

class TestGraphCollator:
    def test_padding_shapes(self):
        N_max, K, d_c = 8, 5, 16
        collator = RiemannFMGraphCollator(max_nodes=N_max, num_edge_types=K)

        graphs = []
        for n in [3, 5, 4]:
            graphs.append(RiemannFMGraphData(
                edge_types=torch.zeros(n, n, K),
                node_text=torch.randn(n, d_c),
                node_mask=torch.ones(n, dtype=torch.bool),
                num_nodes=n,
                node_ids=torch.arange(n),
                num_edge_types=K,
            ))

        batch = collator(graphs)
        B = 3
        assert batch["edge_types"].shape == (B, N_max, N_max, K)
        assert batch["node_text"].shape == (B, N_max, d_c)
        assert batch["node_mask"].shape == (B, N_max)
        assert batch["node_ids"].shape == (B, N_max)
        assert batch["batch_size"] == B

    def test_virtual_node_zeros(self):
        """Virtual nodes should have zero edges, zero text, mask=0 (Def 3.7)."""
        N_max, K, d_c = 6, 3, 8
        collator = RiemannFMGraphCollator(max_nodes=N_max, num_edge_types=K)

        graph = RiemannFMGraphData(
            edge_types=torch.ones(2, 2, K),  # all edges set
            node_text=torch.ones(2, d_c),
            node_mask=torch.ones(2, dtype=torch.bool),
            num_nodes=2,
            node_ids=torch.tensor([10, 20]),
            num_edge_types=K,
        )
        batch = collator([graph])

        # Real nodes (0, 1): mask True
        assert batch["node_mask"][0, :2].all()
        # Virtual nodes (2-5): mask False
        assert not batch["node_mask"][0, 2:].any()
        # Virtual node edges: all zero
        assert batch["edge_types"][0, 2:, :, :].sum() == 0
        assert batch["edge_types"][0, :, 2:, :].sum() == 0
        # Virtual node text: all zero
        assert batch["node_text"][0, 2:].sum() == 0
        # Virtual node IDs: -1
        assert (batch["node_ids"][0, 2:] == -1).all()

    def test_auto_max_nodes(self):
        K, d_c = 3, 8
        collator = RiemannFMGraphCollator(num_edge_types=K)

        graphs = []
        for n in [3, 7, 5]:
            graphs.append(RiemannFMGraphData(
                edge_types=torch.zeros(n, n, K),
                node_text=torch.randn(n, d_c),
                node_mask=torch.ones(n, dtype=torch.bool),
                num_nodes=n,
                node_ids=torch.arange(n),
                num_edge_types=K,
            ))

        batch = collator(graphs)
        assert batch["edge_types"].shape[1] == 7  # max in batch


# ─── Integration: Dataset + Collator on real mini data ─────────────────────

@requires_mini_data
class TestPretrainDataset:
    def test_dataset_loads(self):
        from riemannfm.data.datasets.pretrain_dataset import RiemannFMKGDataset

        ds = RiemannFMKGDataset(
            data_dir=str(MINI_DATA_DIR),
            split="train",
            num_edge_types=822,
            max_nodes=32,
            max_hops=2,
            text_encoder="nomic-embed-text",

        )
        assert len(ds) > 0
        assert ds.entity_emb is not None
        assert ds.relation_emb is not None
        assert ds.relation_text.shape == (822, 768)

    def test_single_sample_shapes(self):
        from riemannfm.data.datasets.pretrain_dataset import RiemannFMKGDataset

        ds = RiemannFMKGDataset(
            data_dir=str(MINI_DATA_DIR),
            split="train",
            num_edge_types=822,
            max_nodes=32,
            max_hops=2,
            text_encoder="nomic-embed-text",

        )
        sample = ds[0]

        # Check types
        assert isinstance(sample, RiemannFMGraphData)
        N = sample.total_nodes
        assert 1 <= N <= 32

        # Check shapes per math spec
        assert sample.edge_types.shape == (N, N, 822)
        assert sample.node_text.shape == (N, 768)
        assert sample.node_mask.shape == (N,)
        assert sample.node_ids.shape == (N,)

        # Check values
        assert sample.node_mask.all()  # no virtual nodes before collation
        assert ((sample.edge_types == 0) | (sample.edge_types == 1)).all()

    def test_collated_batch_shapes(self):
        from riemannfm.data.datasets.pretrain_dataset import RiemannFMKGDataset

        ds = RiemannFMKGDataset(
            data_dir=str(MINI_DATA_DIR),
            split="train",
            num_edge_types=822,
            max_nodes=32,
            max_hops=2,
            text_encoder="nomic-embed-text",

        )
        collator = RiemannFMGraphCollator(max_nodes=32, num_edge_types=822)

        samples = [ds[i] for i in range(4)]
        batch = collator(samples)

        B, N_max, K, d_c = 4, 32, 822, 768
        assert batch["edge_types"].shape == (B, N_max, N_max, K)
        assert batch["node_text"].shape == (B, N_max, d_c)
        assert batch["node_mask"].shape == (B, N_max)
        assert batch["node_ids"].shape == (B, N_max)
        assert batch["batch_size"] == B

        # Virtual nodes should be masked
        for i in range(B):
            n_real = batch["num_real_nodes"][i].item()
            assert batch["node_mask"][i, :n_real].all()
            if n_real < N_max:
                assert not batch["node_mask"][i, n_real:].any()


@requires_mini_data
class TestDataModule:
    _DM_KWARGS: ClassVar[dict[str, object]] = {
        "data_dir": str(MINI_DATA_DIR),
        "num_edge_types": 822,
        "max_nodes": 32,
        "max_hops": 2,
        "num_workers": 0,
        "text_encoder": "nomic-embed-text",
        "val_epoch_size": 100,
        "test_epoch_size": 100,
        "batch_size": 4,
    }

    def test_setup_and_first_batch(self):
        from riemannfm.data.datamodule import RiemannFMDataModule

        dm = RiemannFMDataModule(**self._DM_KWARGS)
        dm.setup("fit")

        # Get first training batch
        loader = dm.train_dataloader()
        batch = next(iter(loader))

        B, N_max, K, d_c = 4, 32, 822, 768
        assert batch["edge_types"].shape == (B, N_max, N_max, K)
        assert batch["node_text"].shape == (B, N_max, d_c)
        assert batch["node_mask"].shape == (B, N_max)
        assert batch["node_ids"].shape == (B, N_max)
        assert batch["batch_size"] == B

    def test_relation_text_property(self):
        from riemannfm.data.datamodule import RiemannFMDataModule

        dm = RiemannFMDataModule(**self._DM_KWARGS)
        dm.setup("fit")

        C_R = dm.relation_text
        assert C_R.shape == (822, 768)

    def test_val_dataloader(self):
        from riemannfm.data.datamodule import RiemannFMDataModule

        dm = RiemannFMDataModule(**self._DM_KWARGS)
        dm.setup("fit")

        val_loader = dm.val_dataloader()
        batch = next(iter(val_loader))
        assert batch["edge_types"].shape[0] == 4  # batch_size
        assert batch["edge_types"].shape[1] == 32  # N_max
