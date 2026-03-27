"""Tests for KG benchmark pipeline: sampler, dataset, link prediction, metrics."""

import torch

from riedfm.manifolds.product import RieDFMProductManifold

DEVICE = torch.device("cpu")


def _make_small_manifold() -> RieDFMProductManifold:
    return RieDFMProductManifold(dim_hyperbolic=4, dim_spherical=4, dim_euclidean=4)


def _make_triples() -> list[tuple[int, int, int]]:
    """Create a small test KG."""
    return [
        (0, 0, 1),
        (0, 1, 2),
        (1, 0, 3),
        (2, 1, 3),
        (3, 0, 4),
        (4, 1, 0),
    ]


class TestSubgraphSampler:
    def test_sample_returns_three_tuple(self):
        from riedfm.data.subgraph_sampler import RieDFMSubgraphSampler

        sampler = RieDFMSubgraphSampler(_make_triples(), max_nodes=10, max_hops=2)
        result = sampler.sample()
        assert len(result) == 3
        node_ids, triples, depth_map = result
        assert isinstance(node_ids, list)
        assert isinstance(triples, list)
        assert isinstance(depth_map, dict)

    def test_sample_from_seed(self):
        from riedfm.data.subgraph_sampler import RieDFMSubgraphSampler

        sampler = RieDFMSubgraphSampler(_make_triples(), max_nodes=10, max_hops=2)
        node_ids, _triples, depth_map = sampler.sample_from_seed(0)
        # Seed node should always be included
        assert 0 in node_ids
        # Seed should have depth 0
        seed_local_idx = node_ids.index(0)
        assert depth_map[seed_local_idx] == 0

    def test_depth_map_monotonic(self):
        from riedfm.data.subgraph_sampler import RieDFMSubgraphSampler

        sampler = RieDFMSubgraphSampler(_make_triples(), max_nodes=10, max_hops=3)
        _node_ids, _triples, depth_map = sampler.sample_from_seed(0)
        # All depths should be non-negative
        for d in depth_map.values():
            assert d >= 0

    def test_sample_fanout_returns_three_tuple(self):
        from riedfm.data.subgraph_sampler import RieDFMSubgraphSampler

        sampler = RieDFMSubgraphSampler(_make_triples(), max_nodes=10)
        result = sampler.sample_fanout(0, 0)
        assert len(result) == 3


class TestKGDataset:
    def test_triple_mode(self):
        """Test that triple mode returns dict with correct keys."""
        from riedfm.data.kg_datasets import RieDFMKGDataset

        # Use a non-existent dir - should return empty dataset
        ds = RieDFMKGDataset(
            data_dir="/tmp/nonexistent_kg_data",
            mode="triple",
            auto_download=False,
        )
        assert ds.num_entities == 0
        assert len(ds) == 0

    def test_filter_masks(self):
        """Test filter mask construction."""
        from riedfm.data.kg_datasets import RieDFMKGDataset

        ds = RieDFMKGDataset(
            data_dir="/tmp/nonexistent_kg_data",
            mode="triple",
            auto_download=False,
        )
        # Manually add some triples
        ds.all_triples = [(0, 0, 1), (0, 0, 2), (0, 0, 3)]
        ds._hr2t.clear()
        ds._rt2h.clear()
        from collections import defaultdict

        ds._hr2t = defaultdict(set)
        ds._rt2h = defaultdict(set)
        for h, r, t in ds.all_triples:
            ds._hr2t[(h, r)].add(t)
            ds._rt2h[(r, t)].add(h)

        tail_filter, _head_filter = ds.get_filter_masks(0, 0, 1)
        # Other tails for (0, 0, ?): {2, 3} (excluding true answer 1)
        assert tail_filter == {2, 3}


class TestLinkPredictionHead:
    def test_score_shape(self):
        from riedfm.models.downstream.link_prediction import RieDFMLinkPredictionHead

        manifold = _make_small_manifold()
        head = RieDFMLinkPredictionHead(manifold, num_entities=100, num_relations=10)

        h = torch.tensor([0, 1, 2])
        r = torch.tensor([0, 1, 2])
        t = torch.tensor([3, 4, 5])

        scores = head.score(h, r, t)
        assert scores.shape == (3,)

    def test_score_all_tails_shape(self):
        from riedfm.models.downstream.link_prediction import RieDFMLinkPredictionHead

        manifold = _make_small_manifold()
        head = RieDFMLinkPredictionHead(manifold, num_entities=50, num_relations=5)

        h = torch.tensor([0, 1])
        r = torch.tensor([0, 1])

        scores = head.score_all_tails(h, r)
        assert scores.shape == (2, 50)

    def test_score_all_heads_shape(self):
        from riedfm.models.downstream.link_prediction import RieDFMLinkPredictionHead

        manifold = _make_small_manifold()
        head = RieDFMLinkPredictionHead(manifold, num_entities=50, num_relations=5)

        t = torch.tensor([3, 4])
        r = torch.tensor([0, 1])

        scores = head.score_all_heads(t, r)
        assert scores.shape == (2, 50)

    def test_gradient_flow(self):
        from riedfm.models.downstream.link_prediction import RieDFMLinkPredictionHead

        manifold = _make_small_manifold()
        head = RieDFMLinkPredictionHead(manifold, num_entities=20, num_relations=3)

        h = torch.tensor([0, 1])
        r = torch.tensor([0, 1])
        t = torch.tensor([2, 3])

        loss = head.link_prediction_loss(h, r, t)
        loss.backward()

        assert head.entity_embeds.weight.grad is not None
        assert head.relation_embeds.weight.grad is not None


class TestFilteredRanking:
    def test_compute_filtered_rank(self):
        from riedfm.utils.metrics import compute_filtered_rank

        scores = torch.tensor([0.5, 0.8, 0.3, 0.9, 0.1])
        # True entity is index 1 (score=0.8). Only index 3 (0.9) scores higher.
        rank = compute_filtered_rank(scores, true_idx=1, filter_ids=None)
        assert rank == 2  # 1-based: rank 2

        # If index 3 is a filter (other correct answer), exclude it
        rank_filtered = compute_filtered_rank(scores, true_idx=1, filter_ids={3})
        assert rank_filtered == 1  # Now rank 1

    def test_perfect_rank(self):
        from riedfm.utils.metrics import compute_filtered_rank

        scores = torch.tensor([1.0, 0.5, 0.3])
        rank = compute_filtered_rank(scores, true_idx=0, filter_ids=None)
        assert rank == 1


class TestCosineScheduler:
    def test_warmup_phase(self):
        from riedfm.utils.scheduler import build_cosine_scheduler

        optimizer = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=1.0)
        scheduler = build_cosine_scheduler(optimizer, warmup_steps=100, total_steps=1000)

        # At step 0, LR should be ~0
        assert scheduler.get_last_lr()[0] < 0.02

        # During warmup, LR should increase
        for _ in range(50):
            optimizer.step()
            scheduler.step()
        lr_mid = scheduler.get_last_lr()[0]
        assert 0.4 < lr_mid < 0.6

    def test_decay_phase(self):
        from riedfm.utils.scheduler import build_cosine_scheduler

        optimizer = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=1.0)
        scheduler = build_cosine_scheduler(optimizer, warmup_steps=10, total_steps=100)

        # Run through all steps
        for _ in range(100):
            optimizer.step()
            scheduler.step()

        # At the end, LR should be near min_lr_ratio (0.01)
        final_lr = scheduler.get_last_lr()[0]
        assert final_lr < 0.05
