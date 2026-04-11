"""Tests for KGC task heads, metrics, and dataset."""

from __future__ import annotations

import torch

from riemannfm.manifolds.product import RiemannFMProductManifold
from riemannfm.tasks.kgc import RiemannFMKGCLinkHead, RiemannFMKGCRelHead
from riemannfm.tasks.metrics import (
    build_true_triples_index,
    compute_ranking_metrics,
    compute_relation_metrics,
    filtered_rank,
)

# Test constants.
B = 8
D_H = 8
D_S = 8
D_E = 8
K = 10  # relations
E = 100  # entities


def _make_manifold() -> RiemannFMProductManifold:
    return RiemannFMProductManifold(
        dim_hyperbolic=D_H,
        dim_spherical=D_S,
        dim_euclidean=D_E,
        learn_curvature=False,
    )


# ── Metrics ──────────────────────────────────────────────────────────


class TestFilteredRank:
    def test_perfect_rank(self) -> None:
        scores = torch.tensor([0.1, 0.2, 0.9, 0.3])
        # Target index 2 has the highest score -> rank 1.
        rank = filtered_rank(scores, target_idx=2, true_indices=torch.tensor([2]))
        assert rank == 1

    def test_filtered_removes_other_trues(self) -> None:
        # Index 0 scores highest (0.95), target is index 2 (0.9).
        # Without filtering, rank would be 2. With filtering index 0, rank is 1.
        scores = torch.tensor([0.95, 0.2, 0.9, 0.3])
        rank = filtered_rank(scores, target_idx=2, true_indices=torch.tensor([0, 2]))
        assert rank == 1

    def test_worst_rank(self) -> None:
        scores = torch.tensor([0.5, 0.6, 0.1, 0.7])
        rank = filtered_rank(scores, target_idx=2, true_indices=torch.tensor([2]))
        assert rank == 4  # lowest score


class TestRankingMetrics:
    def test_perfect_mrr(self) -> None:
        ranks = torch.tensor([1, 1, 1, 1])
        m = compute_ranking_metrics(ranks)
        assert m["mrr"] == 1.0
        assert m["hits@1"] == 1.0
        assert m["hits@10"] == 1.0

    def test_varied_ranks(self) -> None:
        ranks = torch.tensor([1, 2, 5, 20])
        m = compute_ranking_metrics(ranks)
        assert 0.0 < m["mrr"] < 1.0
        assert m["hits@1"] == 0.25  # only rank 1
        assert m["hits@3"] == 0.5   # ranks 1 and 2
        assert m["hits@10"] == 0.75  # ranks 1, 2, 5


class TestRelationMetrics:
    def test_perfect_accuracy(self) -> None:
        logits = torch.eye(K)
        targets = torch.arange(K)
        m = compute_relation_metrics(logits, targets)
        assert m["accuracy"] == 1.0

    def test_random_accuracy(self) -> None:
        logits = torch.randn(100, K)
        targets = torch.randint(0, K, (100,))
        m = compute_relation_metrics(logits, targets)
        assert 0.0 <= m["accuracy"] <= 1.0
        assert "mrr" in m


class TestTrueTripleIndex:
    def test_build_index(self) -> None:
        triples = torch.tensor([[0, 0, 1], [0, 0, 2], [1, 0, 0]])
        hr2t, rt2h = build_true_triples_index(triples)
        assert set(hr2t[(0, 0)].tolist()) == {1, 2}
        assert set(rt2h[(0, 0)].tolist()) == {1}
        assert set(rt2h[(0, 1)].tolist()) == {0}


# ── Link Prediction Head ────────────────────────────────────────────


class TestLinkPredHead:
    def test_forward_shape_manifold_dist(self) -> None:
        manifold = _make_manifold()
        D = manifold.ambient_dim
        head = RiemannFMKGCLinkHead(manifold, D, K, scoring="manifold_dist")

        h = manifold.proj_manifold(torch.randn(B, D))
        t = manifold.proj_manifold(torch.randn(B, D))
        r = torch.randint(0, K, (B,))

        scores = head(h, r, t)
        assert scores.shape == (B,)

    def test_forward_shape_bilinear(self) -> None:
        manifold = _make_manifold()
        D = manifold.ambient_dim
        head = RiemannFMKGCLinkHead(manifold, D, K, scoring="bilinear")

        h = torch.randn(B, D)
        t = torch.randn(B, D)
        r = torch.randint(0, K, (B,))

        scores = head(h, r, t)
        assert scores.shape == (B,)

    def test_score_all_tails(self) -> None:
        manifold = _make_manifold()
        D = manifold.ambient_dim
        head = RiemannFMKGCLinkHead(manifold, D, K, scoring="bilinear")

        h = torch.randn(B, D)
        r = torch.randint(0, K, (B,))
        all_emb = torch.randn(E, D)

        scores = head.score_all_tails(h, r, all_emb)
        assert scores.shape == (B, E)

    def test_score_all_heads(self) -> None:
        manifold = _make_manifold()
        D = manifold.ambient_dim
        head = RiemannFMKGCLinkHead(manifold, D, K, scoring="bilinear")

        t = torch.randn(B, D)
        r = torch.randint(0, K, (B,))
        all_emb = torch.randn(E, D)

        scores = head.score_all_heads(t, r, all_emb)
        assert scores.shape == (B, E)

    def test_gradient_flows(self) -> None:
        manifold = _make_manifold()
        D = manifold.ambient_dim
        head = RiemannFMKGCLinkHead(manifold, D, K, scoring="manifold_dist")

        h = manifold.proj_manifold(torch.randn(B, D))
        t = manifold.proj_manifold(torch.randn(B, D))
        r = torch.randint(0, K, (B,))

        scores = head(h, r, t)
        scores.sum().backward()

        assert head.rel_emb.weight.grad is not None
        assert head.rel_emb.weight.grad.abs().sum() > 0


# ── Relation Prediction Head ────────────────────────────────────────


class TestRelPredHead:
    def test_forward_shape(self) -> None:
        manifold = _make_manifold()
        D = manifold.ambient_dim
        head = RiemannFMKGCRelHead(D, K)

        h = torch.randn(B, D)
        t = torch.randn(B, D)

        logits = head(h, t)
        assert logits.shape == (B, K)

    def test_gradient_flows(self) -> None:
        manifold = _make_manifold()
        D = manifold.ambient_dim
        head = RiemannFMKGCRelHead(D, K)

        h = torch.randn(B, D)
        t = torch.randn(B, D)

        logits = head(h, t)
        logits.sum().backward()

        for p in head.mlp.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ── KGC Dataset ──────────────────────────────────────────────���───────


class TestKGCDataset:
    def test_train_mode(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from riemannfm.data.datasets.kgc_dataset import RiemannFMKGCDataset

        # Create fake triples.
        triples = torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 3]])
        processed = tmp_path / "processed"
        processed.mkdir()
        torch.save(triples, processed / "train_triples.pt")

        ds = RiemannFMKGCDataset(
            data_dir=str(tmp_path), split="train",
            num_entities=10, neg_samples=4, mode="train",
        )
        assert len(ds) == 3
        sample = ds[0]
        assert sample["head"].shape == (5,)  # 1 + 4 neg
        assert sample["label"][0] == 1.0
        assert sample["label"][1:].sum() == 0.0

    def test_eval_mode(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from riemannfm.data.datasets.kgc_dataset import RiemannFMKGCDataset

        triples = torch.tensor([[0, 0, 1], [1, 1, 2]])
        processed = tmp_path / "processed"
        processed.mkdir()
        torch.save(triples, processed / "val_triples.pt")

        ds = RiemannFMKGCDataset(
            data_dir=str(tmp_path), split="val",
            num_entities=10, mode="eval",
        )
        assert len(ds) == 2
        sample = ds[0]
        assert sample["head"].shape == (1,)
        assert "label" not in sample
