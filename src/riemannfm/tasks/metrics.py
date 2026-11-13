"""KGC evaluation metrics: MRR, Hits@k, filtered ranking.

Standard knowledge graph completion metrics following the filtered
evaluation protocol (Bordes et al., 2013).
"""

from __future__ import annotations

import torch
from torch import Tensor


def filtered_rank(
    scores: Tensor,
    target_idx: int,
    true_indices: Tensor,
) -> int:
    """Compute filtered rank of the target entity.

    Filters out other known-true entities before ranking so that the
    target is only compared against genuinely negative candidates.

    Args:
        scores: Scores for all candidates, shape ``(num_entities,)``.
            Higher score = more plausible.
        target_idx: Index of the correct entity.
        true_indices: Indices of all known-true entities for this query
            (including the target itself).

    Returns:
        1-based filtered rank of the target entity.
    """
    target_score = scores[target_idx].item()

    # Mask known-true entities (except the target) with -inf.
    filtered_scores = scores.clone()
    filtered_scores[true_indices] = float("-inf")
    filtered_scores[target_idx] = target_score  # restore target

    # Rank: count how many candidates score strictly higher.
    rank = int((filtered_scores > target_score).sum().item()) + 1
    return rank


def compute_ranking_metrics(
    ranks: Tensor,
    ks: tuple[int, ...] = (1, 3, 10),
) -> dict[str, float]:
    """Compute standard KGC ranking metrics from a tensor of ranks.

    Args:
        ranks: 1-based ranks, shape ``(N,)``.
        ks: Tuple of k values for Hits@k.

    Returns:
        Dict with keys: ``mrr``, ``hits@1``, ``hits@3``, ``hits@10``, etc.
    """
    ranks_f = ranks.float()
    metrics: dict[str, float] = {
        "mrr": (1.0 / ranks_f).mean().item(),
    }
    for k in ks:
        metrics[f"hits@{k}"] = (ranks <= k).float().mean().item()
    return metrics


def compute_relation_metrics(
    logits: Tensor,
    targets: Tensor,
    ks: tuple[int, ...] = (1, 3),
) -> dict[str, float]:
    """Compute relation prediction metrics.

    Args:
        logits: Predicted logits, shape ``(N, K)``.
        targets: Ground-truth relation indices, shape ``(N,)``.
        ks: Tuple of k values for Hits@k.

    Returns:
        Dict with keys: ``accuracy``, ``hits@1``, ``hits@3``, etc.
    """
    preds = logits.argmax(dim=-1)
    metrics: dict[str, float] = {
        "accuracy": (preds == targets).float().mean().item(),
    }
    # Hits@k via ranking
    sorted_indices = logits.argsort(dim=-1, descending=True)
    ranks = torch.zeros(len(targets), dtype=torch.long, device=targets.device)
    for i in range(len(targets)):
        rank_pos = (sorted_indices[i] == targets[i]).nonzero(as_tuple=True)[0]
        ranks[i] = rank_pos.item() + 1  # 1-based
    for k in ks:
        metrics[f"hits@{k}"] = (ranks <= k).float().mean().item()
    metrics["mrr"] = (1.0 / ranks.float()).mean().item()
    return metrics


def build_true_triples_index(
    *triple_sets: Tensor,
) -> tuple[dict[tuple[int, int], Tensor], dict[tuple[int, int], Tensor]]:
    """Build lookup indices for filtered ranking.

    For each (h, r) pair, collects all known tail entities.
    For each (r, t) pair, collects all known head entities.

    Args:
        *triple_sets: One or more tensors of shape ``(M, 3)`` with columns
            ``[head, relation, tail]``.

    Returns:
        Tuple of (hr2t, rt2h) dicts mapping query keys to tensors of
        known-true entity indices.
    """
    hr2t: dict[tuple[int, int], list[int]] = {}
    rt2h: dict[tuple[int, int], list[int]] = {}

    for triples in triple_sets:
        for h, r, t in triples.tolist():
            hr2t.setdefault((h, r), []).append(t)
            rt2h.setdefault((r, t), []).append(h)

    # Convert lists to tensors for efficient indexing.
    hr2t_tensor = {k: torch.tensor(v, dtype=torch.long) for k, v in hr2t.items()}
    rt2h_tensor = {k: torch.tensor(v, dtype=torch.long) for k, v in rt2h.items()}

    return hr2t_tensor, rt2h_tensor
