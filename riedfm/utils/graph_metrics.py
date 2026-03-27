"""Graph-level evaluation metrics: structure F1, BERTScore consistency."""

import torch
from torch import Tensor


def structure_f1(
    pred_adj: Tensor,
    true_adj: Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute structure-level precision, recall, F1 for graph generation.

    Args:
        pred_adj: Predicted adjacency matrix, shape (N, N).
        true_adj: Ground truth adjacency matrix, shape (N, N).
        threshold: Threshold for binarizing predictions.

    Returns:
        Dictionary with "precision", "recall", "f1".
    """
    pred_binary = (pred_adj > threshold).float()
    true_binary = (true_adj > 0).float()

    # Exclude diagonal
    N = pred_binary.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool, device=pred_binary.device)
    pred_flat = pred_binary[mask]
    true_flat = true_binary[mask]

    tp = (pred_flat * true_flat).sum().item()
    fp = (pred_flat * (1 - true_flat)).sum().item()
    fn = ((1 - pred_flat) * true_flat).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def edge_type_accuracy(
    pred_types: Tensor,
    true_types: Tensor,
    ignore_no_edge: bool = True,
) -> float:
    """Accuracy of edge type prediction.

    Args:
        pred_types: Predicted edge types, shape (N, N).
        true_types: True edge types, shape (N, N).
        ignore_no_edge: If True, only evaluate on edges that exist in ground truth.

    Returns:
        Accuracy value.
    """
    if ignore_no_edge:
        mask = true_types > 0
        if mask.sum() == 0:
            return 0.0
        return (pred_types[mask] == true_types[mask]).float().mean().item()
    else:
        N = pred_types.shape[0]
        mask = ~torch.eye(N, dtype=torch.bool, device=pred_types.device)
        return (pred_types[mask] == true_types[mask]).float().mean().item()
