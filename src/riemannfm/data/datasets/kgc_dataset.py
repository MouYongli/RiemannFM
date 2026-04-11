"""Triple-level dataset for KGC fine-tuning and evaluation.

Unlike the pretrain dataset (subgraph sampling), this provides individual
(head, relation, tail) triples with negative sampling for training and
raw triples for evaluation.

Expected data layout:
    {data_dir}/processed/train_triples.pt       shape (M, 3)
    {data_dir}/processed/val_triples.pt         shape (M, 3)
    {data_dir}/processed/test_triples.pt        shape (M, 3)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class RiemannFMKGCDataset(Dataset[dict[str, Tensor]]):
    """Triple-level dataset for KGC fine-tuning and evaluation.

    Training mode: returns positive triples with on-the-fly negative
    sampling (corrupt head or tail uniformly).

    Evaluation mode: returns raw positive triples for external ranking.

    Args:
        data_dir: Path to dataset directory (with processed/ subdirectory).
        split: Dataset split (``"train"``, ``"val"``, ``"test"``).
        num_entities: Total number of entities E.
        neg_samples: Number of negative samples per positive (training only).
        mode: ``"train"`` for negative sampling, ``"eval"`` for raw triples.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_entities: int = 0,
        neg_samples: int = 256,
        mode: str = "train",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_entities = num_entities
        self.neg_samples = neg_samples
        self.mode = mode

        self.triples = self._load_triples(split)
        logger.info(
            "RiemannFMKGCDataset(%s, %s): %d triples, E=%d, neg=%d",
            split, mode, len(self.triples), num_entities, neg_samples,
        )

    def _load_triples(self, split: str) -> Tensor:
        """Load integer triples for a split."""
        split_map = {"val": "val", "valid": "val", "train": "train", "test": "test"}
        split_key = split_map.get(split, split)
        pt_path = self.data_dir / "processed" / f"{split_key}_triples.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Triples file not found: {pt_path}")
        triples: Tensor = torch.load(pt_path, map_location="cpu", weights_only=True)
        logger.info("  Loaded %s_triples.pt: %s", split_key, triples.shape)
        return triples

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return a single sample.

        Training mode returns:
            head: (1 + neg_samples,) — positive head + corrupted heads
            rel: (1 + neg_samples,) — relation (repeated)
            tail: (1 + neg_samples,) — positive tail + corrupted tails
            label: (1 + neg_samples,) — 1.0 for positive, 0.0 for negatives

        Eval mode returns:
            head: (1,)
            rel: (1,)
            tail: (1,)
        """
        h, r, t = self.triples[idx]

        if self.mode == "eval":
            return {
                "head": h.unsqueeze(0),
                "rel": r.unsqueeze(0),
                "tail": t.unsqueeze(0),
            }

        # Training: positive + negative samples.
        N = self.neg_samples
        heads = torch.empty(1 + N, dtype=torch.long)
        rels = torch.empty(1 + N, dtype=torch.long)
        tails = torch.empty(1 + N, dtype=torch.long)
        labels = torch.zeros(1 + N, dtype=torch.float32)

        # Positive sample at index 0.
        heads[0] = h
        rels[0] = r
        tails[0] = t
        labels[0] = 1.0

        # Negative samples: corrupt head or tail with equal probability.
        for i in range(N):
            if torch.rand(1).item() < 0.5:
                # Corrupt tail.
                heads[1 + i] = h
                rels[1 + i] = r
                tails[1 + i] = torch.randint(0, self.num_entities, (1,)).item()
            else:
                # Corrupt head.
                heads[1 + i] = torch.randint(0, self.num_entities, (1,)).item()
                rels[1 + i] = r
                tails[1 + i] = t

        return {
            "head": heads,
            "rel": rels,
            "tail": tails,
            "label": labels,
        }


def kgc_collate_train(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate training samples by stacking.

    Args:
        batch: List of dicts from RiemannFMKGCDataset (training mode).

    Returns:
        Dict with stacked tensors:
            head: (B, 1+N)
            rel: (B, 1+N)
            tail: (B, 1+N)
            label: (B, 1+N)
    """
    return {
        key: torch.stack([sample[key] for sample in batch])
        for key in batch[0]
    }


def kgc_collate_eval(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate evaluation samples by concatenating.

    Args:
        batch: List of dicts from RiemannFMKGCDataset (eval mode).

    Returns:
        Dict with concatenated tensors:
            head: (B,)
            rel: (B,)
            tail: (B,)
    """
    return {
        key: torch.cat([sample[key] for sample in batch])
        for key in batch[0]
    }
