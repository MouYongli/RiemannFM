"""Lightning DataModule for RiemannFM pretraining.

Wraps RiemannFMKGDataset + RiemannFMGraphCollator into a
LightningDataModule that provides train/val/test DataLoaders.

Usage with Hydra instantiate:
    dm = instantiate(cfg.data, batch_size=cfg.training.batch_size)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    # batch["edge_types"].shape == (B, N_max, N_max, K)
    # batch["node_text"].shape  == (B, N_max, d_c)
    # batch["node_mask"].shape  == (B, N_max)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from riemannfm.data.collator import RiemannFMGraphCollator
from riemannfm.data.datasets.pretrain_dataset import RiemannFMKGDataset
from riemannfm.data.sampler import RiemannFMSubgraphSampler

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


class RiemannFMDataModule(LightningDataModule):
    """Lightning DataModule for RiemannFM pretraining.

    Args:
        data_dir: Path to dataset directory.
        num_edge_types: K — number of relation types.
        max_nodes: N_max — max nodes per subgraph.
        max_hops: BFS expansion hops.
        num_workers: DataLoader workers.
        text_encoder: Encoder model name / slug.
        dim_text_emb: d_c — text embedding dimension (0 = auto-detect).
        val_epoch_size: Validation samples per epoch.
        test_epoch_size: Test samples per epoch.
        batch_size: Batch size B.
        **kwargs: Extra config keys (slug, dataset, etc.) absorbed by instantiate.
    """

    def __init__(
        self,
        data_dir: str,
        num_edge_types: int,
        max_nodes: int = 256,
        max_hops: int = 2,
        num_workers: int = 4,
        text_encoder: str | None = None,
        dim_text_emb: int = 0,
        val_epoch_size: int = 1000,
        test_epoch_size: int = 1000,
        batch_size: int = 64,
        mask_ratio_c: float = 0.0,
        mask_ratio_x: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__()
        self.data_dir = data_dir
        self._num_edge_types = num_edge_types
        self.max_nodes = max_nodes
        self.max_hops = max_hops
        self.num_workers = num_workers
        self.text_encoder = text_encoder
        self._dim_text_emb = dim_text_emb
        self.val_epoch_size = val_epoch_size
        self.test_epoch_size = test_epoch_size
        self.batch_size = batch_size
        self.mask_ratio_c = mask_ratio_c
        self.mask_ratio_x = mask_ratio_x
        self._train_dataset: RiemannFMKGDataset | None = None
        self._val_dataset: RiemannFMKGDataset | None = None
        self._test_dataset: RiemannFMKGDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create datasets for the requested stage.

        Loads shared resources (embeddings, train triples, sampler) once
        and passes them to all datasets to avoid redundant I/O and memory.
        Guarded against repeat calls from Lightning Trainer.

        Args:
            stage: "fit", "validate", "test", or None (all).
        """
        # Guard: skip if already set up for this stage.
        if stage in ("fit", None) and self._train_dataset is not None:
            return
        if stage == "test" and self._test_dataset is not None:
            return

        from pathlib import Path

        from riemannfm.data.pipeline.embed import encoder_slug

        data_path = Path(self.data_dir)

        # ── Load shared resources once ─────────────────────────────────
        # Text embeddings (loaded once, shared by all splits).
        entity_emb: Tensor | None = None
        relation_emb: Tensor | None = None
        if self.text_encoder:
            key = encoder_slug(self.text_encoder)
            emb_dir = data_path / "processed" / "text_embeddings"
            for p in sorted(emb_dir.glob(f"entity_emb_{key}_*.pt")):
                entity_emb = torch.load(p, map_location="cpu", weights_only=True)
                logger.info(f"Loaded shared entity embeddings: {entity_emb.shape} from {p.name}")
                break
            for p in sorted(emb_dir.glob(f"relation_emb_{key}_*.pt")):
                relation_emb = torch.load(p, map_location="cpu", weights_only=True)
                logger.info(f"Loaded shared relation embeddings: {relation_emb.shape} from {p.name}")
                break

        # Train triples + sampler (shared by train and val datasets).
        train_triples_path = data_path / "processed" / "train_triples.pt"
        train_triples = torch.load(train_triples_path, map_location="cpu", weights_only=True)
        logger.info(f"Loaded shared train_triples: {train_triples.shape}")

        sampler = RiemannFMSubgraphSampler(
            triples=train_triples,
            num_edge_types=self._num_edge_types,
            max_nodes=self.max_nodes,
            max_hops=self.max_hops,
        )

        common_kwargs: dict[str, Any] = {
            "data_dir": self.data_dir,
            "num_edge_types": self._num_edge_types,
            "max_nodes": self.max_nodes,
            "max_hops": self.max_hops,
            "text_encoder": self.text_encoder,
            "entity_emb": entity_emb,
            "relation_emb": relation_emb,
            "sampler": sampler,
            "train_triples": train_triples,
        }

        # ── Create datasets ────────────────────────────────────────────
        if stage in ("fit", None):
            self._train_dataset = RiemannFMKGDataset(
                split="train",
                **common_kwargs,
            )
            self._val_dataset = RiemannFMKGDataset(
                split="val",
                epoch_size=min(self.val_epoch_size, 1000),
                **common_kwargs,
            )

        if stage in ("test", None):
            self._test_dataset = RiemannFMKGDataset(
                split="test",
                epoch_size=min(self.test_epoch_size, 1000),
                **common_kwargs,
            )

    @property
    def relation_text(self) -> Tensor:
        """Global relation text matrix C_R in R^{Kxd_c} (Def 3.6).

        Shared across all subgraphs. Access after setup().
        """
        ds = self._train_dataset or self._val_dataset or self._test_dataset
        if ds is None:
            raise RuntimeError("Call setup() before accessing relation_text")
        return ds.relation_text

    @property
    def num_edge_types(self) -> int:
        """Number of relation types K."""
        return self._num_edge_types

    @property
    def dim_text_emb(self) -> int:
        """Text embedding dimension d_c (auto-detected from loaded data)."""
        ds = self._train_dataset or self._val_dataset or self._test_dataset
        if ds is not None:
            return ds.dim_text_emb
        return self._dim_text_emb

    def _make_collator(
        self,
        mask_ratio_c: float | None = None,
        mask_ratio_x: float | None = None,
    ) -> RiemannFMGraphCollator:
        return RiemannFMGraphCollator(
            max_nodes=self.max_nodes,
            num_edge_types=self._num_edge_types,
            mask_ratio_c=mask_ratio_c if mask_ratio_c is not None else self.mask_ratio_c,
            mask_ratio_x=mask_ratio_x if mask_ratio_x is not None else self.mask_ratio_x,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataset is not None, "Call setup('fit') first"
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._make_collator(),
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataset is not None, "Call setup('fit') first"
        # Mirror training mask ratios so val/L_mask_c and val/L_mask_x
        # are meaningful.  val_epoch_size averages out mask stochasticity.
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._make_collator(),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        assert self._test_dataset is not None, "Call setup('test') first"
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._make_collator(mask_ratio_c=0.0, mask_ratio_x=0.0),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
