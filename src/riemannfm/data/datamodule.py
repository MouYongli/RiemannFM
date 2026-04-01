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
        self._train_dataset: RiemannFMKGDataset | None = None
        self._val_dataset: RiemannFMKGDataset | None = None
        self._test_dataset: RiemannFMKGDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create datasets for the requested stage.

        Args:
            stage: "fit", "validate", "test", or None (all).
        """
        common_kwargs: dict[str, Any] = {
            "data_dir": self.data_dir,
            "num_edge_types": self._num_edge_types,
            "max_nodes": self.max_nodes,
            "max_hops": self.max_hops,
            "text_encoder": self.text_encoder,
        }

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

    def _make_collator(self) -> RiemannFMGraphCollator:
        return RiemannFMGraphCollator(
            max_nodes=self.max_nodes,
            num_edge_types=self._num_edge_types,
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
            collate_fn=self._make_collator(),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
