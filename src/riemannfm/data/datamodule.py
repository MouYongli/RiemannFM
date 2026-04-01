"""Lightning DataModule for RiemannFM pretraining.

Wraps RiemannFMKGDataset + RiemannFMGraphCollator into a
LightningDataModule that provides train/val/test DataLoaders.

Usage with Hydra config:
    dm = RiemannFMDataModule(cfg)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    # batch["edge_types"].shape == (B, N_max, N_max, K)
    # batch["node_text"].shape  == (B, N_max, d_c)
    # batch["node_mask"].shape  == (B, N_max)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from riemannfm.data.collator import RiemannFMGraphCollator
from riemannfm.data.datasets.pretrain_dataset import RiemannFMKGDataset

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import Tensor

logger = logging.getLogger(__name__)


class RiemannFMDataModule(LightningDataModule):
    """Lightning DataModule for RiemannFM pretraining.

    Hydra config keys used (under cfg.data and cfg.training):
        data.data_dir:          Path to dataset directory
        data.num_edge_types:    K — number of relation types
        data.max_nodes:         N_max — max nodes per subgraph
        data.max_hops:          BFS expansion hops
        data.num_workers:       DataLoader workers
        data.dim_text_emb:      d_c — text embedding dimension
        data.text_encoder:      encoder model name / slug
        training.batch_size:    batch size B

    Args:
        cfg: Hydra DictConfig with data and training sections.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self._train_dataset: RiemannFMKGDataset | None = None
        self._val_dataset: RiemannFMKGDataset | None = None
        self._test_dataset: RiemannFMKGDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create datasets for the requested stage.

        Args:
            stage: "fit", "validate", "test", or None (all).
        """
        data_cfg = self.cfg.data
        common_kwargs = {
            "data_dir": data_cfg.data_dir,
            "num_edge_types": data_cfg.num_edge_types,
            "max_nodes": data_cfg.max_nodes,
            "max_hops": data_cfg.get("max_hops", 2),
            "text_encoder": data_cfg.get("text_encoder", None),
            "dim_text_emb": data_cfg.get("dim_text_emb", 0),
        }

        if stage in ("fit", None):
            self._train_dataset = RiemannFMKGDataset(
                split="train",
                **common_kwargs,
            )
            self._val_dataset = RiemannFMKGDataset(
                split="val",
                epoch_size=min(
                    data_cfg.get("val_epoch_size", 1000),
                    1000,
                ),
                **common_kwargs,
            )

        if stage in ("test", None):
            self._test_dataset = RiemannFMKGDataset(
                split="test",
                epoch_size=min(
                    data_cfg.get("test_epoch_size", 1000),
                    1000,
                ),
                **common_kwargs,
            )

    @property
    def relation_text(self) -> Tensor:
        """Global relation text matrix C_R ∈ R^{Kxd_c} (Def 3.6).

        Shared across all subgraphs. Access after setup().
        """
        ds = self._train_dataset or self._val_dataset or self._test_dataset
        if ds is None:
            raise RuntimeError("Call setup() before accessing relation_text")
        return ds.relation_text

    @property
    def num_edge_types(self) -> int:
        """Number of relation types K."""
        return int(self.cfg.data.num_edge_types)

    @property
    def dim_text_emb(self) -> int:
        """Text embedding dimension d_c."""
        return int(self.cfg.data.get("dim_text_emb", 0))

    def _make_collator(self) -> RiemannFMGraphCollator:
        return RiemannFMGraphCollator(
            max_nodes=self.cfg.data.max_nodes,
            num_edge_types=self.cfg.data.num_edge_types,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataset is not None, "Call setup('fit') first"
        return DataLoader(
            self._train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.get("num_workers", 4),
            collate_fn=self._make_collator(),
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.cfg.data.get("num_workers", 4) > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataset is not None, "Call setup('fit') first"
        return DataLoader(
            self._val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.get("num_workers", 4),
            collate_fn=self._make_collator(),
            pin_memory=True,
            persistent_workers=self.cfg.data.get("num_workers", 4) > 0,
        )

    def test_dataloader(self) -> DataLoader:
        assert self._test_dataset is not None, "Call setup('test') first"
        return DataLoader(
            self._test_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.get("num_workers", 4),
            collate_fn=self._make_collator(),
            pin_memory=True,
            persistent_workers=self.cfg.data.get("num_workers", 4) > 0,
        )
