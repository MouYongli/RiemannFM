"""RiemannFM KGC evaluation CLI.

Evaluate a pretrained or fine-tuned model on KGC benchmarks:

    # Evaluate fine-tuned checkpoint:
    python -m riemannfm.cli.evaluate data=fb15k_237 task.task=kgc_lp ckpt_path=<path>

    # Evaluate pretrained model directly (Wikidata5M transductive):
    python -m riemannfm.cli.evaluate data=wikidata_5m task.task=kgc_lp ckpt_path=<path>
"""

import logging
import warnings
from pathlib import Path

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from riemannfm.data.datasets.kgc_dataset import (
    RiemannFMKGCDataset,
    kgc_collate_eval,
)
from riemannfm.models.finetune_module import RiemannFMKGCModule

logger = logging.getLogger(__name__)


def _resolve_precision(mixed_precision: str | None) -> str:
    if mixed_precision == "bf16":
        return "bf16-mixed"
    if mixed_precision == "fp16":
        return "16-mixed"
    return "32-true"


def _load_triples(data_dir: Path, split: str) -> torch.Tensor:
    split_map = {"val": "val", "valid": "val", "train": "train", "test": "test"}
    pt_path = data_dir / "processed" / f"{split_map.get(split, split)}_triples.pt"
    triples: torch.Tensor = torch.load(pt_path, map_location="cpu", weights_only=True)
    return triples


def _try_load_kgc_module(ckpt_path: str, cfg: DictConfig) -> RiemannFMKGCModule:
    """Try loading as KGC module first, fall back to pretrained.

    If the checkpoint is a fine-tuned KGC module, load directly.
    If it's a pretrained checkpoint, wrap it in a KGC module.
    """
    data_dir = Path(cfg.data.data_dir)
    num_relations = cfg.data.num_edge_types
    task_name = cfg.task.task

    train_triples = _load_triples(data_dir, "train")
    val_triples = _load_triples(data_dir, "val")
    test_triples = _load_triples(data_dir, "test")

    # Try loading as fine-tuned KGC module.
    try:
        module = RiemannFMKGCModule.load_from_checkpoint(
            ckpt_path, map_location="cpu",
            train_triples=train_triples,
            val_triples=val_triples,
            test_triples=test_triples,
        )
        logger.info("Loaded fine-tuned KGC checkpoint: %s", ckpt_path)
        return module
    except Exception:
        logger.info("Not a KGC checkpoint, trying pretrained...")

    # Fall back to pretrained checkpoint.
    scoring = cfg.task.get("scoring", "manifold_dist")
    hidden_dim = cfg.task.get("hidden_dim", 512)
    margin = cfg.task.get("margin", 1.0)

    module = RiemannFMKGCModule.from_pretrained(
        ckpt_path=ckpt_path,
        task=task_name,
        num_relations=num_relations,
        scoring=scoring,
        hidden_dim=hidden_dim,
        train_triples=train_triples,
        val_triples=val_triples,
        test_triples=test_triples,
        backbone_frozen=True,
        backbone_lr=0.0,
        head_lr=0.0,
        margin=margin,
    )
    logger.info("Loaded pretrained checkpoint for evaluation: %s", ckpt_path)
    return module


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """KGC evaluation entry point."""
    warnings.filterwarnings("ignore", message=".*LeafSpec.*")
    warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")

    torch.set_float32_matmul_precision("medium")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    L.seed_everything(cfg.seed, workers=True)

    # ── Data ──────────────────────────────────────────────────────────
    data_dir = Path(cfg.data.data_dir)
    eval_split = cfg.get("eval_split", "test")

    test_ds = RiemannFMKGCDataset(
        data_dir=str(data_dir), split=eval_split,
        num_entities=cfg.data.num_entities, mode="eval",
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg.task.get("eval_batch_size", 64),
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=kgc_collate_eval,
    )

    # ── Model ─────────────────────────────────────────────────────────
    ckpt_path = cfg.ckpt_path
    module = _try_load_kgc_module(ckpt_path, cfg)

    # ── Loggers ───────────────────────────────────────────────────────
    loggers: list = []
    if cfg.get("logger"):
        for lg_conf in cfg.logger.values():
            if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
                loggers.append(hydra.utils.instantiate(lg_conf))

    # ── Trainer ───────────────────────────────────────────────────────
    precision = _resolve_precision(cfg.training.get("mixed_precision", "bf16"))
    trainer = hydra.utils.instantiate(
        cfg.accelerator,
        max_steps=-1,
        precision=precision,
        logger=loggers or None,
        enable_checkpointing=False,
        deterministic=False,
    )

    logger.info("Starting KGC evaluation on %s split...", eval_split)
    trainer.test(module, dataloaders=test_loader)


if __name__ == "__main__":
    main()
