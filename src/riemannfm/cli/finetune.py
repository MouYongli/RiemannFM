"""RiemannFM KGC fine-tuning CLI.

Fine-tune a pretrained RiemannFM model for knowledge graph completion:

    python -m riemannfm.cli.finetune experiment=kgc_fb15k237 pretrain_ckpt=<path>
    python -m riemannfm.cli.finetune data=wn18rr task.task=kgc_lp pretrain_ckpt=<path>
    python -m riemannfm.cli.finetune task.task=kgc_rp data=fb15k_237 pretrain_ckpt=<path>
"""

import logging
import warnings
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from riemannfm.data.datasets.kgc_dataset import (
    RiemannFMKGCDataset,
    kgc_collate_eval,
    kgc_collate_train,
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
    """Load triples tensor for a given split."""
    split_map = {"val": "val", "valid": "val", "train": "train", "test": "test"}
    pt_path = data_dir / "processed" / f"{split_map.get(split, split)}_triples.pt"
    triples: torch.Tensor = torch.load(pt_path, map_location="cpu", weights_only=True)
    return triples


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """KGC fine-tuning entry point."""
    warnings.filterwarnings("ignore", message=".*LeafSpec.*")
    warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")
    warnings.filterwarnings("ignore", message=".*AccumulateGrad.*stream.*")

    torch.set_float32_matmul_precision("medium")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    L.seed_everything(cfg.seed, workers=True)

    # ── Data ──────────────────────────────────────────────────────────
    data_dir = Path(cfg.data.data_dir)
    num_entities = cfg.data.num_entities
    num_relations = cfg.data.num_edge_types
    task_name = cfg.task.task
    neg_samples = cfg.task.get("neg_samples", 256)
    batch_size = cfg.training.batch_size

    # Load all triples for filtered evaluation.
    train_triples = _load_triples(data_dir, "train")
    val_triples = _load_triples(data_dir, "val")
    test_triples = _load_triples(data_dir, "test")

    # Create datasets.
    train_ds = RiemannFMKGCDataset(
        data_dir=str(data_dir), split="train",
        num_entities=num_entities, neg_samples=neg_samples, mode="train",
    )
    val_ds = RiemannFMKGCDataset(
        data_dir=str(data_dir), split="val",
        num_entities=num_entities, mode="eval",
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, collate_fn=kgc_collate_train,
        pin_memory=torch.cuda.is_available(), drop_last=True,
        persistent_workers=cfg.data.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.task.get("eval_batch_size", 64), shuffle=False,
        num_workers=cfg.data.num_workers, collate_fn=kgc_collate_eval,
        persistent_workers=cfg.data.num_workers > 0,
    )

    # ── Model ─────────────────────────────────────────────────────────
    pretrain_ckpt = cfg.pretrain_ckpt
    scoring = cfg.task.get("scoring", "manifold_dist")
    hidden_dim = cfg.task.get("hidden_dim", 512)
    margin = cfg.task.get("margin", 1.0)

    module = RiemannFMKGCModule.from_pretrained(
        ckpt_path=pretrain_ckpt,
        task=task_name,
        num_relations=num_relations,
        scoring=scoring,
        hidden_dim=hidden_dim,
        train_triples=train_triples,
        val_triples=val_triples,
        test_triples=test_triples,
        backbone_frozen=cfg.training.backbone_frozen,
        backbone_lr=cfg.training.backbone_lr,
        head_lr=cfg.training.head_lr,
        curvature_lr=cfg.training.curvature_lr,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        max_steps=cfg.training.max_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        use_riemannian_optim=cfg.training.use_riemannian_optim,
        margin=margin,
    )

    total_params = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    logger.info("Parameters: %s total, %s trainable", f"{total_params:,}", f"{trainable:,}")

    # ── Loggers ───────────────────────────────────────────────────────
    loggers: list = []
    if cfg.get("logger"):
        for lg_conf in cfg.logger.values():
            if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
                loggers.append(hydra.utils.instantiate(lg_conf))

    # ── Callbacks ─────────────────────────────────────────────────────
    ckpt_dir = f"{cfg.paths.output_dir}/checkpoints"
    monitor_metric = "val/mrr" if task_name == "kgc_lp" else "val/accuracy"
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"step={{step}}-{monitor_metric.replace('/', '_')}={{{monitor_metric}:.4f}}",
            save_top_k=3,
            monitor=monitor_metric,
            mode="max",
            auto_insert_metric_name=False,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = hydra.utils.instantiate(
        cfg.accelerator,
        max_steps=cfg.training.max_steps,
        precision=_resolve_precision(cfg.training.mixed_precision),
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        gradient_clip_val=cfg.training.max_grad_norm,
        val_check_interval=cfg.training.val_check_interval,
        check_val_every_n_epoch=None,
        limit_val_batches=cfg.training.limit_val_batches,
        num_sanity_val_steps=0,
        logger=loggers or None,
        callbacks=callbacks,
        enable_checkpointing=True,
        log_every_n_steps=10,
        deterministic=False,
    )

    logger.info("Starting KGC fine-tuning (%s) ...", task_name)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
