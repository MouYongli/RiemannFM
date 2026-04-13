"""RiemannFM pretraining CLI.

Launch pretraining with Hydra configuration:
    python -m riemannfm.cli.pretrain
    python -m riemannfm.cli.pretrain model=small data=wikidata_5m_mini

Logger selection (standard Hydra config group):
    python -m riemannfm.cli.pretrain logger=default      # wandb + csv (default)
    python -m riemannfm.cli.pretrain logger=wandb    # wandb only
    python -m riemannfm.cli.pretrain logger=csv      # csv only (offline)
    python -m riemannfm.cli.pretrain logger=none           # no logger
"""

import logging
import warnings
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from omegaconf import DictConfig, OmegaConf

from riemannfm.losses.combined_loss import RiemannFMCombinedLoss

logger = logging.getLogger(__name__)




def _resolve_precision(mixed_precision: str | None) -> str:
    """Convert config precision string to Lightning format."""
    if mixed_precision == "bf16":
        return "bf16-mixed"
    if mixed_precision == "fp16":
        return "16-mixed"
    return "32-true"


def _find_wandb_run_id(ckpt_path: str) -> str | None:
    """Extract wandb run id from a previous run's output directory.

    Searches for ``wandb/latest-run/run-<id>.wandb`` relative to the
    checkpoint's parent output directory.
    """
    output_dir = Path(ckpt_path).resolve().parent.parent
    latest_run = output_dir / "wandb" / "latest-run"
    if not latest_run.exists():
        return None
    for f in latest_run.glob("run-*.wandb"):
        return f.stem.removeprefix("run-")
    return None


def _carry_over_csv_logs(ckpt_path: str, csv_logger: object) -> None:
    """Copy previous CSV metrics into the new CSVLogger directory.

    CSVLogger appends when ``metrics.csv`` already exists, so copying
    the old file ensures continuous logs across resume boundaries.
    """
    old_output = Path(ckpt_path).resolve().parent.parent
    old_metrics = list(old_output.glob("csv/*/metrics.csv"))
    if not old_metrics:
        return
    new_log_dir = Path(csv_logger.log_dir)  # type: ignore[attr-defined]
    new_log_dir.mkdir(parents=True, exist_ok=True)
    dest = new_log_dir / "metrics.csv"
    if not dest.exists():
        import shutil

        shutil.copy2(old_metrics[0], dest)
        logger.info("Copied previous CSV metrics: %s -> %s", old_metrics[0], dest)


def _instantiate_loggers(cfg: DictConfig, ckpt_path: str | None = None) -> list:
    """Instantiate loggers from cfg.logger dict.

    Each key in cfg.logger is a logger config with a ``_target_`` field.
    When *ckpt_path* is given, wandb loggers are configured to resume
    the previous run and CSV logs are carried over.
    """
    if not cfg.get("logger"):
        return []

    wandb_run_id = _find_wandb_run_id(ckpt_path) if ckpt_path else None

    loggers: list = []
    for lg_conf in cfg.logger.values():
        if not (isinstance(lg_conf, DictConfig) and "_target_" in lg_conf):
            continue
        if wandb_run_id and "WandbLogger" in lg_conf._target_:
            lg = instantiate(lg_conf, id=wandb_run_id, resume="must")
            logger.info("Resuming wandb run: %s", wandb_run_id)
        elif ckpt_path and "CSVLogger" in lg_conf._target_:
            lg = instantiate(lg_conf)
            _carry_over_csv_logs(ckpt_path, lg)
        else:
            lg = instantiate(lg_conf)
        loggers.append(lg)
    return loggers


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """RiemannFM pretraining entry point."""
    # Suppress noisy warnings from Lightning internals.
    warnings.filterwarnings("ignore", message=".*LeafSpec.*")
    warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")
    warnings.filterwarnings("ignore", message=".*AccumulateGrad.*stream.*")

    # Enable TF32 for H100/A100 Tensor Cores (trades minimal precision for speed).
    torch.set_float32_matmul_precision("medium")

    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Seed.
    L.seed_everything(cfg.seed, workers=True)

    # 1. Data — instantiate DataModule from cfg.data + batch_size.
    dm = instantiate(
        cfg.data,
        batch_size=cfg.training.batch_size,
        mask_ratio_c=float(getattr(cfg.training, "mask_ratio_c", 0.0)),
        mask_ratio_x=float(getattr(cfg.training, "mask_ratio_x", 0.0)),
    )
    dm.setup("fit")

    num_edge_types = cfg.data.num_edge_types
    num_entities = cfg.data.num_entities
    max_steps = cfg.training.max_steps

    # Text dimension is auto-detected from disk after dm.setup().
    input_text_dim = dm.dim_text_emb

    # Global relation text embeddings C_R (shared across all subgraphs).
    C_R = dm.relation_text if input_text_dim > 0 else None

    # 2. Manifold — all primitives, direct instantiate.
    manifold = instantiate(cfg.manifold)

    # 3. Model — pass manifold + ablation flags + runtime data.
    model = instantiate(
        cfg.model,
        manifold=manifold,
        num_edge_types=num_edge_types,
        input_text_dim=input_text_dim,
        use_geodesic_kernel=cfg.ablation.use_geodesic_kernel,
        use_ath_norm=cfg.ablation.use_ath_norm,
        use_edge_self_update=cfg.ablation.use_edge_self_update,
        use_dual_stream_cross=cfg.ablation.use_dual_stream_cross,
        use_text_condition=cfg.ablation.use_text_condition,
    )

    # 4. Flow — pass shared manifold.
    flow = instantiate(cfg.flow, manifold=manifold)

    # 5. Loss — constructed directly from training + flow params.
    loss_fn = RiemannFMCombinedLoss(
        manifold=manifold,
        lambda_disc=cfg.training.lambda_disc,
        mu_align=cfg.training.mu_align,
        nu_mask_c=float(getattr(cfg.training, "nu_mask_c", 0.0)),
        nu_mask_x=float(getattr(cfg.training, "nu_mask_x", 0.0)),
        neg_ratio=float(getattr(cfg.training, "neg_ratio", 1.0)),
        temperature=cfg.training.temperature,
        mask_c_temperature=float(
            getattr(cfg.training, "mask_c_temperature", 0.07),
        ),
        input_text_dim=input_text_dim,
        node_dim=cfg.model.node_dim,
        d_a=int(getattr(cfg.model, "text_proj_dim", 256)),
        max_align_nodes=int(getattr(cfg.training, "max_align_nodes", 128)),
    )

    # 6. LitModule — pass all pre-built objects.
    module = instantiate(
        cfg.training,
        manifold=manifold,
        model=model,
        flow=flow,
        loss_fn=loss_fn,
        num_entities=num_entities,
        input_text_dim=input_text_dim,
        C_R=C_R,
    )

    total_params = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    logger.info("Parameters: %s total, %s trainable", f"{total_params:,}", f"{trainable:,}")

    # 7. Loggers (standard Hydra dict-of-loggers).
    ckpt_path = cfg.paths.get("ckpt_path")
    loggers = _instantiate_loggers(cfg, ckpt_path=ckpt_path) or None

    # 8. Callbacks.
    ckpt_dir = f"{cfg.paths.output_dir}/checkpoints"
    callbacks = [
        # Save top-3 checkpoints by val/loss with metric in filename.
        # No `every_n_train_steps`: with `monitor` set, ModelCheckpoint fires
        # at validation_end so the val/loss baked into the filename is always
        # the freshly logged one (the train-step trigger could fire before
        # validation and stamp a stale value).
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="step={step}-val_loss={val/loss:.4f}",
            save_top_k=3,
            monitor="val/loss",
            mode="min",
            auto_insert_metric_name=False,
            save_last=True,  # always save last.ckpt for easy resume
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # 9. Trainer — instantiate from accelerator config + training overrides.
    trainer = instantiate(
        cfg.accelerator,
        max_steps=max_steps,
        precision=_resolve_precision(cfg.training.mixed_precision),
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        gradient_clip_val=cfg.training.max_grad_norm,
        val_check_interval=cfg.training.val_check_interval,
        check_val_every_n_epoch=None,  # interpret val_check_interval as global steps
        limit_val_batches=cfg.training.limit_val_batches,
        num_sanity_val_steps=0,  # skip sanity check (slow with multi-t)
        logger=loggers,
        callbacks=callbacks,
        enable_checkpointing=True,
        log_every_n_steps=10,
        deterministic=False,
    )

    # Train (resume from checkpoint if provided).
    if ckpt_path:
        logger.info("Resuming from checkpoint: %s", ckpt_path)
    else:
        logger.info("Starting pretraining from scratch...")
    trainer.fit(module, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
