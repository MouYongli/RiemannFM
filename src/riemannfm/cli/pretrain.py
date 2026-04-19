"""RiemannFM pretraining CLI (spec §22).

    uv run python -m riemannfm.cli.pretrain
    uv run python -m riemannfm.cli.pretrain model=small data=wikidata_5m_mini

Logger selection:
    uv run python -m riemannfm.cli.pretrain logger=default   # wandb+csv
    uv run python -m riemannfm.cli.pretrain logger=wandb
    uv run python -m riemannfm.cli.pretrain logger=csv
    uv run python -m riemannfm.cli.pretrain logger=none
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
from riemannfm.models.pretrain_heads import RiemannFMPretrainHeads

logger = logging.getLogger(__name__)


def _resolve_precision(mixed_precision: str | None) -> str:
    """Convert config precision string to Lightning format."""
    if mixed_precision == "bf16":
        return "bf16-mixed"
    if mixed_precision == "fp16":
        return "16-mixed"
    return "32-true"


def _find_wandb_run_id(ckpt_path: str) -> str | None:
    """Extract wandb run id from a previous run's output directory."""
    output_dir = Path(ckpt_path).resolve().parent.parent
    latest_run = output_dir / "wandb" / "latest-run"
    if not latest_run.exists():
        return None
    for f in latest_run.glob("run-*.wandb"):
        return f.stem.removeprefix("run-")
    return None


def _carry_over_csv_logs(ckpt_path: str, csv_logger: object) -> None:
    """Copy previous CSV metrics into the new CSVLogger directory."""
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
    warnings.filterwarnings("ignore", message=".*LeafSpec.*")
    warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")
    warnings.filterwarnings("ignore", message=".*AccumulateGrad.*stream.*")

    torch.set_float32_matmul_precision("medium")

    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    L.seed_everything(cfg.seed, workers=True)

    # Modality masking (spec §9) lives on the training config but
    # spans the data (mode sampling + per-node bits) and flow (text-mask
    # Beta distribution) modules.
    mm = cfg.training.get("modality_mask", {})
    mode_probs = (
        float(mm.get("p_full", 1.0)),
        float(mm.get("p_tm", 0.0)),
        float(mm.get("p_cm", 0.0)),
    )
    rho_tm = float(mm.get("rho_tm", 0.30))
    rho_cm = float(mm.get("rho_cm", 0.15))
    beta_a_text_mask = float(mm.get("beta_a_text_mask", 5.0))
    beta_b_text_mask = float(mm.get("beta_b_text_mask", 1.0))

    dm = instantiate(
        cfg.data,
        batch_size=cfg.training.batch_size,
        mode_probs=mode_probs,
        rho_tm=rho_tm,
        rho_cm=rho_cm,
    )
    dm.setup("fit")

    num_edge_types = cfg.data.num_edge_types
    num_entities = cfg.data.num_entities
    max_steps = cfg.training.max_steps

    input_text_dim = dm.dim_text_emb
    relation_text = dm.relation_text if input_text_dim > 0 else None

    manifold = instantiate(cfg.manifold)

    # Ablation flags (spec §11-16 per-module toggles).
    ab = cfg.ablation
    model = instantiate(
        cfg.model,
        manifold=manifold,
        num_edge_types=num_edge_types,
        input_text_dim=input_text_dim,
        use_a_r=bool(getattr(ab, "use_a_r", True)),
        use_c=bool(getattr(ab, "use_c", True)),
        use_d_vr=bool(getattr(ab, "use_d_vr", True)),
        use_d_ve=bool(getattr(ab, "use_d_ve", True)),
        use_e_v=bool(getattr(ab, "use_e_v", True)),
        use_e_r=bool(getattr(ab, "use_e_r", True)),
        use_geodesic_kernel=bool(getattr(ab, "use_geodesic_kernel", True)),
        use_relation_similarity_bias=bool(
            getattr(ab, "use_relation_similarity_bias", True),
        ),
    )

    flow = instantiate(
        cfg.flow,
        manifold=manifold,
        beta_a_text_mask=beta_a_text_mask,
        beta_b_text_mask=beta_b_text_mask,
    )

    lambda_align_R = float(getattr(cfg.training, "lambda_align_R", 0.0))
    loss_fn = RiemannFMCombinedLoss(
        manifold=manifold,
        lambda_X=float(getattr(cfg.training, "lambda_X", 1.0)),
        lambda_ex=float(getattr(cfg.training, "lambda_ex", 1.0)),
        lambda_ty=float(getattr(cfg.training, "lambda_ty", 0.5)),
        lambda_align_R=lambda_align_R,
        align_tau=float(getattr(cfg.training, "align_tau", 0.1)),
    )

    pretrain_heads = RiemannFMPretrainHeads(
        manifold=manifold,
        num_entities=num_entities,
        input_text_dim=input_text_dim,
        lambda_align_R=lambda_align_R,
        d_p=int(getattr(cfg.training, "d_p", 128)),
        rel_emb_dim=cfg.model.rel_emb_dim,
    )

    module = instantiate(
        cfg.training,
        manifold=manifold,
        model=model,
        flow=flow,
        loss_fn=loss_fn,
        pretrain_heads=pretrain_heads,
        relation_text=relation_text,
    )

    total_params = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    logger.info("Parameters: %s total, %s trainable", f"{total_params:,}", f"{trainable:,}")

    ckpt_path = cfg.paths.get("ckpt_path")
    loggers = _instantiate_loggers(cfg, ckpt_path=ckpt_path) or None

    ckpt_dir = f"{cfg.paths.output_dir}/checkpoints"
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="step={step}-val_loss={val/loss:.4f}",
            save_top_k=3,
            monitor="val/loss",
            mode="min",
            auto_insert_metric_name=False,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = instantiate(
        cfg.accelerator,
        max_steps=max_steps,
        precision=_resolve_precision(cfg.training.mixed_precision),
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        gradient_clip_val=cfg.training.max_grad_norm,
        val_check_interval=cfg.training.val_check_interval,
        check_val_every_n_epoch=None,
        limit_val_batches=cfg.training.limit_val_batches,
        num_sanity_val_steps=0,
        logger=loggers,
        callbacks=callbacks,
        enable_checkpointing=True,
        log_every_n_steps=10,
        deterministic=False,
    )

    if ckpt_path:
        logger.info("Resuming from checkpoint: %s", ckpt_path)
    else:
        logger.info("Starting pretraining from scratch...")
    trainer.fit(module, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
