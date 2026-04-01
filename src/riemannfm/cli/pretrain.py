"""RiemannFM pretraining CLI.

Launch pretraining with Hydra configuration:
    python -m riemannfm.cli.pretrain
    python -m riemannfm.cli.pretrain model=rieformer_small data=wikidata_5m_mini
    make pretrain ARGS="model=rieformer_small data=wikidata_5m_mini"

Logger selection (standard Hydra config group):
    python -m riemannfm.cli.pretrain logger=default      # wandb + csv (default)
    python -m riemannfm.cli.pretrain logger=wandb_only    # wandb only
    python -m riemannfm.cli.pretrain logger=csv_only      # csv only (offline)
    python -m riemannfm.cli.pretrain logger=none           # no logger
"""

import logging
import warnings

import hydra
import lightning as L
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


def _instantiate_loggers(cfg: DictConfig) -> list:
    """Instantiate loggers from cfg.logger dict.

    Each key in cfg.logger is a logger config with a ``_target_`` field.
    Returns a list of instantiated logger objects.
    """
    if not cfg.get("logger"):
        return []
    return [
        instantiate(lg_conf)
        for lg_conf in cfg.logger.values()
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf
    ]


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """RiemannFM pretraining entry point."""
    # Suppress noisy warnings from Lightning internals.
    warnings.filterwarnings("ignore", message=".*LeafSpec.*")
    warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")

    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Seed.
    L.seed_everything(cfg.seed, workers=True)

    # 1. Data — instantiate DataModule from cfg.data + batch_size.
    dm = instantiate(cfg.data, batch_size=cfg.training.batch_size)
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
        avg_edge_density=cfg.flow.avg_edge_density,
        w_max=cfg.training.w_max,
        temperature=cfg.training.temperature,
    )

    # 6. LitModule — pass all pre-built objects.
    module = instantiate(
        cfg.training,
        manifold=manifold,
        model=model,
        flow=flow,
        loss_fn=loss_fn,
        num_entities=num_entities,
        C_R=C_R,
    )

    total_params = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    logger.info("Parameters: %s total, %s trainable", f"{total_params:,}", f"{trainable:,}")

    # 7. Loggers (standard Hydra dict-of-loggers).
    loggers = _instantiate_loggers(cfg) or None

    # 8. Callbacks.
    ckpt_dir = f"{cfg.paths.output_dir}/checkpoints"
    callbacks = [
        # Save top-3 checkpoints by val/loss with metric in filename.
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="step={step}-val_loss={val/loss:.4f}",
            save_top_k=3,
            monitor="val/loss",
            mode="min",
            every_n_train_steps=cfg.training.val_check_interval,
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

    # Train.
    logger.info("Starting pretraining...")
    trainer.fit(module, datamodule=dm)


if __name__ == "__main__":
    main()
