"""RiemannFM pretraining CLI.

Launch pretraining with Hydra configuration:
    python -m riemannfm.cli.pretrain
    python -m riemannfm.cli.pretrain model=rieformer_small data=wikidata_5m_mini
    make pretrain ARGS="model=rieformer_small data=wikidata_5m_mini"

Logger selection ("+" to combine):
    python -m riemannfm.cli.pretrain loggers=wandb+csv   # wandb + csv (default)
    python -m riemannfm.cli.pretrain loggers=wandb       # wandb only
    python -m riemannfm.cli.pretrain loggers=csv         # csv only (offline)
    python -m riemannfm.cli.pretrain loggers=null        # no logger
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
from omegaconf import DictConfig, OmegaConf, open_dict

from riemannfm.data.datamodule import RiemannFMDataModule
from riemannfm.models.lightning_module import RiemannFMPretrainModule

logger = logging.getLogger(__name__)


def _instantiate_loggers(cfg: DictConfig) -> list:
    """Instantiate loggers from ``cfg.loggers`` string.

    ``cfg.loggers`` is a "+" separated string of logger names
    (e.g. "wandb+csv", "csv", "wandb", "null").
    Each name maps to ``configs/logger/{name}.yaml`` which contains
    a top-level key wrapping a ``_target_`` (lightning-hydra-template style).

    Args:
        cfg: Full Hydra config (needed to resolve interpolations).

    Returns:
        List of instantiated loggers (empty list if none configured).
    """
    logger_str = cfg.get("loggers", "null")
    if not logger_str or logger_str == "null":
        return []

    loggers = []
    for name in str(logger_str).split("+"):
        name = name.strip()
        if not name:
            continue
        lcfg = OmegaConf.load(f"configs/logger/{name}.yaml")
        # Merge into main config so ${project_name} etc. resolve.
        with open_dict(cfg):
            merged = OmegaConf.merge(cfg, {"_loggers_": lcfg})
        for _key, lg_conf in merged._loggers_.items():
            if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
                logger.info("Instantiating logger <%s>", lg_conf._target_)
                loggers.append(instantiate(lg_conf))
    return loggers


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """RiemannFM pretraining entry point."""
    # Suppress noisy warnings from Lightning internals.
    warnings.filterwarnings("ignore", message=".*LeafSpec.*")
    warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")

    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Seed.
    L.seed_everything(cfg.seed, workers=True)

    # Data.
    dm = RiemannFMDataModule(cfg)
    dm.setup("fit")
    num_edge_types = cfg.data.num_edge_types
    num_entities = cfg.data.get("num_entities", 10000)
    max_steps = cfg.training.max_steps

    # Text dimension is auto-detected from disk after dm.setup().
    input_text_dim = dm.dim_text_emb

    # Global relation text embeddings C_R (shared across all subgraphs).
    C_R = dm.relation_text if input_text_dim > 0 else None

    # Model.
    module = RiemannFMPretrainModule.from_config(
        model_cfg=cfg.model,
        manifold_cfg=cfg.manifold,
        flow_cfg=cfg.flow,
        training_cfg=cfg.training,
        ablation_cfg=cfg.ablation,
        num_edge_types=num_edge_types,
        num_entities=num_entities,
        input_text_dim=input_text_dim,
        C_R=C_R,
        max_steps=max_steps,
    )

    total_params = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    logger.info("Parameters: %s total, %s trainable", f"{total_params:,}", f"{trainable:,}")

    # Loggers (config-driven: wandb, csv, or both).
    loggers = _instantiate_loggers(cfg) or None

    # Callbacks.
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

    # Precision.
    precision = cfg.training.get("mixed_precision", None)
    if precision == "bf16":
        precision_str = "bf16-mixed"
    elif precision == "fp16":
        precision_str = "16-mixed"
    else:
        precision_str = "32-true"

    # Trainer — use max_steps, not max_epochs.
    # Subgraph sampling is an infinite data stream, so epochs are meaningless.
    val_interval = cfg.training.val_check_interval
    trainer = L.Trainer(
        max_steps=max_steps,
        val_check_interval=val_interval,
        check_val_every_n_epoch=None,  # interpret val_check_interval as global steps
        limit_val_batches=cfg.training.get("limit_val_batches", 8),
        num_sanity_val_steps=0,  # skip sanity check (slow with multi-t)
        accelerator=cfg.accelerator.accelerator,
        devices=cfg.accelerator.devices,
        strategy=cfg.accelerator.strategy,
        precision=precision_str,
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        gradient_clip_val=cfg.training.max_grad_norm,
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
