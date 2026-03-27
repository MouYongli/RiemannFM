"""Logging and experiment tracking utilities."""

import logging
from typing import Any

logger = logging.getLogger("riedfm")


def setup_logging(level: str = "INFO"):
    """Configure logging for the project."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_wandb(cfg: dict) -> Any:
    """Initialize Weights & Biases run.

    Args:
        cfg: Configuration dictionary (from Hydra).

    Returns:
        wandb.Run instance, or None if wandb is not available.
    """
    try:
        import wandb

        run = wandb.init(
            project=cfg.get("project_name", "riedfm-g"),
            name=cfg.get("run_name"),
            config=dict(cfg),
            reinit=True,
        )
        return run
    except ImportError:
        logger.warning("wandb not available, skipping experiment tracking")
        return None


def log_metrics(metrics: dict[str, float], step: int, prefix: str = ""):
    """Log metrics to wandb and console.

    Args:
        metrics: Dictionary of metric names to values.
        step: Global step number.
        prefix: Optional prefix for metric names (e.g., "train/", "val/").
    """
    prefixed = {f"{prefix}{k}": v for k, v in metrics.items()}

    try:
        import wandb

        if wandb.run is not None:
            wandb.log(prefixed, step=step)
    except ImportError:
        pass

    logger.info(f"Step {step}: {prefixed}")


def log_manifold_stats(model, step: int):
    """Log manifold-specific statistics (curvatures, norms).

    Args:
        model: RieDFMG model instance.
        step: Global step number.
    """
    stats = {}

    if hasattr(model, "manifold"):
        manifold = model.manifold
        if hasattr(manifold, "curvature_h"):
            stats["manifold/curvature_h"] = manifold.curvature_h
        if hasattr(manifold, "curvature_s"):
            stats["manifold/curvature_s"] = manifold.curvature_s

    if stats:
        log_metrics(stats, step)
