"""Model checkpoint save/load utilities."""

from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    path: str | Path,
    **extra,
):
    """Save a training checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state to save.
        epoch: Current epoch number.
        step: Current global step.
        path: Save path.
        **extra: Additional items to save (e.g., config, metrics).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        **extra,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Load a training checkpoint.

    Args:
        path: Checkpoint path.
        model: Optional model to load weights into.
        optimizer: Optional optimizer to load state into.
        device: Device to map tensors to.

    Returns:
        Full checkpoint dictionary.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def load_backbone(
    path: str | Path,
    model: nn.Module,
    strict: bool = False,
    device: torch.device = torch.device("cpu"),
):
    """Load only the backbone (RED-Former) weights for fine-tuning.

    Filters state dict to only include backbone keys and loads them.

    Args:
        path: Checkpoint path.
        model: Model to load weights into.
        strict: Whether to require exact key match.
        device: Device to map tensors to.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Filter for backbone keys
    backbone_dict = {
        k: v for k, v in state_dict.items() if k.startswith("backbone.")
    }

    model.load_state_dict(backbone_dict, strict=strict)
