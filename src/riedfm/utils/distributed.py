"""Distributed training utilities for multi-GPU DDP training.

Provides setup/cleanup for PyTorch DistributedDataParallel (DDP)
and gradient-preserving all-gather for contrastive loss.
"""

import logging
import os

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger(__name__)


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get the current process rank."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get the total number of processes."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main (rank 0) process."""
    return get_rank() == 0


def setup_distributed() -> torch.device:
    """Initialize distributed training from environment variables.

    Expects RANK, WORLD_SIZE, LOCAL_RANK, and MASTER_ADDR/MASTER_PORT
    to be set (typically by torchrun or torch.distributed.launch).

    Returns:
        The device assigned to this process.
    """
    if "RANK" not in os.environ:
        logger.info("No RANK env var found; running in single-GPU mode.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    logger.info(f"Initialized process {rank}/{world_size} on {device}")
    return device


def cleanup_distributed():
    """Clean up distributed process group."""
    if is_distributed():
        dist.destroy_process_group()


def wrap_ddp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Wrap model in DistributedDataParallel if in distributed mode.

    Args:
        model: The model to wrap.
        device: Device the model is on.

    Returns:
        DDP-wrapped model or original model if not distributed.
    """
    if is_distributed():
        return DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True)
    return model


def unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    """Get the underlying model from a DDP wrapper."""
    if isinstance(model, DistributedDataParallel):
        result: torch.nn.Module = model.module
        return result
    return model


class GatherWithGrad(torch.autograd.Function):
    """All-gather that preserves gradients for contrastive loss."""

    @staticmethod
    def forward(ctx, tensor: Tensor) -> Tensor:  # type: ignore[override]
        world_size = get_world_size()
        if world_size == 1:
            return tensor

        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        result: Tensor = torch.cat(gathered, dim=0)
        return result

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:  # type: ignore[override]
        world_size = get_world_size()
        if world_size == 1:
            return grad_output

        rank = get_rank()
        chunk_size = grad_output.shape[0] // world_size
        return grad_output[rank * chunk_size : (rank + 1) * chunk_size]


def all_gather_with_grad(tensor: Tensor) -> Tensor:
    """All-gather with gradient support for contrastive loss across GPUs."""
    result: Tensor = GatherWithGrad.apply(tensor)  # type: ignore[assignment]
    return result
