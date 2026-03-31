"""Learning rate schedulers for RiemannFM training.

Provides cosine decay with linear warmup, the standard schedule
for transformer pretraining and finetuning.
"""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_cosine_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Build a cosine LR scheduler with linear warmup.

    Schedule:
        - Steps [0, warmup_steps): linear warmup from 0 to peak LR
        - Steps [warmup_steps, total_steps]: cosine decay from peak to min_lr_ratio * peak

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps.
        min_lr_ratio: Minimum LR as fraction of peak LR.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return current_step / max(warmup_steps, 1)
        # Cosine decay
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
