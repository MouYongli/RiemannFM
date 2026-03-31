"""Discrete flow matching for edge types via CTMC (Continuous-Time Markov Chain).

Edge types are discrete variables e_ij in {0, 1, ..., K}.
The interpolation is a probabilistic "reveal" process:
    e_t = e_1 with probability t (reveal true type)
    e_t = e_0 with probability 1-t (keep noise type)

The network learns p_theta(e_1 | E_t, X_t, t, c), trained with cross-entropy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RiemannFMDiscreteFlow(nn.Module):
    """Discrete flow matching for edge types using CTMC framework.

    At time t=0, all edges are random noise.
    At time t=1, all edges show their true type.
    At intermediate t, each edge independently reveals its true type with probability t.
    """

    def __init__(self, num_edge_types: int):
        """
        Args:
            num_edge_types: Total number of edge types including 0 (no-edge).
                           So K real types means num_edge_types = K + 1.
        """
        super().__init__()
        self.num_edge_types = num_edge_types

    def interpolate(self, e_0: Tensor, e_1: Tensor, t: Tensor) -> Tensor:
        """Compute discrete interpolant: reveal true edge type with probability t.

        Args:
            e_0: Noise edge types, shape (N, N), values in {0, ..., K}.
            e_1: True edge types, shape (N, N), values in {0, ..., K}.
            t: Time parameter in [0, 1], scalar or shape (1,).

        Returns:
            Interpolated edge types e_t, shape (N, N).
        """
        t_val = (t.item() if t.numel() == 1 else t) if isinstance(t, Tensor) else t
        # Each edge independently decides whether to reveal
        mask = torch.rand_like(e_0.float()) < t_val
        return torch.where(mask, e_1, e_0)

    def compute_loss(
        self,
        p_pred: Tensor,
        e_1: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute cross-entropy loss for edge type prediction.

        Args:
            p_pred: Predicted logits, shape (N, N, K+1) or (num_edges, K+1).
            e_1: True edge types, shape matching p_pred[..., 0].
            mask: Optional boolean mask for valid edges, same shape as e_1.

        Returns:
            Scalar loss.
        """
        # Flatten for cross-entropy
        if p_pred.dim() == 3:
            N = p_pred.shape[0]
            logits = p_pred.reshape(-1, self.num_edge_types)
            targets = e_1.reshape(-1).long()
            if mask is not None:
                mask_flat = mask.reshape(-1)
            else:
                # Exclude diagonal (self-loops)
                diag_mask = ~torch.eye(N, dtype=torch.bool, device=p_pred.device).reshape(-1)
                mask_flat = diag_mask
        else:
            logits = p_pred
            targets = e_1.long()
            mask_flat = mask

        loss = F.cross_entropy(logits, targets, reduction="none")

        if mask_flat is not None:
            loss = loss * mask_flat.float()
            return loss.sum() / mask_flat.float().sum().clamp(min=1)
        return loss.mean()

    def sample_step(
        self,
        e_t: Tensor,
        p_pred: Tensor,
        t: float,
        dt: float,
    ) -> Tensor:
        """Take one discrete sampling step: probabilistically update edge types.

        With probability tau(t, dt), replace current type with argmax of predicted probs.

        Args:
            e_t: Current edge types, shape (N, N).
            p_pred: Predicted logits, shape (N, N, K+1).
            t: Current time.
            dt: Time step size.

        Returns:
            Updated edge types, shape (N, N).
        """
        # Transition probability: how likely to flip at this step
        # Using the rate: tau = dt / (1 - t) to ensure completion by t=1
        tau = min(dt / max(1.0 - t, 1e-7), 1.0)
        # Decide which edges to update
        update_mask = torch.rand_like(e_t.float()) < tau
        # Get predicted types
        new_types = p_pred.argmax(dim=-1)
        return torch.where(update_mask, new_types, e_t)
