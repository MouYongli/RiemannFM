"""Time and positional embeddings for RiemannFM (Def 5.2).

Sinusoidal time embedding with learnable MLP projection, following the
standard diffusion / flow-matching convention.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class RiemannFMTimeEmbedding(nn.Module):
    """Sinusoidal time embedding with learnable MLP projection (Def 5.2).

    Given a scalar time step t in [0, 1], produces a d-dimensional embedding:
      1. Sinusoidal encoding: [sin(w_1 t), cos(w_1 t), ..., sin(w_F t), cos(w_F t)]
         with log-spaced frequencies w_k = 10000^{-2k/d}.
      2. Two-layer MLP: Linear(d, d) -> SiLU -> Linear(d, d).

    NOTE: Deviations from Def 5.5:
      - sin/cos are block-concatenated [sin..., cos...] instead of interleaved
        [sin,cos,sin,cos,...].  Equivalent after the learnable linear layer.
      - Two-layer MLP instead of single linear projection.  Standard practice
        in diffusion/flow-matching models for richer time conditioning.

    Args:
        dim: Output embedding dimension (must be even for sinusoidal part).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            msg = f"Time embedding dim must be even, got {dim}"
            raise ValueError(msg)

        self.dim = dim
        half = dim // 2
        # Log-spaced frequencies (frozen, not learnable).
        freq = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
        self.register_buffer("freq", freq)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Embed time steps of arbitrary leading shape.

        Args:
            t: Time steps.  Accepts any shape; each element is embedded
                independently.  Typical usage:
                  - ``(B,)`` — scalar per batch (classic flow).
                  - ``(B, N)`` — per-node time (M_x/M_c labels forced to
                    0 or 1 by the collator).

        Returns:
            Time embeddings with shape ``t.shape + (dim,)`` — e.g. ``(B, dim)``
            for scalar input, ``(B, N, dim)`` for per-node input.
        """
        lead_shape = t.shape
        t_flat = t.reshape(-1, 1)  # (*, 1)
        freq: Tensor = self.freq  # type: ignore[assignment]
        angles = t_flat * freq
        sinusoidal = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (*, dim)
        emb: Tensor = self.mlp(sinusoidal)
        return emb.reshape(*lead_shape, self.dim)
