"""Adaptive Tangent-space Hyperbolic Normalization (ATH-Norm) — Def 5.9.

ATH-Norm is a time-conditioned adaptive layer normalization.
Time-conditioned affine parameters (gamma, beta) are predicted from
the time embedding, enabling the model to adjust normalization behavior
across the flow trajectory.  An optional auxiliary conditioning tensor
(e.g. curvature scalars κ_h, κ_s) may be concatenated with t_emb before
the adaLN projection so that gamma/beta also vary with manifold geometry.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RiemannFMATHNorm(nn.Module):
    """Adaptive time-conditioned layer normalization (Def 5.9).

    Steps:
      1. Layer-normalize the input features.
      2. Apply time-conditioned affine: x' = gamma(t, cond) * x_norm
         + beta(t, cond), where ``cond`` is an optional auxiliary scalar
         vector (e.g. [κ_h, κ_s]).

    Args:
        dim: Feature dimension (node_dim or edge_dim).
        time_dim: Dimension of the time embedding.
        cond_dim: Extra per-batch conditioning dimensions concatenated
            to t_emb before the adaLN projection.  0 disables.
        eps: LayerNorm epsilon.
    """

    def __init__(
        self,
        dim: int,
        time_dim: int,
        cond_dim: int = 0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        # Time- (and optionally curvature-) conditioned affine.
        self.adaLN = nn.Linear(time_dim + cond_dim, 2 * dim)
        # Initialize near identity: gamma~1, beta~0.
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)
        self.adaLN.bias.data[:dim] = 1.0  # gamma init = 1

    def forward(
        self, x: Tensor, t_emb: Tensor, node_mask: Tensor | None = None,
        cond: Tensor | None = None,
    ) -> Tensor:
        """Apply adaptive layer normalization.

        Args:
            x: Input features, shape ``(B, N, dim)``.
            t_emb: Time embeddings.  Shape ``(B, time_dim)`` for a
                batch-scalar schedule, or ``(B, N, time_dim)`` for
                per-node time (e.g. M_x/M_c labels from the collator).
            node_mask: Unused (kept for interface compatibility).
            cond: Optional auxiliary conditioning, shape ``(B, cond_dim)``.
                Broadcast to match t_emb's rank before concatenation.

        Returns:
            Normalized features, shape ``(B, N, dim)``.
        """
        # 1. Layer-normalize.
        x_norm = self.norm(x)

        # 2. Time- (and optionally curvature-) conditioned affine.
        #    Predict (gamma, beta) per-batch when t_emb is (B, time_dim);
        #    per-node when (B, N, time_dim).
        if self.cond_dim > 0:
            assert cond is not None, "cond required when cond_dim > 0"
            if t_emb.dim() == 3 and cond.dim() == 2:
                cond = cond.unsqueeze(1).expand(-1, t_emb.shape[1], -1)
            t_emb = torch.cat([t_emb, cond], dim=-1)
        gamma_beta = self.adaLN(t_emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        if gamma.dim() == 2:
            gamma = gamma.unsqueeze(1)  # (B, 1, dim)
            beta = beta.unsqueeze(1)

        result: Tensor = gamma * x_norm + beta
        return result


class RiemannFMPreNorm(nn.Module):
    """Simple pre-norm wrapper (LayerNorm).

    Used as a fallback when ``use_ath_norm=False`` in ablation configs.

    Args:
        dim: Feature dimension.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, x: Tensor, t_emb: Tensor | None = None,
        node_mask: Tensor | None = None,
        cond: Tensor | None = None,
    ) -> Tensor:
        """Apply LayerNorm (ignores t_emb, node_mask, cond).

        Args:
            x: Input features, shape ``(B, N, dim)``.
            t_emb: Unused (kept for interface compatibility).
            node_mask: Unused.
            cond: Unused.

        Returns:
            Normalized features, shape ``(B, N, dim)``.
        """
        result: Tensor = self.norm(x)
        return result
