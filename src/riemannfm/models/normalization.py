"""Adaptive Tangent-space Hyperbolic Normalization (ATH-Norm) — spec def 11.1.

Time-conditioned adaptive layer normalization (FiLM on top of a plain
LayerNorm). ``γ`` and ``β`` are predicted from the time embedding and
optionally an auxiliary conditioning vector (e.g. curvature scalars
``[κ_h, κ_s]`` in A_V's ATH-Norm per spec §12.4).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RiemannFMATHNorm(nn.Module):
    """Adaptive time-conditioned layer normalization (spec def 11.1).

    Args:
        dim: Feature dimension.
        time_dim: Dimension of the time embedding.
        cond_dim: Extra per-batch conditioning dimensions concatenated
            to t_emb before the adaLN projection. 0 disables.
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
        self.adaLN = nn.Linear(time_dim + cond_dim, 2 * dim)
        # Initialise near-identity: γ ≈ 1, β ≈ 0 (spec def 11.1).
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)
        self.adaLN.bias.data[:dim] = 1.0

    def forward(
        self,
        x: Tensor,
        t_emb: Tensor,
        cond: Tensor | None = None,
    ) -> Tensor:
        """Apply adaptive layer normalization.

        Args:
            x: Input features, shape ``(B, N, dim)`` (or ``(B, K, dim)``).
            t_emb: Time embeddings, shape ``(B, time_dim)`` or
                ``(B, N, time_dim)`` for per-node time.
            cond: Optional auxiliary conditioning, shape ``(B, cond_dim)``.
                Required when ``cond_dim > 0``.

        Returns:
            Normalized features, same shape as ``x``.
        """
        x_norm = self.norm(x)

        if self.cond_dim > 0:
            assert cond is not None, "cond required when cond_dim > 0"
            if t_emb.dim() == 3 and cond.dim() == 2:
                cond = cond.unsqueeze(1).expand(-1, t_emb.shape[1], -1)
            t_emb = torch.cat([t_emb, cond], dim=-1)

        gamma_beta = self.adaLN(t_emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        # Broadcast γ, β against x's middle dims (works for node/relation
        # hidden ``(B, N|K, dim)`` and edge hidden ``(B, N, N, dim)``).
        extra = x_norm.dim() - gamma.dim()
        for _ in range(extra):
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        out: Tensor = gamma * x_norm + beta
        return out
