"""Manifold attention with geodesic distance kernel (Def 5.5-5.8).

Multi-head self-attention where the attention logits incorporate a
geodesic distance kernel computed from per-component manifold distances.
Optionally uses M-RoPE (Manifold Rotary Position Embedding) and edge bias.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


class RiemannFMGeodesicAttention(nn.Module):
    """Multi-head attention with geodesic kernel (Def 5.5-5.8).

    Attention logits:
        A_{ij} = (Q_i K_j^T) / sqrt(d_k) + geodesic_kernel(x_i, x_j) + edge_bias_{ij}

    The geodesic kernel uses per-component distances weighted by learnable
    parameters (Def 5.7):
        kernel(x_i, x_j) = sum_c alpha_c * exp(-d_c(x_i, x_j)^2 / (2 * sigma_c^2))

    Args:
        node_dim: Node hidden dimension.
        num_heads: Number of attention heads.
        manifold: Product manifold for distance computation.
        use_geodesic_kernel: Whether to use the geodesic distance kernel.
        dropout: Attention dropout rate.
    """

    def __init__(
        self,
        node_dim: int,
        num_heads: int,
        manifold: RiemannFMProductManifold,
        use_geodesic_kernel: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if node_dim % num_heads != 0:
            msg = f"node_dim ({node_dim}) must be divisible by num_heads ({num_heads})"
            raise ValueError(msg)

        self.node_dim = node_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.manifold = manifold
        self.use_geodesic_kernel = use_geodesic_kernel

        # Q, K, V projections.
        self.W_q = nn.Linear(node_dim, node_dim)
        self.W_k = nn.Linear(node_dim, node_dim)
        self.W_v = nn.Linear(node_dim, node_dim)
        self.W_o = nn.Linear(node_dim, node_dim)

        # Geodesic kernel parameters (Def 5.7).
        if use_geodesic_kernel:
            num_comp = manifold.num_components
            # Per-component learnable alpha and sigma.
            self.alpha = nn.Parameter(torch.ones(num_heads, num_comp))
            self.log_sigma = nn.Parameter(torch.zeros(num_heads, num_comp))

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_bias: Tensor | None = None,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute manifold attention.

        Args:
            h: Node hidden states, shape ``(B, N, node_dim)``.
            x: Manifold coordinates, shape ``(B, N, D)``.
            edge_bias: Pre-computed edge bias, shape ``(B, num_heads, N, N)``
                or ``(B, 1, N, N)``.  Added to attention logits.
            node_mask: Bool mask, shape ``(B, N)``.  False positions are
                masked out of attention.

        Returns:
            Updated node hidden states, shape ``(B, N, node_dim)``.
        """
        B, N, _ = h.shape

        # QKV projections -> (B, num_heads, N, head_dim).
        Q = self._reshape_heads(self.W_q(h))
        K = self._reshape_heads(self.W_k(h))
        V = self._reshape_heads(self.W_v(h))

        # Standard dot-product attention logits.
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        # Geodesic distance kernel (Def 5.7).
        if self.use_geodesic_kernel:
            geo_bias = self._geodesic_kernel(x)  # (B, H, N, N)
            attn = attn + geo_bias

        # Edge bias from edge encoder (Def 5.6).
        if edge_bias is not None:
            attn = attn + edge_bias

        # Mask virtual nodes.
        if node_mask is not None:
            # Mask shape: (B, 1, 1, N) — mask out keys.
            key_mask = node_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn = attn.masked_fill(~key_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum of values.
        out = torch.matmul(attn, V)  # (B, H, N, d_k)
        out = out.transpose(1, 2).contiguous().view(B, N, self.node_dim)
        return self.W_o(out)

    def _reshape_heads(self, x: Tensor) -> Tensor:
        """Reshape (B, N, D) -> (B, H, N, d_k)."""
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _geodesic_kernel(self, x: Tensor) -> Tensor:
        """Compute geodesic distance kernel bias (Def 5.7).

        kernel_h(x_i, x_j) = sum_c alpha_{h,c} * exp(-d_c^2 / (2 * sigma_{h,c}^2))

        Args:
            x: Manifold coordinates, shape ``(B, N, D)``.

        Returns:
            Kernel bias, shape ``(B, num_heads, N, N)``.
        """
        B, N, _ = x.shape
        # Force float32 for manifold distance (arccosh/arccos unstable in fp16/bf16).
        x_f32 = x.float()

        # Compute per-component pairwise distances.
        x_i = x_f32.unsqueeze(2)  # (B, N, 1, D)
        x_j = x_f32.unsqueeze(1)  # (B, 1, N, D)
        x_i_flat = x_i.expand(B, N, N, -1).reshape(B * N * N, -1)
        x_j_flat = x_j.expand(B, N, N, -1).reshape(B * N * N, -1)

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            comp_dists = self.manifold.component_dists(
                x_i_flat, x_j_flat,
            )  # dict[str, (B*N*N,)]

        # Stack component distances: (B*N*N, num_comp).
        dist_stack = torch.stack(list(comp_dists.values()), dim=-1)
        dist_stack = dist_stack.view(B, N, N, -1)  # (B, N, N, C)

        # Gaussian kernel: alpha * exp(-d^2 / (2*sigma^2)).
        sigma = self.log_sigma.exp().clamp(min=1e-4)  # (H, C)
        # Reshape for broadcast: (B, 1, N, N, C) vs (1, H, 1, 1, C).
        d_sq = dist_stack.unsqueeze(1).pow(2)  # (B, 1, N, N, C)
        alpha = self.alpha[None, :, None, None, :]  # (1, H, 1, 1, C)
        sigma = sigma[None, :, None, None, :]  # (1, H, 1, 1, C)
        kernel = alpha * torch.exp(
            -d_sq / (2.0 * sigma.pow(2)),
        )  # (B, H, N, N, C)

        return kernel.sum(dim=-1)  # (B, H, N, N)
