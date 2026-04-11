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
    """Multi-head attention with M-RoPE and geodesic kernel (Def 5.5-5.8).

    Attention logits:
        A_{ij} = (R(theta_ij) Q_i)^T K_j / sqrt(d_k)
                 + beta^(s) * kernel^(s)(x_i, x_j) + edge_bias_{ij}

    M-RoPE (Def 5.6): Rotates Q by pairwise manifold distance theta_{ij,l}
    = omega_l * d_M(x_i, x_j), using block-diagonal 2x2 rotation matrices.

    Geodesic kernel (Def 5.7) uses per-component geometry-aware kernels:
        H: -d_H,  S: kappa_s * a^T b,  E: -||a-b||^2

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

        # M-RoPE frequencies (Def 5.6): omega_l = 10000^{-2l/d_head}.
        half = self.head_dim // 2
        rope_freq = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
        self.register_buffer("rope_freq", rope_freq)

        # Geodesic kernel parameters (Def 5.7).
        if use_geodesic_kernel:
            num_comp = manifold.num_components
            # Per-head per-component weights w_c^(s).
            self.w_kernel = nn.Parameter(torch.ones(num_heads, num_comp))
            # Per-head scaling coefficient beta^(s) (Def 5.8).
            self.beta = nn.Parameter(torch.ones(num_heads))

        self.attn_dropout = nn.Dropout(dropout)

    def _compute_pairwise_geometry(
        self, x: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        """Compute pairwise manifold geometry once, shared by M-RoPE and kernel.

        Args:
            x: Manifold coordinates, shape ``(B, N, D)``.

        Returns:
            Tuple of:
              - dist: Product distance, shape ``(B, N, N)``.
              - comp_dists: Per-component distances, dict[str, ``(B*N*N,)``].
              - x_i_parts: Per-component coords for x_i, dict[str, ``(B*N*N, d)``].
              - x_j_parts: Per-component coords for x_j, dict[str, ``(B*N*N, d)``].
        """
        B, N, _ = x.shape
        x_f32 = x.float()

        x_i_flat = x_f32.unsqueeze(2).expand(B, N, N, -1).reshape(B * N * N, -1)
        x_j_flat = x_f32.unsqueeze(1).expand(B, N, N, -1).reshape(B * N * N, -1)

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            comp_dists = self.manifold.component_dists(x_i_flat, x_j_flat)
            # Product distance from component distances (avoids recomputation).
            d_sq = torch.stack(
                [d.pow(2) for d in comp_dists.values()], dim=0,
            ).sum(dim=0)
            dist = d_sq.sqrt().view(B, N, N)

        x_i_parts = self.manifold.split(x_i_flat)
        x_j_parts = self.manifold.split(x_j_flat)

        return dist, comp_dists, x_i_parts, x_j_parts

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

        # Compute pairwise geometry ONCE — shared by M-RoPE and geodesic kernel.
        dist, comp_dists, x_i_parts, x_j_parts = self._compute_pairwise_geometry(x)

        # M-RoPE attention logits (Def 5.6 / 5.8).
        # theta_{ij,l} = omega_l * d_M(x_i, x_j)  — (B, N, N, half)
        theta = dist.unsqueeze(-1) * self.rope_freq  # type: ignore[operator]
        cos_t = theta.cos().unsqueeze(1)  # (B, 1, N, N, half)
        sin_t = theta.sin().unsqueeze(1)  # (B, 1, N, N, half)

        # Split Q, K into even/odd pairs for 2x2 rotation blocks.
        q1, q2 = Q[..., 0::2], Q[..., 1::2]  # (B, H, N, half)
        k1, k2 = K[..., 0::2], K[..., 1::2]

        # Expanded rotary dot product:
        # (R(theta_ij) q_i)^T k_j = cos(theta) * (q1·k1 + q2·k2)
        #                          + sin(theta) * (q1·k2 - q2·k1)
        qk_same = q1.unsqueeze(3) * k1.unsqueeze(2) + q2.unsqueeze(3) * k2.unsqueeze(2)
        qk_cross = q1.unsqueeze(3) * k2.unsqueeze(2) - q2.unsqueeze(3) * k1.unsqueeze(2)
        # (B, H, N, N, half) -> sum over half -> (B, H, N, N)
        attn = (cos_t * qk_same + sin_t * qk_cross).sum(dim=-1) * self.scale

        # Geodesic distance kernel (Def 5.7) with beta scaling (Def 5.8).
        if self.use_geodesic_kernel:
            geo_bias = self._geodesic_kernel_from_cache(
                x, comp_dists, x_i_parts, x_j_parts,
            )
            beta = self.beta[None, :, None, None]  # (1, H, 1, 1)
            attn = attn + beta * geo_bias

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
        result: Tensor = self.W_o(out)
        return result

    def _reshape_heads(self, x: Tensor) -> Tensor:
        """Reshape (B, N, D) -> (B, H, N, d_k)."""
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _geodesic_kernel_from_cache(
        self,
        x: Tensor,
        comp_dists: dict[str, Tensor],
        x_i_parts: dict[str, Tensor],
        x_j_parts: dict[str, Tensor],
    ) -> Tensor:
        """Compute geodesic kernel from pre-computed pairwise geometry.

        Reuses component distances and split coordinates already computed
        for M-RoPE, avoiding redundant manifold.dist() calls.

        Args:
            x: Manifold coordinates, shape ``(B, N, D)``.
            comp_dists: Per-component distances, dict[str, ``(B*N*N,)``].
            x_i_parts: Per-component coords for x_i, dict[str, ``(B*N*N, d)``].
            x_j_parts: Per-component coords for x_j, dict[str, ``(B*N*N, d)``].

        Returns:
            Kernel bias, shape ``(B, num_heads, N, N)``.
        """
        B, N, _ = x.shape

        # Build per-component kernels (Def 5.7).
        kernels: list[Tensor] = []
        for name in self.manifold._component_names:
            if name == "hyperbolic":
                kernels.append(-comp_dists[name])
            elif name == "spherical":
                ks = self.manifold.spherical.curvature  # type: ignore[union-attr]
                kernels.append(ks * (x_i_parts[name] * x_j_parts[name]).sum(-1))
            elif name == "euclidean":
                kernels.append(-comp_dists[name].pow(2))

        # Stack and reshape: (B*N*N, C) -> (B, N, N, C).
        kernel_stack = torch.stack(kernels, dim=-1).view(B, N, N, -1)

        # Per-head per-component weights: w_c^(s).
        w = self.w_kernel[None, :, None, None, :]  # (1, H, 1, 1, C)
        return (w * kernel_stack.unsqueeze(1)).sum(dim=-1)  # (B, H, N, N)
