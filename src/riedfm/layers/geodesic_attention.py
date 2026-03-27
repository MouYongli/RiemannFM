"""Geodesic Kernel Attention: manifold-aware multi-head attention.

Combines standard QK attention with manifold geometry:
    alpha_ij = softmax(q_i^T k_j / sqrt(d) + beta * kappa(x_i, x_j))

where kappa is a weighted sum of sub-manifold kernels:
    kappa = w_H * kappa_H + w_S * kappa_S + w_R * kappa_R

Value aggregation uses tangent space aggregation:
    o_i = Exp_{x_i}(sum_j alpha_ij * Log_{x_i}(v_j))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from riedfm.layers.manifold_rope import RieDFMManifoldRoPE
from riedfm.manifolds.product import RieDFMProductManifold


class RieDFMGeodesicAttention(nn.Module):
    """Multi-head attention with geodesic kernel bias and manifold-aware aggregation.

    Args:
        hidden_dim: Input/output feature dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head (default: hidden_dim // num_heads).
        dropout: Attention dropout rate.
        beta: Strength of geometric prior (learnable per head).
        use_mrope: Whether to use Manifold RoPE.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int | None = None,
        dropout: float = 0.0,
        beta: float = 1.0,
        use_mrope: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_mrope = use_mrope

        # QKV projections
        inner_dim = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, hidden_dim)

        # Per-head learnable kernel weights
        # w_H, w_S, w_R for each head
        self.kernel_weights = nn.Parameter(torch.ones(num_heads, 3) / 3.0)
        # Per-head beta (geometric prior strength)
        self.beta = nn.Parameter(torch.full((num_heads,), beta))

        # M-RoPE
        if use_mrope:
            self.mrope = RieDFMManifoldRoPE(self.head_dim)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: Tensor,
        manifold: RieDFMProductManifold,
        positions: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            h: Node features, shape (N, hidden_dim).
            manifold: Product manifold for geometry.
            positions: Node positions on manifold, shape (N, total_dim).
            mask: Optional attention mask, shape (N, N). True = attend.

        Returns:
            Updated features, shape (N, hidden_dim).
        """
        N = h.shape[0]

        # Project to Q, K, V
        q = rearrange(self.q_proj(h), "n (h d) -> n h d", h=self.num_heads)
        k = rearrange(self.k_proj(h), "n (h d) -> n h d", h=self.num_heads)
        v = rearrange(self.v_proj(h), "n (h d) -> n h d", h=self.num_heads)

        # Apply M-RoPE if enabled
        if self.use_mrope:
            q, k = self.mrope(q, k, manifold, positions)

        # Standard QK attention scores: (N, N, num_heads)
        attn_logits = torch.einsum("nhd,mhd->nmh", q, k) * self.scale

        # Compute manifold kernel bias
        k_h, k_s, k_e = manifold.compute_kernels(
            positions.unsqueeze(1).expand(-1, N, -1),
            positions.unsqueeze(0).expand(N, -1, -1),
        )  # Each (N, N)

        # Stack kernels and weight by per-head learned weights
        kernels = torch.stack([k_h, k_s, k_e], dim=-1)  # (N, N, 3)
        w = F.softmax(self.kernel_weights, dim=-1)  # (num_heads, 3)
        # Weighted kernel: (N, N, num_heads)
        kernel_bias = torch.einsum("nms,hs->nmh", kernels, w)

        # Add geometric bias scaled by beta
        attn_logits = attn_logits + self.beta.unsqueeze(0).unsqueeze(0) * kernel_bias

        # Apply mask
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=1)  # Softmax over source dimension
        attn_weights = self.attn_dropout(attn_weights)

        # Standard value aggregation (in feature space, not manifold)
        # For efficiency, we do Euclidean aggregation of feature vectors
        # The manifold-aware part is in the attention weights (kernel bias)
        out = torch.einsum("nmh,mhd->nhd", attn_weights, v)
        out = rearrange(out, "n h d -> n (h d)")

        result: Tensor = self.out_proj(out)
        return result
