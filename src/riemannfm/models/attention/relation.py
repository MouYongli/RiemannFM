"""Relation self-attention A_R (spec §13).

Standard multi-head self-attention over the K relation tokens with a
per-head **cosine-similarity bias** computed from the raw relation
embeddings (``R``, not the evolving hidden). The absolute-similarity
bias is shared across layers so every layer sees the same geometric
prior on relation-to-relation closeness.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RiemannFMRelationSelfAttention(nn.Module):
    """Multi-head self-attention over K relations with similarity bias (def 13.2).

    The attention logit is

        ``a_{kk'}^{(s)} = (q_k^{(s)}^T k_{k'}^{(s)}) / sqrt(d_head_R)
                        + w^{R,(s)} * cos(r_k, r_{k'})``

    No RoPE, no edge bias.

    Args:
        rel_dim: Relation hidden dimension ``d_r``.
        num_heads: Number of attention heads.
        rel_emb_dim: Relation embedding ``R`` dimension. Must equal
            ``rel_dim`` when it is used unchanged, but can differ if
            the relation encoder up-projects ``R``.
        use_similarity_bias: If ``False``, drop the cosine-similarity
            bias term (ablation).
        dropout: Dropout on attention weights.
    """

    def __init__(
        self,
        rel_dim: int,
        num_heads: int,
        rel_emb_dim: int,
        use_similarity_bias: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if rel_dim % num_heads != 0:
            msg = (
                f"rel_dim ({rel_dim}) must be divisible by num_heads ({num_heads})"
            )
            raise ValueError(msg)

        self.rel_dim = rel_dim
        self.num_heads = num_heads
        self.head_dim = rel_dim // num_heads
        self.use_similarity_bias = use_similarity_bias
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_q = nn.Linear(rel_dim, rel_dim, bias=False)
        self.W_k = nn.Linear(rel_dim, rel_dim, bias=False)
        self.W_v = nn.Linear(rel_dim, rel_dim, bias=False)
        self.W_o = nn.Linear(rel_dim, rel_dim, bias=False)

        if use_similarity_bias:
            # Per-head learnable weights on cosine(R_k, R_{k'}).
            self.w_sim = nn.Parameter(torch.zeros(num_heads))
            self._rel_emb_dim = rel_emb_dim
        else:
            self.register_parameter("w_sim", None)
            self._rel_emb_dim = rel_emb_dim

        self.dropout = nn.Dropout(dropout)

    def forward(self, h_R: Tensor, R: Tensor) -> Tensor:
        """Apply relation self-attention.

        Args:
            h_R: Relation hidden, shape ``(B, K, rel_dim)``.
            R: Raw relation embedding parameter, shape ``(K, rel_emb_dim)``.
                Used only for the cosine-similarity bias.

        Returns:
            Updated relation hidden, shape ``(B, K, rel_dim)``.
        """
        B, K, _ = h_R.shape
        H = self.num_heads
        D = self.head_dim

        q = self.W_q(h_R).reshape(B, K, H, D).transpose(1, 2)  # (B, H, K, D)
        k = self.W_k(h_R).reshape(B, K, H, D).transpose(1, 2)
        v = self.W_v(h_R).reshape(B, K, H, D).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, K, K)

        if self.use_similarity_bias:
            R_norm = F.normalize(R.to(h_R.dtype), dim=-1)
            sim = R_norm @ R_norm.T  # (K, K)
            # Broadcast per-head weight: (H, 1, 1) * (K, K) -> (H, K, K).
            bias = self.w_sim.view(H, 1, 1) * sim.unsqueeze(0)
            attn = attn + bias.unsqueeze(0)  # broadcast batch

        alpha = F.softmax(attn, dim=-1)
        alpha = self.dropout(alpha)
        out = torch.matmul(alpha, v)  # (B, H, K, D)
        out = out.transpose(1, 2).reshape(B, K, H * D)
        projected: Tensor = self.W_o(out)
        return projected
