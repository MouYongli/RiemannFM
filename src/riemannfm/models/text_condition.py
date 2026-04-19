"""Per-layer text-condition injection E_V and E_R (spec §16).

Standard cross-attention with pre-norm via ATH-Norm (applied by the
caller).  Both modules operate on the **already text-masked** C_V
(spec §9.3: text-mask positions carry ``mask_emb``) and C_R.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _TextCross(nn.Module):
    """Cross-attention: queries from hidden stream, keys/values from text."""

    def __init__(
        self,
        q_dim: int,
        text_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if q_dim % num_heads != 0:
            msg = f"q_dim ({q_dim}) must be divisible by num_heads ({num_heads})"
            raise ValueError(msg)

        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_q = nn.Linear(q_dim, q_dim, bias=False)
        self.W_k = nn.Linear(text_dim, q_dim, bias=False)
        self.W_v = nn.Linear(text_dim, q_dim, bias=False)
        self.W_o = nn.Linear(q_dim, q_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q_in: Tensor,
        text: Tensor,
        key_mask: Tensor | None = None,
    ) -> Tensor:
        B, N_q, _ = q_in.shape
        N_kv = text.shape[1]
        H = self.num_heads
        D = self.head_dim

        q = self.W_q(q_in).reshape(B, N_q, H, D).transpose(1, 2)
        k = self.W_k(text).reshape(B, N_kv, H, D).transpose(1, 2)
        v = self.W_v(text).reshape(B, N_kv, H, D).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if key_mask is not None:
            attn = attn.masked_fill(
                ~key_mask.unsqueeze(1).unsqueeze(1), -1e4,
            )
        alpha = F.softmax(attn, dim=-1)
        alpha = self.dropout(alpha)
        out = torch.matmul(alpha, v).transpose(1, 2).reshape(B, N_q, H * D)
        projected: Tensor = self.W_o(out)
        return projected


class RiemannFMNodeTextCross(nn.Module):
    """E_V: each node token cross-attends to the node text matrix (def 16.1).

    Args:
        node_dim: d_v.
        text_dim: Projected text dimension (text_proj_dim).
        num_heads: Head count.
        dropout: Attention dropout.
    """

    def __init__(
        self,
        node_dim: int,
        text_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cross = _TextCross(node_dim, text_dim, num_heads, dropout)

    def forward(
        self,
        h_V_bar: Tensor,
        C_V: Tensor,
        node_mask: Tensor,
    ) -> Tensor:
        """Cross-attend from nodes to node-text rows.

        Keys are masked by ``node_mask`` so virtual-node text rows do
        not contaminate the attention.

        Args:
            h_V_bar: Pre-normalised node hidden, shape ``(B, N, d_v)``.
            C_V: Projected node text (already text-masked by the
                lightning module when applicable), shape ``(B, N, text_dim)``.
            node_mask: Bool mask, shape ``(B, N)``.

        Returns:
            Additive update for the node stream, shape ``(B, N, d_v)``.
        """
        out: Tensor = self.cross(h_V_bar, C_V, key_mask=node_mask)
        return out


class RiemannFMRelationTextCross(nn.Module):
    """E_R: each relation token cross-attends to relation text (def 16.2).

    Args:
        rel_dim: d_r.
        text_dim: Projected text dimension.
        num_heads: Head count.
        dropout: Attention dropout.
    """

    def __init__(
        self,
        rel_dim: int,
        text_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cross = _TextCross(rel_dim, text_dim, num_heads, dropout)

    def forward(self, h_R_bar: Tensor, C_R: Tensor) -> Tensor:
        """Cross-attend from relation tokens to relation-text rows.

        Args:
            h_R_bar: Pre-normalised relation hidden, shape ``(B, K, d_r)``.
            C_R: Projected relation text, shape ``(B, K, text_dim)`` or
                ``(K, text_dim)``.

        Returns:
            Additive update for the relation stream, shape ``(B, K, d_r)``.
        """
        if C_R.dim() == 2:
            B = h_R_bar.shape[0]
            C_R = C_R.unsqueeze(0).expand(B, -1, -1)
        out: Tensor = self.cross(h_R_bar, C_R)
        return out
