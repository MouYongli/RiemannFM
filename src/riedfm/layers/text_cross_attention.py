"""Text-Graph Cross-Attention for condition injection.

Graph node features attend to text token embeddings from a frozen
text encoder (XLM-R or mE5). This injects text semantics into
the graph generation process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


class RieDFMTextCrossAttention(nn.Module):
    """Cross-attention from graph nodes (queries) to text tokens (keys/values).

    Args:
        node_dim: Node feature dimension (query dim).
        text_dim: Text encoder output dimension (key/value dim).
        num_heads: Number of attention heads.
        dropout: Attention dropout rate.
    """

    def __init__(
        self,
        node_dim: int,
        text_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads

        self.q_proj = nn.Linear(node_dim, node_dim, bias=False)
        self.k_proj = nn.Linear(text_dim, node_dim, bias=False)
        self.v_proj = nn.Linear(text_dim, node_dim, bias=False)
        self.out_proj = nn.Linear(node_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_v: Tensor,
        text_embeds: Tensor,
        text_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            h_v: Node features, shape (N, node_dim).
            text_embeds: Text token embeddings, shape (T, text_dim) or (B, T, text_dim).
            text_mask: Optional mask for text tokens, shape (T,) or (B, T).

        Returns:
            Updated node features, shape (N, node_dim).
        """
        residual = h_v

        # Handle batch dimension
        if text_embeds.dim() == 2:
            text_embeds = text_embeds.unsqueeze(0)  # (1, T, text_dim)
        if text_mask is not None and text_mask.dim() == 1:
            text_mask = text_mask.unsqueeze(0)

        q = rearrange(self.q_proj(h_v), "n (h d) -> 1 h n d", h=self.num_heads)
        k = rearrange(self.k_proj(text_embeds), "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(self.v_proj(text_embeds), "b t (h d) -> b h t d", h=self.num_heads)

        # Attention scores: (1, H, N, T)
        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if text_mask is not None:
            # text_mask: (B, T) -> (B, 1, 1, T)
            attn = attn.masked_fill(~text_mask[:, None, None, :], float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Aggregate: (1, H, N, D)
        out = torch.matmul(attn, v)
        out = rearrange(out, "1 h n d -> n (h d)")
        out = self.out_proj(out)

        result: Tensor = self.norm(residual + out)
        return result
