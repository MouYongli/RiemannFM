"""Edge self-update and edge-to-attention bias (Def 5.10-5.11).

The edge self-update uses factorized head/tail attention to aggregate
information from other edges sharing the same head or tail node.
Edge bias projects edge features into per-head scalars that bias the
attention logits.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RiemannFMEdgeBias(nn.Module):
    """Project edge features into per-head attention bias (Def 5.6).

    Args:
        edge_dim: Edge hidden dimension.
        num_heads: Number of attention heads.
    """

    def __init__(self, edge_dim: int, num_heads: int) -> None:
        super().__init__()
        self.proj = nn.Linear(edge_dim, num_heads, bias=False)

    def forward(self, g: Tensor) -> Tensor:
        """Compute edge bias for attention.

        Args:
            g: Edge hidden states, shape ``(B, N, N, edge_dim)``.

        Returns:
            Per-head bias, shape ``(B, num_heads, N, N)``.
        """
        # (B, N, N, edge_dim) -> (B, N, N, H) -> (B, H, N, N)
        result: Tensor = self.proj(g).permute(0, 3, 1, 2)
        return result


class RiemannFMEdgeSelfUpdate(nn.Module):
    """Factorized edge self-update via decomposed attention (Def 5.11).

    For each edge (i, j), aggregates information from:
      - Head-side: other edges sharing head node i, i.e. {h_{ip} : p != j}
      - Tail-side: other edges sharing tail node j, i.e. {h_{pj} : p != i}

    Each side uses independent QKV projections (6 matrices total).

    Residual update:
      g'_{ij} = g_{ij} + MLP([g_{ij} || g_head || g_tail])

    Args:
        edge_dim: Edge hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        edge_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.scale = 1.0 / math.sqrt(edge_dim)

        # Head-side attention: query from (i,j), key/value from (i,p).
        self.W_eq_head = nn.Linear(edge_dim, edge_dim, bias=False)
        self.W_ek_head = nn.Linear(edge_dim, edge_dim, bias=False)
        self.W_ev_head = nn.Linear(edge_dim, edge_dim, bias=False)

        # Tail-side attention: query from (i,j), key/value from (p,j).
        self.W_eq_tail = nn.Linear(edge_dim, edge_dim, bias=False)
        self.W_ek_tail = nn.Linear(edge_dim, edge_dim, bias=False)
        self.W_ev_tail = nn.Linear(edge_dim, edge_dim, bias=False)

        # Residual MLP: [g_{ij} || g_head || g_tail] -> edge_dim.
        self.mlp = nn.Sequential(
            nn.Linear(3 * edge_dim, edge_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim, edge_dim),
        )

    def forward(self, g: Tensor) -> Tensor:
        """Update edge features via factorized attention.

        Args:
            g: Edge hidden states, shape ``(B, N, N, edge_dim)``.

        Returns:
            Updated edge features, shape ``(B, N, N, edge_dim)``.
        """
        N = g.shape[1]

        # Self-exclusion mask: prevent edge (i,j) from attending to itself.
        # For head-side: exclude p=j; for tail-side: exclude p=i.
        # Both use the same NxN identity mask on the last two dims.
        eye_mask = torch.eye(N, device=g.device, dtype=torch.bool)  # (N, N)

        # --- Head-side: fix row i, attend over columns (j queries p) ---
        # g[:, i, j, :] queries; g[:, i, p, :] keys/values
        # This is attention along dim 2 for each fixed i (dim 1).
        q_head = self.W_eq_head(g)  # (B, N, N, D)
        k_head = self.W_ek_head(g)  # (B, N, N, D)
        v_head = self.W_ev_head(g)  # (B, N, N, D)
        # attn_head[b, i, j, p] = q[b,i,j] · k[b,i,p] / sqrt(D)
        attn_head = torch.matmul(q_head, k_head.transpose(-2, -1)) * self.scale  # (B, N, N, N)
        # Mask p=j: eye_mask[j, p] is True when j==p
        attn_head = attn_head.masked_fill(eye_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_head = F.softmax(attn_head, dim=-1)  # softmax over p
        g_head = torch.matmul(attn_head, v_head)  # (B, N, N, D)

        # --- Tail-side: fix col j, attend over rows (i queries p) ---
        # g[:, i, j, :] queries; g[:, p, j, :] keys/values
        # Transpose to (B, N_j, N_i, D), do attention along dim 2, transpose back.
        g_t = g.permute(0, 2, 1, 3)  # (B, j, i, D)
        q_tail = self.W_eq_tail(g_t)
        k_tail = self.W_ek_tail(g_t)
        v_tail = self.W_ev_tail(g_t)
        attn_tail = torch.matmul(q_tail, k_tail.transpose(-2, -1)) * self.scale  # (B, N, N, N)
        # Mask p=i: after transpose, dim 2 is i and dim 3 is p
        attn_tail = attn_tail.masked_fill(eye_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_tail = F.softmax(attn_tail, dim=-1)
        g_tail = torch.matmul(attn_tail, v_tail).permute(0, 2, 1, 3)  # (B, N, N, D)

        # MLP([g || g_head || g_tail]).
        # Residual connection is applied externally by the caller (RieFormerBlock).
        combined = torch.cat([g, g_head, g_tail], dim=-1)  # (B, N, N, 3D)
        result: Tensor = self.mlp(combined)
        return result
