"""Cross-interaction modules D_VR and D_VE (spec §15).

  - ``RiemannFMNodeRelationCross`` (D_VR, def 15.1): two multi-head
    cross-attentions. Nodes read from relations; relations read from
    (masked) nodes.
  - ``RiemannFMNodeEdgeCross`` (D_VE, def 15.2): nodes cross-attend over
    the union ``{h_{E,i,:}} ∪ {h_{E,:,i}}`` of incident edges; edges
    update via an MLP on ``[h_V_i ‖ h_V_j]``.

Each cross-attention is a standard scaled-dot-product MHA with an
output projection.  The caller is responsible for pre-norm (ATH-Norm).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _CrossAttn(nn.Module):
    """Plain multi-head cross-attention (Q from stream A, K/V from stream B)."""

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
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
        self.W_k = nn.Linear(kv_dim, q_dim, bias=False)
        self.W_v = nn.Linear(kv_dim, q_dim, bias=False)
        self.W_o = nn.Linear(q_dim, q_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q_in: Tensor,
        kv_in: Tensor,
        key_mask: Tensor | None = None,
    ) -> Tensor:
        """Cross-attention forward.

        Args:
            q_in: Query tokens, shape ``(B, N_q, q_dim)``.
            kv_in: Key/value tokens, shape ``(B, N_kv, kv_dim)``.
            key_mask: Bool mask over keys, shape ``(B, N_kv)``. True =
                valid (attend); False = padded (suppress).

        Returns:
            Shape ``(B, N_q, q_dim)``.
        """
        B, N_q, _ = q_in.shape
        N_kv = kv_in.shape[1]
        H = self.num_heads
        D = self.head_dim

        q = self.W_q(q_in).reshape(B, N_q, H, D).transpose(1, 2)
        k = self.W_k(kv_in).reshape(B, N_kv, H, D).transpose(1, 2)
        v = self.W_v(kv_in).reshape(B, N_kv, H, D).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, Nq, Nkv)
        if key_mask is not None:
            # Finite negative bias so fully-masked rows still produce a
            # finite softmax (prevents NaN gradients from empty support).
            attn = attn.masked_fill(
                ~key_mask.unsqueeze(1).unsqueeze(1), -1e4,
            )
        alpha = F.softmax(attn, dim=-1)
        alpha = self.dropout(alpha)
        out = torch.matmul(alpha, v).transpose(1, 2).reshape(B, N_q, H * D)
        projected: Tensor = self.W_o(out)
        return projected


class RiemannFMNodeRelationCross(nn.Module):
    """D_VR: bidirectional node ↔ relation cross-attention (spec def 15.1).

    - ``V ← R``: each node queries all K relations.
    - ``R ← V``: each relation queries all real nodes (virtual nodes
      masked out).

    Args:
        node_dim: d_v.
        rel_dim: d_r.
        num_heads_V: Head count for the V-side (V ← R) query projection.
            Must divide ``node_dim``.
        num_heads_R: Head count for the R-side (R ← V) query projection.
            Must divide ``rel_dim``.
        dropout: Attention dropout.
    """

    def __init__(
        self,
        node_dim: int,
        rel_dim: int,
        num_heads_V: int,
        num_heads_R: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.v_from_r = _CrossAttn(node_dim, rel_dim, num_heads_V, dropout)
        self.r_from_v = _CrossAttn(rel_dim, node_dim, num_heads_R, dropout)

    def forward(
        self,
        h_V_bar: Tensor,
        h_R_bar: Tensor,
        node_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply both cross-attention directions.

        Args:
            h_V_bar: Pre-normalised node hidden, shape ``(B, N, d_v)``.
            h_R_bar: Pre-normalised relation hidden, shape ``(B, K, d_r)``.
            node_mask: Bool mask, shape ``(B, N)``.

        Returns:
            ``(delta_V, delta_R)``: additive updates, same shapes as the
            pre-norm inputs. The block applies the residual add.
        """
        delta_V = self.v_from_r(h_V_bar, h_R_bar, key_mask=None)
        delta_R = self.r_from_v(h_R_bar, h_V_bar, key_mask=node_mask)
        return delta_V, delta_R


class RiemannFMNodeEdgeCross(nn.Module):
    """D_VE: node ← edge cross-attention + edge ← node MLP (spec def 15.2).

    - ``V ← E``: each node ``i`` queries the union of incident edges
      ``{h_E[b, i, :]} ∪ {h_E[b, :, i]}`` (both outgoing and incoming).
    - ``E ← V``: each edge ``(i, j)`` updates via an MLP on
      ``[h_V_i ‖ h_V_j]`` (attention degenerates here because each edge
      has a single deterministic node pair).

    Args:
        node_dim: d_v.
        edge_dim: d_b.
        num_heads: Head count for the V ← E cross-attention.
        dropout: Dropout.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.v_from_e = _CrossAttn(node_dim, edge_dim, num_heads, dropout)
        self.e_from_v = nn.Sequential(
            nn.Linear(2 * node_dim, edge_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim, edge_dim),
        )

    def forward(
        self,
        h_V_bar: Tensor,
        h_E_bar: Tensor,
        h_V_post: Tensor,
        node_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply D_VE.

        Args:
            h_V_bar: Pre-normalised node hidden for the V ← E query,
                shape ``(B, N, d_v)``.
            h_E_bar: Pre-normalised edge hidden, shape ``(B, N, N, d_b)``.
            h_V_post: The node stream **after** the prior D_VR update,
                used as the input to the E ← V MLP so edges see the
                latest node representations.
            node_mask: Bool mask, shape ``(B, N)``.

        Returns:
            ``(delta_V, delta_E)``: additive updates. Shapes match the
            inputs (delta_E is ``(B, N, N, d_b)``).
        """
        B, N, d_v = h_V_bar.shape
        d_b = h_E_bar.shape[-1]

        # V ← E: concatenate outgoing and incoming edges per node.
        outgoing = h_E_bar                                 # (B, N, N, d_b)  edges (i,:)
        incoming = h_E_bar.transpose(1, 2)                 # (B, N, N, d_b)  edges (:,i)
        kv = torch.cat([outgoing, incoming], dim=2)        # (B, N, 2N, d_b)

        # Flatten (B, N) queries into the cross-attn's batch dim so each
        # node attends only to its own 2N incident edges.
        q_flat = h_V_bar.reshape(B * N, 1, d_v)
        kv_flat = kv.reshape(B * N, 2 * N, d_b)
        # Key mask: valid iff the "other" endpoint is a real node.
        #   outgoing (i, j) valid  ⇔  node_mask[b, j]
        #   incoming (j, i) valid  ⇔  node_mask[b, j]
        # Both halves share the same per-node validity pattern.
        km = node_mask.unsqueeze(1).expand(B, N, N)        # (B, N, N)
        key_mask = torch.cat([km, km], dim=2).reshape(B * N, 2 * N)
        delta_V = self.v_from_e(q_flat, kv_flat, key_mask=key_mask).reshape(B, N, d_v)

        # E ← V: MLP on [h_V_i || h_V_j].
        h_i = h_V_post.unsqueeze(2).expand(B, N, N, d_v)
        h_j = h_V_post.unsqueeze(1).expand(B, N, N, d_v)
        pair = torch.cat([h_i, h_j], dim=-1)
        delta_E = self.e_from_v(pair).reshape(B, N, N, d_b)

        return delta_V, delta_E
