"""Graph-text contrastive alignment loss L_align (Def 6.9-6.10).

Node-level InfoNCE contrastive loss between graph-derived node features
and text condition embeddings.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def contrastive_alignment_loss(
    h: Tensor,
    node_text: Tensor,
    node_mask: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """Graph-text contrastive alignment loss L_align (Def 6.9-6.10).

    InfoNCE between node hidden states and text embeddings:
        L_align = -log( exp(sim(h_i, c_i)/tau) / sum_j exp(sim(h_i, c_j)/tau) )

    Only computed over real nodes (m_i = 1) with non-zero text embeddings.

    Args:
        h: Node hidden states (graph-derived), shape ``(B, N, d)``.
        node_text: Text condition embeddings, shape ``(B, N, d_c)``.
        node_mask: Bool mask, shape ``(B, N)``.
        temperature: InfoNCE temperature tau.

    Returns:
        Scalar contrastive loss.  Returns 0 if no valid nodes.
    """
    # Flatten batch and node dims: (B*N, d).
    B, N, d = h.shape
    d_c = node_text.shape[-1]

    if d_c == 0:
        return torch.tensor(0.0, device=h.device, dtype=h.dtype)

    mask_flat = node_mask.reshape(-1)  # (B*N,)
    valid_idx = mask_flat.nonzero(as_tuple=True)[0]

    if valid_idx.numel() < 2:
        return torch.tensor(0.0, device=h.device, dtype=h.dtype)

    # Gather valid node features and text embeddings.
    h_flat = h.reshape(B * N, d)
    t_flat = node_text.reshape(B * N, d_c)

    h_valid = h_flat[valid_idx]  # (M, d)
    t_valid = t_flat[valid_idx]  # (M, d_c)

    # Skip if text embeddings are all zero.
    if t_valid.norm(dim=-1).max() < 1e-8:
        return torch.tensor(0.0, device=h.device, dtype=h.dtype)

    # Project to same dimension if needed (use simple linear projection).
    # For MVP, assume d == d_c or use dot product on min(d, d_c).
    min_dim = min(d, d_c)
    h_proj = F.normalize(h_valid[:, :min_dim], dim=-1)
    t_proj = F.normalize(t_valid[:, :min_dim], dim=-1)

    # Cosine similarity matrix: (M, M).
    logits = h_proj @ t_proj.T / temperature

    # Labels: diagonal (each node matches its own text).
    labels = torch.arange(logits.shape[0], device=logits.device)

    # Symmetric InfoNCE.
    loss_h2t = F.cross_entropy(logits, labels)
    loss_t2h = F.cross_entropy(logits.T, labels)

    return (loss_h2t + loss_t2h) / 2
