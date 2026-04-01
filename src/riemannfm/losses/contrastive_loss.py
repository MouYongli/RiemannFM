"""Graph-text contrastive alignment loss L_align (Def 6.9-6.10).

Node-level symmetric InfoNCE between projected manifold coordinates
and projected text embeddings.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def contrastive_alignment_loss(
    g: Tensor,
    c: Tensor,
    node_mask: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """Graph-text contrastive alignment loss L_align (Def 6.9-6.10).

    Symmetric InfoNCE between projected graph features and projected
    text embeddings (both already in the shared alignment space d_a):

        L_align = 1/2 (L_{g->c} + L_{c->g})
        L_{g->c} = -1/|B| sum_i log[ exp(sim(g_i, c_i)/tau) /
                                       sum_j exp(sim(g_i, c_j)/tau) ]

    Args:
        g: Projected graph features, shape ``(B, N, d_a)``.
        c: Projected text embeddings, shape ``(B, N, d_a)``.
        node_mask: Bool mask, shape ``(B, N)``.
        temperature: InfoNCE temperature tau.

    Returns:
        Scalar contrastive loss.  Returns 0 if fewer than 2 valid nodes.
    """
    d_a = g.shape[-1]

    if d_a == 0:
        return torch.tensor(0.0, device=g.device, dtype=g.dtype)

    # Flatten batch and node dims: (B*N,).
    mask_flat = node_mask.reshape(-1)
    valid_idx = mask_flat.nonzero(as_tuple=True)[0]

    if valid_idx.numel() < 2:
        return torch.tensor(0.0, device=g.device, dtype=g.dtype)

    B, N, _ = g.shape
    g_valid = g.reshape(B * N, d_a)[valid_idx]  # (M, d_a)
    c_valid = c.reshape(B * N, d_a)[valid_idx]  # (M, d_a)

    # Skip if text embeddings are all zero.
    if c_valid.norm(dim=-1).max() < 1e-8:
        return torch.tensor(0.0, device=g.device, dtype=g.dtype)

    # L2 normalize for cosine similarity.
    g_norm = F.normalize(g_valid, dim=-1)
    c_norm = F.normalize(c_valid, dim=-1)

    # Cosine similarity matrix: (M, M).
    logits = g_norm @ c_norm.T / temperature

    # Labels: diagonal (each node matches its own text).
    labels = torch.arange(logits.shape[0], device=logits.device)

    # Symmetric InfoNCE.
    loss_g2c = F.cross_entropy(logits, labels)
    loss_c2g = F.cross_entropy(logits.T, labels)

    return (loss_g2c + loss_c2g) / 2
