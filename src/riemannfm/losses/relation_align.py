"""Relation-text alignment loss L_align^R (spec def 19.1).

Optional auxiliary loss that ties learnable relation embeddings R to
their text descriptions (spec C_R) via a symmetric cosine InfoNCE over the K
relations. Used as a light regulariser; λ_align^R ∈ [0.01, 0.05].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def relation_align_infonce(
    z_R: Tensor,
    z_C: Tensor,
    tau: float = 0.1,
) -> Tensor:
    """Symmetric InfoNCE over K relations (spec def 19.1).

    Given pre-projected relation embeddings ``z_R = W_p · R`` and text
    embeddings ``z_C = W_p^c · relation_text`` both of shape ``(K, d_p)``, compute

    ``L = -(1 / 2K) Σ_k [log p(k | R_k→C) + log p(k | C_k→R)]``

    where ``p(k | ·)`` is the cosine-softmax with temperature ``tau``.

    Args:
        z_R: Projected relation embeddings, shape ``(K, d_p)``.
        z_C: Projected relation text embeddings, shape ``(K, d_p)``.
        tau: Temperature.

    Returns:
        Scalar InfoNCE loss. Zero when K < 2 or text is all zero.
    """
    if z_R.shape[0] < 2:
        return torch.tensor(0.0, device=z_R.device, dtype=z_R.dtype)
    if z_C.norm(dim=-1).max() < 1e-8:
        return torch.tensor(0.0, device=z_R.device, dtype=z_R.dtype)

    r_n = F.normalize(z_R, dim=-1)
    c_n = F.normalize(z_C, dim=-1)
    logits = r_n @ c_n.T / tau  # (K, K)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_r2c = F.cross_entropy(logits, labels)
    loss_c2r = F.cross_entropy(logits.T, labels)
    return (loss_r2c + loss_c2r) / 2
