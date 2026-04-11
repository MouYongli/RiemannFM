"""Flow matching losses: continuous (Def 6.6) and discrete (Def 6.8).

L_cont: Riemannian norm MSE of vector field prediction error.
L_disc: Weighted binary cross-entropy for edge prediction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


def continuous_flow_loss(
    manifold: RiemannFMProductManifold,
    V_hat: Tensor,
    u_t: Tensor,
    x_t: Tensor,
    node_mask: Tensor,
) -> Tensor:
    """Continuous flow matching loss L_cont (Def 6.6).

    L_cont = (1/|V_real|) * sum_{i: m_i=1} ||V_hat_i - u_t_i||^2_{T_{x_t} M}

    The squared norm is computed via :meth:`tangent_norm_sq`, which avoids
    the ``sqrt`` of :meth:`tangent_norm`.  Going through ``sqrt`` then
    ``pow(2)`` is mathematically equivalent in the forward pass but produces
    ``NaN`` gradients whenever the residual vanishes at any token (the
    backward of ``sqrt`` at zero is ``inf``, multiplied by the mask zero).

    Args:
        manifold: Product manifold for tangent norm computation.
        V_hat: Predicted vector field, shape ``(B, N, D)``.
        u_t: Target vector field, shape ``(B, N, D)``.
        x_t: Base points for tangent norm, shape ``(B, N, D)``.
        node_mask: Bool mask, shape ``(B, N)``. True = real node.

    Returns:
        Scalar loss (mean over real nodes).
    """
    # Residual in tangent space.
    residual = V_hat - u_t  # (B, N, D)

    # Riemannian norm squared per node, computed without an intermediate sqrt.
    # Force float32 for Lorentz inner product stability under AMP.
    norm_sq = manifold.tangent_norm_sq(
        x_t.float(), residual.float(),
    )  # (B, N)

    # Mask virtual nodes and average over real nodes.
    mask_float = node_mask.float()  # (B, N)
    num_real = mask_float.sum().clamp(min=1)
    loss = (norm_sq * mask_float).sum() / num_real

    return loss


def discrete_flow_loss(
    P_hat: Tensor,
    E_1: Tensor,
    node_mask: Tensor,
    neg_ratio: float = 1.0,
) -> Tensor:
    """Discrete flow matching loss L_disc (Def 6.8) with negative sampling.

    Instead of computing BCE over all N*N*K elements (dominated by trivial
    negatives), we gather only positive node pairs + sampled negative pairs,
    then compute BCE on the gathered subset. This is both faster (smaller
    tensors) and produces stronger gradient signal for edge prediction.

    Args:
        P_hat: Predicted edge logits (pre-sigmoid), shape ``(B, N, N, K)``.
        E_1: Target edge types, shape ``(B, N, N, K)``.
        node_mask: Bool mask, shape ``(B, N)``. True = real node.
        neg_ratio: Ratio of negative to positive pairs to sample.

    Returns:
        Scalar loss (mean over sampled pairs and relations).
    """
    # Valid node pairs only.
    pair_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)  # (B, N, N)

    # Positive pairs: (i, j) where at least one relation is active.
    has_edge = E_1.sum(dim=-1) > 0.5  # (B, N, N)
    pos_mask = has_edge & pair_mask  # (B, N, N)

    # Flatten to (B*N*N,) for indexing.
    pos_flat = pos_mask.reshape(-1)  # (L,)
    pos_idx = pos_flat.nonzero(as_tuple=False).squeeze(1)  # (P,)

    if pos_idx.numel() == 0:
        return torch.tensor(0.0, device=P_hat.device, dtype=P_hat.dtype)

    # Negative pool: valid pairs with no edge.
    neg_flat = ((~has_edge) & pair_mask).reshape(-1).float()  # (L,)

    # Sample negatives via multinomial.
    num_pos = pos_idx.shape[0]
    num_neg_pool = int(neg_flat.sum().item())
    n_neg = min(int(num_pos * neg_ratio), num_neg_pool)

    if n_neg > 0:
        neg_idx = torch.multinomial(neg_flat, n_neg, replacement=False)
        sample_idx = torch.cat([pos_idx, neg_idx], dim=0)
    else:
        sample_idx = pos_idx

    # Gather only sampled pairs — avoids BCE on full (B, N, N, K).
    K = E_1.shape[-1]
    P_flat = P_hat.reshape(-1, K)  # (L, K)
    E_flat = E_1.reshape(-1, K)    # (L, K)

    P_sampled = P_flat[sample_idx]  # (S, K)
    E_sampled = E_flat[sample_idx]  # (S, K)

    # BCE on the compact gathered tensor.
    loss = F.binary_cross_entropy_with_logits(P_sampled, E_sampled)

    return loss
