"""Combined loss for RiemannFM pretraining.

L_total = L_cont + lambda_disc * L_disc + mu_align * L_align
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from riemannfm.losses.flow_matching_loss import (
    continuous_flow_loss,
    discrete_flow_loss,
)

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


def _contrastive_loss_from_pairs(
    g: Tensor, c: Tensor, temperature: float,
) -> Tensor:
    """Symmetric InfoNCE on pre-gathered valid node pairs.

    Args:
        g: Projected graph features for valid nodes, shape ``(M, d_a)``.
        c: Projected text embeddings for valid nodes, shape ``(M, d_a)``.
        temperature: InfoNCE temperature.

    Returns:
        Scalar contrastive loss.
    """
    import torch.nn.functional as F

    if c.norm(dim=-1).max() < 1e-8:
        return torch.tensor(0.0, device=g.device, dtype=g.dtype)

    g_norm = F.normalize(g, dim=-1)
    c_norm = F.normalize(c, dim=-1)
    logits = g_norm @ c_norm.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_g2c = F.cross_entropy(logits, labels)
    loss_c2g = F.cross_entropy(logits.T, labels)
    return (loss_g2c + loss_c2g) / 2


class RiemannFMCombinedLoss(nn.Module):
    """Combined pretraining loss (Algorithm 1).

    L_total = L_cont + lambda_disc * L_disc + mu_align * L_align

    Contains learnable projection layers for the contrastive alignment
    loss (Definition 6.9):
      - proj_g: R^D -> R^{d_a}  (manifold ambient coords -> alignment space)
      - proj_c: R^{d_c} -> R^{d_a}  (text embeddings -> alignment space)

    Args:
        manifold: Product manifold for tangent norm in L_cont.
        lambda_disc: Weight for discrete flow loss.
        mu_align: Weight for contrastive alignment loss.
        neg_ratio: Ratio of negative to positive pairs for L_disc sampling.
        temperature: InfoNCE temperature for L_align.
        input_text_dim: Raw text embedding dimension d_c (0 to disable alignment).
        d_a: Alignment space dimension for projection layers.
    """

    def __init__(
        self,
        manifold: RiemannFMProductManifold,
        lambda_disc: float = 1.0,
        mu_align: float = 0.1,
        neg_ratio: float = 1.0,
        temperature: float = 0.07,
        input_text_dim: int = 0,
        d_a: int = 256,
        max_align_nodes: int = 128,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.lambda_disc = lambda_disc
        self.mu_align = mu_align
        self.neg_ratio = neg_ratio
        self.temperature = temperature
        self.max_align_nodes = max_align_nodes

        # Projection layers for L_align (Def 6.9):
        #   g_i = proj_g(pi(x_{1,i}))
        #   c_i = proj_c(text_emb_i)
        self.proj_g: nn.Sequential | None
        self.proj_c: nn.Sequential | None
        if mu_align > 0 and input_text_dim > 0:
            D = manifold.ambient_dim
            self.proj_g = nn.Sequential(
                nn.Linear(D, d_a),
                nn.GELU(),
                nn.Linear(d_a, d_a),
            )
            self.proj_c = nn.Sequential(
                nn.Linear(input_text_dim, d_a),
                nn.GELU(),
                nn.Linear(d_a, d_a),
            )
        else:
            self.proj_g = None
            self.proj_c = None

    def forward(
        self,
        V_hat: Tensor,
        u_t: Tensor,
        x_t: Tensor,
        P_hat: Tensor,
        E_1: Tensor,
        node_mask: Tensor,
        x_1: Tensor | None = None,
        node_text: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute combined loss.

        Args:
            V_hat: Predicted vector field, shape ``(B, N, D)``.
            u_t: Target vector field, shape ``(B, N, D)``.
            x_t: Interpolated manifold coords, shape ``(B, N, D)``.
            P_hat: Predicted edge logits, shape ``(B, N, N, K)``.
            E_1: Target edge types, shape ``(B, N, N, K)``.
            node_mask: Bool mask, shape ``(B, N)``.
            x_1: Data-end manifold coordinates for L_align, shape ``(B, N, D)``.
                None to skip L_align.
            node_text: Raw text embeddings for L_align, shape ``(B, N, d_c)``.
                None to skip L_align.

        Returns:
            Tuple of (total_loss, metrics_dict).
            metrics_dict contains individual loss components.
        """
        # L_cont: continuous flow matching loss.
        l_cont = continuous_flow_loss(
            self.manifold, V_hat, u_t, x_t, node_mask,
        )

        # L_disc: discrete flow matching loss (Def 6.8) with negative sampling.
        l_disc = discrete_flow_loss(
            P_hat, E_1, node_mask,
            neg_ratio=self.neg_ratio,
        )

        # L_align: contrastive alignment loss (Def 6.9).
        if (
            self.mu_align > 0
            and self.proj_g is not None
            and self.proj_c is not None
            and x_1 is not None
            and node_text is not None
            and node_text.shape[-1] > 0
        ):
            # Project only valid (real) nodes to avoid zero-input gradient
            # starvation on proj_c/proj_g weights from virtual node padding.
            B, N, _ = x_1.shape
            mask_flat = node_mask.reshape(-1)
            valid_idx = mask_flat.nonzero(as_tuple=True)[0]

            # Subsample to cap contrastive matrix size.
            if valid_idx.numel() > self.max_align_nodes:
                perm = torch.randperm(
                    valid_idx.numel(), device=valid_idx.device,
                )[:self.max_align_nodes]
                valid_idx = valid_idx[perm]

            if valid_idx.numel() < 2:
                l_align = torch.tensor(0.0, device=x_1.device, dtype=x_1.dtype)
            else:
                x_valid = x_1.reshape(B * N, -1)[valid_idx]  # (M, D)
                t_valid = node_text.reshape(B * N, -1)[valid_idx]  # (M, d_c)
                g_valid = self.proj_g(x_valid)  # (M, d_a)
                c_valid = self.proj_c(t_valid)  # (M, d_a)
                l_align = _contrastive_loss_from_pairs(
                    g_valid, c_valid, self.temperature,
                )
        else:
            l_align = torch.tensor(0.0, device=V_hat.device, dtype=V_hat.dtype)

        # Total loss.
        total = l_cont + self.lambda_disc * l_disc + self.mu_align * l_align

        metrics = {
            "loss/total": total.detach(),
            "loss/cont": l_cont.detach(),
            "loss/disc": l_disc.detach(),
            "loss/align": l_align.detach(),
        }

        return total, metrics
