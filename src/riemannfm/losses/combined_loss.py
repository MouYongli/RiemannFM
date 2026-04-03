"""Combined loss for RiemannFM pretraining.

L_total = L_cont + lambda_disc * L_disc + mu_align * L_align
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from riemannfm.losses.contrastive_loss import contrastive_alignment_loss
from riemannfm.losses.flow_matching_loss import (
    continuous_flow_loss,
    discrete_flow_loss,
)

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


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
        avg_edge_density: Edge density for L_disc class weights.
        w_max: Maximum positive class weight for L_disc.
        temperature: InfoNCE temperature for L_align.
        input_text_dim: Raw text embedding dimension d_c (0 to disable alignment).
        d_a: Alignment space dimension for projection layers.
    """

    def __init__(
        self,
        manifold: RiemannFMProductManifold,
        lambda_disc: float = 1.0,
        mu_align: float = 0.1,
        avg_edge_density: float = 0.05,
        w_max: float = 10.0,
        temperature: float = 0.07,
        input_text_dim: int = 0,
        d_a: int = 256,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.lambda_disc = lambda_disc
        self.mu_align = mu_align
        self.avg_edge_density = avg_edge_density
        self.w_max = w_max
        self.temperature = temperature

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

        # L_disc: discrete flow matching loss.
        l_disc = discrete_flow_loss(
            P_hat, E_1, node_mask,
            avg_edge_density=self.avg_edge_density,
            w_max=self.w_max,
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
            g = self.proj_g(x_1)         # (B, N, d_a)
            c = self.proj_c(node_text)    # (B, N, d_a)
            l_align = contrastive_alignment_loss(
                g, c, node_mask,
                temperature=self.temperature,
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
