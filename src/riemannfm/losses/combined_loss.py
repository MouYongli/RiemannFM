"""Combined loss: L = L_cont + lambda * L_disc + mu * L_align."""

import torch.nn as nn
from torch import Tensor

from riemannfm.losses.contrastive_loss import RiemannFMContrastiveLoss
from riemannfm.losses.flow_matching_loss import RiemannFMFlowMatchingLoss
from riemannfm.manifolds.product import RiemannFMProductManifold


class RiemannFMCombinedLoss(nn.Module):
    """Weighted combination of flow matching and contrastive alignment losses.

    L = L_cont + lambda * L_disc + mu * L_align

    Args:
        manifold: Product manifold.
        num_edge_types: Number of edge types.
        text_dim: Text embedding dimension.
        lambda_disc: Weight for discrete flow matching loss.
        mu_align: Weight for contrastive alignment loss.
        temperature: Temperature for contrastive loss.
    """

    def __init__(
        self,
        manifold: RiemannFMProductManifold,
        num_edge_types: int,
        text_dim: int = 1024,
        lambda_disc: float = 1.0,
        mu_align: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.lambda_disc = lambda_disc
        self.mu_align = mu_align

        self.flow_loss = RiemannFMFlowMatchingLoss(
            manifold=manifold,
            num_edge_types=num_edge_types,
            lambda_disc=lambda_disc,
        )
        self.align_loss = RiemannFMContrastiveLoss(
            manifold=manifold,
            text_dim=text_dim,
            temperature=temperature,
        )

    def forward(
        self,
        outputs: dict[str, Tensor],
        graph_embeds: Tensor | None = None,
        text_embeds: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute combined loss.

        Args:
            outputs: Dictionary from RiemannFM.forward().
            graph_embeds: Pooled graph embeddings for contrastive loss, shape (B, D).
            text_embeds: Pooled text embeddings for contrastive loss, shape (B, D).

        Returns:
            Dictionary with all loss components and total.
        """
        losses = self.flow_loss(outputs)

        if graph_embeds is not None and text_embeds is not None:
            l_align = self.align_loss(graph_embeds, text_embeds)
            losses["loss_align"] = l_align
            losses["loss"] = losses["loss"] + self.mu_align * l_align
        else:
            losses["loss_align"] = outputs["v_pred"].new_tensor(0.0)

        return dict(losses)
