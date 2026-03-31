"""Graph-text contrastive alignment loss (InfoNCE).

Aligns graph embeddings with text embeddings using contrastive learning.
The graph embedding is obtained by projecting the Frechet mean of node
coordinates from the manifold to Euclidean space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from riemannfm.manifolds.product import RiemannFMProductManifold


class RiemannFMContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss for graph-text alignment.

    L_align = -1/B * sum_i log( exp(sim(f(x_1^i), c_i) / tau) / sum_j exp(sim(f(x_1^j), c_j) / tau) )

    Args:
        manifold: Product manifold for graph embeddings.
        graph_dim: Dimension of projected graph embedding.
        text_dim: Dimension of text embedding.
        proj_dim: Shared projection dimension for contrastive loss.
        temperature: Temperature parameter tau.
    """

    def __init__(
        self,
        manifold: RiemannFMProductManifold,
        graph_dim: int | None = None,
        text_dim: int = 1024,
        proj_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        graph_input_dim = graph_dim or manifold.total_dim

        # Projection heads: manifold -> shared space, text -> shared space
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_input_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(
        self,
        graph_embeds: Tensor,
        text_embeds: Tensor,
    ) -> Tensor:
        """Compute InfoNCE contrastive loss.

        Args:
            graph_embeds: Graph-level embeddings, shape (B, graph_dim).
                         Typically the mean of node coordinates or pooled features.
            text_embeds: Text embeddings, shape (B, text_dim).

        Returns:
            Scalar loss.
        """
        # Project to shared space and normalize
        g = F.normalize(self.graph_proj(graph_embeds), dim=-1)  # (B, proj_dim)
        t = F.normalize(self.text_proj(text_embeds), dim=-1)  # (B, proj_dim)

        # Cosine similarity matrix
        sim = torch.mm(g, t.T) / self.temperature  # (B, B)

        # InfoNCE: diagonal entries are positive pairs
        labels = torch.arange(sim.shape[0], device=sim.device)
        loss_g2t = F.cross_entropy(sim, labels)
        loss_t2g = F.cross_entropy(sim.T, labels)

        return (loss_g2t + loss_t2g) / 2
