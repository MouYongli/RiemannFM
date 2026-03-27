"""Top-level RieDFM-G model: wraps RED-Former with flow matching.

This is the main model class used for pretraining and inference.
It orchestrates noise sampling, interpolation, backbone forward pass,
loss computation, and generation.
"""

import torch
import torch.nn as nn
from torch import Tensor

from riedfm.flow.joint_flow import RieDFMJointFlow
from riedfm.manifolds.product import RieDFMProductManifold
from riedfm.models.red_former import RieDFMREDFormer


class RieDFMG(nn.Module):
    """RieDFM-G: Riemannian Discrete Flow Matching on Graphs.

    Top-level model that combines:
    - RieDFMProductManifold for geometry
    - RieDFMJointFlow for noise/interpolation/sampling
    - RieDFMREDFormer as the denoising backbone

    Args:
        manifold: Product manifold configuration.
        num_layers: Number of RED-Former layers.
        node_dim: Node hidden dimension.
        edge_dim: Edge hidden dimension.
        num_heads: Number of attention heads.
        num_edge_types: Number of edge types (including no-edge).
        text_dim: Text encoder output dimension.
        text_cross_attn_every: Frequency of text injection layers.
        avg_edge_density: Average edge density for sparse noise prior.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        manifold: RieDFMProductManifold,
        num_layers: int = 12,
        node_dim: int = 768,
        edge_dim: int = 256,
        num_heads: int = 12,
        num_edge_types: int = 1201,
        text_dim: int = 1024,
        text_cross_attn_every: int = 3,
        avg_edge_density: float = 0.05,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.manifold = manifold

        # Flow matching
        self.flow_matcher = RieDFMJointFlow(
            manifold=manifold,
            num_edge_types=num_edge_types,
            avg_edge_density=avg_edge_density,
        )

        # Backbone
        self.backbone = RieDFMREDFormer(
            manifold=manifold,
            num_layers=num_layers,
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            num_edge_types=num_edge_types,
            text_dim=text_dim,
            text_cross_attn_every=text_cross_attn_every,
            dropout=dropout,
        )

    def forward(
        self,
        x_1: Tensor,
        e_1: Tensor,
        text_embeds: Tensor | None = None,
        text_mask: Tensor | None = None,
        depth: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Training forward pass: sample t, create noisy data, predict, compute targets.

        Args:
            x_1: True node coordinates, shape (N, total_dim).
            e_1: True edge type matrix, shape (N, N).
            text_embeds: Optional text embeddings, shape (T, text_dim).
            text_mask: Optional text mask, shape (T,).
            depth: Optional hierarchy depth, shape (N,).

        Returns:
            Dictionary with:
                - v_pred: Predicted velocity field, shape (N, total_dim).
                - p_pred: Predicted edge logits, shape (N, N, K+1).
                - x_t: Noisy interpolated coordinates, shape (N, total_dim).
                - e_t: Noisy interpolated edge types, shape (N, N).
                - u_target: Target velocity field, shape (N, total_dim).
                - e_1: True edge types (for loss computation).
                - t: Sampled time.
        """
        device = x_1.device

        # Sample time
        t = self.flow_matcher.sample_time(1, device).squeeze()

        # Create noisy interpolated data
        flow_data = self.flow_matcher.prepare_training_data(x_1, e_1, t)

        # Backbone forward
        v_pred, p_pred = self.backbone(
            x_t=flow_data["x_t"],
            e_t=flow_data["e_t"],
            t=t,
            text_embeds=text_embeds,
            text_mask=text_mask,
            depth=depth,
        )

        return {
            "v_pred": v_pred,
            "p_pred": p_pred,
            "x_t": flow_data["x_t"],
            "e_t": flow_data["e_t"],
            "u_target": flow_data["u_target"],
            "e_1": e_1,
            "t": t,
        }

    @torch.no_grad()
    def generate(
        self,
        num_nodes: int,
        num_steps: int = 100,
        device: torch.device | None = None,
        text_embeds: Tensor | None = None,
        text_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Generate a graph by integrating the learned flow.

        Args:
            num_nodes: Number of nodes (including virtual).
            num_steps: Number of ODE integration steps.
            device: Torch device.
            text_embeds: Optional text condition.
            text_mask: Optional text mask.

        Returns:
            (x_1, e_1): Generated node coordinates and edge type matrix.
        """
        if device is None:
            device = torch.device("cpu")

        def network_fn(x_t, e_t, t, condition):
            return self.backbone(
                x_t=x_t,
                e_t=e_t,
                t=t,
                text_embeds=condition,
                text_mask=text_mask,
            )

        x_gen, e_gen = self.flow_matcher.sample(
            network_fn=network_fn,
            num_nodes=num_nodes,
            num_steps=num_steps,
            device=device,
            condition=text_embeds,
        )
        return (x_gen, e_gen)
