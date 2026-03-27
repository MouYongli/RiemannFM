"""Joint continuous-discrete flow matching scheduler.

Coordinates the simultaneous evolution of:
- Continuous node coordinates on the product manifold
- Discrete edge types via CTMC
"""

import torch
import torch.nn as nn
from torch import Tensor

from riedfm.flow.continuous_flow import RieDFMContinuousFlow
from riedfm.flow.discrete_flow import RieDFMDiscreteFlow
from riedfm.flow.noise import RieDFMManifoldNoise, RieDFMSparseEdgeNoise
from riedfm.manifolds.product import RieDFMProductManifold


class RieDFMJointFlow(nn.Module):
    """Joint continuous-discrete flow matching for graph generation.

    Coordinates sampling of time t, noise generation, interpolation,
    and loss computation for both node coordinates and edge types.
    """

    def __init__(
        self,
        manifold: RieDFMProductManifold,
        num_edge_types: int,
        avg_edge_density: float = 0.05,
        time_distribution: str = "uniform",
    ):
        """
        Args:
            manifold: Product manifold for node coordinates.
            num_edge_types: Number of edge types (including 0 = no-edge).
            avg_edge_density: Average edge density for sparse noise prior.
            time_distribution: How to sample t - "uniform" or "logit_normal".
        """
        super().__init__()
        self.continuous_flow = RieDFMContinuousFlow(manifold)
        self.discrete_flow = RieDFMDiscreteFlow(num_edge_types)
        self.node_noise = RieDFMManifoldNoise(manifold)
        self.edge_noise = RieDFMSparseEdgeNoise(
            num_edge_types=num_edge_types - 1,  # K non-zero types
            avg_density=avg_edge_density,
        )
        self.time_distribution = time_distribution

    def sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample time parameter t from the chosen distribution.

        Args:
            batch_size: Number of time samples.
            device: Torch device.

        Returns:
            Time values in (0, 1), shape (batch_size,).
        """
        if self.time_distribution == "uniform":
            return torch.rand(batch_size, device=device)
        elif self.time_distribution == "logit_normal":
            # Logit-normal concentrates samples toward t=0.5, which can
            # help with training stability
            z = torch.randn(batch_size, device=device)
            return torch.sigmoid(z)
        else:
            raise ValueError(f"Unknown time distribution: {self.time_distribution}")

    def prepare_training_data(
        self,
        x_1: Tensor,
        e_1: Tensor,
        t: Tensor,
    ) -> dict[str, Tensor]:
        """Prepare noisy interpolated data for one training step.

        Args:
            x_1: True node coordinates, shape (N, D).
            e_1: True edge type matrix, shape (N, N).
            t: Time parameter, scalar or shape (1,).

        Returns:
            Dictionary with keys:
                - x_0: Noise node coordinates
                - x_t: Interpolated node coordinates
                - e_0: Noise edge types
                - e_t: Interpolated edge types
                - u_target: Target vector field for continuous part
                - t: Time parameter
        """
        N = x_1.shape[0]
        device = x_1.device

        # Sample noise
        x_0 = self.node_noise.sample(N, device)
        e_0 = self.edge_noise.sample(N, device)

        # Interpolate
        x_t = self.continuous_flow.interpolate(x_0, x_1, t)
        e_t = self.discrete_flow.interpolate(e_0, e_1, t)

        # Target vector field
        u_target = self.continuous_flow.target_vector_field(x_t, x_1, t)

        return {
            "x_0": x_0,
            "x_t": x_t,
            "e_0": e_0,
            "e_t": e_t,
            "u_target": u_target,
            "t": t,
        }

    @torch.no_grad()
    def sample(
        self,
        network_fn,
        num_nodes: int,
        num_steps: int = 100,
        device: torch.device | None = None,
        condition: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Generate a graph by integrating the learned flow.

        Args:
            network_fn: Callable (x_t, e_t, t, condition) -> (v_pred, p_pred).
            num_nodes: Number of nodes (including virtual nodes).
            num_steps: Number of ODE integration steps.
            device: Torch device.
            condition: Optional conditioning tensor (e.g., text embeddings).

        Returns:
            (x_1, e_1): Generated node coordinates and edge types.
        """
        if device is None:
            device = torch.device("cpu")
        dt = 1.0 / num_steps

        # Initialize from noise
        x_t = self.node_noise.sample(num_nodes, device)
        e_t = self.edge_noise.sample(num_nodes, device)

        for step in range(num_steps):
            t = step * dt
            t_tensor = torch.tensor(t, device=device)

            # Get network predictions
            v_pred, p_pred = network_fn(x_t, e_t, t_tensor, condition)

            # Update continuous coordinates (Euler step on manifold)
            x_t = self.continuous_flow.ode_step(x_t, v_pred, dt)

            # Update discrete edge types
            e_t = self.discrete_flow.sample_step(e_t, p_pred, t, dt)

        return x_t, e_t
