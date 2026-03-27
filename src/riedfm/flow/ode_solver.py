"""Manifold-aware ODE solver wrapping torchdiffeq.

Provides adaptive step-size ODE integration on product manifolds,
using the exponential map to ensure points stay on the manifold.
"""

import torch
import torch.nn as nn
from torch import Tensor

from riedfm.manifolds.product import ProductManifold


class ManifoldODEFunc(nn.Module):
    """Wraps a velocity field network as an ODE function for torchdiffeq.

    The ODE is defined in the tangent space:
        dx/dt = v_theta(x_t, e_t, t, condition)

    Integration is done via repeated exp_map steps to stay on the manifold.
    """

    def __init__(
        self,
        network_fn,
        manifold: ProductManifold,
        edge_types: Tensor,
        condition: Tensor | None = None,
    ):
        """
        Args:
            network_fn: Callable (x_t, e_t, t, condition) -> (v_pred, p_pred).
            manifold: Product manifold instance.
            edge_types: Current edge type matrix (held fixed during ODE solve).
            condition: Optional conditioning tensor.
        """
        super().__init__()
        self.network_fn = network_fn
        self.manifold = manifold
        self.edge_types = edge_types
        self.condition = condition

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute the velocity field at time t and state x.

        Note: torchdiffeq calls this as f(t, x) where t is scalar.
        """
        v_pred, _ = self.network_fn(x, self.edge_types, t, self.condition)
        # Project onto tangent space for numerical safety
        v_pred = self.manifold.proj_tangent(x, v_pred)
        return v_pred


class ManifoldODESolver(nn.Module):
    """Adaptive ODE solver on product manifolds.

    Uses Euler method with manifold exponential map by default.
    Can optionally use torchdiffeq for adaptive stepping (Dormand-Prince).
    """

    def __init__(
        self,
        manifold: ProductManifold,
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        """
        Args:
            manifold: Product manifold instance.
            method: ODE solver method - "euler" for fixed-step manifold Euler,
                   "dopri5" for adaptive Dormand-Prince (via torchdiffeq).
            atol: Absolute tolerance for adaptive solver.
            rtol: Relative tolerance for adaptive solver.
        """
        super().__init__()
        self.manifold = manifold
        self.method = method
        self.atol = atol
        self.rtol = rtol

    @torch.no_grad()
    def solve(
        self,
        network_fn,
        x_0: Tensor,
        e_t: Tensor,
        num_steps: int = 100,
        condition: Tensor | None = None,
    ) -> Tensor:
        """Integrate the ODE from t=0 to t=1.

        Args:
            network_fn: Velocity field network.
            x_0: Initial noise points, shape (N, D).
            e_t: Edge types (may be updated externally between calls).
            num_steps: Number of steps for fixed-step methods.
            condition: Optional conditioning tensor.

        Returns:
            Final points x_1 on the manifold, shape (N, D).
        """
        if self.method == "euler":
            return self._euler_solve(network_fn, x_0, e_t, num_steps, condition)
        elif self.method == "dopri5":
            return self._adaptive_solve(network_fn, x_0, e_t, condition)
        else:
            raise ValueError(f"Unknown ODE method: {self.method}")

    def _euler_solve(
        self,
        network_fn,
        x_t: Tensor,
        e_t: Tensor,
        num_steps: int,
        condition: Tensor | None,
    ) -> Tensor:
        """Fixed-step Euler method with manifold exponential map."""
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.tensor(step * dt, device=x_t.device, dtype=x_t.dtype)
            v_pred, _ = network_fn(x_t, e_t, t, condition)
            v_pred = self.manifold.proj_tangent(x_t, v_pred)
            x_t = self.manifold.exp_map(x_t, dt * v_pred)
        return x_t

    def _adaptive_solve(
        self,
        network_fn,
        x_0: Tensor,
        e_t: Tensor,
        condition: Tensor | None,
    ) -> Tensor:
        """Adaptive Dormand-Prince solver via torchdiffeq.

        Note: This operates in the ambient Euclidean space and requires
        periodic re-projection onto the manifold.
        """
        from torchdiffeq import odeint

        ode_func = ManifoldODEFunc(network_fn, self.manifold, e_t, condition)
        t_span = torch.tensor([0.0, 1.0], device=x_0.device, dtype=x_0.dtype)
        solution = odeint(
            ode_func,
            x_0,
            t_span,
            method="dopri5",
            atol=self.atol,
            rtol=self.rtol,
        )
        # solution shape: (2, N, D), take the final state
        x_1 = solution[-1]
        # Re-project onto manifold for safety
        x_h, x_s, x_e = self.manifold.split(x_1)
        from riedfm.utils.manifold_utils import project_to_lorentz, project_to_sphere
        x_h = project_to_lorentz(x_h)
        x_s = project_to_sphere(x_s)
        return self.manifold.combine(x_h, x_s, x_e)
