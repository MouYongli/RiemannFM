"""Continuous flow matching on product manifolds via geodesic interpolation.

Defines the conditional flow path x_t = geodesic(x_0, x_1, t) and the
target vector field u_t(x_t | x_1) = log_{x_t}(x_1) / (1 - t).
"""

import torch.nn as nn
from torch import Tensor

from riedfm.manifolds.product import RieDFMProductManifold
from riedfm.utils.manifold_utils import EPS


class RieDFMContinuousFlow(nn.Module):
    """Continuous flow matching for node coordinates on a product manifold.

    The conditional flow path follows geodesics:
        x_t = Exp_{x_0}(t * Log_{x_0}(x_1))

    The target vector field at x_t is:
        u_t = Log_{x_t}(x_1) / (1 - t)

    A neural network v_theta learns to predict u_t from (x_t, t, context).
    """

    def __init__(self, manifold: RieDFMProductManifold):
        super().__init__()
        self.manifold = manifold

    def interpolate(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Compute the geodesic interpolant x_t between noise x_0 and data x_1.

        Args:
            x_0: Noise points on manifold, shape (N, D).
            x_1: Data points on manifold, shape (N, D).
            t: Time parameter in [0, 1], shape (N,) or scalar.

        Returns:
            Interpolated points x_t, shape (N, D).
        """
        return self.manifold.geodesic(x_0, x_1, t)

    def target_vector_field(self, x_t: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Compute the target conditional vector field u_t(x_t | x_1).

        u_t = Log_{x_t}(x_1) / (1 - t)

        Args:
            x_t: Current interpolated points, shape (N, D).
            x_1: Target data points, shape (N, D).
            t: Current time, shape (N,) or scalar.

        Returns:
            Target tangent vectors at x_t, shape (N, D).
        """
        log_vec = self.manifold.log_map(x_t, x_1)
        # Compute 1/(1-t) with clamping for numerical stability near t=1
        if isinstance(t, int | float):
            inv_remaining = 1.0 / max(1.0 - t, EPS)
            return log_vec * inv_remaining
        while t.dim() < log_vec.dim():
            t = t.unsqueeze(-1)
        inv_remaining = 1.0 / (1.0 - t).clamp(min=EPS)
        return log_vec * inv_remaining

    def compute_loss(
        self,
        v_pred: Tensor,
        x_t: Tensor,
        x_1: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Compute the continuous flow matching loss (Riemannian MSE).

        L = sum_i || v_pred_i - u_target_i ||^2_{T_{x_t} M}

        Args:
            v_pred: Predicted vector field from network, shape (N, D).
            x_t: Current interpolated points, shape (N, D).
            x_1: Target data points, shape (N, D).
            t: Current time, shape (N,) or scalar.

        Returns:
            Scalar loss.
        """
        u_target = self.target_vector_field(x_t, x_1, t)
        diff = v_pred - u_target
        # Riemannian squared norm: split into sub-manifolds and compute
        # For product manifold, the metric is block-diagonal
        x_h, x_s, x_e = self.manifold.split(x_t)
        d_h, d_s, d_e = self.manifold.split(diff)
        mH, mS, mE = self.manifold._get_manifolds()

        loss_h = mH.inner(x_h, d_h, d_h)  # Lorentz inner product
        loss_s = mS.inner(x_s, d_s, d_s)  # Euclidean inner product on tangent space
        loss_e = mE.inner(x_e, d_e, d_e)  # Euclidean inner product

        return (loss_h + loss_s + loss_e).mean()

    def ode_step(self, x_t: Tensor, v: Tensor, dt: float) -> Tensor:
        """Take one Euler step on the manifold ODE: x_{t+dt} = Exp_{x_t}(dt * v).

        Args:
            x_t: Current points, shape (N, D).
            v: Velocity vector field, shape (N, D).
            dt: Time step size.

        Returns:
            Updated points x_{t+dt}, shape (N, D).
        """
        return self.manifold.exp_map(x_t, dt * v)
