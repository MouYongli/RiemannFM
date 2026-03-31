"""Flow matching losses: continuous (Riemannian MSE) + discrete (cross-entropy)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from riemannfm.manifolds.product import RiemannFMProductManifold


class RiemannFMFlowMatchingLoss(nn.Module):
    """Combined continuous + discrete flow matching loss.

    L = L_cont + lambda * L_disc

    L_cont: Riemannian MSE between predicted and target vector fields
    L_disc: Cross-entropy for edge type prediction

    Args:
        manifold: Product manifold for Riemannian norm computation.
        num_edge_types: Number of edge types (K+1).
        lambda_disc: Weight for discrete loss.
    """

    def __init__(
        self,
        manifold: RiemannFMProductManifold,
        num_edge_types: int,
        lambda_disc: float = 1.0,
    ):
        super().__init__()
        self.manifold = manifold
        self.num_edge_types = num_edge_types
        self.lambda_disc = lambda_disc

    def continuous_loss(
        self,
        v_pred: Tensor,
        u_target: Tensor,
        x_t: Tensor,
    ) -> Tensor:
        """Riemannian MSE loss for the continuous vector field.

        L_cont = mean_i || v_pred_i - u_target_i ||^2_{T_{x_t} M}

        Args:
            v_pred: Predicted tangent vectors, shape (N, D).
            u_target: Target tangent vectors, shape (N, D).
            x_t: Current positions (for computing Riemannian norm), shape (N, D).

        Returns:
            Scalar loss.
        """
        diff = v_pred - u_target
        x_h, x_s, x_e = self.manifold.split(x_t)
        d_h, d_s, d_e = self.manifold.split(diff)
        mH, mS, mE = self.manifold._get_manifolds()

        # Riemannian squared norms on each sub-manifold
        loss_h = mH.inner(x_h, d_h, d_h)
        loss_s = mS.inner(x_s, d_s, d_s)
        loss_e = mE.inner(x_e, d_e, d_e)

        return (loss_h + loss_s + loss_e).mean()

    def discrete_loss(
        self,
        p_pred: Tensor,
        e_1: Tensor,
    ) -> Tensor:
        """Cross-entropy loss for edge type prediction.

        Args:
            p_pred: Predicted logits, shape (N, N, K+1).
            e_1: True edge types, shape (N, N).

        Returns:
            Scalar loss.
        """
        N = p_pred.shape[0]
        # Exclude diagonal (self-loops)
        mask = ~torch.eye(N, dtype=torch.bool, device=p_pred.device)
        logits = p_pred[mask]  # (N*(N-1), K+1)
        targets = e_1[mask].long()  # (N*(N-1),)
        return F.cross_entropy(logits, targets)

    def forward(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute combined flow matching loss.

        Args:
            outputs: Dictionary from RiemannFM.forward() with keys:
                v_pred, p_pred, x_t, u_target, e_1.

        Returns:
            Dictionary with loss values: total, continuous, discrete.
        """
        l_cont = self.continuous_loss(outputs["v_pred"], outputs["u_target"], outputs["x_t"])
        l_disc = self.discrete_loss(outputs["p_pred"], outputs["e_1"])
        total = l_cont + self.lambda_disc * l_disc

        return {
            "loss": total,
            "loss_continuous": l_cont,
            "loss_discrete": l_disc,
        }
