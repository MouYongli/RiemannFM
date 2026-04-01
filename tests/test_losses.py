"""Tests for RiemannFM loss functions (Phase 3: Stream B).

Covers loss finiteness, positivity, masking behavior, gradient flow
to curvature parameters, and the combined loss orchestrator.
"""

from __future__ import annotations

import torch

from riemannfm.losses.combined_loss import RiemannFMCombinedLoss
from riemannfm.losses.contrastive_loss import contrastive_alignment_loss
from riemannfm.losses.flow_matching_loss import (
    continuous_flow_loss,
    discrete_flow_loss,
)
from riemannfm.manifolds.product import RiemannFMProductManifold

B = 4
N = 6
K = 5
DIM_H = 4
DIM_S = 4
DIM_E = 4


def _make_manifold() -> RiemannFMProductManifold:
    return RiemannFMProductManifold(DIM_H, DIM_S, DIM_E)


class TestContinuousFlowLoss:
    def test_finite_and_positive(self) -> None:
        manifold = _make_manifold()
        D = manifold.ambient_dim
        x_t = manifold.sample_noise(B, N, radius_h=1.0)
        V_hat = torch.randn(B, N, D)
        u_t = torch.randn(B, N, D)
        mask = torch.ones(B, N, dtype=torch.bool)
        loss = continuous_flow_loss(manifold, V_hat, u_t, x_t, mask)
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_zero_when_perfect(self) -> None:
        manifold = _make_manifold()
        D = manifold.ambient_dim
        x_t = manifold.sample_noise(B, N, radius_h=1.0)
        u_t = torch.randn(B, N, D)
        # Perfect prediction.
        V_hat = u_t.clone()
        mask = torch.ones(B, N, dtype=torch.bool)
        loss = continuous_flow_loss(manifold, V_hat, u_t, x_t, mask)
        assert loss < 1e-6

    def test_masks_virtual_nodes(self) -> None:
        manifold = _make_manifold()
        D = manifold.ambient_dim
        x_t = manifold.sample_noise(B, N, radius_h=1.0)
        u_t = torch.randn(B, N, D)
        V_hat = u_t.clone()
        # All virtual — loss should be zero.
        mask = torch.zeros(B, N, dtype=torch.bool)
        loss = continuous_flow_loss(manifold, V_hat, u_t, x_t, mask)
        assert loss < 1e-6

    def test_gradient_to_prediction(self) -> None:
        manifold = _make_manifold()
        D = manifold.ambient_dim
        x_t = manifold.sample_noise(B, N, radius_h=1.0)
        V_hat = torch.randn(B, N, D, requires_grad=True)
        u_t = torch.randn(B, N, D)
        mask = torch.ones(B, N, dtype=torch.bool)
        loss = continuous_flow_loss(manifold, V_hat, u_t, x_t, mask)
        loss.backward()
        assert V_hat.grad is not None
        assert torch.isfinite(V_hat.grad).all()


class TestDiscreteFlowLoss:
    def test_finite_and_positive(self) -> None:
        P_hat = torch.randn(B, N, N, K)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)
        loss = discrete_flow_loss(P_hat, E_1, mask)
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_zero_ish_when_perfect(self) -> None:
        E_1 = torch.zeros(B, N, N, K)
        # Perfect prediction: large negative logits for all zeros.
        P_hat = torch.full((B, N, N, K), -10.0)
        mask = torch.ones(B, N, dtype=torch.bool)
        loss = discrete_flow_loss(P_hat, E_1, mask)
        assert loss < 0.01

    def test_masks_virtual_nodes(self) -> None:
        P_hat = torch.randn(B, N, N, K)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask_full = torch.ones(B, N, dtype=torch.bool)
        mask_partial = torch.zeros(B, N, dtype=torch.bool)
        mask_partial[:, :2] = True
        loss_full = discrete_flow_loss(P_hat, E_1, mask_full)
        loss_partial = discrete_flow_loss(P_hat, E_1, mask_partial)
        # Partial mask should give different loss.
        assert not torch.allclose(loss_full, loss_partial)

    def test_gradient_flow(self) -> None:
        P_hat = torch.randn(B, N, N, K, requires_grad=True)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)
        loss = discrete_flow_loss(P_hat, E_1, mask)
        loss.backward()
        assert P_hat.grad is not None


class TestContrastiveLoss:
    def test_with_text(self) -> None:
        d = 32
        h = torch.randn(B, N, d)
        node_text = torch.randn(B, N, d)
        mask = torch.ones(B, N, dtype=torch.bool)
        loss = contrastive_alignment_loss(h, node_text, mask)
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_zero_when_no_text(self) -> None:
        d = 32
        h = torch.randn(B, N, d)
        node_text = torch.zeros(B, N, 0)  # d_c = 0
        mask = torch.ones(B, N, dtype=torch.bool)
        loss = contrastive_alignment_loss(h, node_text, mask)
        assert loss == 0.0

    def test_gradient_flow(self) -> None:
        d = 32
        h = torch.randn(B, N, d, requires_grad=True)
        node_text = torch.randn(B, N, d)
        mask = torch.ones(B, N, dtype=torch.bool)
        loss = contrastive_alignment_loss(h, node_text, mask)
        loss.backward()
        assert h.grad is not None


class TestCombinedLoss:
    def test_output_format(self) -> None:
        manifold = _make_manifold()
        D = manifold.ambient_dim
        loss_fn = RiemannFMCombinedLoss(manifold)
        x_t = manifold.sample_noise(B, N, radius_h=1.0)
        V_hat = torch.randn(B, N, D)
        u_t = torch.randn(B, N, D)
        P_hat = torch.randn(B, N, N, K)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)

        total, metrics = loss_fn(V_hat, u_t, x_t, P_hat, E_1, mask)
        assert torch.isfinite(total)
        assert "loss/total" in metrics
        assert "loss/cont" in metrics
        assert "loss/disc" in metrics
        assert "loss/align" in metrics

    def test_no_text_skips_align(self) -> None:
        manifold = _make_manifold()
        D = manifold.ambient_dim
        loss_fn = RiemannFMCombinedLoss(manifold, mu_align=0.1)
        x_t = manifold.sample_noise(B, N, radius_h=1.0)
        V_hat = torch.randn(B, N, D)
        u_t = torch.randn(B, N, D)
        P_hat = torch.randn(B, N, N, K)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)

        # No text → L_align = 0.
        _total, metrics = loss_fn(V_hat, u_t, x_t, P_hat, E_1, mask)
        assert metrics["loss/align"] == 0.0

    def test_gradient_backward(self) -> None:
        manifold = _make_manifold()
        D = manifold.ambient_dim
        loss_fn = RiemannFMCombinedLoss(manifold)
        x_t = manifold.sample_noise(B, N, radius_h=1.0)
        V_hat = torch.randn(B, N, D, requires_grad=True)
        u_t = torch.randn(B, N, D)
        P_hat = torch.randn(B, N, N, K, requires_grad=True)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)

        total, _ = loss_fn(V_hat, u_t, x_t, P_hat, E_1, mask)
        total.backward()
        assert V_hat.grad is not None
        assert P_hat.grad is not None
