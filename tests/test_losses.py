"""Tests for RiemannFM loss functions (spec §19)."""

from __future__ import annotations

import torch

from riemannfm.losses.combined_loss import RiemannFMCombinedLoss
from riemannfm.losses.flow_matching_loss import (
    continuous_flow_loss,
    edge_existence_loss,
    edge_type_loss,
)
from riemannfm.losses.relation_align import relation_align_infonce
from riemannfm.manifolds.product import RiemannFMProductManifold

B = 4
N = 6
K = 5
DIM_H = 4
DIM_S = 4
DIM_E = 4


def _make_manifold() -> RiemannFMProductManifold:
    return RiemannFMProductManifold(DIM_H, DIM_S, DIM_E)


def _masked_edges(node_mask: torch.Tensor) -> torch.Tensor:
    """Return mu_t = 1 on every valid pair (matches full-mask training scenarios)."""
    pm = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
    return pm.float()


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
        mask = torch.zeros(B, N, dtype=torch.bool)
        loss = continuous_flow_loss(manifold, V_hat, u_t, x_t, mask)
        assert loss < 1e-6

    def test_gradient_finite_when_residual_zero_at_masked_token(self) -> None:
        manifold = _make_manifold()
        D = manifold.ambient_dim
        x_t = manifold.sample_noise(B, N, radius_h=1.0)
        u_t = torch.randn(B, N, D)
        V_hat = u_t.clone().detach().requires_grad_(True)
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[0, 0] = False
        loss = continuous_flow_loss(manifold, V_hat, u_t, x_t, mask)
        loss.backward()
        assert V_hat.grad is not None
        assert torch.isfinite(V_hat.grad).all()
        for comp in (manifold.hyperbolic, manifold.spherical):
            if comp is not None and comp._curvature.grad is not None:
                assert torch.isfinite(comp._curvature.grad).all()


class TestEdgeExistenceLoss:
    def test_finite_and_non_negative(self) -> None:
        ell = torch.randn(B, N, N)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)
        mu_t = _masked_edges(mask)
        loss = edge_existence_loss(ell, E_1, mask, mu_t)
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_zero_when_no_masked_positions(self) -> None:
        ell = torch.randn(B, N, N)
        E_1 = torch.zeros(B, N, N, K)
        mask = torch.ones(B, N, dtype=torch.bool)
        mu_t = torch.zeros(B, N, N)
        loss = edge_existence_loss(ell, E_1, mask, mu_t)
        assert loss.item() == 0.0

    def test_low_loss_when_prediction_matches(self) -> None:
        E_1 = torch.zeros(B, N, N, K)
        # Strong negative logits for "no edge" → BCE small.
        ell = torch.full((B, N, N), -10.0)
        mask = torch.ones(B, N, dtype=torch.bool)
        mu_t = _masked_edges(mask)
        loss = edge_existence_loss(ell, E_1, mask, mu_t)
        assert loss < 0.01

    def test_gradient_flow(self) -> None:
        ell = torch.randn(B, N, N, requires_grad=True)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)
        mu_t = _masked_edges(mask)
        edge_existence_loss(ell, E_1, mask, mu_t).backward()
        assert ell.grad is not None


class TestEdgeTypeLoss:
    def test_gated_on_positive_masked_positions(self) -> None:
        ell_type = torch.randn(B, N, N, K)
        E_1 = torch.zeros(B, N, N, K)
        # Set some positive edges at arbitrary positions.
        E_1[:, 0, 1, 0] = 1.0
        E_1[:, 2, 3, 2] = 1.0
        mask = torch.ones(B, N, dtype=torch.bool)
        mu_t = _masked_edges(mask)
        loss = edge_type_loss(ell_type, E_1, mask, mu_t)
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_zero_when_no_positive_masked(self) -> None:
        ell_type = torch.randn(B, N, N, K)
        E_1 = torch.zeros(B, N, N, K)  # no positives anywhere
        mask = torch.ones(B, N, dtype=torch.bool)
        mu_t = _masked_edges(mask)
        loss = edge_type_loss(ell_type, E_1, mask, mu_t)
        assert loss.item() == 0.0

    def test_gradient_flow(self) -> None:
        ell_type = torch.randn(B, N, N, K, requires_grad=True)
        E_1 = torch.zeros(B, N, N, K)
        E_1[:, 0, 1, 0] = 1.0
        mask = torch.ones(B, N, dtype=torch.bool)
        mu_t = _masked_edges(mask)
        edge_type_loss(ell_type, E_1, mask, mu_t).backward()
        assert ell_type.grad is not None


class TestRelationAlign:
    def test_finite_and_non_negative(self) -> None:
        z_R = torch.randn(K, 24)
        z_C = torch.randn(K, 24)
        loss = relation_align_infonce(z_R, z_C)
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_zero_when_singleton(self) -> None:
        loss = relation_align_infonce(torch.randn(1, 16), torch.randn(1, 16))
        assert loss.item() == 0.0

    def test_zero_when_text_empty(self) -> None:
        loss = relation_align_infonce(torch.randn(K, 8), torch.zeros(K, 8))
        assert loss.item() == 0.0

    def test_gradient_flow(self) -> None:
        z_R = torch.randn(K, 24, requires_grad=True)
        z_C = torch.randn(K, 24, requires_grad=True)
        relation_align_infonce(z_R, z_C).backward()
        assert z_R.grad is not None
        assert z_C.grad is not None


class TestCombinedLossV101:
    def _inputs(self) -> dict[str, torch.Tensor]:
        manifold = _make_manifold()
        D = manifold.ambient_dim
        x_t = manifold.sample_noise(B, N, radius_h=1.0)
        V_hat = torch.randn(B, N, D, requires_grad=True)
        u_t = torch.randn(B, N, D)
        ell_ex = torch.randn(B, N, N, requires_grad=True)
        ell_type = torch.randn(B, N, N, K, requires_grad=True)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        E_1[:, 0, 1, 0] = 1.0  # guarantee at least one positive
        mask = torch.ones(B, N, dtype=torch.bool)
        mu_t = _masked_edges(mask)
        return {
            "manifold": manifold,
            "V_hat": V_hat, "u_t": u_t, "x_t": x_t,
            "ell_ex": ell_ex, "ell_type": ell_type,
            "E_1": E_1, "mu_t": mu_t, "node_mask": mask,
        }

    def test_metrics_keys_and_finiteness(self) -> None:
        inp = self._inputs()
        loss_fn = RiemannFMCombinedLoss(inp["manifold"])
        total, metrics = loss_fn(
            V_hat=inp["V_hat"], u_t=inp["u_t"], x_t=inp["x_t"],
            ell_ex=inp["ell_ex"], ell_type=inp["ell_type"],
            E_1=inp["E_1"], mu_t=inp["mu_t"], node_mask=inp["node_mask"],
        )
        assert torch.isfinite(total)
        for k in ("loss/total", "loss/X", "loss/ex", "loss/ty", "loss/align_R"):
            assert k in metrics

    def test_align_R_zero_when_disabled(self) -> None:
        inp = self._inputs()
        loss_fn = RiemannFMCombinedLoss(inp["manifold"], lambda_align_R=0.0)
        _, metrics = loss_fn(
            V_hat=inp["V_hat"], u_t=inp["u_t"], x_t=inp["x_t"],
            ell_ex=inp["ell_ex"], ell_type=inp["ell_type"],
            E_1=inp["E_1"], mu_t=inp["mu_t"], node_mask=inp["node_mask"],
            z_R=torch.randn(K, 16), z_C=torch.randn(K, 16),
        )
        assert metrics["loss/align_R"].item() == 0.0

    def test_align_R_active(self) -> None:
        inp = self._inputs()
        loss_fn = RiemannFMCombinedLoss(inp["manifold"], lambda_align_R=0.05)
        z_R = torch.randn(K, 16, requires_grad=True)
        z_C = torch.randn(K, 16, requires_grad=True)
        total, metrics = loss_fn(
            V_hat=inp["V_hat"], u_t=inp["u_t"], x_t=inp["x_t"],
            ell_ex=inp["ell_ex"], ell_type=inp["ell_type"],
            E_1=inp["E_1"], mu_t=inp["mu_t"], node_mask=inp["node_mask"],
            z_R=z_R, z_C=z_C,
        )
        assert metrics["loss/align_R"] > 0.0
        total.backward()
        assert z_R.grad is not None
        assert z_C.grad is not None

    def test_backward_flows_to_all_predictions(self) -> None:
        inp = self._inputs()
        loss_fn = RiemannFMCombinedLoss(inp["manifold"])
        total, _ = loss_fn(
            V_hat=inp["V_hat"], u_t=inp["u_t"], x_t=inp["x_t"],
            ell_ex=inp["ell_ex"], ell_type=inp["ell_type"],
            E_1=inp["E_1"], mu_t=inp["mu_t"], node_mask=inp["node_mask"],
        )
        total.backward()
        assert inp["V_hat"].grad is not None
        assert inp["ell_ex"].grad is not None
        assert inp["ell_type"].grad is not None

    def test_m_coord_gates_L_X(self) -> None:
        """m_coord = 0 on all nodes → L_X = 0 (no gating mask survives)."""
        inp = self._inputs()
        loss_fn = RiemannFMCombinedLoss(inp["manifold"])
        m_coord = torch.zeros(B, N, dtype=torch.bool)
        _, metrics = loss_fn(
            V_hat=inp["V_hat"], u_t=inp["u_t"], x_t=inp["x_t"],
            ell_ex=inp["ell_ex"], ell_type=inp["ell_type"],
            E_1=inp["E_1"], mu_t=inp["mu_t"], node_mask=inp["node_mask"],
            m_coord=m_coord,
        )
        assert metrics["loss/X"].item() == 0.0
