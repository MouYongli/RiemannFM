"""Tests for RiemannFM loss functions (Phase 3: Stream B).

Covers loss finiteness, positivity, masking behavior, gradient flow
to curvature parameters, and the combined loss orchestrator.
"""

from __future__ import annotations

import torch

from riemannfm.data.collator import (
    MASK_C,
    MASK_REAL,
    MASK_VIRTUAL,
    MASK_X,
    RiemannFMGraphCollator,
    _apply_node_masking,
)
from riemannfm.losses.combined_loss import (
    RiemannFMCombinedLoss,
    masked_node_loss,
)
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

    def test_gradient_finite_when_residual_zero_at_masked_token(self) -> None:
        """Regression: zero residual at any token must not produce NaN grads."""
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
        d_a = 32
        g = torch.randn(B, N, d_a)
        c = torch.randn(B, N, d_a)
        mask = torch.ones(B, N, dtype=torch.bool)
        loss = contrastive_alignment_loss(g, c, mask)
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_zero_when_no_text(self) -> None:
        g = torch.randn(B, N, 0)
        c = torch.zeros(B, N, 0)
        mask = torch.ones(B, N, dtype=torch.bool)
        loss = contrastive_alignment_loss(g, c, mask)
        assert loss == 0.0

    def test_gradient_flow(self) -> None:
        d_a = 32
        g = torch.randn(B, N, d_a, requires_grad=True)
        c = torch.randn(B, N, d_a, requires_grad=True)
        mask = torch.ones(B, N, dtype=torch.bool)
        loss = contrastive_alignment_loss(g, c, mask)
        loss.backward()
        assert g.grad is not None
        assert c.grad is not None

    def test_fewer_than_two_valid(self) -> None:
        d_a = 32
        g = torch.randn(B, N, d_a)
        c = torch.randn(B, N, d_a)
        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[0, 0] = True
        loss = contrastive_alignment_loss(g, c, mask)
        assert loss == 0.0


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
        assert "loss/mask_c" in metrics
        assert "loss/mask_x" in metrics

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

        _total, metrics = loss_fn(V_hat, u_t, x_t, P_hat, E_1, mask)
        assert metrics["loss/align"] == 0.0

    def test_with_alignment(self) -> None:
        manifold = _make_manifold()
        D = manifold.ambient_dim
        d_c = 16
        d_a = 8
        node_dim = 32
        loss_fn = RiemannFMCombinedLoss(
            manifold, mu_align=0.1, input_text_dim=d_c,
            node_dim=node_dim, d_a=d_a,
        )
        x_t = manifold.sample_noise(B, N, radius_h=1.0)
        h = torch.randn(B, N, node_dim)
        V_hat = torch.randn(B, N, D)
        u_t = torch.randn(B, N, D)
        P_hat = torch.randn(B, N, N, K)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        node_text = torch.randn(B, N, d_c)
        mask = torch.ones(B, N, dtype=torch.bool)

        total, metrics = loss_fn(
            V_hat, u_t, x_t, P_hat, E_1, mask,
            h=h, node_text=node_text,
        )
        assert torch.isfinite(total)
        assert metrics["loss/align"] > 0.0

    def test_gradient_backward(self) -> None:
        manifold = _make_manifold()
        D = manifold.ambient_dim
        d_c = 16
        node_dim = 32
        loss_fn = RiemannFMCombinedLoss(
            manifold, mu_align=0.1, input_text_dim=d_c,
            node_dim=node_dim, d_a=8,
        )
        x_t = manifold.sample_noise(B, N, radius_h=1.0)
        h = torch.randn(B, N, node_dim, requires_grad=True)
        V_hat = torch.randn(B, N, D, requires_grad=True)
        u_t = torch.randn(B, N, D)
        P_hat = torch.randn(B, N, N, K, requires_grad=True)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        node_text = torch.randn(B, N, d_c)
        mask = torch.ones(B, N, dtype=torch.bool)

        total, _ = loss_fn(
            V_hat, u_t, x_t, P_hat, E_1, mask,
            h=h, node_text=node_text,
        )
        total.backward()
        assert V_hat.grad is not None
        assert P_hat.grad is not None
        assert loss_fn.proj_g[0].weight.grad is not None
        assert loss_fn.proj_c[0].weight.grad is not None
        assert h.grad is not None


class TestNodeMasking:
    """Tests for collator-level node partition (Def 6.9a)."""

    def test_apply_node_masking_disjoint_counts(self) -> None:
        # Initialise padding positions as VIRTUAL so we can count REAL
        # anchors without accidentally counting padding.
        mask_type = torch.full((4, 10), MASK_VIRTUAL, dtype=torch.long)
        num_real = torch.tensor([8, 6, 10, 3])
        for i, nr in enumerate(num_real.tolist()):
            mask_type[i, :nr] = MASK_REAL
        t_node = torch.full((4, 10), float("nan"))
        _apply_node_masking(
            mask_type, t_node, num_real,
            mask_ratio_c=0.25, mask_ratio_x=0.15,
        )

        for i in range(4):
            nr = int(num_real[i].item())
            n_c = int((mask_type[i] == MASK_C).sum().item())
            n_x = int((mask_type[i] == MASK_X).sum().item())
            n_real = int((mask_type[i, :nr] == MASK_REAL).sum().item())
            # Disjoint partition covers all real nodes.
            assert n_c + n_x + n_real == nr
            # At least one REAL anchor survives.
            assert n_real >= 1

    def test_t_node_labels_correct(self) -> None:
        mask_type = torch.zeros(2, 8, dtype=torch.long)
        t_node = torch.full((2, 8), float("nan"))
        num_real = torch.tensor([6, 6])
        _apply_node_masking(
            mask_type, t_node, num_real,
            mask_ratio_c=0.3, mask_ratio_x=0.3,
        )
        # M_x positions must be exactly 0.0.
        assert ((mask_type == MASK_X) == (t_node == 0.0)).all()
        # M_c positions must be exactly 1.0.
        assert ((mask_type == MASK_C) == (t_node == 1.0)).all()
        # REAL / VIRTUAL positions remain NaN.
        assert torch.isnan(t_node[mask_type == MASK_REAL]).all()

    def test_masking_does_not_touch_virtual(self) -> None:
        mask_type = torch.full((2, 8), MASK_VIRTUAL, dtype=torch.long)
        mask_type[:, :4] = MASK_REAL
        t_node = torch.full((2, 8), float("nan"))
        num_real = torch.tensor([4, 4])
        _apply_node_masking(
            mask_type, t_node, num_real,
            mask_ratio_c=0.3, mask_ratio_x=0.2,
        )
        assert (mask_type[:, 4:] == MASK_VIRTUAL).all()
        assert torch.isnan(t_node[:, 4:]).all()

    def test_collator_emits_t_node(self) -> None:
        from riemannfm.data.graph import RiemannFMGraphData

        g = RiemannFMGraphData(
            edge_types=torch.zeros(6, 6, K),
            node_text=torch.randn(6, 8),
            node_mask=torch.tensor([True] * 4 + [False] * 2),
            num_nodes=4,
            node_ids=torch.tensor([0, 1, 2, 3, -1, -1]),
            num_edge_types=K,
        )
        collator = RiemannFMGraphCollator(
            max_nodes=6, num_edge_types=K,
            mask_ratio_c=0.25, mask_ratio_x=0.25,
        )
        batch = collator([g, g])
        mt = batch["mask_type"]
        tn = batch["t_node"]
        assert mt.shape == (2, 6)
        assert tn.shape == (2, 6)
        # Virtual nodes are VIRTUAL with NaN t.
        assert (mt[:, 4:] == MASK_VIRTUAL).all()
        assert torch.isnan(tn[:, 4:]).all()

    def test_collator_no_masking_when_zero(self) -> None:
        from riemannfm.data.graph import RiemannFMGraphData

        g = RiemannFMGraphData(
            edge_types=torch.zeros(4, 4, K),
            node_text=torch.randn(4, 8),
            node_mask=torch.ones(4, dtype=torch.bool),
            num_nodes=4,
            node_ids=torch.arange(4),
            num_edge_types=K,
        )
        collator = RiemannFMGraphCollator(
            max_nodes=4, num_edge_types=K,
            mask_ratio_c=0.0, mask_ratio_x=0.0,
        )
        batch = collator([g])
        assert (batch["mask_type"][0] == MASK_REAL).all()
        assert torch.isnan(batch["t_node"][0]).all()


class TestMaskedNodeLoss:
    """Tests for the masked_node_loss utility (used inside L_mask_c)."""

    def test_finite_and_positive(self) -> None:
        node_dim = 32
        D_emb = 16
        M = 8
        proj = torch.nn.Sequential(
            torch.nn.Linear(node_dim, D_emb, bias=False),
        )
        h_masked = torch.randn(M, node_dim)
        true_emb = torch.randn(M, D_emb)

        loss = masked_node_loss(h_masked, proj, true_emb)
        assert torch.isfinite(loss)
        assert loss > 0

    def test_zero_when_fewer_than_two(self) -> None:
        node_dim = 32
        D_emb = 16
        proj = torch.nn.Sequential(
            torch.nn.Linear(node_dim, D_emb, bias=False),
        )
        loss0 = masked_node_loss(torch.randn(0, node_dim), proj, torch.randn(0, D_emb))
        assert loss0.item() == 0.0
        loss1 = masked_node_loss(torch.randn(1, node_dim), proj, torch.randn(1, D_emb))
        assert loss1.item() == 0.0

    def test_gradient_flows_to_proj(self) -> None:
        node_dim = 32
        D_emb = 16
        M = 4
        proj = torch.nn.Sequential(
            torch.nn.Linear(node_dim, D_emb, bias=False),
        )
        h_masked = torch.randn(M, node_dim, requires_grad=True)
        true_emb = torch.randn(M, D_emb)

        loss = masked_node_loss(h_masked, proj, true_emb)
        loss.backward()
        assert proj[0].weight.grad is not None
        assert h_masked.grad is not None


class TestCombinedLossWithMask:
    """Tests for L_mask_c / L_mask_x integration in RiemannFMCombinedLoss."""

    def test_l_mask_disabled_by_default(self) -> None:
        manifold = _make_manifold()
        D = manifold.ambient_dim
        loss_fn = RiemannFMCombinedLoss(manifold)
        x_t = manifold.sample_noise(B, N, radius_h=1.0)
        V_hat = torch.randn(B, N, D)
        u_t = torch.randn(B, N, D)
        P_hat = torch.randn(B, N, N, K)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)

        _, metrics = loss_fn(V_hat, u_t, x_t, P_hat, E_1, mask)
        assert metrics["loss/mask_c"].item() == 0.0
        assert metrics["loss/mask_x"].item() == 0.0

    def test_l_mask_c_active(self) -> None:
        manifold = _make_manifold()
        D = manifold.ambient_dim
        d_c = 16
        node_dim = 32
        loss_fn = RiemannFMCombinedLoss(
            manifold, nu_mask_c=1.0, input_text_dim=d_c,
            node_dim=node_dim,
        )
        x_t = manifold.sample_noise(B, N, radius_h=1.0)
        h = torch.randn(B, N, node_dim)
        V_hat = torch.randn(B, N, D)
        u_t = torch.randn(B, N, D)
        P_hat = torch.randn(B, N, N, K)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)

        # Tag two nodes per graph as M_c.
        mask_type = torch.zeros(B, N, dtype=torch.long)
        mask_type[:, 0] = MASK_C
        mask_type[:, 1] = MASK_C

        true_text_emb = torch.randn(B, N, d_c)

        total, metrics = loss_fn(
            V_hat, u_t, x_t, P_hat, E_1, mask,
            h=h, mask_type=mask_type,
            true_text_emb=true_text_emb,
        )
        assert torch.isfinite(total)
        assert metrics["loss/mask_c"] > 0.0
        # L_mask_x still zero because no MASK_X nodes.
        assert metrics["loss/mask_x"].item() == 0.0

    def test_l_mask_x_active(self) -> None:
        """MASK_X subset produces a non-zero L_mask_x via continuous_flow_loss."""
        manifold = _make_manifold()
        D = manifold.ambient_dim
        node_dim = 32
        loss_fn = RiemannFMCombinedLoss(
            manifold, nu_mask_x=1.0, node_dim=node_dim,
        )
        x_t = manifold.sample_noise(B, N, radius_h=1.0)
        V_hat = torch.randn(B, N, D)
        u_t = torch.randn(B, N, D)
        P_hat = torch.randn(B, N, N, K)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)

        mask_type = torch.zeros(B, N, dtype=torch.long)
        mask_type[:, 2] = MASK_X
        mask_type[:, 3] = MASK_X

        _, metrics = loss_fn(
            V_hat, u_t, x_t, P_hat, E_1, mask,
            mask_type=mask_type,
        )
        assert metrics["loss/mask_x"] > 0.0

    def test_masked_nodes_excluded_from_l_cont(self) -> None:
        """MASK_C and MASK_X nodes must not contribute to L_cont."""
        manifold = _make_manifold()
        D = manifold.ambient_dim
        node_dim = 32
        loss_fn = RiemannFMCombinedLoss(
            manifold, nu_mask_c=1.0, input_text_dim=D,
            node_dim=node_dim, lambda_disc=0.0, mu_align=0.0,
        )

        x_t = manifold.sample_noise(B, N, radius_h=1.0)
        V_hat = torch.randn(B, N, D)
        u_t = torch.randn(B, N, D)
        P_hat = torch.randn(B, N, N, K)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)
        h = torch.randn(B, N, node_dim)

        mt_none = torch.zeros(B, N, dtype=torch.long)
        _, m1 = loss_fn(
            V_hat, u_t, x_t, P_hat, E_1, mask, h=h,
            mask_type=mt_none,
            true_text_emb=torch.randn(B, N, D),
        )

        mt_half = torch.zeros(B, N, dtype=torch.long)
        mt_half[:, :N // 2] = MASK_C
        _, m2 = loss_fn(
            V_hat, u_t, x_t, P_hat, E_1, mask, h=h,
            mask_type=mt_half,
            true_text_emb=torch.randn(B, N, D),
        )

        assert m1["loss/cont"] != m2["loss/cont"]

    def test_l_mask_c_gradient_backward(self) -> None:
        """Gradient flows from L_mask_c through proj_mask_c to h."""
        manifold = _make_manifold()
        D = manifold.ambient_dim
        node_dim = 32
        loss_fn = RiemannFMCombinedLoss(
            manifold, nu_mask_c=1.0, input_text_dim=D,
            node_dim=node_dim, mu_align=0.0, lambda_disc=0.0,
        )
        x_t = manifold.sample_noise(B, N, radius_h=1.0)
        h = torch.randn(B, N, node_dim, requires_grad=True)
        V_hat = torch.randn(B, N, D, requires_grad=True)
        u_t = torch.randn(B, N, D)
        P_hat = torch.randn(B, N, N, K)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)

        mask_type = torch.zeros(B, N, dtype=torch.long)
        mask_type[:, 0] = MASK_C
        mask_type[:, 1] = MASK_C
        true_text_emb = torch.randn(B, N, D)

        total, _ = loss_fn(
            V_hat, u_t, x_t, P_hat, E_1, mask, h=h,
            mask_type=mask_type,
            true_text_emb=true_text_emb,
        )
        total.backward()
        assert loss_fn.proj_mask_c[1].weight.grad is not None
        assert h.grad is not None
