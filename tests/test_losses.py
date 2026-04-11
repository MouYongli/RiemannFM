"""Tests for RiemannFM loss functions (Phase 3: Stream B).

Covers loss finiteness, positivity, masking behavior, gradient flow
to curvature parameters, and the combined loss orchestrator.
"""

from __future__ import annotations

import torch

from riemannfm.data.collator import (
    MASK_MASKED,
    MASK_REAL,
    MASK_VIRTUAL,
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

    def test_gradient_finite_when_residual_zero_at_masked_token(self) -> None:
        """Regression: zero residual at any token must not produce NaN grads.

        Going through ``tangent_norm`` (sqrt) then ``pow(2)`` is forward-equal
        to a direct squared norm, but the backward of ``sqrt`` at zero is
        ``inf``; multiplied by the mask zero this gave ``0 * inf = NaN`` and
        poisoned the curvature parameter via global gradient clipping.
        """
        manifold = _make_manifold()
        D = manifold.ambient_dim
        x_t = manifold.sample_noise(B, N, radius_h=1.0)
        u_t = torch.randn(B, N, D)
        # Make V_hat exactly equal to u_t at one token so residual = 0 there.
        V_hat = u_t.clone().detach().requires_grad_(True)
        # Mask out that token to mimic a virtual / padding node.
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[0, 0] = False
        loss = continuous_flow_loss(manifold, V_hat, u_t, x_t, mask)
        loss.backward()
        assert V_hat.grad is not None
        assert torch.isfinite(V_hat.grad).all()
        # Curvatures must also receive finite gradients.
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
        d_a = 32
        g = torch.randn(B, N, d_a)
        c = torch.randn(B, N, d_a)
        mask = torch.ones(B, N, dtype=torch.bool)
        loss = contrastive_alignment_loss(g, c, mask)
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_zero_when_no_text(self) -> None:
        g = torch.randn(B, N, 0)
        c = torch.zeros(B, N, 0)  # d_a = 0
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
        # Only 1 valid node across all batches.
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

        # No x_1 or text → L_align = 0.
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
        # Projection layers should also have gradients.
        assert loss_fn.proj_g[0].weight.grad is not None
        assert loss_fn.proj_c[0].weight.grad is not None
        # Gradient flows back to backbone hidden states.
        assert h.grad is not None

    def test_per_group_gradient_clipping(self) -> None:
        """Alignment proj grads survive clipping independently of backbone."""
        manifold = _make_manifold()
        D = manifold.ambient_dim
        d_c = 16
        node_dim = 32
        loss_fn = RiemannFMCombinedLoss(
            manifold, mu_align=1.0, input_text_dim=d_c,
            node_dim=node_dim, d_a=8,
        )

        # Create a mock backbone param with large gradients.
        backbone_param = torch.nn.Parameter(torch.randn(512, 512))

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
        # Add a large fake backbone loss to simulate L_cont dominance.
        fake_backbone_loss = (backbone_param ** 2).sum() * 1000
        (total + fake_backbone_loss).backward()

        # Before clipping: backbone gradient norm dwarfs proj gradient norm.
        align_params = list(loss_fn.parameters())
        align_norm_before = torch.nn.utils.clip_grad_norm_(
            align_params, float("inf"),
        )

        backbone_norm = backbone_param.grad.norm()
        assert backbone_norm > 100 * align_norm_before, (
            "Test setup: backbone grads should dominate"
        )

        # Re-run backward for a fresh set of gradients.
        loss_fn.zero_grad()
        backbone_param.grad = None
        total2, _ = loss_fn(
            V_hat.detach().requires_grad_(), u_t, x_t,
            P_hat.detach().requires_grad_(), E_1, mask,
            h=h.detach().requires_grad_(), node_text=node_text,
        )
        fake2 = (backbone_param ** 2).sum() * 1000
        (total2 + fake2).backward()

        # Per-group clipping: clip alignment and backbone independently.
        clip_val = 1.0
        torch.nn.utils.clip_grad_norm_(align_params, clip_val)
        torch.nn.utils.clip_grad_norm_([backbone_param], clip_val)

        # After per-group clipping, alignment grads should be at clip_val.
        align_norm_after = torch.cat(
            [p.grad.flatten() for p in align_params if p.grad is not None],
        ).norm()
        # The alignment norm should be close to clip_val (not near-zero).
        assert align_norm_after > 0.5 * clip_val, (
            f"Per-group clipping should preserve alignment grad norm, "
            f"got {align_norm_after:.4f}"
        )


class TestNodeMasking:
    """Tests for collator-level node masking."""

    def test_apply_node_masking_counts(self) -> None:
        """Correct number of nodes are masked per graph."""
        mask_type = torch.zeros(4, 10, dtype=torch.long)
        num_real = torch.tensor([8, 6, 10, 3])
        _apply_node_masking(mask_type, num_real, mask_ratio=0.15)

        for i in range(4):
            nr = int(num_real[i].item())
            n_masked = (mask_type[i] == MASK_MASKED).sum().item()
            # At least 1 masked, at most nr-1 (keep 1 unmasked).
            assert n_masked >= 1, f"Graph {i}: expected >=1 masked"
            assert n_masked < nr, f"Graph {i}: all nodes masked"
            # Masked nodes only within real range.
            assert (mask_type[i, nr:] != MASK_MASKED).all()

    def test_masking_does_not_touch_virtual(self) -> None:
        mask_type = torch.full((2, 8), MASK_VIRTUAL, dtype=torch.long)
        mask_type[:, :4] = MASK_REAL
        num_real = torch.tensor([4, 4])
        _apply_node_masking(mask_type, num_real, mask_ratio=0.3)
        # Virtual nodes remain unchanged.
        assert (mask_type[:, 4:] == MASK_VIRTUAL).all()

    def test_collator_mask_type_shape(self) -> None:
        """Collator produces mask_type with correct shape and values."""
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
            max_nodes=6, num_edge_types=K, mask_ratio=0.25,
        )
        batch = collator([g, g])
        mt = batch["mask_type"]
        assert mt.shape == (2, 6)
        # Virtual nodes are MASK_VIRTUAL.
        assert (mt[:, 4:] == MASK_VIRTUAL).all()
        # At least 1 masked per graph.
        for i in range(2):
            assert (mt[i] == MASK_MASKED).any()

    def test_collator_no_masking_when_zero(self) -> None:
        """mask_ratio=0 produces no masked nodes."""
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
            max_nodes=4, num_edge_types=K, mask_ratio=0.0,
        )
        batch = collator([g])
        assert (batch["mask_type"][0] == MASK_REAL).all()


class TestMaskedNodeLoss:
    """Tests for the masked node prediction loss function."""

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
        # 0 masked nodes.
        loss0 = masked_node_loss(torch.randn(0, node_dim), proj, torch.randn(0, D_emb))
        assert loss0.item() == 0.0
        # 1 masked node — need >=2 for in-batch contrastive.
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
    """Tests for L_mask integration in RiemannFMCombinedLoss."""

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
        assert metrics["loss/mask"].item() == 0.0

    def test_l_mask_active(self) -> None:
        manifold = _make_manifold()
        D = manifold.ambient_dim
        node_dim = 32
        loss_fn = RiemannFMCombinedLoss(
            manifold, nu_mask=1.0, entity_emb_dim=D,
            node_dim=node_dim,
        )
        x_t = manifold.sample_noise(B, N, radius_h=1.0)
        h = torch.randn(B, N, node_dim)
        V_hat = torch.randn(B, N, D)
        u_t = torch.randn(B, N, D)
        P_hat = torch.randn(B, N, N, K)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)

        # Create mask_type: mask 2 nodes per graph.
        mask_type = torch.zeros(B, N, dtype=torch.long)
        mask_type[:, 0] = MASK_MASKED
        mask_type[:, 1] = MASK_MASKED

        true_entity_emb = torch.randn(B, N, D)

        total, metrics = loss_fn(
            V_hat, u_t, x_t, P_hat, E_1, mask,
            h=h, mask_type=mask_type,
            true_entity_emb=true_entity_emb,
        )
        assert torch.isfinite(total)
        assert metrics["loss/mask"] > 0.0

    def test_masked_nodes_excluded_from_l_cont(self) -> None:
        """Masked nodes should not contribute to L_cont."""
        manifold = _make_manifold()
        D = manifold.ambient_dim
        node_dim = 32
        loss_fn = RiemannFMCombinedLoss(
            manifold, nu_mask=1.0, entity_emb_dim=D,
            node_dim=node_dim, lambda_disc=0.0, mu_align=0.0,
        )

        x_t = manifold.sample_noise(B, N, radius_h=1.0)
        V_hat = torch.randn(B, N, D)
        u_t = torch.randn(B, N, D)
        P_hat = torch.randn(B, N, N, K)
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        mask = torch.ones(B, N, dtype=torch.bool)
        h = torch.randn(B, N, node_dim)

        # All real, no masking.
        mt_none = torch.zeros(B, N, dtype=torch.long)
        _, m1 = loss_fn(
            V_hat, u_t, x_t, P_hat, E_1, mask, h=h,
            mask_type=mt_none,
            true_entity_emb=torch.randn(B, N, D),
        )

        # Mask half the nodes — L_cont should change (fewer contributing nodes).
        mt_half = torch.zeros(B, N, dtype=torch.long)
        mt_half[:, :N // 2] = MASK_MASKED
        _, m2 = loss_fn(
            V_hat, u_t, x_t, P_hat, E_1, mask, h=h,
            mask_type=mt_half,
            true_entity_emb=torch.randn(B, N, D),
        )

        # L_cont values should differ since masked nodes are excluded.
        assert m1["loss/cont"] != m2["loss/cont"]

    def test_l_mask_gradient_backward(self) -> None:
        """Gradient flows from L_mask through proj_mask to h."""
        manifold = _make_manifold()
        D = manifold.ambient_dim
        node_dim = 32
        loss_fn = RiemannFMCombinedLoss(
            manifold, nu_mask=1.0, entity_emb_dim=D,
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
        mask_type[:, 0] = MASK_MASKED
        true_entity_emb = torch.randn(B, N, D)

        total, _ = loss_fn(
            V_hat, u_t, x_t, P_hat, E_1, mask, h=h,
            mask_type=mask_type,
            true_entity_emb=true_entity_emb,
        )
        total.backward()
        # proj_mask should have gradients.
        assert loss_fn.proj_mask[0].weight.grad is not None
        # Gradient flows back to backbone hidden states.
        assert h.grad is not None
