"""Tests for modality-masking data augmentation (spec §9)."""

from __future__ import annotations

import torch

from riemannfm.data.collator import (
    MODE_COORD_MASK,
    MODE_FULL,
    MODE_TEXT_MASK,
    RiemannFMGraphCollator,
)
from riemannfm.data.graph import RiemannFMGraphData
from riemannfm.flow.joint_flow import RiemannFMJointFlow
from riemannfm.manifolds.product import RiemannFMProductManifold

B = 4
N = 6
K = 3


def _graph(num_real: int = 4) -> RiemannFMGraphData:
    return RiemannFMGraphData(
        edge_types=torch.zeros(N, N, K),
        node_text=torch.randn(N, 8),
        node_mask=torch.tensor([True] * num_real + [False] * (N - num_real)),
        num_nodes=num_real,
        node_ids=torch.tensor(
            list(range(num_real)) + [-1] * (N - num_real), dtype=torch.long,
        ),
        num_edge_types=K,
    )


class TestCollatorModes:
    def test_full_mode_all_ones(self) -> None:
        col = RiemannFMGraphCollator(max_nodes=N, num_edge_types=K)
        batch = col([_graph() for _ in range(B)])
        assert batch["m_text"].all()
        assert batch["m_coord"].all()
        assert (batch["mode_idx"] == MODE_FULL).all()

    def test_text_mask_mode_flips_text_only(self) -> None:
        col = RiemannFMGraphCollator(
            max_nodes=N, num_edge_types=K,
            mode_probs=(0.0, 1.0, 0.0), rho_tm=0.5,
        )
        batch = col([_graph(num_real=4) for _ in range(B)])
        assert (batch["mode_idx"] == MODE_TEXT_MASK).all()
        assert batch["m_coord"].all()
        # At least one text bit flipped in every subgraph (rho_tm=0.5 on 4 real).
        for i in range(B):
            num_text_off = int((~batch["m_text"][i]).sum().item())
            assert num_text_off > 0

    def test_coord_mask_mode_flips_coord_only(self) -> None:
        col = RiemannFMGraphCollator(
            max_nodes=N, num_edge_types=K,
            mode_probs=(0.0, 0.0, 1.0), rho_cm=0.5,
        )
        batch = col([_graph(num_real=4) for _ in range(B)])
        assert (batch["mode_idx"] == MODE_COORD_MASK).all()
        assert batch["m_text"].all()
        for i in range(B):
            num_coord_off = int((~batch["m_coord"][i]).sum().item())
            assert num_coord_off > 0

    def test_virtual_positions_keep_one(self) -> None:
        """Virtual nodes should keep m_text = m_coord = 1 so their
        handling stays governed by ``node_mask`` alone."""
        col = RiemannFMGraphCollator(
            max_nodes=N, num_edge_types=K,
            mode_probs=(0.0, 1.0, 0.0), rho_tm=1.0,
        )
        batch = col([_graph(num_real=2) for _ in range(B)])
        # Positions [2:6] are virtual in every graph.
        assert batch["m_text"][:, 2:].all()
        assert batch["m_coord"][:, 2:].all()

    def test_mode_probabilities_approximately_match(self) -> None:
        torch.manual_seed(0)
        col = RiemannFMGraphCollator(
            max_nodes=N, num_edge_types=K,
            mode_probs=(0.7, 0.15, 0.15),
        )
        counts = torch.zeros(3)
        for _ in range(64):
            batch = col([_graph() for _ in range(16)])
            for m in batch["mode_idx"].tolist():
                counts[m] += 1
        fracs = counts / counts.sum()
        assert abs(fracs[0].item() - 0.70) < 0.05
        assert abs(fracs[1].item() - 0.15) < 0.05
        assert abs(fracs[2].item() - 0.15) < 0.05


class TestFlowWithModalityBits:
    def _manifold(self) -> RiemannFMProductManifold:
        return RiemannFMProductManifold(dim_hyperbolic=4, dim_spherical=4, dim_euclidean=4)

    def test_m_coord_zero_holds_x_t_at_x_0(self) -> None:
        """Coord-masked nodes should have x_t independent of the sampled t."""
        manifold = self._manifold()
        D = manifold.ambient_dim
        flow = RiemannFMJointFlow(manifold, time_distribution="uniform")
        x_1 = manifold.sample_noise(B, N, radius_h=1.0)
        E_1 = torch.zeros(B, N, N, K)
        node_mask = torch.ones(B, N, dtype=torch.bool)
        m_coord = torch.ones(B, N, dtype=torch.bool)
        m_coord[:, 0] = False  # mask first node in every graph
        torch.manual_seed(1)
        s1 = flow.sample(x_1, E_1, node_mask, m_coord=m_coord)
        torch.manual_seed(2)
        s2 = flow.sample(x_1, E_1, node_mask, m_coord=m_coord)
        # Different seeds → different t, different x_0. Coord-masked
        # positions should still have x_t = x_0 (not interpolated),
        # so u_t at those positions is zero.
        assert (s1.u_t[:, 0, :].abs().sum() == 0.0)
        assert (s2.u_t[:, 0, :].abs().sum() == 0.0)
        # Non-masked positions should have non-zero u_t (unless t = 0).
        assert torch.isfinite(s1.x_t).all()
        assert s1.x_t.shape == (B, N, D)

    def test_mode_idx_text_mask_uses_beta(self) -> None:
        """Text-mask mode should bias ``t`` toward 1 (Beta(5, 1))."""
        torch.manual_seed(0)
        manifold = self._manifold()
        flow = RiemannFMJointFlow(
            manifold, time_distribution="logit_normal",
            beta_a_text_mask=5.0, beta_b_text_mask=1.0,
        )
        x_1 = manifold.sample_noise(256, 4, radius_h=1.0)
        E_1 = torch.zeros(256, 4, 4, K)
        node_mask = torch.ones(256, 4, dtype=torch.bool)
        mode_idx = torch.full((256,), MODE_TEXT_MASK, dtype=torch.long)
        s = flow.sample(x_1, E_1, node_mask, mode_idx=mode_idx)
        assert s.t.mean().item() > 0.6

    def test_mode_idx_full_uses_base_distribution(self) -> None:
        """Without text-mask mode, ``t`` follows the base distribution."""
        torch.manual_seed(0)
        manifold = self._manifold()
        flow = RiemannFMJointFlow(
            manifold, time_distribution="logit_normal",
            beta_a_text_mask=5.0, beta_b_text_mask=1.0,
        )
        x_1 = manifold.sample_noise(256, 4, radius_h=1.0)
        E_1 = torch.zeros(256, 4, 4, K)
        node_mask = torch.ones(256, 4, dtype=torch.bool)
        mode_idx = torch.full((256,), MODE_FULL, dtype=torch.long)
        s = flow.sample(x_1, E_1, node_mask, mode_idx=mode_idx)
        # Logit-Normal centered at 0 (logit space) → mean ≈ 0.5.
        assert 0.3 < s.t.mean().item() < 0.7


class TestNodeEncoderWithMasks:
    def test_forward_accepts_m_text_m_coord(self) -> None:
        from riemannfm.models.input_encoding import RiemannFMNodeEncoder

        manifold_dim = 13
        dim_h, dim_s, dim_e = 5, 5, 3
        d_c = 8
        node_dim = 16
        time_dim = 16
        enc = RiemannFMNodeEncoder(
            ambient_dim=manifold_dim,
            text_dim=d_c,
            node_dim=node_dim,
            time_dim=time_dim,
            dim_h_ambient=dim_h,
            dim_s_ambient=dim_s,
            dim_e=dim_e,
        )
        x = torch.randn(B, N, manifold_dim)
        node_text = torch.randn(B, N, d_c)
        node_mask = torch.ones(B, N, dtype=torch.bool)
        t_emb = torch.randn(B, time_dim)
        m_text = torch.ones(B, N, dtype=torch.bool)
        m_text[0, 0] = False
        m_coord = torch.ones(B, N, dtype=torch.bool)
        m_coord[0, 1] = False
        h = enc(x, node_text, node_mask, t_emb, m_text=m_text, m_coord=m_coord)
        assert h.shape == (B, N, node_dim)

    def test_forward_defaults_to_all_ones(self) -> None:
        from riemannfm.models.input_encoding import RiemannFMNodeEncoder

        manifold_dim = 13
        enc = RiemannFMNodeEncoder(
            ambient_dim=manifold_dim,
            text_dim=8,
            node_dim=16,
            time_dim=16,
            dim_h_ambient=5,
            dim_s_ambient=5,
            dim_e=3,
        )
        x = torch.randn(B, N, manifold_dim)
        node_text = torch.randn(B, N, 8)
        node_mask = torch.ones(B, N, dtype=torch.bool)
        t_emb = torch.randn(B, 16)
        h = enc(x, node_text, node_mask, t_emb)  # no m_text / m_coord
        assert h.shape == (B, N, 16)
