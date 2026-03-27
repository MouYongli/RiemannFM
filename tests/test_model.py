"""Tests for full model forward pass and gradient flow."""

import pytest
import torch

from riedfm.manifolds.product import RieDFMProductManifold

DEVICE = torch.device("cpu")


@pytest.fixture
def small_manifold():
    """Small manifold for fast testing."""
    return RieDFMProductManifold(dim_hyperbolic=4, dim_spherical=4, dim_euclidean=4)


class TestREDFormerBlock:
    def test_forward_shape(self, small_manifold):
        from riedfm.layers.ath_norm import RieDFMTimestepEmbedding
        from riedfm.models.red_former_block import RieDFMREDFormerBlock

        block = RieDFMREDFormerBlock(
            node_dim=64,
            edge_dim=32,
            num_heads=4,
            edge_heads=2,
            time_embed_dim=32,
            text_dim=0,
            dropout=0.0,
            use_mrope=False,
        )
        t_embed_fn = RieDFMTimestepEmbedding(32)

        N = 6
        h_v = torch.randn(N, 64)
        h_e = torch.randn(N, N, 32)
        positions = small_manifold.sample_uniform((N,), DEVICE)
        t_embed = t_embed_fn(torch.tensor(0.5)).unsqueeze(0)

        h_v_out, h_e_out = block(h_v, h_e, small_manifold, positions, t_embed)
        assert h_v_out.shape == (N, 64)
        assert h_e_out.shape == (N, N, 32)


class TestREDFormer:
    def test_forward_shape(self, small_manifold):
        from riedfm.models.red_former import RieDFMREDFormer

        model = RieDFMREDFormer(
            manifold=small_manifold,
            num_layers=2,
            node_dim=64,
            edge_dim=32,
            num_heads=4,
            num_edge_types=11,
            text_dim=0,
            dropout=0.0,
            use_mrope=False,
        )

        N = 6
        x_t = small_manifold.sample_uniform((N,), DEVICE)
        e_t = torch.randint(0, 11, (N, N))
        t = torch.tensor(0.5)

        v_pred, p_pred = model(x_t, e_t, t)
        assert v_pred.shape == (N, small_manifold.total_dim)
        assert p_pred.shape == (N, N, 11)

    def test_gradient_flow(self, small_manifold):
        from riedfm.models.red_former import RieDFMREDFormer

        model = RieDFMREDFormer(
            manifold=small_manifold,
            num_layers=2,
            node_dim=64,
            edge_dim=32,
            num_heads=4,
            num_edge_types=11,
            text_dim=0,
            dropout=0.0,
            use_mrope=False,
        )

        N = 4
        x_t = small_manifold.sample_uniform((N,), DEVICE)
        e_t = torch.randint(0, 11, (N, N))
        t = torch.tensor(0.5)

        v_pred, p_pred = model(x_t, e_t, t)
        loss = v_pred.sum() + p_pred.sum()
        loss.backward()

        # Check gradients exist for all parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestRieDFMG:
    def test_forward_shape(self, small_manifold):
        from riedfm.models.riedfm_g import RieDFMG

        model = RieDFMG(
            manifold=small_manifold,
            num_layers=2,
            node_dim=64,
            edge_dim=32,
            num_heads=4,
            num_edge_types=11,
            text_dim=0,
            dropout=0.0,
        )

        N = 6
        x_1 = small_manifold.sample_uniform((N,), DEVICE)
        e_1 = torch.randint(0, 11, (N, N))

        outputs = model(x_1, e_1)
        assert "v_pred" in outputs
        assert "p_pred" in outputs
        assert "u_target" in outputs
        assert outputs["v_pred"].shape == (N, small_manifold.total_dim)
        assert outputs["p_pred"].shape == (N, N, 11)

    def test_generate_shape(self, small_manifold):
        from riedfm.models.riedfm_g import RieDFMG

        model = RieDFMG(
            manifold=small_manifold,
            num_layers=2,
            node_dim=64,
            edge_dim=32,
            num_heads=4,
            num_edge_types=11,
            text_dim=0,
            dropout=0.0,
        )

        x, e = model.generate(num_nodes=4, num_steps=3, device=DEVICE)
        assert x.shape == (4, small_manifold.total_dim)
        assert e.shape == (4, 4)
