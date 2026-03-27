"""Tests for RED-Former layer components.

Verifies output shapes and basic properties for each layer.
"""

import pytest
import torch

from riedfm.manifolds.product import RieDFMProductManifold

DEVICE = torch.device("cpu")


@pytest.fixture
def manifold():
    return RieDFMProductManifold(dim_hyperbolic=4, dim_spherical=4, dim_euclidean=4)


class TestATHNorm:
    def test_output_shape(self):
        from riedfm.layers.ath_norm import RieDFMATHNorm, RieDFMTimestepEmbedding

        hidden_dim = 64
        time_dim = 32
        norm = RieDFMATHNorm(hidden_dim, time_dim)
        t_embed_fn = RieDFMTimestepEmbedding(time_dim)

        h = torch.randn(10, hidden_dim)
        t_embed = t_embed_fn(torch.tensor(0.5)).unsqueeze(0)
        depth = torch.randint(0, 10, (10,))

        out = norm(h, t_embed, depth)
        assert out.shape == (10, hidden_dim)


class TestTimestepEmbedding:
    def test_output_shape(self):
        from riedfm.layers.ath_norm import RieDFMTimestepEmbedding

        embed = RieDFMTimestepEmbedding(64)
        t = torch.tensor(0.5)
        out = embed(t)
        assert out.shape == (1, 64)

    def test_batch_output(self):
        from riedfm.layers.ath_norm import RieDFMTimestepEmbedding

        embed = RieDFMTimestepEmbedding(64)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = embed(t)
        assert out.shape == (3, 64)


class TestGeodesicKernelAttention:
    def test_output_shape(self, manifold):
        from riedfm.layers.geodesic_attention import RieDFMGeodesicAttention

        attn = RieDFMGeodesicAttention(hidden_dim=64, num_heads=4, use_mrope=False)
        h = torch.randn(8, 64)
        positions = manifold.sample_uniform((8,), DEVICE)
        out = attn(h, manifold, positions)
        assert out.shape == (8, 64)


class TestEdgeStreamAttention:
    def test_output_shape(self):
        from riedfm.layers.edge_attention import RieDFMEdgeAttention

        attn = RieDFMEdgeAttention(edge_dim=64, num_heads=4)
        h_e = torch.randn(8, 8, 64)
        out = attn(h_e)
        assert out.shape == (8, 8, 64)


class TestDualStreamInteraction:
    def test_output_shapes(self):
        from riedfm.layers.dual_stream_interaction import RieDFMDualStreamInteraction

        cross = RieDFMDualStreamInteraction(node_dim=64, edge_dim=32)
        h_v = torch.randn(8, 64)
        h_e = torch.randn(8, 8, 32)
        h_v_out, h_e_out = cross(h_v, h_e)
        assert h_v_out.shape == (8, 64)
        assert h_e_out.shape == (8, 8, 32)


class TestTextCrossAttention:
    def test_output_shape(self):
        from riedfm.layers.text_cross_attention import RieDFMTextCrossAttention

        cross = RieDFMTextCrossAttention(node_dim=64, text_dim=128, num_heads=4)
        h_v = torch.randn(8, 64)
        text = torch.randn(20, 128)  # 20 text tokens
        out = cross(h_v, text)
        assert out.shape == (8, 64)


class TestVectorFieldHead:
    def test_continuous_head_shape(self, manifold):
        from riedfm.layers.vector_field_head import RieDFMContinuousVFHead

        head = RieDFMContinuousVFHead(hidden_dim=64, manifold=manifold)
        h = torch.randn(8, 64)
        x = manifold.sample_uniform((8,), DEVICE)
        v = head(h, x)
        assert v.shape == (8, manifold.total_dim)

    def test_discrete_head_shape(self):
        from riedfm.layers.vector_field_head import RieDFMDiscreteEdgeHead

        head = RieDFMDiscreteEdgeHead(edge_dim=32, num_edge_types=11)
        h_e = torch.randn(8, 8, 32)
        logits = head(h_e)
        assert logits.shape == (8, 8, 11)
