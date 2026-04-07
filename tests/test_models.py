"""Tests for RiemannFM model sub-modules (Phase 1: Stream C).

Covers forward pass shapes, gradient flow, and edge cases for all
independent model components before they are assembled into RieFormer.
"""

from __future__ import annotations

import pytest
import torch

from riemannfm.manifolds.product import RiemannFMProductManifold
from riemannfm.models.attention.edge import (
    RiemannFMEdgeBias,
    RiemannFMEdgeSelfUpdate,
)
from riemannfm.models.attention.geodesic import RiemannFMGeodesicAttention
from riemannfm.models.heads import (
    RiemannFMDualStreamCross,
    RiemannFMEdgeHead,
    RiemannFMVFHead,
)
from riemannfm.models.input_encoding import (
    RiemannFMEdgeEncoder,
    RiemannFMNodeEncoder,
)
from riemannfm.models.normalization import RiemannFMATHNorm, RiemannFMPreNorm
from riemannfm.models.positional import RiemannFMTimeEmbedding

# Test constants matching small model config.
B = 4
N = 8
K = 10
NODE_DIM = 64  # smaller than 384 for fast tests
EDGE_DIM = 32
NUM_HEADS = 4
EDGE_HEADS = 2
DIM_H = 8
DIM_S = 8
DIM_E = 8


@pytest.fixture()
def manifold() -> RiemannFMProductManifold:
    return RiemannFMProductManifold(DIM_H, DIM_S, DIM_E)


@pytest.fixture()
def ambient_dim(manifold: RiemannFMProductManifold) -> int:
    return manifold.ambient_dim  # (8+1)+(8+1)+8 = 26


class TestTimeEmbedding:
    def test_output_shape(self) -> None:
        emb = RiemannFMTimeEmbedding(NODE_DIM)
        t = torch.rand(B)
        out = emb(t)
        assert out.shape == (B, NODE_DIM)

    def test_output_shape_2d(self) -> None:
        emb = RiemannFMTimeEmbedding(NODE_DIM)
        t = torch.rand(B, 1)
        out = emb(t)
        assert out.shape == (B, NODE_DIM)

    def test_different_times_different_embeddings(self) -> None:
        emb = RiemannFMTimeEmbedding(NODE_DIM)
        t1 = torch.tensor([0.0])
        t2 = torch.tensor([1.0])
        out1 = emb(t1)
        out2 = emb(t2)
        assert not torch.allclose(out1, out2)

    def test_odd_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="even"):
            RiemannFMTimeEmbedding(63)

    def test_gradient_flow(self) -> None:
        emb = RiemannFMTimeEmbedding(NODE_DIM)
        t = torch.rand(B)
        out = emb(t)
        out.sum().backward()
        for p in emb.parameters():
            assert p.grad is not None


class TestATHNorm:
    def test_output_shape(self) -> None:
        norm = RiemannFMATHNorm(NODE_DIM, NODE_DIM)
        x = torch.randn(B, N, NODE_DIM)
        t_emb = torch.randn(B, NODE_DIM)
        out = norm(x, t_emb)
        assert out.shape == (B, N, NODE_DIM)

    def test_identity_init(self) -> None:
        norm = RiemannFMATHNorm(NODE_DIM, NODE_DIM)
        x = torch.randn(B, N, NODE_DIM)
        t_emb = torch.zeros(B, NODE_DIM)
        out = norm(x, t_emb)
        # With zero t_emb, gamma=1 and beta=0, so output = LayerNorm(x).
        expected = torch.nn.functional.layer_norm(x, [NODE_DIM])
        assert torch.allclose(out, expected, atol=1e-5)

    def test_gradient_flow(self) -> None:
        norm = RiemannFMATHNorm(NODE_DIM, NODE_DIM)
        x = torch.randn(B, N, NODE_DIM, requires_grad=True)
        t_emb = torch.randn(B, NODE_DIM)
        out = norm(x, t_emb)
        out.sum().backward()
        assert x.grad is not None


class TestPreNorm:
    def test_output_shape(self) -> None:
        norm = RiemannFMPreNorm(NODE_DIM)
        x = torch.randn(B, N, NODE_DIM)
        out = norm(x)
        assert out.shape == (B, N, NODE_DIM)


class TestNodeEncoder:
    def test_output_shape_no_text(
        self, ambient_dim: int,
    ) -> None:
        enc = RiemannFMNodeEncoder(ambient_dim, 0, NODE_DIM, NODE_DIM)
        x = torch.randn(B, N, ambient_dim)
        node_text = torch.zeros(B, N, 0)
        mask = torch.ones(B, N, dtype=torch.bool)
        t_emb = torch.randn(B, NODE_DIM)
        out = enc(x, node_text, mask, t_emb)
        assert out.shape == (B, N, NODE_DIM)

    def test_output_shape_with_text(
        self, ambient_dim: int,
    ) -> None:
        d_c = 16
        enc = RiemannFMNodeEncoder(ambient_dim, d_c, NODE_DIM, NODE_DIM)
        x = torch.randn(B, N, ambient_dim)
        node_text = torch.randn(B, N, d_c)
        mask = torch.ones(B, N, dtype=torch.bool)
        t_emb = torch.randn(B, NODE_DIM)
        out = enc(x, node_text, mask, t_emb)
        assert out.shape == (B, N, NODE_DIM)


class TestEdgeEncoder:
    def test_output_shape_no_text(self) -> None:
        enc = RiemannFMEdgeEncoder(K, 16, 0, EDGE_DIM)
        E_t = torch.rand(B, N, N, K)
        out = enc(E_t)
        assert out.shape == (B, N, N, EDGE_DIM)

    def test_output_shape_with_text(self) -> None:
        d_c = 16
        enc = RiemannFMEdgeEncoder(K, 16, d_c, EDGE_DIM)
        E_t = torch.rand(B, N, N, K)
        C_R = torch.randn(K, d_c)
        out = enc(E_t, C_R)
        assert out.shape == (B, N, N, EDGE_DIM)


class TestEdgeBias:
    def test_output_shape(self) -> None:
        mod = RiemannFMEdgeBias(EDGE_DIM, NUM_HEADS)
        g = torch.randn(B, N, N, EDGE_DIM)
        out = mod(g)
        assert out.shape == (B, NUM_HEADS, N, N)


class TestGeodesicAttention:
    def test_output_shape(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        attn = RiemannFMGeodesicAttention(
            NODE_DIM, NUM_HEADS, manifold, use_geodesic_kernel=True,
        )
        h = torch.randn(B, N, NODE_DIM)
        x = manifold.sample_noise(B, N)
        mask = torch.ones(B, N, dtype=torch.bool)
        out = attn(h, x, node_mask=mask)
        assert out.shape == (B, N, NODE_DIM)

    def test_with_edge_bias(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        attn = RiemannFMGeodesicAttention(
            NODE_DIM, NUM_HEADS, manifold,
        )
        h = torch.randn(B, N, NODE_DIM)
        x = manifold.sample_noise(B, N)
        bias = torch.randn(B, NUM_HEADS, N, N)
        mask = torch.ones(B, N, dtype=torch.bool)
        out = attn(h, x, edge_bias=bias, node_mask=mask)
        assert out.shape == (B, N, NODE_DIM)

    def test_without_geodesic_kernel(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        attn = RiemannFMGeodesicAttention(
            NODE_DIM, NUM_HEADS, manifold, use_geodesic_kernel=False,
        )
        h = torch.randn(B, N, NODE_DIM)
        x = manifold.sample_noise(B, N)
        out = attn(h, x)
        assert out.shape == (B, N, NODE_DIM)

    def test_masking(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        attn = RiemannFMGeodesicAttention(
            NODE_DIM, NUM_HEADS, manifold, use_geodesic_kernel=False,
        )
        h = torch.randn(B, N, NODE_DIM)
        x = manifold.sample_noise(B, N)
        # Mask out all but first 2 nodes.
        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[:, :2] = True
        out = attn(h, x, node_mask=mask)
        assert out.shape == (B, N, NODE_DIM)
        assert torch.isfinite(out[:, :2]).all()

    def test_gradient_flow(
        self, manifold: RiemannFMProductManifold,
    ) -> None:
        attn = RiemannFMGeodesicAttention(
            NODE_DIM, NUM_HEADS, manifold,
        )
        h = torch.randn(B, N, NODE_DIM, requires_grad=True)
        x = manifold.sample_noise(B, N)
        out = attn(h, x)
        out.sum().backward()
        assert h.grad is not None
        assert torch.isfinite(h.grad).all()


class TestEdgeSelfUpdate:
    def test_output_shape(self) -> None:
        mod = RiemannFMEdgeSelfUpdate(EDGE_DIM)
        g = torch.randn(B, N, N, EDGE_DIM)
        out = mod(g)
        assert out.shape == (B, N, N, EDGE_DIM)

    def test_residual(self) -> None:
        mod = RiemannFMEdgeSelfUpdate(EDGE_DIM)
        g = torch.randn(B, N, N, EDGE_DIM)
        out = mod(g)
        assert out.shape == g.shape

    def test_gradient_flow(self) -> None:
        mod = RiemannFMEdgeSelfUpdate(EDGE_DIM)
        g = torch.randn(B, N, N, EDGE_DIM, requires_grad=True)
        out = mod(g)
        out.sum().backward()
        assert g.grad is not None
        assert torch.isfinite(g.grad).all()


class TestDualStreamCross:
    def test_output_shapes(self) -> None:
        mod = RiemannFMDualStreamCross(NODE_DIM, EDGE_DIM)
        h = torch.randn(B, N, NODE_DIM)
        g = torch.randn(B, N, N, EDGE_DIM)
        mask = torch.ones(B, N, dtype=torch.bool)
        h_out, g_out = mod(h, g, mask)
        assert h_out.shape == (B, N, NODE_DIM)
        assert g_out.shape == (B, N, N, EDGE_DIM)

    def test_without_mask(self) -> None:
        mod = RiemannFMDualStreamCross(NODE_DIM, EDGE_DIM)
        h = torch.randn(B, N, NODE_DIM)
        g = torch.randn(B, N, N, EDGE_DIM)
        h_out, g_out = mod(h, g)
        assert h_out.shape == (B, N, NODE_DIM)
        assert g_out.shape == (B, N, N, EDGE_DIM)


class TestVFHead:
    def test_output_shape(
        self, manifold: RiemannFMProductManifold, ambient_dim: int,
    ) -> None:
        head = RiemannFMVFHead(NODE_DIM, ambient_dim, manifold)
        h = torch.randn(B, N, NODE_DIM)
        x = manifold.sample_noise(B, N)
        out = head(h, x)
        assert out.shape == (B, N, ambient_dim)

    def test_output_is_tangent(
        self, manifold: RiemannFMProductManifold, ambient_dim: int,
    ) -> None:
        head = RiemannFMVFHead(NODE_DIM, ambient_dim, manifold)
        h = torch.randn(B, N, NODE_DIM) * 0.1
        x = manifold.sample_noise(B, N)
        v = head(h, x)
        # Re-projecting should be near-identity (already tangent).
        v_proj = manifold.proj_tangent(x, v)
        # Relative error check: |v - v_proj| / (|v| + eps) should be small.
        rel_err = (v - v_proj).norm() / (v.norm() + 1e-8)
        assert rel_err < 1e-3, f"relative tangent error: {rel_err:.6f}"


class TestEdgeHead:
    def test_output_shape_no_text(self) -> None:
        head = RiemannFMEdgeHead(EDGE_DIM, K)
        g = torch.randn(B, N, N, EDGE_DIM)
        out = head(g)
        assert out.shape == (B, N, N, K)

    def test_output_shape_with_text(self) -> None:
        d_c = 16
        head = RiemannFMEdgeHead(EDGE_DIM, K, text_proj_dim=d_c)
        g = torch.randn(B, N, N, EDGE_DIM)
        C_R = torch.randn(K, d_c)
        out = head(g, C_R)
        assert out.shape == (B, N, N, K)

    def test_gradient_flow(self) -> None:
        d_c = 16
        head = RiemannFMEdgeHead(EDGE_DIM, K, text_proj_dim=d_c)
        g = torch.randn(B, N, N, EDGE_DIM, requires_grad=True)
        C_R = torch.randn(K, d_c)
        out = head(g, C_R)
        out.sum().backward()
        assert g.grad is not None


class TestEndToEndForwardPass:
    """Smoke test wiring all sub-modules together."""

    def test_full_pipeline(
        self, manifold: RiemannFMProductManifold, ambient_dim: int,
    ) -> None:
        d_c = 0
        # Build all modules.
        time_emb = RiemannFMTimeEmbedding(NODE_DIM)
        node_enc = RiemannFMNodeEncoder(ambient_dim, d_c, NODE_DIM, NODE_DIM)
        edge_enc = RiemannFMEdgeEncoder(K, 16, d_c, EDGE_DIM)
        edge_bias_mod = RiemannFMEdgeBias(EDGE_DIM, NUM_HEADS)
        geo_attn = RiemannFMGeodesicAttention(
            NODE_DIM, NUM_HEADS, manifold,
        )
        edge_update = RiemannFMEdgeSelfUpdate(EDGE_DIM)
        cross = RiemannFMDualStreamCross(NODE_DIM, EDGE_DIM)
        vf_head = RiemannFMVFHead(NODE_DIM, ambient_dim, manifold)
        edge_head = RiemannFMEdgeHead(EDGE_DIM, K)

        # Inputs.
        t = torch.rand(B)
        x = manifold.sample_noise(B, N)
        E_t = torch.rand(B, N, N, K)
        node_text = torch.zeros(B, N, d_c)
        mask = torch.ones(B, N, dtype=torch.bool)

        # Forward pass.
        t_emb = time_emb(t)
        h = node_enc(x, node_text, mask, t_emb)
        g = edge_enc(E_t)
        bias = edge_bias_mod(g)
        h = geo_attn(h, x, edge_bias=bias, node_mask=mask)
        g = edge_update(g)
        h, g = cross(h, g, mask)
        V_hat = vf_head(h, x)
        P_hat = edge_head(g)

        assert V_hat.shape == (B, N, ambient_dim)
        assert P_hat.shape == (B, N, N, K)

        # Backward.
        loss = V_hat.sum() + P_hat.sum()
        loss.backward()
        # Check gradient flows to manifold curvature.
        if manifold.hyperbolic is not None:
            assert manifold.hyperbolic._curvature.grad is not None
