"""Tests for RiemannFM model sub-modules (spec §10-17)."""

from __future__ import annotations

import pytest
import torch

from riemannfm.manifolds.product import RiemannFMProductManifold
from riemannfm.models.attention.cross import (
    RiemannFMNodeEdgeCross,
    RiemannFMNodeRelationCross,
)
from riemannfm.models.attention.edge import (
    RiemannFMEdgeBias,
    RiemannFMEdgeSelfUpdate,
)
from riemannfm.models.attention.geodesic import RiemannFMGeodesicAttention
from riemannfm.models.attention.relation import RiemannFMRelationSelfAttention
from riemannfm.models.heads import (
    RiemannFMEdgeExHead,
    RiemannFMEdgeTypeHead,
    RiemannFMVFHead,
)
from riemannfm.models.input_encoding import (
    RiemannFMEdgeEncoder,
    RiemannFMNodeEncoder,
    RiemannFMRelationEncoder,
)
from riemannfm.models.normalization import RiemannFMATHNorm
from riemannfm.models.positional import RiemannFMTimeEmbedding
from riemannfm.models.text_condition import (
    RiemannFMNodeTextCross,
    RiemannFMRelationTextCross,
)

# Test constants matching small model config.
B = 4
N = 8
K = 10
NODE_DIM = 64
REL_DIM = 32
EDGE_DIM = 32
NUM_HEADS = 4
REL_HEADS = 4
REL_EMB_DIM = 16
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

    def test_odd_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="even"):
            RiemannFMTimeEmbedding(63)

    def test_gradient_flow(self) -> None:
        emb = RiemannFMTimeEmbedding(NODE_DIM)
        t = torch.rand(B)
        emb(t).sum().backward()
        for p in emb.parameters():
            assert p.grad is not None


class TestATHNorm:
    def test_output_shape(self) -> None:
        norm = RiemannFMATHNorm(NODE_DIM, NODE_DIM)
        x = torch.randn(B, N, NODE_DIM)
        t_emb = torch.randn(B, NODE_DIM)
        assert norm(x, t_emb).shape == (B, N, NODE_DIM)

    def test_identity_init(self) -> None:
        norm = RiemannFMATHNorm(NODE_DIM, NODE_DIM)
        x = torch.randn(B, N, NODE_DIM)
        t_emb = torch.zeros(B, NODE_DIM)
        out = norm(x, t_emb)
        expected = torch.nn.functional.layer_norm(x, [NODE_DIM])
        assert torch.allclose(out, expected, atol=1e-5)

    def test_gradient_flow(self) -> None:
        norm = RiemannFMATHNorm(NODE_DIM, NODE_DIM)
        x = torch.randn(B, N, NODE_DIM, requires_grad=True)
        t_emb = torch.randn(B, NODE_DIM)
        norm(x, t_emb).sum().backward()
        assert x.grad is not None

    def test_cond_film_injection(self) -> None:
        norm = RiemannFMATHNorm(NODE_DIM, NODE_DIM, cond_dim=2)
        x = torch.randn(B, N, NODE_DIM)
        t_emb = torch.randn(B, NODE_DIM)
        cond_a = torch.tensor([[-1.0, 1.0]]).expand(B, 2)
        cond_b = torch.tensor([[2.0, -3.0]]).expand(B, 2)
        out_a = norm(x, t_emb, cond=cond_a)
        out_b = norm(x, t_emb, cond=cond_b)
        assert out_a.shape == (B, N, NODE_DIM)
        opt = torch.optim.SGD(norm.parameters(), lr=1.0)
        (out_a.sum() + out_b.sum()).backward()
        opt.step()
        out_a2 = norm(x, t_emb, cond=cond_a)
        out_b2 = norm(x, t_emb, cond=cond_b)
        assert not torch.allclose(out_a2, out_b2)

    def test_cond_required_when_cond_dim_set(self) -> None:
        norm = RiemannFMATHNorm(NODE_DIM, NODE_DIM, cond_dim=2)
        x = torch.randn(B, N, NODE_DIM)
        t_emb = torch.randn(B, NODE_DIM)
        with pytest.raises(AssertionError):
            norm(x, t_emb, cond=None)


class TestNodeEncoder:
    def test_output_shape_no_text(self, ambient_dim: int) -> None:
        enc = RiemannFMNodeEncoder(
            ambient_dim, 0, NODE_DIM, NODE_DIM,
            dim_h_ambient=DIM_H + 1, dim_s_ambient=DIM_S + 1, dim_e=DIM_E,
        )
        x = torch.randn(B, N, ambient_dim)
        out = enc(
            x, torch.zeros(B, N, 0), torch.ones(B, N, dtype=torch.bool),
            torch.randn(B, NODE_DIM),
        )
        assert out.shape == (B, N, NODE_DIM)

    def test_output_shape_with_text(self, ambient_dim: int) -> None:
        d_c = 16
        enc = RiemannFMNodeEncoder(
            ambient_dim, d_c, NODE_DIM, NODE_DIM,
            dim_h_ambient=DIM_H + 1, dim_s_ambient=DIM_S + 1, dim_e=DIM_E,
        )
        x = torch.randn(B, N, ambient_dim)
        out = enc(
            x, torch.randn(B, N, d_c), torch.ones(B, N, dtype=torch.bool),
            torch.randn(B, NODE_DIM),
        )
        assert out.shape == (B, N, NODE_DIM)

    def test_drops_lorentz_time_coord(self, ambient_dim: int) -> None:
        enc = RiemannFMNodeEncoder(
            ambient_dim, 0, NODE_DIM, NODE_DIM,
            dim_h_ambient=DIM_H + 1, dim_s_ambient=DIM_S + 1, dim_e=DIM_E,
        ).eval()
        # in_dim = ambient_dim - 1 (drop x_0) + 0 (no text) + 0 (no pe)
        #        + 3 (node_mask + m_text + m_coord).
        assert enc.mlp[0].in_features == ambient_dim - 1 + 3

        x = torch.randn(B, N, ambient_dim)
        mask = torch.ones(B, N, dtype=torch.bool)
        t_emb = torch.randn(B, NODE_DIM)
        out_a = enc(x, torch.zeros(B, N, 0), mask, t_emb)
        x2 = x.clone()
        x2[..., 0] += 10.0
        out_b = enc(x2, torch.zeros(B, N, 0), mask, t_emb)
        assert torch.allclose(out_a, out_b)


class TestEdgeEncoder:
    def test_output_shape_no_text(self) -> None:
        rel_emb_dim = 16
        enc = RiemannFMEdgeEncoder(K, rel_emb_dim, 0, EDGE_DIM)
        E_t = torch.rand(B, N, N, K)
        R = torch.randn(K, rel_emb_dim)
        mu_t = torch.randint(0, 2, (B, N, N)).float()
        out = enc(E_t, R, mu_t)
        assert out.shape == (B, N, N, EDGE_DIM)

    def test_output_shape_with_text(self) -> None:
        d_c = 16
        rel_emb_dim = 16
        enc = RiemannFMEdgeEncoder(K, rel_emb_dim, d_c, EDGE_DIM)
        E_t = torch.rand(B, N, N, K)
        R = torch.randn(K, rel_emb_dim)
        mu_t = torch.randint(0, 2, (B, N, N)).float()
        relation_text = torch.randn(K, d_c)
        out = enc(E_t, R, mu_t, relation_text)
        assert out.shape == (B, N, N, EDGE_DIM)

    def test_mu_t_bit_is_consumed(self) -> None:
        rel_emb_dim = 16
        enc = RiemannFMEdgeEncoder(K, rel_emb_dim, 0, EDGE_DIM).eval()
        E_t = torch.zeros(B, N, N, K)
        R = torch.randn(K, rel_emb_dim)
        mu_a = torch.zeros(B, N, N)
        mu_b = torch.ones(B, N, N)
        out_a = enc(E_t, R, mu_a)
        out_b = enc(E_t, R, mu_b)
        assert not torch.allclose(out_a, out_b)


class TestRelationEncoder:
    def test_output_shape_no_text(self) -> None:
        rel_emb_dim = REL_EMB_DIM
        enc = RiemannFMRelationEncoder(rel_emb_dim, 0, REL_DIM, NODE_DIM)
        R = torch.randn(K, rel_emb_dim)
        t_emb = torch.randn(B, NODE_DIM)
        out = enc(R, t_emb)
        assert out.shape == (B, K, REL_DIM)

    def test_output_shape_with_text(self) -> None:
        d_c = 16
        enc = RiemannFMRelationEncoder(REL_EMB_DIM, d_c, REL_DIM, NODE_DIM)
        R = torch.randn(K, REL_EMB_DIM)
        relation_text = torch.randn(K, d_c)
        t_emb = torch.randn(B, NODE_DIM)
        out = enc(R, t_emb, relation_text)
        assert out.shape == (B, K, REL_DIM)


class TestEdgeBias:
    def test_output_shape(self) -> None:
        mod = RiemannFMEdgeBias(EDGE_DIM, NUM_HEADS)
        g = torch.randn(B, N, N, EDGE_DIM)
        assert mod(g).shape == (B, NUM_HEADS, N, N)


class TestGeodesicAttention:
    def test_output_shape(self, manifold: RiemannFMProductManifold) -> None:
        attn = RiemannFMGeodesicAttention(NODE_DIM, NUM_HEADS, manifold)
        h = torch.randn(B, N, NODE_DIM)
        x = manifold.sample_noise(B, N)
        mask = torch.ones(B, N, dtype=torch.bool)
        assert attn(h, x, node_mask=mask).shape == (B, N, NODE_DIM)

    def test_with_edge_bias(self, manifold: RiemannFMProductManifold) -> None:
        attn = RiemannFMGeodesicAttention(NODE_DIM, NUM_HEADS, manifold)
        h = torch.randn(B, N, NODE_DIM)
        x = manifold.sample_noise(B, N)
        bias = torch.randn(B, NUM_HEADS, N, N)
        mask = torch.ones(B, N, dtype=torch.bool)
        assert attn(h, x, edge_bias=bias, node_mask=mask).shape == (B, N, NODE_DIM)

    def test_gradient_flow(self, manifold: RiemannFMProductManifold) -> None:
        attn = RiemannFMGeodesicAttention(NODE_DIM, NUM_HEADS, manifold)
        h = torch.randn(B, N, NODE_DIM, requires_grad=True)
        x = manifold.sample_noise(B, N)
        attn(h, x).sum().backward()
        assert h.grad is not None
        assert torch.isfinite(h.grad).all()


class TestRelationSelfAttention:
    def test_output_shape(self) -> None:
        attn = RiemannFMRelationSelfAttention(REL_DIM, REL_HEADS, REL_EMB_DIM)
        h_R = torch.randn(B, K, REL_DIM)
        R = torch.randn(K, REL_EMB_DIM)
        assert attn(h_R, R).shape == (B, K, REL_DIM)

    def test_gradient_flow(self) -> None:
        attn = RiemannFMRelationSelfAttention(REL_DIM, REL_HEADS, REL_EMB_DIM)
        h_R = torch.randn(B, K, REL_DIM, requires_grad=True)
        R = torch.randn(K, REL_EMB_DIM, requires_grad=True)
        attn(h_R, R).sum().backward()
        assert h_R.grad is not None
        assert R.grad is not None

    def test_similarity_bias_off(self) -> None:
        attn_on = RiemannFMRelationSelfAttention(
            REL_DIM, REL_HEADS, REL_EMB_DIM, use_similarity_bias=True,
        ).eval()
        attn_off = RiemannFMRelationSelfAttention(
            REL_DIM, REL_HEADS, REL_EMB_DIM, use_similarity_bias=False,
        ).eval()
        # Same params except the bias term — copy weights and compare.
        attn_off.W_q.load_state_dict(attn_on.W_q.state_dict())
        attn_off.W_k.load_state_dict(attn_on.W_k.state_dict())
        attn_off.W_v.load_state_dict(attn_on.W_v.state_dict())
        attn_off.W_o.load_state_dict(attn_on.W_o.state_dict())
        h_R = torch.randn(B, K, REL_DIM)
        R = torch.randn(K, REL_EMB_DIM)
        # On init w_sim is zero so outputs match; perturb for contrast.
        attn_on.w_sim.data.fill_(1.0)
        assert not torch.allclose(attn_on(h_R, R), attn_off(h_R, R))


class TestEdgeSelfUpdate:
    def test_output_shape(self, manifold: RiemannFMProductManifold) -> None:
        mod = RiemannFMEdgeSelfUpdate(
            manifold=manifold,
            node_dim=NODE_DIM, edge_dim=EDGE_DIM,
            dim_h_ambient=DIM_H + 1, dim_s_ambient=DIM_S + 1, dim_e=DIM_E,
        )
        h_E = torch.randn(B, N, N, EDGE_DIM)
        h_V = torch.randn(B, N, NODE_DIM)
        x = manifold.sample_noise(B, N, radius_h=1.0)
        out = mod(h_E, h_V, x)
        assert out.shape == (B, N, N, EDGE_DIM)

    def test_gradient_flow(self, manifold: RiemannFMProductManifold) -> None:
        mod = RiemannFMEdgeSelfUpdate(
            manifold=manifold,
            node_dim=NODE_DIM, edge_dim=EDGE_DIM,
            dim_h_ambient=DIM_H + 1, dim_s_ambient=DIM_S + 1, dim_e=DIM_E,
        )
        h_E = torch.randn(B, N, N, EDGE_DIM, requires_grad=True)
        h_V = torch.randn(B, N, NODE_DIM, requires_grad=True)
        x = manifold.sample_noise(B, N, radius_h=1.0)
        mod(h_E, h_V, x).sum().backward()
        assert h_E.grad is not None
        assert h_V.grad is not None


class TestNodeRelationCross:
    def test_output_shapes(self) -> None:
        mod = RiemannFMNodeRelationCross(
            NODE_DIM, REL_DIM, num_heads_V=NUM_HEADS, num_heads_R=REL_HEADS,
        )
        h_V = torch.randn(B, N, NODE_DIM)
        h_R = torch.randn(B, K, REL_DIM)
        mask = torch.ones(B, N, dtype=torch.bool)
        dV, dR = mod(h_V, h_R, mask)
        assert dV.shape == (B, N, NODE_DIM)
        assert dR.shape == (B, K, REL_DIM)

    def test_node_mask_respected(self) -> None:
        """Masked-out nodes shouldn't leak into relation updates."""
        mod = RiemannFMNodeRelationCross(
            NODE_DIM, REL_DIM, num_heads_V=NUM_HEADS, num_heads_R=REL_HEADS,
        ).eval()
        h_R = torch.randn(B, K, REL_DIM)
        h_V = torch.randn(B, N, NODE_DIM)
        mask_a = torch.ones(B, N, dtype=torch.bool)
        mask_a[:, N // 2:] = False
        _, dR_a = mod(h_V, h_R, mask_a)
        # Perturb the masked-out nodes; dR should not change.
        h_V_b = h_V.clone()
        h_V_b[:, N // 2:] = 5.0
        _, dR_b = mod(h_V_b, h_R, mask_a)
        assert torch.allclose(dR_a, dR_b, atol=1e-5)


class TestNodeEdgeCross:
    def test_output_shapes(self) -> None:
        mod = RiemannFMNodeEdgeCross(NODE_DIM, EDGE_DIM, NUM_HEADS)
        h_V = torch.randn(B, N, NODE_DIM)
        h_E = torch.randn(B, N, N, EDGE_DIM)
        mask = torch.ones(B, N, dtype=torch.bool)
        dV, dE = mod(h_V, h_E, h_V, mask)
        assert dV.shape == (B, N, NODE_DIM)
        assert dE.shape == (B, N, N, EDGE_DIM)

    def test_gradient_flow(self) -> None:
        mod = RiemannFMNodeEdgeCross(NODE_DIM, EDGE_DIM, NUM_HEADS)
        h_V = torch.randn(B, N, NODE_DIM, requires_grad=True)
        h_E = torch.randn(B, N, N, EDGE_DIM, requires_grad=True)
        mask = torch.ones(B, N, dtype=torch.bool)
        dV, dE = mod(h_V, h_E, h_V, mask)
        (dV.sum() + dE.sum()).backward()
        assert h_V.grad is not None
        assert h_E.grad is not None


class TestNodeTextCross:
    def test_output_shape(self) -> None:
        text_dim = 24
        mod = RiemannFMNodeTextCross(NODE_DIM, text_dim, NUM_HEADS)
        h_V = torch.randn(B, N, NODE_DIM)
        C_V = torch.randn(B, N, text_dim)
        mask = torch.ones(B, N, dtype=torch.bool)
        assert mod(h_V, C_V, mask).shape == (B, N, NODE_DIM)


class TestRelationTextCross:
    def test_output_shape_2d_text(self) -> None:
        text_dim = 24
        mod = RiemannFMRelationTextCross(REL_DIM, text_dim, REL_HEADS)
        h_R = torch.randn(B, K, REL_DIM)
        relation_text = torch.randn(K, text_dim)  # 2D, auto-broadcast to batch
        assert mod(h_R, relation_text).shape == (B, K, REL_DIM)

    def test_output_shape_3d_text(self) -> None:
        text_dim = 24
        mod = RiemannFMRelationTextCross(REL_DIM, text_dim, REL_HEADS)
        h_R = torch.randn(B, K, REL_DIM)
        relation_text = torch.randn(B, K, text_dim)
        assert mod(h_R, relation_text).shape == (B, K, REL_DIM)


class TestVFHead:
    def test_output_shape(
        self, manifold: RiemannFMProductManifold, ambient_dim: int,
    ) -> None:
        head = RiemannFMVFHead(NODE_DIM, ambient_dim, manifold)
        h = torch.randn(B, N, NODE_DIM)
        x = manifold.sample_noise(B, N)
        assert head(h, x).shape == (B, N, ambient_dim)

    def test_output_is_tangent(
        self, manifold: RiemannFMProductManifold, ambient_dim: int,
    ) -> None:
        head = RiemannFMVFHead(NODE_DIM, ambient_dim, manifold)
        h = torch.randn(B, N, NODE_DIM) * 0.1
        x = manifold.sample_noise(B, N)
        v = head(h, x)
        v_proj = manifold.proj_tangent(x, v)
        rel_err = (v - v_proj).norm() / (v.norm() + 1e-8)
        assert rel_err < 1e-3


class TestEdgeExHead:
    def test_output_shape(self) -> None:
        head = RiemannFMEdgeExHead(EDGE_DIM)
        assert head(torch.randn(B, N, N, EDGE_DIM)).shape == (B, N, N)

    def test_gradient_flow(self) -> None:
        head = RiemannFMEdgeExHead(EDGE_DIM)
        g = torch.randn(B, N, N, EDGE_DIM, requires_grad=True)
        head(g).sum().backward()
        assert g.grad is not None


class TestEdgeTypeHead:
    def test_output_shape(self) -> None:
        head = RiemannFMEdgeTypeHead(EDGE_DIM, REL_EMB_DIM, K)
        g = torch.randn(B, N, N, EDGE_DIM)
        R = torch.randn(K, REL_EMB_DIM)
        assert head(g, R).shape == (B, N, N, K)

    def test_gradient_flows_to_relation_embeddings(self) -> None:
        head = RiemannFMEdgeTypeHead(EDGE_DIM, REL_EMB_DIM, K)
        g = torch.randn(B, N, N, EDGE_DIM, requires_grad=True)
        R = torch.randn(K, REL_EMB_DIM, requires_grad=True)
        head(g, R).sum().backward()
        assert g.grad is not None
        assert R.grad is not None


class TestEndToEndModel:
    """Smoke test: instantiate the top-level RiemannFM and run forward/backward."""

    def test_full_forward_backward(
        self, manifold: RiemannFMProductManifold, ambient_dim: int,
    ) -> None:
        from riemannfm.models.riemannfm import RiemannFM

        model = RiemannFM(
            manifold=manifold,
            num_layers=2,
            node_dim=NODE_DIM,
            rel_dim=REL_DIM,
            edge_dim=EDGE_DIM,
            num_heads_V=NUM_HEADS,
            num_heads_R=REL_HEADS,
            num_edge_types=K,
            input_text_dim=16,
            text_proj_dim=16,
            rel_emb_dim=REL_EMB_DIM,
            dropout=0.0,
        )

        x_t = manifold.sample_noise(B, N, radius_h=1.0)
        E_t = torch.zeros(B, N, N, K)
        E_t[:, 0, 1, 0] = 1.0
        mu_t = torch.rand(B, N, N).round()
        t = torch.rand(B)
        node_text = torch.randn(B, N, 16)
        node_mask = torch.ones(B, N, dtype=torch.bool)
        relation_text = torch.randn(K, 16)
        m_text = torch.ones(B, N, dtype=torch.bool)
        m_coord = torch.ones(B, N, dtype=torch.bool)

        V_hat, ell_ex, ell_type, h = model(
            x_t=x_t, E_t=E_t, mu_t=mu_t, t=t,
            node_text=node_text, node_mask=node_mask,
            relation_text=relation_text, m_text=m_text, m_coord=m_coord,
        )
        assert V_hat.shape == (B, N, ambient_dim)
        assert ell_ex.shape == (B, N, N)
        assert ell_type.shape == (B, N, N, K)
        assert h.shape == (B, N, NODE_DIM)

        (V_hat.sum() + ell_ex.sum() + ell_type.sum()).backward()
        # Gradient must reach the global relation embedding (rel_emb is
        # consumed by the edge encoder, A_R bias, D_VR, and the type head).
        assert model.rel_emb.grad is not None
