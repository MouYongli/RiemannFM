"""Top-level RiemannFM model.

Wires together: text projection -> input encoding -> RieFormer backbone -> prediction heads.
Takes raw flow-matching inputs and produces vector field + edge predictions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from riemannfm.models.heads import RiemannFMEdgeHead, RiemannFMVFHead
from riemannfm.models.input_encoding import (
    RiemannFMEdgeEncoder,
    RiemannFMNodeEncoder,
)
from riemannfm.models.positional import RiemannFMTimeEmbedding
from riemannfm.models.rieformer import RiemannFMRieFormer

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


class RiemannFM(nn.Module):
    """Top-level RiemannFM model.

    Text embeddings from the data pipeline (``input_text_dim``, e.g. 768
    from nomic, 4096 from qwen3) are first projected to a fixed internal
    dimension ``text_proj_dim`` via a learned linear layer.  This decouples
    the model architecture from the choice of text encoder.

    Full forward pass:
      0. Text projection: C_V (input_text_dim) -> C_V (text_proj_dim),
                           C_R (input_text_dim) -> C_R (text_proj_dim)
      1. Time embedding: t -> t_emb
      2. Node encoding: (x_t, C_V_proj, m) -> h^0
      3. Edge encoding: (E_t, C_R_proj) -> g^0
      4. RieFormer backbone: (h^0, g^0, x_t, t_emb, m) -> (h^L, g^L)
      5. VF head: (h^L, x_t) -> V_hat (tangent vectors)
      6. Edge head: g^L -> P_hat (edge logits)

    Args:
        manifold: Product manifold M = H x S x E.
        num_layers: Number of RieFormer blocks.
        node_dim: Node hidden dimension.
        edge_dim: Edge hidden dimension.
        num_heads: Number of attention heads.
        edge_heads: Number of edge factorization heads.
        num_edge_types: Number of relation types K.
        input_text_dim: Raw text embedding dimension from data pipeline
            (e.g. 768, 4096).  0 means no text.
        text_proj_dim: Internal text condition dimension after projection
            (Def 3.5).  All model components use this fixed dimension.
        use_text_cross_attn: Enable text cross-attention (Module E).
        text_cross_attn_every: Insert text cross-attn every N layers.
        rel_emb_dim: Learnable relation embedding dimension.
        use_geodesic_kernel: Ablation flag.
        use_ath_norm: Ablation flag.
        use_edge_self_update: Ablation flag.
        use_dual_stream_cross: Ablation flag.
        use_text_condition: Ablation flag.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        manifold: RiemannFMProductManifold,
        num_layers: int = 6,
        node_dim: int = 384,
        edge_dim: int = 128,
        num_heads: int = 6,
        edge_heads: int = 2,
        num_edge_types: int = 10,
        input_text_dim: int = 0,
        text_proj_dim: int = 256,
        pe_dim: int = 0,
        use_text_cross_attn: bool = False,
        text_cross_attn_every: int = 999,
        rel_emb_dim: int = 32,
        use_geodesic_kernel: bool = True,
        use_ath_norm: bool = True,
        use_edge_self_update: bool = True,
        use_dual_stream_cross: bool = True,
        use_text_condition: bool = True,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.text_proj_dim = text_proj_dim if input_text_dim > 0 else 0
        self.pe_dim = pe_dim

        ambient_dim = manifold.ambient_dim
        time_dim = node_dim

        # Per-component ambient dims (H→S→E order), used by the node
        # encoder to drop the Lorentz time-like coord x_0.  Absent
        # components contribute 0.
        dim_h_ambient = (
            manifold.hyperbolic.ambient_dim if manifold.hyperbolic is not None else 0
        )
        dim_s_ambient = (
            manifold.spherical.ambient_dim if manifold.spherical is not None else 0
        )
        dim_e = manifold.euclidean.ambient_dim if manifold.euclidean is not None else 0

        # Curvature conditioning dim: one scalar per curved component
        # present in the product manifold (H, S).  Recovered into the
        # network via ATH-Norm FiLM after x_0 / s_0 are stripped from
        # the encoder input by the block-wise LayerNorm.
        self._has_h_cond = manifold.hyperbolic is not None
        self._has_s_cond = manifold.spherical is not None
        cond_dim = int(self._has_h_cond) + int(self._has_s_cond)

        # Text projection: input_text_dim -> text_proj_dim.
        self.text_proj: nn.Linear | None
        if input_text_dim > 0 and self.text_proj_dim > 0:
            self.text_proj = nn.Linear(input_text_dim, self.text_proj_dim)
        else:
            self.text_proj = None

        # Time embedding.
        self.time_emb = RiemannFMTimeEmbedding(time_dim)

        # Input encoders use text_proj_dim (projected dimension).
        self.node_encoder = RiemannFMNodeEncoder(
            ambient_dim=ambient_dim,
            text_dim=self.text_proj_dim,
            node_dim=node_dim,
            time_dim=time_dim,
            pe_dim=pe_dim,
            dim_h_ambient=dim_h_ambient,
            dim_s_ambient=dim_s_ambient,
            dim_e=dim_e,
            dropout=dropout,
        )
        self.edge_encoder = RiemannFMEdgeEncoder(
            num_edge_types=num_edge_types,
            rel_emb_dim=rel_emb_dim,
            text_dim=self.text_proj_dim if use_text_condition else 0,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        # RieFormer backbone.
        # Cross-attention uses text_proj_dim as kdim/vdim.
        cross_attn_dim = self.text_proj_dim if use_text_cross_attn else 0
        self.backbone = RiemannFMRieFormer(
            num_layers=num_layers,
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            edge_heads=edge_heads,
            manifold=manifold,
            time_dim=time_dim,
            text_dim=cross_attn_dim,
            text_cross_attn_every=text_cross_attn_every,
            use_geodesic_kernel=use_geodesic_kernel,
            use_ath_norm=use_ath_norm,
            use_edge_self_update=use_edge_self_update,
            use_dual_stream_cross=use_dual_stream_cross,
            use_text_condition=use_text_condition,
            cond_dim=cond_dim,
            dropout=dropout,
        )
        self._cond_dim = cond_dim

        # Prediction heads.
        self.vf_head = RiemannFMVFHead(node_dim, ambient_dim, manifold)
        self.edge_head = RiemannFMEdgeHead(
            edge_dim, num_edge_types, text_proj_dim=self.text_proj_dim,
        )

    def _project_text(self, x: Tensor) -> Tensor:
        """Project raw text embeddings to internal text_proj_dim dimension.

        Args:
            x: Raw text embeddings, shape ``(..., input_text_dim)``.

        Returns:
            Projected embeddings, shape ``(..., text_proj_dim)``.
            Returns zero-width tensor if no text projection.
        """
        if self.text_proj is not None and x.shape[-1] > 0:
            result: Tensor = self.text_proj(x)
            return result
        return torch.zeros(
            *x.shape[:-1], self.text_proj_dim, device=x.device, dtype=x.dtype,
        )

    def _build_curvature_cond(
        self, batch_size: int, device: torch.device, dtype: torch.dtype,
    ) -> Tensor | None:
        """Build per-batch curvature conditioning tensor for ATH-Norm FiLM.

        Returns ``(B, cond_dim)`` with available curvature scalars in H→S
        order, or None when no curved component is present.  Curvature
        gradient is preserved (not detached) so FiLM adds a direct path
        alongside the Riemannian-loss path.
        """
        if self._cond_dim == 0:
            return None
        parts: list[Tensor] = []
        if self._has_h_cond:
            parts.append(self.manifold.hyperbolic.curvature.reshape(1))  # type: ignore[union-attr]
        if self._has_s_cond:
            parts.append(self.manifold.spherical.curvature.reshape(1))  # type: ignore[union-attr]
        cond = torch.cat(parts, dim=-1).to(device=device, dtype=dtype)
        return cond.unsqueeze(0).expand(batch_size, -1)

    def forward(
        self,
        x_t: Tensor,
        E_t: Tensor,
        t: Tensor,
        node_text: Tensor,
        node_mask: Tensor,
        C_R: Tensor | None = None,
        node_pe: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Full forward pass.

        Args:
            x_t: Interpolated manifold coordinates, shape ``(B, N, D)``.
            E_t: Interpolated edge types, shape ``(B, N, N, K)``.
            t: Time steps, shape ``(B,)``.
            node_text: Node text embeddings, shape ``(B, N, input_text_dim)``.
            node_mask: Bool mask, shape ``(B, N)``.
            C_R: Relation text embeddings, shape ``(K, input_text_dim)``.

        Returns:
            Tuple of:
              - V_hat: Predicted vector field, shape ``(B, N, D)``.
              - P_hat: Predicted edge logits, shape ``(B, N, N, K)``.
              - h: Final node hidden states, shape ``(B, N, node_dim)``.
        """
        # 0. Project text embeddings: input_text_dim -> text_proj_dim.
        node_text_proj = self._project_text(node_text)  # (B, N, text_proj_dim)
        C_R_proj = self._project_text(C_R) if C_R is not None else None

        # 1. Time embedding.
        t_emb = self.time_emb(t)  # (B, time_dim)

        # 2. Input encoding (uses projected text_proj_dim-dim text).
        h = self.node_encoder(x_t, node_text_proj, node_mask, t_emb, node_pe=node_pe)
        g = self.edge_encoder(E_t, C_R_proj)

        # 3. RieFormer backbone.
        C_V = node_text_proj if self.text_proj_dim > 0 else None
        cond = self._build_curvature_cond(x_t.shape[0], x_t.device, x_t.dtype)
        h, g = self.backbone(h, g, x_t, t_emb, node_mask, C_V, cond=cond)

        # 4. Prediction heads.
        V_hat = self.vf_head(h, x_t)
        P_hat = self.edge_head(g, C_R_proj)

        return V_hat, P_hat, h

    @classmethod
    def from_config(
        cls,
        model_cfg: Any,
        manifold_cfg: Any,
        ablation_cfg: Any,
        num_edge_types: int,
        input_text_dim: int = 0,
    ) -> RiemannFM:
        """Construct from Hydra configs.

        Args:
            model_cfg: Model config (small/base/large).
            manifold_cfg: Manifold config (product_h_s_e etc.).
            ablation_cfg: Ablation config (full, no_mrope, etc.).
            num_edge_types: Number of relation types K.
            input_text_dim: Data text embedding dimension (from disk).
        """
        from riemannfm.manifolds.product import RiemannFMProductManifold

        manifold = RiemannFMProductManifold.from_config(manifold_cfg)

        return cls(
            manifold=manifold,
            num_layers=model_cfg.num_layers,
            node_dim=model_cfg.node_dim,
            edge_dim=model_cfg.edge_dim,
            num_heads=model_cfg.num_heads,
            edge_heads=model_cfg.edge_heads,
            num_edge_types=num_edge_types,
            input_text_dim=input_text_dim,
            text_proj_dim=int(getattr(model_cfg, "text_proj_dim", 256)),
            pe_dim=int(getattr(model_cfg, "pe_dim", 0)),
            use_text_cross_attn=bool(getattr(model_cfg, "use_text_cross_attn", False)),
            text_cross_attn_every=model_cfg.text_cross_attn_every,
            rel_emb_dim=model_cfg.rel_emb_dim,
            use_geodesic_kernel=ablation_cfg.use_geodesic_kernel,
            use_ath_norm=ablation_cfg.use_ath_norm,
            use_edge_self_update=ablation_cfg.use_edge_self_update,
            use_dual_stream_cross=ablation_cfg.use_dual_stream_cross,
            use_text_condition=ablation_cfg.use_text_condition,
            dropout=model_cfg.dropout,
        )
