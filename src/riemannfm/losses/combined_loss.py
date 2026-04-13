"""Combined loss for RiemannFM pretraining.

L_total = L_cont
        + lambda_disc * L_disc
        + mu_align * L_align
        + nu_mask_c * L_mask_c      (semantic masked-node ID)
        + nu_mask_x * L_mask_x      (geometric masked-node reconstruction)

L_mask_x is an explicit term — structurally it is ``continuous_flow_loss``
restricted to the ``MASK_X`` subset (where the collator forced t=0 so
``x_t = x_0`` is pure noise).  Keeping it named separately makes the
ablation/metrics/math-doc story crisp.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from riemannfm.losses.flow_matching_loss import (
    continuous_flow_loss,
    discrete_flow_loss,
)

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


def _contrastive_loss_from_pairs(
    g: Tensor, c: Tensor, temperature: float,
) -> Tensor:
    """Symmetric InfoNCE on pre-gathered valid node pairs.

    Args:
        g: Projected graph features for valid nodes, shape ``(M, d_a)``.
        c: Projected text embeddings for valid nodes, shape ``(M, d_a)``.
        temperature: InfoNCE temperature.

    Returns:
        Scalar contrastive loss.
    """
    if c.norm(dim=-1).max() < 1e-8:
        return torch.tensor(0.0, device=g.device, dtype=g.dtype)

    g_norm = F.normalize(g, dim=-1)
    c_norm = F.normalize(c, dim=-1)
    logits = g_norm @ c_norm.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_g2c = F.cross_entropy(logits, labels)
    loss_c2g = F.cross_entropy(logits.T, labels)
    return (loss_g2c + loss_c2g) / 2


def masked_node_loss(
    h_masked: Tensor,
    proj_mask: nn.Module,
    true_emb: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """In-batch contrastive masked node prediction loss (L_mask_c).

    Each masked node must identify its true text embedding among all
    other M_c nodes' text embeddings in the same batch.  Symmetric
    InfoNCE over the (M_c x M_c) similarity matrix.

    Args:
        h_masked: Backbone hidden states for M_c nodes, ``(M, node_dim)``.
        proj_mask: Projection head ``node_dim -> d_c``.
        true_emb: True text embeddings for M_c nodes, ``(M, d_c)``.
        temperature: InfoNCE temperature.

    Returns:
        Scalar loss.
    """
    if h_masked.shape[0] < 2:
        return torch.tensor(0.0, device=h_masked.device, dtype=h_masked.dtype)

    pred = proj_mask(h_masked)  # (M, d_c)

    # Remove per-batch shared mean direction before cosine similarity.
    # entity_emb collapses to ‖mean‖/mean(‖x‖) ≈ 0.9, hiding real
    # discriminability under a single shared component.
    pred = pred - pred.mean(dim=0, keepdim=True)
    true_emb = true_emb - true_emb.mean(dim=0, keepdim=True)

    pred_norm = F.normalize(pred, dim=-1)
    true_norm = F.normalize(true_emb, dim=-1)

    logits = pred_norm @ true_norm.T / temperature  # (M, M)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_p2t = F.cross_entropy(logits, labels)
    loss_t2p = F.cross_entropy(logits.T, labels)
    return (loss_p2t + loss_t2p) / 2


class RiemannFMCombinedLoss(nn.Module):
    """Combined pretraining loss (Algorithm 1).

    L_total = L_cont
            + lambda_disc * L_disc
            + mu_align   * L_align
            + nu_mask_c  * L_mask_c
            + nu_mask_x  * L_mask_x

    Node partition (Def 6.9a) — each real node is exactly one of:
      - ``REAL``  : both x and c real   — contributes to L_cont + L_align
      - ``MASK_C``: text masked (c ← mask_emb), geometry real (t=1)
                    — contributes to L_mask_c
      - ``MASK_X``: geometry masked (x = x_0 at t=0), text real
                    — contributes to L_mask_x (via continuous_flow_loss)

    Learnable projection heads:
      - proj_g      : R^{node_dim} -> R^{d_a}  (L_align graph side)
      - proj_c      : R^{d_c}     -> R^{d_a}  (L_align text side)
      - proj_mask_c : R^{node_dim} -> R^{d_c}  (L_mask_c; InfoNCE to text)

    L_mask_x reuses the backbone's V_hat output — no new head.

    Args:
        manifold: Product manifold for tangent norm in L_cont / L_mask_x.
        lambda_disc: Weight for discrete flow loss.
        mu_align: Weight for contrastive alignment loss.
        nu_mask_c: Weight for L_mask_c (text-masked nodes).
        nu_mask_x: Weight for L_mask_x (geometry-masked nodes at t=0).
        neg_ratio: Ratio of negative to positive pairs for L_disc sampling.
        edge_loss_mode: L_disc formulation, ``"bce"`` (multi-hot per-channel)
            or ``"softmax_ce"`` (K-way on positive pairs).
        temperature: InfoNCE temperature for L_align.
        mask_c_temperature: InfoNCE temperature for L_mask_c.
        input_text_dim: Raw text embedding dimension d_c (0 to disable
            alignment and L_mask_c).
        node_dim: Backbone hidden state dimension for graph-side projection.
        d_a: Alignment space dimension for projection layers.
        max_align_nodes: Maximum nodes for L_align contrastive matrix.
    """

    def __init__(
        self,
        manifold: RiemannFMProductManifold,
        lambda_disc: float = 1.0,
        mu_align: float = 0.1,
        nu_mask_c: float = 0.0,
        nu_mask_x: float = 0.0,
        neg_ratio: float = 1.0,
        edge_loss_mode: str = "bce",
        temperature: float = 0.07,
        mask_c_temperature: float = 0.07,
        input_text_dim: int = 0,
        node_dim: int = 512,
        d_a: int = 256,
        max_align_nodes: int = 128,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.lambda_disc = lambda_disc
        self.mu_align = mu_align
        self.nu_mask_c = nu_mask_c
        self.nu_mask_x = nu_mask_x
        self.neg_ratio = neg_ratio
        self.edge_loss_mode = edge_loss_mode
        self.temperature = temperature
        self.mask_c_temperature = mask_c_temperature
        self.max_align_nodes = max_align_nodes

        # L_align projections (Def 6.9).
        self.ln_g: nn.LayerNorm | None = None
        self.ln_c: nn.LayerNorm | None = None
        self.proj_g: nn.Sequential | None = None
        self.proj_c: nn.Sequential | None = None
        if mu_align > 0 and input_text_dim > 0:
            # Bias-free projections: randomly-initialised bias creates a
            # large shared offset that, after L2-normalisation in InfoNCE,
            # collapses all embeddings to cos-sim > 0.99 and makes the
            # contrastive loss degenerate (loss = log M).  Without bias
            # the MLP preserves input diversity (cos-sim stays ~0.17).
            self.ln_g = nn.LayerNorm(node_dim)
            self.ln_c = nn.LayerNorm(input_text_dim)
            self.proj_g = nn.Sequential(
                nn.Linear(node_dim, d_a, bias=False),
                nn.GELU(),
                nn.Linear(d_a, d_a, bias=False),
            )
            self.proj_c = nn.Sequential(
                nn.Linear(input_text_dim, d_a, bias=False),
                nn.GELU(),
                nn.Linear(d_a, d_a, bias=False),
            )

        # L_mask_c head (semantic identification from geometry-real h).
        # Targets are text embeddings (d_c), NOT manifold coordinates.
        # Entity embeddings cluster near the manifold origin (cos-sim
        # > 0.998) making manifold-coordinate targets indistinguishable;
        # text embeddings have healthy diversity (cos-sim ~0.20).
        self.proj_mask_c: nn.Sequential | None
        if nu_mask_c > 0 and input_text_dim > 0:
            self.proj_mask_c = nn.Sequential(
                nn.LayerNorm(node_dim),
                nn.Linear(node_dim, node_dim, bias=False),
                nn.GELU(),
                nn.Linear(node_dim, input_text_dim, bias=False),
            )
        else:
            self.proj_mask_c = None

    def forward(
        self,
        V_hat: Tensor,
        u_t: Tensor,
        x_t: Tensor,
        P_hat: Tensor,
        E_1: Tensor,
        node_mask: Tensor,
        h: Tensor | None = None,
        node_text: Tensor | None = None,
        mask_type: Tensor | None = None,
        true_text_emb: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute combined loss.

        Args:
            V_hat: Predicted vector field, shape ``(B, N, D)``.
            u_t: Target vector field, shape ``(B, N, D)``.
            x_t: Interpolated manifold coords, shape ``(B, N, D)``.
            P_hat: Predicted edge logits, shape ``(B, N, N, K)``.
            E_1: Target edge types, shape ``(B, N, N, K)``.
            node_mask: Bool mask, shape ``(B, N)``.
            h: Backbone hidden states for L_align / L_mask_c, shape
                ``(B, N, node_dim)``.  None to skip both.
            node_text: Raw text embeddings (AFTER masking replacement) for
                L_align, shape ``(B, N, d_c)``.  None to skip L_align.
            mask_type: Node partition labels, shape ``(B, N)``.
                ``{-1: VIRTUAL, 0: REAL, 1: MASK_C, 2: MASK_X}``.
                None to skip L_mask_c / L_mask_x.
            true_text_emb: Pre-masking text embeddings for M_c targets,
                shape ``(B, N, d_c)``.  Only needed when mask_type is
                provided and L_mask_c is enabled.

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        from riemannfm.data.collator import MASK_C, MASK_REAL, MASK_X

        if mask_type is not None:
            real_mask = node_mask & (mask_type == MASK_REAL)
            mx_mask = node_mask & (mask_type == MASK_X)
            mc_mask = node_mask & (mask_type == MASK_C)
        else:
            real_mask = node_mask
            mx_mask = torch.zeros_like(node_mask)
            mc_mask = torch.zeros_like(node_mask)

        # L_cont: standard flow matching on REAL nodes only.
        l_cont = continuous_flow_loss(
            self.manifold, V_hat, u_t, x_t, real_mask,
        )

        # L_mask_x: identical flow matching formula on M_x subset (t=0).
        # Reuses continuous_flow_loss; the collator already forced t=0
        # on these positions so x_t[M_x] = x_0 and u_t is the initial
        # velocity pointing to x_1.
        if mask_type is not None and mx_mask.any():
            l_mask_x = continuous_flow_loss(
                self.manifold, V_hat, u_t, x_t, mx_mask,
            )
        else:
            l_mask_x = torch.tensor(0.0, device=V_hat.device, dtype=V_hat.dtype)

        # L_disc: edge flow matching (full node_mask — edges span pairs,
        # unaffected by per-node mask labels).
        l_disc = discrete_flow_loss(
            P_hat, E_1, node_mask,
            neg_ratio=self.neg_ratio,
            mode=self.edge_loss_mode,
        )

        # L_align: contrastive alignment on REAL nodes only.
        if (
            self.mu_align > 0
            and self.proj_g is not None
            and self.proj_c is not None
            and h is not None
            and node_text is not None
            and node_text.shape[-1] > 0
        ):
            B, N, _ = h.shape
            mask_flat = real_mask.reshape(-1)
            valid_idx = mask_flat.nonzero(as_tuple=True)[0]

            if valid_idx.numel() > self.max_align_nodes:
                perm = torch.randperm(
                    valid_idx.numel(), device=valid_idx.device,
                )[:self.max_align_nodes]
                valid_idx = valid_idx[perm]

            if valid_idx.numel() < 2:
                l_align = torch.tensor(0.0, device=h.device, dtype=h.dtype)
            else:
                h_valid = h.reshape(B * N, -1)[valid_idx]
                t_valid = node_text.reshape(B * N, -1)[valid_idx]
                assert self.ln_g is not None and self.ln_c is not None
                g_valid = self.proj_g(self.ln_g(h_valid))
                c_valid = self.proj_c(self.ln_c(t_valid))
                l_align = _contrastive_loss_from_pairs(
                    g_valid, c_valid, self.temperature,
                )
        else:
            l_align = torch.tensor(0.0, device=V_hat.device, dtype=V_hat.dtype)

        # L_mask_c: InfoNCE on M_c nodes against their true text embeddings.
        if (
            self.nu_mask_c > 0
            and self.proj_mask_c is not None
            and h is not None
            and mask_type is not None
            and true_text_emb is not None
            and mc_mask.any()
        ):
            B, N, _ = h.shape
            mc_flat = mc_mask.reshape(-1)
            mc_idx = mc_flat.nonzero(as_tuple=True)[0]

            if mc_idx.numel() < 2:
                l_mask_c = torch.tensor(0.0, device=h.device, dtype=h.dtype)
            else:
                h_mc = h.reshape(B * N, -1)[mc_idx]
                true_mc = true_text_emb.reshape(B * N, -1)[mc_idx]
                l_mask_c = masked_node_loss(
                    h_mc, self.proj_mask_c, true_mc,
                    temperature=self.mask_c_temperature,
                )
        else:
            l_mask_c = torch.tensor(0.0, device=V_hat.device, dtype=V_hat.dtype)

        # Total loss.
        total = (
            l_cont
            + self.lambda_disc * l_disc
            + self.mu_align * l_align
            + self.nu_mask_c * l_mask_c
            + self.nu_mask_x * l_mask_x
        )

        metrics = {
            "loss/total": total.detach(),
            "loss/cont": l_cont.detach(),
            "loss/disc": l_disc.detach(),
            "loss/align": l_align.detach(),
            "loss/mask_c": l_mask_c.detach(),
            "loss/mask_x": l_mask_x.detach(),
        }

        return total, metrics
