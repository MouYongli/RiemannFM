"""Combined loss for RiemannFM pretraining.

L_total = L_cont + lambda_disc * L_disc + mu_align * L_align + nu_mask * L_mask
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
    """In-batch contrastive masked node prediction loss (L_mask).

    Each masked node must identify its true entity embedding among
    all other masked nodes' entity embeddings in the same batch.
    This is the same symmetric InfoNCE formulation as L_align.

    Args:
        h_masked: Backbone hidden states for masked nodes, ``(M, node_dim)``.
        proj_mask: Projection head ``node_dim -> D_emb``.
        true_emb: True entity embeddings for masked nodes, ``(M, D_emb)``.
        temperature: InfoNCE temperature.

    Returns:
        Scalar loss.
    """
    if h_masked.shape[0] < 2:
        return torch.tensor(0.0, device=h_masked.device, dtype=h_masked.dtype)

    pred = proj_mask(h_masked)  # (M, D_emb)

    pred_norm = F.normalize(pred, dim=-1)       # (M, D_emb)
    true_norm = F.normalize(true_emb, dim=-1)   # (M, D_emb)

    # Symmetric InfoNCE: pred[i] should match true[i], not true[j].
    logits = pred_norm @ true_norm.T / temperature  # (M, M)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_p2t = F.cross_entropy(logits, labels)
    loss_t2p = F.cross_entropy(logits.T, labels)
    return (loss_p2t + loss_t2p) / 2


class RiemannFMCombinedLoss(nn.Module):
    """Combined pretraining loss (Algorithm 1).

    L_total = L_cont + lambda_disc * L_disc + mu_align * L_align + nu_mask * L_mask

    Contains learnable projection layers for the contrastive alignment
    loss (Definition 6.9):
      - proj_g: R^{node_dim} -> R^{d_a}  (backbone hidden -> alignment space)
      - proj_c: R^{d_c} -> R^{d_a}       (text embeddings -> alignment space)
    And for the masked node prediction loss:
      - proj_mask: R^{node_dim} -> R^{D_emb}  (backbone hidden -> entity emb space)

    Args:
        manifold: Product manifold for tangent norm in L_cont.
        lambda_disc: Weight for discrete flow loss.
        mu_align: Weight for contrastive alignment loss.
        nu_mask: Weight for masked node prediction loss.
        neg_ratio: Ratio of negative to positive pairs for L_disc sampling.
        temperature: InfoNCE temperature for L_align.
        mask_temperature: InfoNCE temperature for L_mask.
        input_text_dim: Raw text embedding dimension d_c (0 to disable alignment).
        node_dim: Backbone hidden state dimension for graph-side projection.
        entity_emb_dim: Entity embedding dimension D for proj_mask.
        d_a: Alignment space dimension for projection layers.
        max_align_nodes: Maximum nodes for L_align contrastive matrix.
    """

    def __init__(
        self,
        manifold: RiemannFMProductManifold,
        lambda_disc: float = 1.0,
        mu_align: float = 0.1,
        nu_mask: float = 0.0,
        neg_ratio: float = 1.0,
        temperature: float = 0.07,
        mask_temperature: float = 0.07,
        input_text_dim: int = 0,
        node_dim: int = 512,
        entity_emb_dim: int = 0,
        d_a: int = 256,
        max_align_nodes: int = 128,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.lambda_disc = lambda_disc
        self.mu_align = mu_align
        self.nu_mask = nu_mask
        self.neg_ratio = neg_ratio
        self.temperature = temperature
        self.mask_temperature = mask_temperature
        self.max_align_nodes = max_align_nodes

        # Projection layers for L_align (Def 6.9):
        #   g_i = proj_g(h_i)       — backbone hidden state -> alignment space
        #   c_i = proj_c(text_emb_i) — raw text embedding -> alignment space
        # We use the backbone hidden state h instead of raw manifold
        # coordinates x_1 because entity embeddings initialise near the
        # manifold origin and remain directionally collapsed (cosine
        # sim > 0.99) during early training, making InfoNCE degenerate
        # (loss = log(M)).  h has rich inter-sample variance from the
        # start thanks to diverse text cross-attention and noise inputs.
        self.proj_g: nn.Sequential | None
        self.proj_c: nn.Sequential | None
        if mu_align > 0 and input_text_dim > 0:
            # Bias-free projections: randomly-initialised bias creates a
            # large shared offset that, after L2-normalisation in InfoNCE,
            # collapses all embeddings to cos-sim > 0.99 and makes the
            # contrastive loss degenerate (loss = log M).  Without bias
            # the MLP preserves input diversity (cos-sim stays ~0.17).
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
        else:
            self.proj_g = None
            self.proj_c = None

        # Projection head for L_mask: h_masked -> entity embedding space.
        self.proj_mask: nn.Sequential | None
        if nu_mask > 0 and entity_emb_dim > 0:
            self.proj_mask = nn.Sequential(
                nn.Linear(node_dim, node_dim, bias=False),
                nn.GELU(),
                nn.Linear(node_dim, entity_emb_dim, bias=False),
            )
        else:
            self.proj_mask = None

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
        true_entity_emb: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute combined loss.

        Args:
            V_hat: Predicted vector field, shape ``(B, N, D)``.
            u_t: Target vector field, shape ``(B, N, D)``.
            x_t: Interpolated manifold coords, shape ``(B, N, D)``.
            P_hat: Predicted edge logits, shape ``(B, N, N, K)``.
            E_1: Target edge types, shape ``(B, N, N, K)``.
            node_mask: Bool mask, shape ``(B, N)``.
            h: Backbone hidden states for L_align/L_mask, shape
                ``(B, N, node_dim)``.  None to skip L_align and L_mask.
            node_text: Raw text embeddings for L_align, shape ``(B, N, d_c)``.
                None to skip L_align.
            mask_type: Node type labels, shape ``(B, N)``.
                0=real, 1=masked, -1=virtual.  None to skip L_mask.
            true_entity_emb: Original (pre-mask) entity embeddings for
                masked nodes, shape ``(B, N, D_emb)``.  Only needed when
                mask_type is provided.

        Returns:
            Tuple of (total_loss, metrics_dict).
            metrics_dict contains individual loss components.
        """
        from riemannfm.data.collator import MASK_MASKED

        # L_cont: continuous flow matching loss.
        # Exclude masked nodes — their x_1 was replaced, so u_t is invalid.
        cont_mask = node_mask & (mask_type != MASK_MASKED) if mask_type is not None else node_mask
        l_cont = continuous_flow_loss(
            self.manifold, V_hat, u_t, x_t, cont_mask,
        )

        # L_disc: discrete flow matching loss (Def 6.8) with negative sampling.
        # Masked nodes keep their real edges, so L_disc uses full node_mask.
        l_disc = discrete_flow_loss(
            P_hat, E_1, node_mask,
            neg_ratio=self.neg_ratio,
        )

        # L_align: contrastive alignment loss (Def 6.9).
        # Only unmasked real nodes participate (masked nodes have no text).
        if (
            self.mu_align > 0
            and self.proj_g is not None
            and self.proj_c is not None
            and h is not None
            and node_text is not None
            and node_text.shape[-1] > 0
        ):
            B, N, _ = h.shape
            align_mask = node_mask & (mask_type != MASK_MASKED) if mask_type is not None else node_mask
            mask_flat = align_mask.reshape(-1)
            valid_idx = mask_flat.nonzero(as_tuple=True)[0]

            # Subsample to cap contrastive matrix size.
            if valid_idx.numel() > self.max_align_nodes:
                perm = torch.randperm(
                    valid_idx.numel(), device=valid_idx.device,
                )[:self.max_align_nodes]
                valid_idx = valid_idx[perm]

            if valid_idx.numel() < 2:
                l_align = torch.tensor(0.0, device=h.device, dtype=h.dtype)
            else:
                h_valid = h.reshape(B * N, -1)[valid_idx]  # (M, node_dim)
                t_valid = node_text.reshape(B * N, -1)[valid_idx]  # (M, d_c)
                g_valid = self.proj_g(h_valid)  # (M, d_a)
                c_valid = self.proj_c(t_valid)  # (M, d_a)
                l_align = _contrastive_loss_from_pairs(
                    g_valid, c_valid, self.temperature,
                )
        else:
            l_align = torch.tensor(0.0, device=V_hat.device, dtype=V_hat.dtype)

        # L_mask: masked node prediction loss (in-batch contrastive).
        # Each masked node's proj_mask(h) must match its true entity
        # embedding among all other masked nodes' embeddings in the batch.
        if (
            self.nu_mask > 0
            and self.proj_mask is not None
            and h is not None
            and mask_type is not None
            and true_entity_emb is not None
        ):
            B, N, _ = h.shape
            masked_flat = (mask_type == MASK_MASKED).reshape(-1)
            masked_idx = masked_flat.nonzero(as_tuple=True)[0]

            if masked_idx.numel() < 2:
                l_mask = torch.tensor(0.0, device=h.device, dtype=h.dtype)
            else:
                h_masked = h.reshape(B * N, -1)[masked_idx]       # (M, node_dim)
                true_emb = true_entity_emb.reshape(B * N, -1)[masked_idx]  # (M, D_emb)
                l_mask = masked_node_loss(
                    h_masked, self.proj_mask, true_emb,
                    temperature=self.mask_temperature,
                )
        else:
            l_mask = torch.tensor(0.0, device=V_hat.device, dtype=V_hat.dtype)

        # Total loss.
        total = (
            l_cont
            + self.lambda_disc * l_disc
            + self.mu_align * l_align
            + self.nu_mask * l_mask
        )

        metrics = {
            "loss/total": total.detach(),
            "loss/cont": l_cont.detach(),
            "loss/disc": l_disc.detach(),
            "loss/align": l_align.detach(),
            "loss/mask": l_mask.detach(),
        }

        return total, metrics
