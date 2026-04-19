"""Pretrain-only parameters for RiemannFM.

Bundles learnable state specific to pretraining. Fine-tune loads the
backbone checkpoint with ``strict=False`` and drops the entire
``pretrain_heads.*`` prefix.

Owns:
  - ``entity_emb``  : entity embedding table → manifold coordinates.
  - ``mask_emb``    : shared learnable text [MASK] token for text-mask
                      modality-masking mode (spec §9.3). Stage 2 will
                      consume it when the four-mode scheme is wired in.
  - ``W_p, W_p_c``  : optional relation-/text-alignment projections for
                      L_align^R (spec def 19.1). Instantiated only when
                      ``lambda_align_R > 0`` and text is available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


class RiemannFMPretrainHeads(nn.Module):
    """Pretrain-only parameters, discarded at fine-tune time.

    Args:
        manifold: Product manifold (for entity_emb origin seeding).
        num_entities: Entity vocabulary size.
        input_text_dim: Raw text embedding dimension ``d_c``. ``0``
            disables text-dependent heads.
        lambda_align_R: Relation-text alignment weight. Projections
            ``W_p / W_p_c`` are instantiated only when ``> 0`` and
            text is available.
        d_p: Relation-alignment projection dimension.
        rel_emb_dim: Relation embedding dimension ``d_r``. Needed to
            size ``W_p`` when relation alignment is active.
    """

    mask_emb: Tensor

    def __init__(
        self,
        manifold: RiemannFMProductManifold,
        num_entities: int,
        input_text_dim: int,
        lambda_align_R: float = 0.0,
        d_p: int = 128,
        rel_emb_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_text_dim = input_text_dim
        self.lambda_align_R = lambda_align_R

        # Entity embedding table → manifold coordinates (spec §10.5).
        # std=0.1 gives inter-node diversity around the origin.
        D = manifold.ambient_dim
        self.entity_emb = nn.Embedding(num_entities, D)
        nn.init.normal_(self.entity_emb.weight, std=0.1)
        with torch.no_grad():
            origin = manifold.origin(device=self.entity_emb.weight.device)
            self.entity_emb.weight.add_(origin)

        # Learnable [MASK] text vector for text-masked nodes (spec §9.3).
        # Kept even when text is disabled (1-dim placeholder) so
        # state_dict shape is stable across runs.
        self.mask_emb = nn.Parameter(torch.zeros(max(input_text_dim, 1)))
        if input_text_dim > 0:
            nn.init.normal_(self.mask_emb, std=0.02)

        # Optional L_align^R projections (spec def 19.1).
        self.W_p: nn.Linear | None
        self.W_p_c: nn.Linear | None
        if lambda_align_R > 0 and input_text_dim > 0:
            self.W_p = nn.Linear(rel_emb_dim, d_p, bias=False)
            self.W_p_c = nn.Linear(input_text_dim, d_p, bias=False)
        else:
            self.W_p = None
            self.W_p_c = None

    def project_relation_align(
        self, R: Tensor, C_R: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Project relation embeddings + text for L_align^R.

        Args:
            R: Relation embedding parameter, shape ``(K, rel_emb_dim)``.
            C_R: Relation text embeddings, shape ``(K, input_text_dim)``.

        Returns:
            ``(z_R, z_C)`` each of shape ``(K, d_p)``.
        """
        assert self.W_p is not None and self.W_p_c is not None, (
            "L_align^R heads not instantiated (lambda_align_R == 0 or "
            "input_text_dim == 0)"
        )
        return self.W_p(R), self.W_p_c(C_R.to(R.dtype))
