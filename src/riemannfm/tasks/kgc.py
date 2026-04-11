"""KGC task heads: link prediction and relation prediction.

Link prediction scores (h, r, t) triple plausibility.
Relation prediction classifies the relation given (h, t).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


class RiemannFMKGCLinkHead(nn.Module):
    """Score (h, r, t) triples for link prediction.

    Two scoring modes controlled by ``scoring``:

    - ``"manifold_dist"``: Translate head by a learned relation tangent
      vector via exp_map, then compute negative geodesic distance to tail.
      ``score = -d_M(exp_h(r_vec), t)``

    - ``"bilinear"``: Diagonal bilinear scoring in ambient space.
      ``score = sum(h * diag(r) * t)``

    Args:
        manifold: Product manifold for geometric operations.
        ambient_dim: Product manifold ambient dimension D.
        num_relations: Number of relation types K.
        scoring: Scoring mode (``"manifold_dist"`` or ``"bilinear"``).
    """

    def __init__(
        self,
        manifold: RiemannFMProductManifold,
        ambient_dim: int,
        num_relations: int,
        scoring: str = "manifold_dist",
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.scoring = scoring

        # Learnable relation vectors in tangent/ambient space.
        self.rel_emb = nn.Embedding(num_relations, ambient_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(
        self,
        h: Tensor,
        r: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Score a batch of (h, r, t) triples.

        Args:
            h: Head entity manifold coordinates, shape ``(B, D)``.
            r: Relation indices, shape ``(B,)``.
            t: Tail entity manifold coordinates, shape ``(B, D)``.

        Returns:
            Scores, shape ``(B,)``. Higher = more plausible.
        """
        r_vec = self.rel_emb(r)  # (B, D)

        if self.scoring == "manifold_dist":
            return self._score_manifold_dist(h, r_vec, t)
        if self.scoring == "bilinear":
            return self._score_bilinear(h, r_vec, t)
        msg = f"Unknown scoring mode: {self.scoring}"
        raise ValueError(msg)

    def _score_manifold_dist(
        self, h: Tensor, r_vec: Tensor, t: Tensor,
    ) -> Tensor:
        """Manifold distance scoring: -d(exp_h(r), t)."""
        # Project r_vec onto tangent space at h.
        r_tangent = self.manifold.proj_tangent(h, r_vec)
        # Translate head: h' = exp_h(r_tangent).
        h_translated = self.manifold.exp_map(h, r_tangent)
        h_translated = self.manifold.proj_manifold(h_translated)
        # Negative geodesic distance as score.
        return -self.manifold.dist(h_translated, t)

    def _score_bilinear(
        self, h: Tensor, r_vec: Tensor, t: Tensor,
    ) -> Tensor:
        """Diagonal bilinear scoring: sum(h * r * t)."""
        result: Tensor = (h * r_vec * t).sum(dim=-1)
        return result

    def score_all_tails(
        self,
        h: Tensor,
        r: Tensor,
        all_entity_emb: Tensor,
    ) -> Tensor:
        """Score head against all possible tail entities.

        Used during evaluation for ranking all candidates.

        Args:
            h: Head coordinates, shape ``(B, D)``.
            r: Relation indices, shape ``(B,)``.
            all_entity_emb: All entity coordinates, shape ``(E, D)``.

        Returns:
            Scores, shape ``(B, E)``.
        """
        r_vec = self.rel_emb(r)  # (B, D)

        if self.scoring == "manifold_dist":
            r_tangent = self.manifold.proj_tangent(h, r_vec)
            h_translated = self.manifold.exp_map(h, r_tangent)
            h_translated = self.manifold.proj_manifold(h_translated)
            # Batch distance: (B, 1, D) vs (1, E, D) -> (B, E)
            return -self._batch_dist(h_translated, all_entity_emb)
        # bilinear: (B, D) * (B, D) -> (B, D), then (B, D) @ (E, D).T -> (B, E)
        hr = h * r_vec
        result: Tensor = hr @ all_entity_emb.t()
        return result

    def score_all_heads(
        self,
        t: Tensor,
        r: Tensor,
        all_entity_emb: Tensor,
    ) -> Tensor:
        """Score tail against all possible head entities.

        Args:
            t: Tail coordinates, shape ``(B, D)``.
            r: Relation indices, shape ``(B,)``.
            all_entity_emb: All entity coordinates, shape ``(E, D)``.

        Returns:
            Scores, shape ``(B, E)``.
        """
        r_vec = self.rel_emb(r)  # (B, D)
        E = all_entity_emb.shape[0]

        if self.scoring == "manifold_dist":
            # For head prediction: find h such that exp_h(r) ≈ t.
            # Approximate: score = -d(exp_{h_cand}(r), t) for each candidate.
            # Expand: (E, D) heads, (B, D) r_vec -> need (B, E) scores.
            B = t.shape[0]
            scores = torch.empty(B, E, device=t.device, dtype=t.dtype)
            # Process in chunks to avoid OOM on large entity sets.
            chunk_size = 4096
            for start in range(0, E, chunk_size):
                end = min(start + chunk_size, E)
                h_chunk = all_entity_emb[start:end]  # (C, D)
                # Expand r_vec: (B, 1, D) for each candidate
                h_exp = h_chunk.unsqueeze(0).expand(B, -1, -1)  # (B, C, D)
                r_exp = r_vec.unsqueeze(1).expand(-1, end - start, -1)  # (B, C, D)
                r_tan = self.manifold.proj_tangent(
                    h_exp.reshape(-1, h_exp.shape[-1]),
                    r_exp.reshape(-1, r_exp.shape[-1]),
                ).reshape(B, -1, h_exp.shape[-1])
                h_trans = self.manifold.exp_map(
                    h_exp.reshape(-1, h_exp.shape[-1]),
                    r_tan.reshape(-1, r_tan.shape[-1]),
                ).reshape(B, -1, h_exp.shape[-1])
                h_trans = self.manifold.proj_manifold(
                    h_trans.reshape(-1, h_trans.shape[-1]),
                ).reshape(B, -1, h_trans.shape[-1])
                t_exp = t.unsqueeze(1).expand(-1, end - start, -1)
                scores[:, start:end] = -self.manifold.dist(
                    h_trans.reshape(-1, h_trans.shape[-1]),
                    t_exp.reshape(-1, t_exp.shape[-1]),
                ).reshape(B, -1)
            return scores
        # bilinear: score = h_cand * r * t for all h_cand
        rt = r_vec * t  # (B, D)
        result: Tensor = rt @ all_entity_emb.t()
        return result

    def _batch_dist(
        self, x: Tensor, y_all: Tensor,
    ) -> Tensor:
        """Compute distance from x to each row in y_all.

        Args:
            x: shape ``(B, D)``.
            y_all: shape ``(E, D)``.

        Returns:
            Distances, shape ``(B, E)``.
        """
        B = x.shape[0]
        E = y_all.shape[0]
        dists = torch.empty(B, E, device=x.device, dtype=x.dtype)
        chunk_size = 4096
        for start in range(0, E, chunk_size):
            end = min(start + chunk_size, E)
            y_chunk = y_all[start:end]  # (C, D)
            x_exp = x.unsqueeze(1).expand(-1, end - start, -1)  # (B, C, D)
            y_exp = y_chunk.unsqueeze(0).expand(B, -1, -1)  # (B, C, D)
            dists[:, start:end] = self.manifold.dist(
                x_exp.reshape(-1, x.shape[-1]),
                y_exp.reshape(-1, x.shape[-1]),
            ).reshape(B, -1)
        return dists


class RiemannFMKGCRelHead(nn.Module):
    """Classify relation type given (h, t) entity pair.

    Architecture: MLP([h; t; h*t]) -> (K,) logits.

    Args:
        ambient_dim: Entity embedding dimension D.
        num_relations: Number of relation types K.
        hidden_dim: MLP hidden dimension.
    """

    def __init__(
        self,
        ambient_dim: int,
        num_relations: int,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * ambient_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_relations),
        )

    def forward(self, h: Tensor, t: Tensor) -> Tensor:
        """Predict relation logits given head and tail embeddings.

        Args:
            h: Head entity coordinates, shape ``(B, D)``.
            t: Tail entity coordinates, shape ``(B, D)``.

        Returns:
            Logits over K relations, shape ``(B, K)``.
        """
        features = torch.cat([h, t, h * t], dim=-1)  # (B, 3D)
        result: Tensor = self.mlp(features)
        return result
