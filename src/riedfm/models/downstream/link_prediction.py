"""Link prediction head using manifold-aware scoring.

Implements a TransE-style scoring function on the product manifold:
    score(h, r, t) = -dist(exp_map(h_embed, r_embed), t_embed)

Entity embeddings live on the product manifold; relation embeddings are
tangent vectors that translate head entities toward tail entities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from riedfm.manifolds.product import RieDFMProductManifold


class RieDFMLinkPredictionHead(nn.Module):
    """Manifold-aware link prediction head.

    Args:
        manifold: Product manifold for entity embeddings.
        num_entities: Total number of entities in the KG.
        num_relations: Total number of relation types.
        gamma: Margin/temperature for scoring.
    """

    def __init__(
        self,
        manifold: RieDFMProductManifold,
        num_entities: int,
        num_relations: int,
        gamma: float = 12.0,
    ):
        super().__init__()
        self.manifold = manifold
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.gamma = gamma

        total_dim = manifold.total_dim

        # Entity embeddings: initialized on the manifold
        self.entity_embeds = nn.Embedding(num_entities, total_dim)
        # Relation embeddings: tangent vectors (translations)
        self.relation_embeds = nn.Embedding(num_relations, total_dim)

        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embeddings on the manifold."""
        with torch.no_grad():
            # Sample entities uniformly on the manifold
            uniform = self.manifold.sample_uniform((self.num_entities,), self.entity_embeds.weight.device)
            self.entity_embeds.weight.copy_(uniform)
            # Relations as small tangent vectors
            nn.init.xavier_uniform_(self.relation_embeds.weight, gain=0.1)

    def project_embeddings(self):
        """Re-project entity embeddings onto the manifold after gradient update."""
        with torch.no_grad():
            x = self.entity_embeds.weight
            x_h, x_s, x_e = self.manifold.split(x)
            from riedfm.utils.manifold_utils import project_to_lorentz, project_to_sphere

            x_h = project_to_lorentz(x_h)
            x_s = project_to_sphere(x_s)
            projected = self.manifold.combine(x_h, x_s, x_e)
            self.entity_embeds.weight.copy_(projected)

    def _translate(self, h_embed: Tensor, r_embed: Tensor) -> Tensor:
        """Translate head embedding by relation vector on the manifold.

        Uses exponential map: translated = exp_map(h, proj_tangent(h, r))
        """
        r_tangent = self.manifold.proj_tangent(h_embed, r_embed)
        return self.manifold.exp_map(h_embed, r_tangent)

    def score(self, h_ids: Tensor, r_ids: Tensor, t_ids: Tensor) -> Tensor:
        """Score triples.

        Args:
            h_ids: Head entity IDs, shape (B,).
            r_ids: Relation IDs, shape (B,).
            t_ids: Tail entity IDs, shape (B,).

        Returns:
            Scores (higher = more likely), shape (B,).
        """
        h = self.entity_embeds(h_ids)
        r = self.relation_embeds(r_ids)
        t = self.entity_embeds(t_ids)

        h_translated = self._translate(h, r)
        dist = self.manifold.dist(h_translated, t)
        return self.gamma - dist

    def score_all_tails(self, h_ids: Tensor, r_ids: Tensor) -> Tensor:
        """Score all entities as potential tails for given (h, r) pairs.

        Args:
            h_ids: Head entity IDs, shape (B,).
            r_ids: Relation IDs, shape (B,).

        Returns:
            Scores for all entities, shape (B, num_entities).
        """
        h = self.entity_embeds(h_ids)  # (B, D)
        r = self.relation_embeds(r_ids)  # (B, D)
        h_translated = self._translate(h, r)  # (B, D)

        all_entities = self.entity_embeds.weight  # (E, D)
        # Compute distances: (B, E)
        # Expand for pairwise: (B, 1, D) vs (1, E, D)
        h_exp = h_translated.unsqueeze(1)  # (B, 1, D)
        e_exp = all_entities.unsqueeze(0)  # (1, E, D)
        # Chunked distance computation for memory efficiency
        B = h_ids.shape[0]
        E = self.num_entities
        scores = torch.empty(B, E, device=h_ids.device)

        chunk_size = min(E, 4096)
        for start in range(0, E, chunk_size):
            end = min(start + chunk_size, E)
            chunk = e_exp[:, start:end]  # (1, chunk, D)
            # Flatten for dist computation
            h_flat = h_exp.expand(-1, end - start, -1).reshape(-1, h_exp.shape[-1])
            e_flat = chunk.expand(B, -1, -1).reshape(-1, chunk.shape[-1])
            dist_flat = self.manifold.dist(h_flat, e_flat)
            scores[:, start:end] = self.gamma - dist_flat.reshape(B, -1)

        return scores

    def score_all_heads(self, t_ids: Tensor, r_ids: Tensor) -> Tensor:
        """Score all entities as potential heads for given (r, t) pairs.

        Args:
            t_ids: Tail entity IDs, shape (B,).
            r_ids: Relation IDs, shape (B,).

        Returns:
            Scores for all entities, shape (B, num_entities).
        """
        t = self.entity_embeds(t_ids)  # (B, D)
        r = self.relation_embeds(r_ids)  # (B, D)
        all_entities = self.entity_embeds.weight  # (E, D)

        B = t_ids.shape[0]
        E = self.num_entities
        scores = torch.empty(B, E, device=t_ids.device)

        chunk_size = min(E, 4096)
        for start in range(0, E, chunk_size):
            end = min(start + chunk_size, E)
            # Translate each candidate head by r
            h_chunk = all_entities[start:end].unsqueeze(0).expand(B, -1, -1)  # (B, chunk, D)
            r_exp = r.unsqueeze(1).expand(-1, end - start, -1)  # (B, chunk, D)
            h_flat = h_chunk.reshape(-1, h_chunk.shape[-1])
            r_flat = r_exp.reshape(-1, r_exp.shape[-1])
            translated = self._translate(h_flat, r_flat)  # (B*chunk, D)
            t_flat = t.unsqueeze(1).expand(-1, end - start, -1).reshape(-1, t.shape[-1])
            dist_flat = self.manifold.dist(translated, t_flat)
            scores[:, start:end] = self.gamma - dist_flat.reshape(B, -1)

        return scores

    def link_prediction_loss(
        self,
        h_ids: Tensor,
        r_ids: Tensor,
        t_ids: Tensor,
    ) -> Tensor:
        """Compute cross-entropy link prediction loss (both directions).

        Args:
            h_ids: Head entity IDs, shape (B,).
            r_ids: Relation IDs, shape (B,).
            t_ids: Tail entity IDs, shape (B,).

        Returns:
            Scalar loss.
        """
        # Tail prediction
        tail_scores = self.score_all_tails(h_ids, r_ids)  # (B, E)
        tail_loss = F.cross_entropy(tail_scores, t_ids)

        # Head prediction
        head_scores = self.score_all_heads(t_ids, r_ids)  # (B, E)
        head_loss = F.cross_entropy(head_scores, h_ids)

        return (tail_loss + head_loss) / 2
