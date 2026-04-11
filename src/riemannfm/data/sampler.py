"""Hierarchical subgraph sampling from knowledge graphs.

Implements BFS-based neighborhood expansion with multi-hot edge construction
aligned with math spec Definition 3.4:
    E ∈ {0,1}^{NxNxK},  E[i,j,k] = 1 iff (v_i, r_k, v_j) ∈ E_G
"""

import random
from collections import defaultdict

import numpy as np
import torch
from torch import Tensor


class RiemannFMSubgraphSampler:
    """Hierarchical subgraph sampler for knowledge graphs.

    Given a full KG represented as integer triples (h, r, t),
    samples training subgraphs via BFS and builds multi-hot edge tensors.

    Args:
        triples: Integer triples tensor, shape (num_triples, 3).
            Each row is [head_entity_id, relation_id, tail_entity_id].
        num_edge_types: Total number of relation types K.
        max_nodes: Maximum number of nodes per subgraph (N_max).
        max_hops: Maximum BFS hops from seed.
        importance_sample: Whether to use inverse-frequency importance sampling.
    """

    def __init__(
        self,
        triples: Tensor,
        num_edge_types: int,
        max_nodes: int = 64,
        max_hops: int = 2,
        importance_sample: bool = True,
    ):
        self.num_edge_types = num_edge_types
        self.max_nodes = max_nodes
        self.max_hops = max_hops
        self.importance_sample = importance_sample

        # Convert to numpy once — avoids slow torch tensor indexing in the loop.
        h_col = triples[:, 0].numpy()
        r_col = triples[:, 1].numpy()
        t_col = triples[:, 2].numpy()

        # Build adjacency and directed edge index in a single pass.
        self.adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self._edge_index: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.relation_count: dict[int, int] = defaultdict(int)

        for i in range(len(h_col)):
            h, r, t = int(h_col[i]), int(r_col[i]), int(t_col[i])
            self.adj[h].append((t, r))
            self.adj[t].append((h, r))  # undirected for BFS traversal
            self._edge_index[(h, t)].append(r)
            self.relation_count[r] += 1

        self.all_nodes = list(self.adj.keys())

        # Precompute inverse frequency for rare relation sampling
        if self.relation_count:
            max_count = max(self.relation_count.values())
            self.relation_weight = {r: max_count / c for r, c in self.relation_count.items()}
        else:
            self.relation_weight = {}

    def sample(self) -> tuple[list[int], Tensor]:
        """Sample a subgraph from a random seed.

        Returns:
            (node_ids, edge_types):
                - node_ids: List of global entity IDs in the subgraph.
                - edge_types: Multi-hot tensor (N, N, K) per Definition 3.4.
        """
        seed = random.choice(self.all_nodes)
        return self.sample_from_seed(seed)

    def sample_from_seed(self, seed: int) -> tuple[list[int], Tensor]:
        """Sample a subgraph starting from a specific seed node.

        Args:
            seed: Seed entity ID to start BFS from.

        Returns:
            (node_ids, edge_types):
                - node_ids: Global node IDs in the subgraph (length N ≤ max_nodes).
                - edge_types: Multi-hot tensor (N, N, K).
        """
        if seed not in self.adj:
            node_ids = [seed]
            edge_types = torch.zeros(1, 1, self.num_edge_types, dtype=torch.float32)
            return node_ids, edge_types

        # BFS expansion
        visited = {seed}
        frontier = [seed]

        for _hop in range(1, self.max_hops + 1):
            if len(visited) >= self.max_nodes:
                break
            next_frontier: list[int] = []
            for node in frontier:
                neighbors = self.adj[node]
                if self.importance_sample and neighbors:
                    w = np.array(
                        [self.relation_weight.get(r, 1.0) for _, r in neighbors],
                    )
                    w /= w.sum()
                    k = min(len(neighbors), self.max_nodes - len(visited))
                    if k > 0:
                        indices = np.random.choice(len(neighbors), size=k, replace=True, p=w)
                        selected = [neighbors[i] for i in set(indices.tolist())]
                    else:
                        selected = []
                else:
                    selected = neighbors[: self.max_nodes - len(visited)]

                for neighbor, _rel in selected:
                    if neighbor not in visited and len(visited) < self.max_nodes:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)

            frontier = next_frontier

        node_list = list(visited)
        return node_list, self._build_edge_types(node_list)

    def _build_edge_types(self, node_list: list[int]) -> Tensor:
        """Build multi-hot edge type tensor for a subgraph.

        For each directed triple (h, r, t) in the full KG where both h and t
        are in the subgraph, set E[local_h, local_t, r] = 1.

        This naturally supports multi-relational edges: if (i, r1, j) and
        (i, r2, j) both exist, then E[i,j,r1] = E[i,j,r2] = 1.

        Collects all (local_i, local_j, relation) triples and sets them
        in one vectorized ``index_put_`` call.

        Args:
            node_list: Global entity IDs in the subgraph.

        Returns:
            Multi-hot edge type tensor, shape (N, N, K).
        """
        N = len(node_list)
        K = self.num_edge_types
        edge_types = torch.zeros(N, N, K, dtype=torch.float32)

        # Map global node IDs to local indices for O(1) lookup.
        global_to_local = {nid: idx for idx, nid in enumerate(node_list)}

        # Collect all (local_i, local_j, r) in one pass over the subgraph.
        rows: list[int] = []
        cols: list[int] = []
        rels: list[int] = []
        for i, ni in enumerate(node_list):
            for nj, r_list in (
                (nj, self._edge_index.get((ni, nj)))
                for nj in node_list
                if (ni, nj) in self._edge_index
            ):
                local_j = global_to_local[nj]
                for r in r_list:  # type: ignore[union-attr]
                    if 0 <= r < K:
                        rows.append(i)
                        cols.append(local_j)
                        rels.append(r)

        if rows:
            idx = (
                torch.tensor(rows, dtype=torch.long),
                torch.tensor(cols, dtype=torch.long),
                torch.tensor(rels, dtype=torch.long),
            )
            edge_types.index_put_(idx, torch.tensor(1.0))

        return edge_types
