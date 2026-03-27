"""Hierarchical subgraph sampling from knowledge graphs.

Implements the sampling strategies described in the paper:
1. Random seed entity selection
2. k-hop neighborhood expansion
3. Importance sampling (high-degree nodes, rare relations)
4. Fan-out sampling for one-to-many relations
"""

import random
from collections import defaultdict

import torch
from torch import Tensor


class SubgraphSampler:
    """Hierarchical subgraph sampler for knowledge graphs.

    Given a full KG represented as a list of triples (h, r, t),
    samples training subgraphs with configurable strategies.

    Args:
        triples: List of (head_id, relation_id, tail_id) tuples.
        max_nodes: Maximum number of nodes per subgraph.
        max_hops: Maximum BFS hops from seed.
        importance_sample: Whether to use importance sampling.
    """

    def __init__(
        self,
        triples: list[tuple[int, int, int]],
        max_nodes: int = 256,
        max_hops: int = 3,
        importance_sample: bool = True,
    ):
        self.max_nodes = max_nodes
        self.max_hops = max_hops
        self.importance_sample = importance_sample

        # Build adjacency structures
        self.adj: dict[int, list[tuple[int, int]]] = defaultdict(list)  # node -> [(neighbor, relation)]
        self.node_degree: dict[int, int] = defaultdict(int)
        self.relation_count: dict[int, int] = defaultdict(int)
        self.all_nodes: list[int] = []

        for h, r, t in triples:
            self.adj[h].append((t, r))
            self.adj[t].append((h, r))
            self.node_degree[h] += 1
            self.node_degree[t] += 1
            self.relation_count[r] += 1

        self.all_nodes = list(self.adj.keys())

        # Precompute inverse frequency for rare relation sampling
        max_count = max(self.relation_count.values()) if self.relation_count else 1
        self.relation_weight = {r: max_count / c for r, c in self.relation_count.items()}

    def sample(self) -> tuple[list[int], list[tuple[int, int, int]]]:
        """Sample a subgraph.

        Returns:
            (node_ids, subgraph_triples): List of node IDs and list of
            (local_head, relation, local_tail) triples within the subgraph.
        """
        # 1. Select seed entity
        seed = random.choice(self.all_nodes)

        # 2. BFS expansion
        visited = {seed}
        frontier = [seed]

        for hop in range(self.max_hops):
            if len(visited) >= self.max_nodes:
                break
            next_frontier = []
            for node in frontier:
                neighbors = self.adj[node]
                if self.importance_sample:
                    # Weight by inverse relation frequency (prefer rare relations)
                    weights = [self.relation_weight.get(r, 1.0) for _, r in neighbors]
                    total_w = sum(weights)
                    weights = [w / total_w for w in weights]
                    # Sample subset of neighbors
                    k = min(len(neighbors), self.max_nodes - len(visited))
                    if k > 0 and len(neighbors) > 0:
                        indices = random.choices(range(len(neighbors)), weights=weights, k=k)
                        selected = [neighbors[i] for i in set(indices)]
                    else:
                        selected = []
                else:
                    selected = neighbors[: self.max_nodes - len(visited)]

                for neighbor, rel in selected:
                    if neighbor not in visited and len(visited) < self.max_nodes:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)

            frontier = next_frontier

        # 3. Extract subgraph triples
        node_list = list(visited)
        node_set = visited
        node_to_local = {n: i for i, n in enumerate(node_list)}

        subgraph_triples = []
        for node in node_list:
            for neighbor, rel in self.adj[node]:
                if neighbor in node_set:
                    local_h = node_to_local[node]
                    local_t = node_to_local[neighbor]
                    subgraph_triples.append((local_h, rel, local_t))

        return node_list, subgraph_triples

    def sample_fanout(self, head: int, relation: int) -> tuple[list[int], list[tuple[int, int, int]]]:
        """Sample a complete fan-out subgraph for one-to-many relations.

        Collects all tails for a given (head, relation) pair.

        Args:
            head: Head entity ID.
            relation: Relation type ID.

        Returns:
            (node_ids, subgraph_triples) for the fan-out subgraph.
        """
        tails = [n for n, r in self.adj[head] if r == relation]
        nodes = [head] + tails[: self.max_nodes - 1]
        node_to_local = {n: i for i, n in enumerate(nodes)}

        triples = [(0, relation, node_to_local[t]) for t in tails if t in node_to_local]
        return nodes, triples
