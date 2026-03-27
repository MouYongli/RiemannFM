"""Hierarchical subgraph sampling from knowledge graphs.

Implements the sampling strategies described in the paper:
1. Random seed entity selection or specific seed
2. k-hop neighborhood expansion with BFS depth tracking
3. Importance sampling (high-degree nodes, rare relations)
4. Fan-out sampling for one-to-many relations
"""

import random
from collections import defaultdict


class RieDFMSubgraphSampler:
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

    def sample(self) -> tuple[list[int], list[tuple[int, int, int]], dict[int, int]]:
        """Sample a subgraph from a random seed.

        Returns:
            (node_ids, subgraph_triples, depth_map):
                - node_ids: Global node IDs in the subgraph.
                - subgraph_triples: (local_head, relation, local_tail) triples.
                - depth_map: Mapping from local node index to BFS hop distance from seed.
        """
        seed = random.choice(self.all_nodes)
        return self.sample_from_seed(seed)

    def sample_from_seed(self, seed: int) -> tuple[list[int], list[tuple[int, int, int]], dict[int, int]]:
        """Sample a subgraph starting from a specific seed node.

        Args:
            seed: Seed entity ID to start BFS from.

        Returns:
            (node_ids, subgraph_triples, depth_map):
                - node_ids: Global node IDs in the subgraph.
                - subgraph_triples: (local_head, relation, local_tail) triples.
                - depth_map: Mapping from local node index to BFS hop distance from seed.
        """
        if seed not in self.adj:
            # Seed not in graph, return minimal subgraph
            return [seed], [], {0: 0}

        # BFS expansion with depth tracking
        visited = {seed}
        node_depth: dict[int, int] = {seed: 0}
        frontier = [seed]

        for hop in range(1, self.max_hops + 1):
            if len(visited) >= self.max_nodes:
                break
            next_frontier = []
            for node in frontier:
                neighbors = self.adj[node]
                if self.importance_sample:
                    weights = [self.relation_weight.get(r, 1.0) for _, r in neighbors]
                    total_w = sum(weights)
                    weights = [w / total_w for w in weights]
                    k = min(len(neighbors), self.max_nodes - len(visited))
                    if k > 0 and len(neighbors) > 0:
                        indices = random.choices(range(len(neighbors)), weights=weights, k=k)
                        selected = [neighbors[i] for i in set(indices)]
                    else:
                        selected = []
                else:
                    selected = neighbors[: self.max_nodes - len(visited)]

                for neighbor, _rel in selected:
                    if neighbor not in visited and len(visited) < self.max_nodes:
                        visited.add(neighbor)
                        node_depth[neighbor] = hop
                        next_frontier.append(neighbor)

            frontier = next_frontier

        # Extract subgraph triples with local indexing
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

        # Build depth_map with local indices
        depth_map = {node_to_local[n]: d for n, d in node_depth.items()}

        return node_list, subgraph_triples, depth_map

    def sample_fanout(self, head: int, relation: int) -> tuple[list[int], list[tuple[int, int, int]], dict[int, int]]:
        """Sample a complete fan-out subgraph for one-to-many relations.

        Args:
            head: Head entity ID.
            relation: Relation type ID.

        Returns:
            (node_ids, subgraph_triples, depth_map) for the fan-out subgraph.
        """
        tails = [n for n, r in self.adj[head] if r == relation]
        nodes = [head, *tails[: self.max_nodes - 1]]
        node_to_local = {n: i for i, n in enumerate(nodes)}

        triples = [(0, relation, node_to_local[t]) for t in tails if t in node_to_local]
        depth_map = {0: 0}
        for i in range(1, len(nodes)):
            depth_map[i] = 1
        return nodes, triples, depth_map
