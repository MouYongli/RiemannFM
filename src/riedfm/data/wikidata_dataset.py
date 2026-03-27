"""WikiData subgraph dataset for pretraining.

Loads preprocessed WikiData subgraphs and provides them as RieDFMGraphData objects
for training. Supports hierarchical sampling and text feature extraction.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from riedfm.data.graph_data import RieDFMGraphData
from riedfm.data.subgraph_sampler import RieDFMSubgraphSampler
from riedfm.manifolds.product import RieDFMProductManifold


class RieDFMWikiDataDataset(Dataset):
    """WikiData knowledge graph dataset with on-the-fly subgraph sampling.

    Expects preprocessed data in the following format:
    - entities.json: {entity_id: {"label": str, "description": str, "depth": int}}
    - triples.txt: tab-separated (head_id, relation_id, tail_id) per line
    - embeddings.pt: precomputed entity embeddings on the product manifold

    Args:
        data_dir: Path to preprocessed WikiData directory.
        manifold: Product manifold for coordinate representation.
        max_nodes: Maximum nodes per subgraph.
        max_hops: Maximum BFS hops for sampling.
        num_edge_types: Number of relation types.
        split: Dataset split ("train", "val", "test").
    """

    def __init__(
        self,
        data_dir: str,
        manifold: RieDFMProductManifold,
        max_nodes: int = 256,
        max_hops: int = 3,
        num_edge_types: int = 1200,
        split: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.manifold = manifold
        self.max_nodes = max_nodes
        self.num_edge_types = num_edge_types
        self.split = split

        # Load data (lazy — files may not exist yet during development)
        self.triples: list[tuple[int, int, int]] = []
        self.entity_info: dict[int, dict] = {}
        self.embeddings: torch.Tensor | None = None
        self.sampler: RieDFMSubgraphSampler | None = None

        self._load_data()

    def _load_data(self):
        """Load preprocessed data files if they exist."""
        triples_path = self.data_dir / f"{self.split}_triples.txt"
        entities_path = self.data_dir / "entities.json"
        embeddings_path = self.data_dir / f"{self.split}_embeddings.pt"

        if triples_path.exists():
            self.triples = []
            with open(triples_path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 3:
                        self.triples.append((int(parts[0]), int(parts[1]), int(parts[2])))
            self.sampler = RieDFMSubgraphSampler(
                triples=self.triples,
                max_nodes=self.max_nodes,
            )

        if entities_path.exists():
            with open(entities_path) as f:
                self.entity_info = {int(k): v for k, v in json.load(f).items()}

        if embeddings_path.exists():
            self.embeddings = torch.load(embeddings_path, map_location="cpu", weights_only=True)

    def __len__(self) -> int:
        # For on-the-fly sampling, return a large number (epoch-based training)
        return max(len(self.triples) // 10, 10000)

    def __getitem__(self, idx: int) -> RieDFMGraphData:
        """Sample a subgraph and return as RieDFMGraphData."""
        if self.sampler is None:
            # Return dummy data for development
            return self._dummy_graph()

        node_ids, subgraph_triples = self.sampler.sample()
        N = len(node_ids)

        # Build node coordinates
        if self.embeddings is not None:
            x = self.embeddings[node_ids]
        else:
            # Random initialization on manifold
            x = self.manifold.sample_uniform((N,), torch.device("cpu"))

        # Build edge type matrix
        edge_types = torch.zeros(N, N, dtype=torch.long)
        for h, r, t in subgraph_triples:
            edge_types[h, t] = r

        # Get hierarchy depths
        depth = torch.zeros(N, dtype=torch.long)
        for i, nid in enumerate(node_ids):
            if nid in self.entity_info:
                depth[i] = self.entity_info[nid].get("depth", 0)

        # Get entity IDs and labels
        entity_ids = [str(nid) for nid in node_ids]
        node_labels = [self.entity_info.get(nid, {}).get("label", f"Entity_{nid}") for nid in node_ids]

        return RieDFMGraphData(
            x=x,
            edge_types=edge_types,
            num_nodes=N,
            depth=depth,
            entity_ids=entity_ids,
            node_labels=node_labels,
        )

    def _dummy_graph(self) -> RieDFMGraphData:
        """Generate a dummy graph for development/testing."""
        N = min(8, self.max_nodes)
        x = self.manifold.sample_uniform((N,), torch.device("cpu"))
        edge_types = torch.zeros(N, N, dtype=torch.long)
        # Add a few random edges
        for _ in range(N):
            i, j = torch.randint(0, N, (2,))
            if i != j:
                edge_types[i, j] = torch.randint(1, min(self.num_edge_types, 10), (1,)).item()
        return RieDFMGraphData(x=x, edge_types=edge_types, num_nodes=N)
