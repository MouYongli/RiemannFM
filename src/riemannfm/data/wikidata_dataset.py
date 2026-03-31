"""WikiData subgraph dataset for pretraining.

Loads preprocessed WikiData subgraphs and provides them as RiemannFMGraphData objects
for training. Supports hierarchical sampling and text feature extraction.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from riemannfm.data.download import embedding_filename, encoder_slug
from riemannfm.data.graph_data import RiemannFMGraphData
from riemannfm.data.subgraph_sampler import RiemannFMSubgraphSampler
from riemannfm.manifolds.product import RiemannFMProductManifold


class RiemannFMWikiDataDataset(Dataset):
    """WikiData knowledge graph dataset with on-the-fly subgraph sampling.

    Expects data in the following layout:
    - raw/entities.json: {entity_id: {"label": str, "description": str, "depth": int}}
    - raw/{split}_triples.txt: tab-separated (head_id, relation_id, tail_id)
    - processed/text_embeddings/entity_emb_{encoder}_{dim}.pt: precomputed embeddings

    Args:
        data_dir: Path to preprocessed WikiData directory.
        manifold: Product manifold for coordinate representation.
        max_nodes: Maximum nodes per subgraph.
        max_hops: Maximum BFS hops for sampling.
        num_edge_types: Number of relation types.
        split: Dataset split ("train", "val", "test").
        text_encoder: Text encoder model name or slug for loading embeddings.
        dim_text_emb: Dimension of text embeddings.
    """

    def __init__(
        self,
        data_dir: str,
        manifold: RiemannFMProductManifold,
        max_nodes: int = 256,
        max_hops: int = 3,
        num_edge_types: int = 1200,
        split: str = "train",
        text_encoder: str | None = None,
        dim_text_emb: int = 0,
    ):
        self.data_dir = Path(data_dir)
        self.manifold = manifold
        self.max_nodes = max_nodes
        self.num_edge_types = num_edge_types
        self.split = split
        self.text_encoder = text_encoder
        self.dim_text_emb = dim_text_emb

        # Load data (lazy — files may not exist yet during development)
        self.triples: list[tuple[int, int, int]] = []
        self.entity_info: dict[int, dict] = {}
        self.embeddings: torch.Tensor | None = None
        self.sampler: RiemannFMSubgraphSampler | None = None

        self._load_data()

    def _load_data(self):
        """Load data files from raw/ and processed/ subdirectories."""
        raw_dir = self.data_dir / "raw"
        triples_path = raw_dir / f"{self.split}_triples.txt"
        entities_path = raw_dir / "entities.json"

        if triples_path.exists():
            self.triples = []
            with open(triples_path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 3:
                        self.triples.append((int(parts[0]), int(parts[1]), int(parts[2])))
            self.sampler = RiemannFMSubgraphSampler(
                triples=self.triples,
                max_nodes=self.max_nodes,
            )

        if entities_path.exists():
            with open(entities_path) as f:
                self.entity_info = {int(k): v for k, v in json.load(f).items()}

        # Load precomputed text embeddings from processed/
        if self.text_encoder and self.dim_text_emb > 0:
            key = encoder_slug(self.text_encoder)
            emb_path = self.data_dir / "processed" / "text_embeddings" / embedding_filename(key, self.dim_text_emb)
            if emb_path.exists():
                self.embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)

    def __len__(self) -> int:
        # For on-the-fly sampling, return a large number (epoch-based training)
        return max(len(self.triples) // 10, 10000)

    def __getitem__(self, idx: int) -> RiemannFMGraphData:
        """Sample a subgraph and return as RiemannFMGraphData."""
        if self.sampler is None:
            # Return dummy data for development
            return self._dummy_graph()

        node_ids, subgraph_triples, depth_map = self.sampler.sample()
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

        # Get hierarchy depths from sampler depth_map or entity info
        depth = torch.zeros(N, dtype=torch.long)
        for local_idx, hop_dist in depth_map.items():
            depth[local_idx] = hop_dist
        # Override with entity-level depth info if available
        for i, nid in enumerate(node_ids):
            if nid in self.entity_info:
                entity_depth = self.entity_info[nid].get("depth", None)
                if entity_depth is not None:
                    depth[i] = entity_depth

        # Get entity IDs and labels
        entity_ids = [str(nid) for nid in node_ids]
        node_labels = [self.entity_info.get(nid, {}).get("label", f"Entity_{nid}") for nid in node_ids]

        return RiemannFMGraphData(
            x=x,
            edge_types=edge_types,
            num_nodes=N,
            depth=depth,
            entity_ids=entity_ids,
            node_labels=node_labels,
        )

    def _dummy_graph(self) -> RiemannFMGraphData:
        """Generate a dummy graph for development/testing."""
        N = min(8, self.max_nodes)
        x = self.manifold.sample_uniform((N,), torch.device("cpu"))
        edge_types = torch.zeros(N, N, dtype=torch.long)
        # Add a few random edges
        for _ in range(N):
            i, j = torch.randint(0, N, (2,))
            if i != j:
                edge_types[i, j] = torch.randint(1, min(self.num_edge_types, 10), (1,)).item()
        return RiemannFMGraphData(x=x, edge_types=edge_types, num_nodes=N)
