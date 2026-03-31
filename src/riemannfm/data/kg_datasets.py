"""Standard KG benchmark datasets: FB15k-237, WN18RR, CoDEx.

Provides dataset classes for pretraining, fine-tuning, and evaluation on
standard knowledge graph completion benchmarks.

Supports two modes:
- "subgraph": Returns RiemannFMGraphData objects (for pretraining / graph generation)
- "triple": Returns (h, r, t) dicts (for link prediction training / evaluation)
"""

import logging
import shutil
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

import torch
from torch.utils.data import Dataset

from riemannfm.data.download import embedding_filename, encoder_slug
from riemannfm.data.graph_data import RiemannFMGraphData
from riemannfm.data.subgraph_sampler import RiemannFMSubgraphSampler
from riemannfm.manifolds.product import RiemannFMProductManifold

logger = logging.getLogger(__name__)

# Known dataset download URLs
_DATASET_URLS: dict[str, str] = {
    "fb15k-237": "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237.tar.gz",
    "wn18rr": "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR.tar.gz",
}


class RiemannFMKGDataset(Dataset):
    """KG benchmark dataset with subgraph sampling and triple modes.

    Args:
        data_dir: Path to dataset directory.
        manifold: Product manifold (required for subgraph mode).
        split: "train", "valid", or "test".
        mode: "subgraph" (returns RiemannFMGraphData) or "triple" (returns dict).
        max_nodes: Maximum subgraph size (subgraph mode only).
        max_hops: BFS hops for subgraph sampling.
        auto_download: If True, attempt to download dataset if not found.
    """

    def __init__(
        self,
        data_dir: str,
        manifold: RiemannFMProductManifold | None = None,
        split: str = "train",
        mode: str = "triple",
        max_nodes: int = 64,
        max_hops: int = 2,
        auto_download: bool = True,
        text_encoder: str | None = None,
        dim_text_emb: int = 0,
    ):
        self.data_dir = Path(data_dir)
        self.manifold = manifold
        self.split = split
        self.mode = mode
        self.max_nodes = max_nodes
        self.max_hops = max_hops
        self.text_encoder = text_encoder
        self.dim_text_emb = dim_text_emb

        self.triples: list[tuple[int, int, int]] = []
        self.all_triples: list[tuple[int, int, int]] = []
        self.entity2id: dict[str, int] = {}
        self.relation2id: dict[str, int] = {}
        self.id2entity: dict[int, str] = {}
        self.id2relation: dict[int, str] = {}
        self.num_entities = 0
        self.num_relations = 0
        self.sampler: RiemannFMSubgraphSampler | None = None
        self.text_embeddings: torch.Tensor | None = None

        # True triples per (h, r) and (r, t) for filtered ranking
        self._hr2t: dict[tuple[int, int], set[int]] = defaultdict(set)
        self._rt2h: dict[tuple[int, int], set[int]] = defaultdict(set)

        if auto_download:
            self._download_if_missing()
        self._load_data()

    def _download_if_missing(self):
        """Download dataset files if not present."""
        raw_dir = self.data_dir / "raw"
        train_file = raw_dir / "train.txt"
        if train_file.exists():
            return

        dataset_name = self.data_dir.name.lower().replace("_", "-")
        url = _DATASET_URLS.get(dataset_name)
        if url is None:
            return

        logger.info(f"Downloading {dataset_name} to {raw_dir}...")
        raw_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "dataset.tar.gz"
            urlretrieve(url, str(archive_path))
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(tmpdir)

            for extracted_dir in Path(tmpdir).iterdir():
                if extracted_dir.is_dir() and extracted_dir.name != "__MACOSX":
                    for f in extracted_dir.iterdir():
                        if f.is_file():
                            shutil.copy2(f, raw_dir / f.name)
                    break

        logger.info(f"Downloaded {dataset_name} successfully.")

    def _load_data(self):
        """Load entity/relation mappings and triples from raw/ subdirectory."""
        raw_dir = self.data_dir / "raw"
        # Load entity mapping
        entity_file = raw_dir / "entities.dict"
        if entity_file.exists():
            with open(entity_file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        idx, name = int(parts[0]), parts[1]
                        self.entity2id[name] = idx
                        self.id2entity[idx] = name
            self.num_entities = len(self.entity2id)

        # Load relation mapping
        relation_file = raw_dir / "relations.dict"
        if relation_file.exists():
            with open(relation_file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        idx, name = int(parts[0]), parts[1]
                        self.relation2id[name] = idx
                        self.id2relation[idx] = name
            self.num_relations = len(self.relation2id)

        # Load triples for the requested split
        self.triples = self._load_triples(self.split)

        # Load ALL triples (train + valid + test) for graph structure and filtering
        self.all_triples = []
        for s in ["train", "valid", "test"]:
            self.all_triples.extend(self._load_triples(s))

        # Build filter sets for filtered ranking
        for h, r, t in self.all_triples:
            self._hr2t[(h, r)].add(t)
            self._rt2h[(r, t)].add(h)

        # Build sampler from all triples (full graph structure)
        if self.all_triples and self.mode == "subgraph":
            self.sampler = RiemannFMSubgraphSampler(
                triples=self.all_triples,
                max_nodes=self.max_nodes,
                max_hops=self.max_hops,
            )

        # Load precomputed text embeddings from processed/ if available
        if self.text_encoder and self.dim_text_emb > 0:
            key = encoder_slug(self.text_encoder)
            emb_path = self.data_dir / "processed" / "text_embeddings" / embedding_filename(key, self.dim_text_emb)
            if emb_path.exists():
                self.text_embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)
                logger.info(f"Loaded text embeddings ({self.text_embeddings.shape}) from {emb_path}")
            else:
                logger.debug(f"Text embeddings not found at {emb_path}")

    def _load_triples(self, split: str) -> list[tuple[int, int, int]]:
        """Load triples from a split file in raw/ subdirectory."""
        triples_file = self.data_dir / "raw" / f"{split}.txt"
        triples = []
        if triples_file.exists():
            with open(triples_file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 3:
                        h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                        triples.append((h, r, t))
        return triples

    def get_filter_masks(self, h: int, r: int, t: int) -> tuple[set[int], set[int]]:
        """Get filter sets for filtered ranking.

        Returns:
            (tail_filter, head_filter): Sets of valid entity IDs to exclude
            when ranking tails/heads (other correct answers).
        """
        tail_filter = self._hr2t.get((h, r), set()) - {t}
        head_filter = self._rt2h.get((r, t), set()) - {h}
        return tail_filter, head_filter

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> RiemannFMGraphData | dict:
        """Return a sample based on mode.

        mode="triple": Returns dict with head/relation/tail IDs.
        mode="subgraph": Returns RiemannFMGraphData with sampled subgraph.
        """
        h, r, t = self.triples[idx]

        if self.mode == "triple":
            return {"head": h, "relation": r, "tail": t}

        # Subgraph mode
        return self._build_subgraph(h, r, t)

    def _build_subgraph(self, h: int, r: int, t: int) -> RiemannFMGraphData:
        """Build a subgraph centered on the head entity of a triple."""
        if self.sampler is None or self.manifold is None:
            # Fallback: minimal 2-node graph
            N = 2
            x = torch.randn(N, 96)
            edge_types = torch.zeros(N, N, dtype=torch.long)
            edge_types[0, 1] = r
            return RiemannFMGraphData(x=x, edge_types=edge_types, num_nodes=N)

        node_ids, subgraph_triples, depth_map = self.sampler.sample_from_seed(h)
        N = len(node_ids)

        # Initialize coordinates on manifold
        x = self.manifold.sample_uniform((N,), torch.device("cpu"))

        # Build edge type matrix
        edge_types = torch.zeros(N, N, dtype=torch.long)
        for lh, lr, lt in subgraph_triples:
            # Relation types in benchmarks are 0-indexed; shift by 1 so 0 = no edge
            edge_types[lh, lt] = lr + 1

        # Build depth tensor from BFS depth map
        depth = torch.zeros(N, dtype=torch.long)
        for local_idx, hop_dist in depth_map.items():
            depth[local_idx] = hop_dist

        # Attach text embeddings if available
        text_input_ids = None
        if self.text_embeddings is not None:
            node_embs = self.text_embeddings[node_ids]  # (N, dim_text_emb)
            text_input_ids = node_embs

        return RiemannFMGraphData(
            x=x,
            edge_types=edge_types,
            num_nodes=N,
            depth=depth,
            text_input_ids=text_input_ids,
        )
