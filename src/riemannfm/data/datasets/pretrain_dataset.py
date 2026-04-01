"""KG subgraph dataset with on-the-fly sampling.

Serves both pretraining and KGC fine-tuning. Loads preprocessed integer
triples and precomputed text embeddings, samples subgraphs on-the-fly,
and returns RiemannFMGraphData objects aligned with the math spec
5-tuple (X, E, C_V, C_R, m).

Expected data layout:
    {data_dir}/processed/train_triples.pt       shape (num_triples, 3)
    {data_dir}/processed/val_triples.pt         shape (num_triples, 3)
    {data_dir}/processed/test_triples.pt        shape (num_triples, 3)
    {data_dir}/processed/entity2id.tsv          string_id → int
    {data_dir}/processed/relation2id.tsv        string_id → int
    {data_dir}/processed/text_embeddings/entity_emb_{encoder}_{dim}.pt
    {data_dir}/processed/text_embeddings/relation_emb_{encoder}_{dim}.pt
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from riemannfm.data.graph import RiemannFMGraphData
from riemannfm.data.pipeline.embed import embedding_filename, encoder_slug
from riemannfm.data.sampler import RiemannFMSubgraphSampler

logger = logging.getLogger(__name__)


class RiemannFMKGDataset(Dataset[RiemannFMGraphData]):
    """Pretraining dataset with on-the-fly subgraph sampling.

    Each __getitem__ call samples a fresh subgraph via BFS and returns
    it as a RiemannFMGraphData with multi-hot edge types and precomputed
    text embeddings.

    Args:
        data_dir: Path to dataset directory (with processed/ subdirectory).
        split: Dataset split ("train", "val", "test").
        num_edge_types: Total number of relation types K.
        max_nodes: Maximum nodes per subgraph N_max.
        max_hops: Maximum BFS hops from seed.
        text_encoder: Text encoder model name or slug for loading embeddings.
        dim_text_emb: Dimension of text embeddings d_c.
        epoch_size: Virtual epoch size (number of samples per epoch).
            Since sampling is on-the-fly, this controls __len__.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_edge_types: int = 822,
        max_nodes: int = 64,
        max_hops: int = 2,
        text_encoder: str | None = None,
        dim_text_emb: int = 0,
        epoch_size: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_edge_types = num_edge_types
        self.max_nodes = max_nodes
        self.max_hops = max_hops
        self.text_encoder = text_encoder
        self.dim_text_emb = dim_text_emb

        # Load data
        self.triples: Tensor = self._load_triples(split)
        self.entity_emb: Tensor | None = None
        self.relation_emb: Tensor | None = None
        self._load_text_embeddings()

        # Build sampler from training triples (always use train for graph structure)
        train_triples = self._load_triples("train") if split != "train" else self.triples
        self.sampler = RiemannFMSubgraphSampler(
            triples=train_triples,
            num_edge_types=num_edge_types,
            max_nodes=max_nodes,
            max_hops=max_hops,
        )

        # Virtual epoch size
        self._epoch_size = epoch_size or len(self.triples)

        logger.info(
            f"RiemannFMKGDataset({split}): "
            f"{len(self.triples)} triples, K={num_edge_types}, N_max={max_nodes}, "
            f"d_c={dim_text_emb}, epoch_size={self._epoch_size}"
        )

    def _load_triples(self, split: str) -> Tensor:
        """Load integer triples for a split."""
        split_map = {"val": "val", "valid": "val", "train": "train", "test": "test"}
        split_key = split_map.get(split, split)
        pt_path = self.data_dir / "processed" / f"{split_key}_triples.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Triples file not found: {pt_path}")
        triples: Tensor = torch.load(pt_path, map_location="cpu", weights_only=True)
        logger.info(f"  Loaded {split_key}_triples.pt: {triples.shape}")
        return triples

    def _load_text_embeddings(self) -> None:
        """Load precomputed text embeddings for entities and relations."""
        if not self.text_encoder or self.dim_text_emb <= 0:
            return

        key = encoder_slug(self.text_encoder)
        emb_dir = self.data_dir / "processed" / "text_embeddings"

        entity_path = emb_dir / embedding_filename(key, self.dim_text_emb, prefix="entity")
        if entity_path.exists():
            self.entity_emb = torch.load(entity_path, map_location="cpu", weights_only=True)
            logger.info(f"  Loaded entity embeddings: {self.entity_emb.shape}")

        relation_path = emb_dir / embedding_filename(key, self.dim_text_emb, prefix="relation")
        if relation_path.exists():
            self.relation_emb = torch.load(relation_path, map_location="cpu", weights_only=True)
            logger.info(f"  Loaded relation embeddings: {self.relation_emb.shape}")

    @property
    def relation_text(self) -> Tensor:
        """Global relation text matrix C_R ∈ R^{Kxd_c} (Def 3.6).

        Shared across all subgraphs.
        """
        if self.relation_emb is not None:
            return self.relation_emb
        return torch.zeros(self.num_edge_types, max(self.dim_text_emb, 1))

    def __len__(self) -> int:
        return self._epoch_size

    def __getitem__(self, idx: int) -> RiemannFMGraphData:
        """Sample a subgraph and return as RiemannFMGraphData.

        Note: idx is not used directly — each call samples a fresh random
        subgraph for on-the-fly data augmentation.
        """
        node_ids, edge_types = self.sampler.sample()
        N = len(node_ids)
        node_ids_tensor = torch.tensor(node_ids, dtype=torch.long)

        # Build node text embeddings C_V (Def 3.5-3.6)
        if self.entity_emb is not None:
            node_text = self.entity_emb[node_ids_tensor]  # (N, d_c)
        else:
            node_text = torch.zeros(N, max(self.dim_text_emb, 1))

        # Node mask: all real nodes (padding happens in collator)
        node_mask = torch.ones(N, dtype=torch.bool)

        return RiemannFMGraphData(
            edge_types=edge_types,       # (N, N, K)
            node_text=node_text,         # (N, d_c)
            node_mask=node_mask,         # (N,)
            num_nodes=N,
            node_ids=node_ids_tensor,    # (N,)
            num_edge_types=self.num_edge_types,
        )
