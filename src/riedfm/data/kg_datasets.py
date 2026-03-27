"""Standard KG benchmark datasets: FB15k-237, WN18RR, ogbl-wikikg2, CoDEx.

Provides dataset classes for evaluation and fine-tuning on standard
knowledge graph completion benchmarks.
"""

from pathlib import Path

from torch.utils.data import Dataset

from riedfm.manifolds.product import RieDFMProductManifold


class RieDFMKGDataset(Dataset):
    """Generic KG benchmark dataset.

    Loads triples from standard format files and creates subgraph samples.
    Supports FB15k-237, WN18RR, and CoDEx.

    Args:
        data_dir: Path to dataset directory.
        manifold: Product manifold.
        split: "train", "val", or "test".
        max_nodes: Maximum subgraph size for evaluation.
    """

    def __init__(
        self,
        data_dir: str,
        manifold: RieDFMProductManifold,
        split: str = "test",
        max_nodes: int = 256,
    ):
        self.data_dir = Path(data_dir)
        self.manifold = manifold
        self.split = split
        self.max_nodes = max_nodes

        self.triples: list[tuple[int, int, int]] = []
        self.entity2id: dict[str, int] = {}
        self.relation2id: dict[str, int] = {}
        self.num_entities = 0
        self.num_relations = 0

        self._load_data()

    def _load_data(self):
        """Load dataset files."""
        # Load entity and relation mappings
        entity_file = self.data_dir / "entities.dict"
        relation_file = self.data_dir / "relations.dict"

        if entity_file.exists():
            with open(entity_file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        self.entity2id[parts[1]] = int(parts[0])
            self.num_entities = len(self.entity2id)

        if relation_file.exists():
            with open(relation_file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        self.relation2id[parts[1]] = int(parts[0])
            self.num_relations = len(self.relation2id)

        # Load triples
        triples_file = self.data_dir / f"{self.split}.txt"
        if triples_file.exists():
            with open(triples_file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 3:
                        h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                        self.triples.append((h, r, t))

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> dict:
        """Return a single triple for link prediction evaluation.

        Returns:
            Dictionary with head, relation, tail IDs and the full triple.
        """
        h, r, t = self.triples[idx]
        return {
            "head": h,
            "relation": r,
            "tail": t,
            "triple": (h, r, t),
        }
