"""Molecular graph datasets for transfer learning: ZINC, QM9.

These datasets are used to evaluate RieDFM-G's ability to transfer
from knowledge graph pretraining to molecular graph generation.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from riedfm.data.graph_data import RieDFMGraphData
from riedfm.manifolds.product import RieDFMProductManifold


class RieDFMMolecularDataset(Dataset):
    """Molecular graph dataset wrapper.

    Wraps PyG molecular datasets (ZINC, QM9) into our RieDFMGraphData format.
    Molecular graphs have different node/edge types than KGs, so this
    adapter maps them to our product manifold representation.

    Args:
        dataset_name: "zinc" or "qm9".
        data_dir: Path to dataset directory.
        manifold: Product manifold for coordinate representation.
        split: Dataset split.
    """

    def __init__(
        self,
        dataset_name: str = "zinc",
        data_dir: str = "data/molecular",
        manifold: RieDFMProductManifold | None = None,
        split: str = "train",
    ):
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.manifold = manifold
        self.split = split
        self._pyg_dataset = None

    def _load_pyg_dataset(self):
        """Lazy-load PyG dataset."""
        if self._pyg_dataset is not None:
            return

        if self.dataset_name == "zinc":
            from torch_geometric.datasets import ZINC

            self._pyg_dataset = ZINC(
                root=str(self.data_dir / "zinc"),
                subset=True,
                split=self.split,
            )
        elif self.dataset_name == "qm9":
            from torch_geometric.datasets import QM9

            self._pyg_dataset = QM9(root=str(self.data_dir / "qm9"))
        else:
            raise ValueError(f"Unknown molecular dataset: {self.dataset_name}")

    def __len__(self) -> int:
        self._load_pyg_dataset()
        assert self._pyg_dataset is not None
        return len(self._pyg_dataset)

    def __getitem__(self, idx: int) -> RieDFMGraphData:
        """Convert a PyG molecular graph to RieDFMGraphData."""
        self._load_pyg_dataset()
        assert self._pyg_dataset is not None
        data = self._pyg_dataset[idx]
        N = data.num_nodes

        # Initialize coordinates on product manifold
        x = (
            self.manifold.sample_uniform((N,), torch.device("cpu")) if self.manifold is not None else torch.randn(N, 48)
        )  # Fallback

        # Build edge type matrix from edge_index and edge_attr
        edge_types = torch.zeros(N, N, dtype=torch.long)
        if data.edge_index is not None:
            src, dst = data.edge_index
            if data.edge_attr is not None:
                etypes = data.edge_attr.argmax(dim=-1) + 1 if data.edge_attr.dim() > 1 else data.edge_attr.long() + 1
            else:
                etypes = torch.ones(src.shape[0], dtype=torch.long)
            edge_types[src, dst] = etypes

        return RieDFMGraphData(x=x, edge_types=edge_types, num_nodes=N)
