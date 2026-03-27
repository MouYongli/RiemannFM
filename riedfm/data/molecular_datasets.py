"""Molecular graph datasets for transfer learning: ZINC, QM9.

These datasets are used to evaluate RieDFM-G's ability to transfer
from knowledge graph pretraining to molecular graph generation.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from riedfm.data.graph_data import GraphData
from riedfm.manifolds.product import ProductManifold


class MolecularGraphDataset(Dataset):
    """Molecular graph dataset wrapper.

    Wraps PyG molecular datasets (ZINC, QM9) into our GraphData format.
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
        manifold: ProductManifold | None = None,
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
        return len(self._pyg_dataset)

    def __getitem__(self, idx: int) -> GraphData:
        """Convert a PyG molecular graph to GraphData."""
        self._load_pyg_dataset()
        data = self._pyg_dataset[idx]
        N = data.num_nodes

        # Initialize coordinates on product manifold
        if self.manifold is not None:
            x = self.manifold.sample_uniform((N,), torch.device("cpu"))
        else:
            x = torch.randn(N, 48)  # Fallback

        # Build edge type matrix from edge_index and edge_attr
        edge_types = torch.zeros(N, N, dtype=torch.long)
        if data.edge_index is not None:
            src, dst = data.edge_index
            if data.edge_attr is not None:
                if data.edge_attr.dim() > 1:
                    etypes = data.edge_attr.argmax(dim=-1) + 1
                else:
                    etypes = data.edge_attr.long() + 1
            else:
                etypes = torch.ones(src.shape[0], dtype=torch.long)
            edge_types[src, dst] = etypes

        return GraphData(x=x, edge_types=edge_types, num_nodes=N)
