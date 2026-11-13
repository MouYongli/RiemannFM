"""Data loading and processing for RiemannFM."""

__all__ = [
    "RiemannFMDataModule",
    "RiemannFMGraphCollator",
    "RiemannFMGraphData",
    "RiemannFMKGDataset",
    "RiemannFMSubgraphSampler",
]


def __getattr__(name: str):
    if name == "RiemannFMDataModule":
        from riemannfm.data.datamodule import RiemannFMDataModule

        return RiemannFMDataModule
    if name == "RiemannFMGraphCollator":
        from riemannfm.data.collator import RiemannFMGraphCollator

        return RiemannFMGraphCollator
    if name == "RiemannFMGraphData":
        from riemannfm.data.graph import RiemannFMGraphData

        return RiemannFMGraphData
    if name == "RiemannFMKGDataset":
        from riemannfm.data.datasets.pretrain_dataset import RiemannFMKGDataset

        return RiemannFMKGDataset
    if name == "RiemannFMSubgraphSampler":
        from riemannfm.data.sampler import RiemannFMSubgraphSampler

        return RiemannFMSubgraphSampler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
