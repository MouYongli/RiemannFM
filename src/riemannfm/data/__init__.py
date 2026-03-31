"""Data loading and processing for RiemannFM."""

__all__ = ["RiemannFMGraphCollator", "RiemannFMGraphData"]


def __getattr__(name: str):
    if name == "RiemannFMGraphCollator":
        from riemannfm.data.collator import RiemannFMGraphCollator

        return RiemannFMGraphCollator
    if name == "RiemannFMGraphData":
        from riemannfm.data.graph_data import RiemannFMGraphData

        return RiemannFMGraphData
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
