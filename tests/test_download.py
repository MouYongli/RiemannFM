"""Smoke tests for the data download pipeline.

No network I/O — these tests only import the modules and inspect the
download registry. End-to-end download is validated manually.
"""

from riemannfm.data.pipeline.download import _DOWNLOAD_REGISTRY, ALL_DATASETS, DownloadMeta

EXPECTED_DATASETS = {
    "wikidata_5m",
    "fb15k_237",
    "wn18rr",
    "codex_l",
    "wiki27k",
    "yago3_10",
}


def test_all_datasets_enumerated() -> None:
    assert set(ALL_DATASETS) == EXPECTED_DATASETS


def test_registry_shapes() -> None:
    for slug, meta in _DOWNLOAD_REGISTRY.items():
        assert isinstance(meta, DownloadMeta), slug
        assert isinstance(meta.download_format, str) and meta.download_format, slug


def test_cli_imports() -> None:
    import riemannfm.cli.download as mod

    assert callable(mod.main)
