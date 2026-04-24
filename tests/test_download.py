"""Smoke and integration tests for the data download pipeline.

The integration test mocks ``urlretrieve`` to avoid network I/O but still
exercises ``run_pipeline`` end-to-end through all four stages
(download_graph → entity texts → relation texts → optional validate),
which would have caught the earlier validate_raw ImportError regression.
End-to-end against real upstreams is validated manually.
"""

from pathlib import Path

import pytest

from riemannfm.data.pipeline.download import _DOWNLOAD_REGISTRY, ALL_DATASETS, DownloadMeta, run_pipeline

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


def test_mini_not_in_download_registry() -> None:
    assert "wikidata_5m_mini" not in ALL_DATASETS


def test_registry_shapes() -> None:
    for slug, meta in _DOWNLOAD_REGISTRY.items():
        assert isinstance(meta, DownloadMeta), slug
        assert isinstance(meta.download_format, str) and meta.download_format, slug


def test_cli_imports() -> None:
    import riemannfm.cli.download as mod

    assert callable(mod.main)


@pytest.mark.parametrize(
    "slug,expected_format",
    [
        ("wikidata_5m", "wikidata5m"),
        ("fb15k_237", "github_files"),
        ("wn18rr", "github_files"),
        ("yago3_10", "github_files"),
        ("codex_l", "codex"),
        ("wiki27k", "wiki27k"),
    ],
)
def test_registry_download_format(slug: str, expected_format: str) -> None:
    assert _DOWNLOAD_REGISTRY[slug].download_format == expected_format


def test_run_pipeline_fb15k_237_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock urlretrieve, run the full pipeline, verify every stage produced files.

    Regression test for the validate_raw ImportError: the full
    run_pipeline body must be reachable without optional preprocess
    modules installed.
    """
    slug = "fb15k_237"
    data_dir = tmp_path / slug

    def fake_urlretrieve(url: str, dest: str) -> None:
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith("train.txt") or url.endswith("valid.txt") or url.endswith("test.txt"):
            dest_path.write_text(
                "/m/0a\t/film/genre\t/m/0b\n/m/0c\t/location/contains\t/m/0d\n",
                encoding="utf-8",
            )
        elif "entity2textlong" in url:
            dest_path.write_text(
                "/m/0a\tEntity A description\n/m/0b\tEntity B description\n"
                "/m/0c\tEntity C description\n/m/0d\tEntity D description\n",
                encoding="utf-8",
            )
        else:
            raise OSError(f"mocked: optional resource unavailable ({url})")

    monkeypatch.setattr("riemannfm.data.pipeline.download.urlretrieve", fake_urlretrieve)

    run_pipeline(slug=slug, data_dir=str(data_dir), text_source="wikipedia_mapping")

    raw = data_dir / "raw"
    assert (raw / "train.txt").exists()
    assert (raw / "valid.txt").exists()
    assert (raw / "test.txt").exists()
    assert (raw / "entity_texts.tsv").exists()
    assert (raw / "relation_texts.tsv").exists()

    relation_texts = (raw / "relation_texts.tsv").read_text(encoding="utf-8")
    # freebase-style stripping: /film/genre -> "film genre"
    assert "film genre" in relation_texts
    assert "location contains" in relation_texts
