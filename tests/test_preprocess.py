"""Smoke tests for the preprocess CLI and embed dispatch.

No network or model loading — these tests exercise helpers that don't
require a live embedding backend. End-to-end preprocess is validated
manually against real datasets.
"""

import pytest

from riemannfm.data.pipeline.embed import (
    ENCODER_DIMS,
    embedding_filename,
    encoder_model_name,
    encoder_slug,
)


def test_cli_imports() -> None:
    import riemannfm.cli.preprocess as mod

    assert callable(mod.main)


def test_preprocess_imports() -> None:
    from riemannfm.data.pipeline import preprocess as mod

    for name in ("build_id_mappings", "precompute_embeddings", "build_mini_wikidata_5m", "run_preprocess"):
        assert hasattr(mod, name), f"preprocess.{name} missing"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("sentence-transformers/all-MiniLM-L6-v2", "sbert"),
        ("all-minilm:l6-v2", "sbert"),
        ("qwen3-embedding", "qwen3"),
        ("qwen3-embedding:8b", "qwen3"),
        ("nomic-embed-text", "nomic"),
    ],
)
def test_encoder_slug_canonical(raw: str, expected: str) -> None:
    assert encoder_slug(raw) == expected


def test_encoder_model_name_roundtrip() -> None:
    for slug in ("sbert", "qwen3", "nomic"):
        model = encoder_model_name(slug)
        assert encoder_slug(model) == slug


def test_embedding_filename_shape() -> None:
    assert embedding_filename("qwen3", 768, prefix="entity") == "entity_emb_qwen3_768.pt"
    assert embedding_filename("sbert", 384, prefix="relation") == "relation_emb_sbert_384.pt"


def test_encoder_dims_complete() -> None:
    assert set(ENCODER_DIMS.keys()) == {"sbert", "qwen3", "nomic"}
    for slug, dim in ENCODER_DIMS.items():
        assert dim > 0, slug
