"""Unit tests for the post-download validate module."""

from pathlib import Path

from riemannfm.data.pipeline.validate import validate_raw


def _write(raw_dir: Path, name: str, content: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / name).write_text(content, encoding="utf-8")


def test_validate_raw_clean_dataset(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write(
        raw,
        "train.txt",
        "e1\tr1\te2\ne2\tr2\te3\ne3\tr1\te1\n",
    )
    _write(raw, "valid.txt", "e1\tr2\te3\n")
    _write(raw, "test.txt", "e3\tr1\te2\n")
    _write(
        raw,
        "entity_texts.tsv",
        "e1\tentity one\ne2\tentity two\ne3\tentity three\n",
    )
    _write(
        raw,
        "relation_texts.tsv",
        "r1\tfirst relation\nr2\tsecond relation\n",
    )

    assert validate_raw(str(tmp_path), slug="fake") is True


def test_validate_raw_missing_train(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    assert validate_raw(str(tmp_path), slug="fake") is False


def test_validate_raw_malformed_triples(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write(raw, "train.txt", "e1\tr1\te2\nonly-two-cols\n")
    # bad lines cause the train check to fail; hard failure.
    assert validate_raw(str(tmp_path), slug="fake") is False


def test_validate_raw_split_leakage_is_soft_warning(tmp_path: Path) -> None:
    """Split leakage is logged as a warning but does not fail validation."""
    raw = tmp_path / "raw"
    _write(raw, "train.txt", "e1\tr1\te2\ne2\tr2\te3\n")
    _write(raw, "valid.txt", "e1\tr1\te2\n")  # identical to train row
    _write(raw, "test.txt", "e2\tr2\te3\n")  # identical to train row
    _write(raw, "entity_texts.tsv", "e1\ta\ne2\tb\ne3\tc\n")
    _write(raw, "relation_texts.tsv", "r1\tx\nr2\ty\n")

    assert validate_raw(str(tmp_path), slug="fake") is True
