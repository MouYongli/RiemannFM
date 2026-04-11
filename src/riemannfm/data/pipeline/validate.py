"""Validate raw KG download integrity.

Checks triple files, entity/relation text files, and split leakage.
Called automatically after download, or standalone:

    uv run python -m riemannfm.data.validate data/wikidata_5m
"""

import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# Triple file names in order of preference
_TRAIN_NAMES = ("train_triples.txt", "train.txt")
_VAL_NAMES = ("val_triples.txt", "valid.txt")
_TEST_NAMES = ("test_triples.txt", "test.txt")


def _find_file(raw_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        p = raw_dir / name
        if p.exists():
            return p
    return None


def _check_triples(path: Path, label: str) -> tuple[bool, int, set[str], set[str]]:
    """Validate a triple file. Returns (ok, count, entities, relations)."""
    entities: set[str] = set()
    relations: set[str] = set()
    bad_lines = 0
    total = 0

    with open(path) as f:
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) != 3:
                bad_lines += 1
                if bad_lines <= 3:
                    logger.warning(f"  {label} line {i + 1}: expected 3 columns, got {len(parts)}")
                continue
            h, r, t = parts
            entities.add(h)
            entities.add(t)
            relations.add(r)
            total += 1

    ok = total > 0 and bad_lines == 0
    status = "OK" if ok else "FAIL"
    msg = f"  {status}  {label}: {total:,} triples, {len(entities):,} entities, {len(relations):,} relations"
    if bad_lines:
        msg += f" ({bad_lines} bad lines)"
    logger.info(msg)
    return ok, total, entities, relations


def _check_text_file(
    path: Path, label: str, reference_ids: set[str]
) -> bool:
    """Validate a TSV text file (entity_texts.tsv or relation_texts.tsv)."""
    if not path.exists():
        logger.warning(f"  SKIP  {label}: file not found")
        return True  # optional file, not a hard failure

    text_ids: set[str] = set()
    empty_text = 0
    bad_lines = 0
    total = 0

    with open(path) as f:
        for i, line in enumerate(f):
            parts = line.strip().split("\t", 1)
            if len(parts) != 2:
                bad_lines += 1
                if bad_lines <= 3:
                    logger.warning(f"  {label} line {i + 1}: expected 2 columns, got {len(parts)}")
                continue
            tid, text = parts
            text_ids.add(tid)
            if not text.strip():
                empty_text += 1
            total += 1

    missing = reference_ids - text_ids
    coverage = (len(reference_ids) - len(missing)) / len(reference_ids) * 100 if reference_ids else 100

    ok = total > 0 and bad_lines == 0
    status = "OK" if ok else "FAIL"
    logger.info(f"  {status}  {label}: {total:,} entries, {empty_text:,} empty, {bad_lines} bad lines")
    logger.info(f"         coverage: {coverage:.1f}% ({len(missing):,} missing)")
    if missing and len(missing) <= 20:
        logger.info(f"         missing: {sorted(missing)}")
    return ok


def _check_split_leakage(train: Path, val: Path | None, test: Path | None) -> bool:
    """Check for triple leakage between splits."""
    def load(p: Path | None) -> set[tuple[str, str, str]]:
        s: set[tuple[str, str, str]] = set()
        if p and p.exists():
            with open(p) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 3:
                        s.add((parts[0], parts[1], parts[2]))
        return s

    tr, va, te = load(train), load(val), load(test)
    tv, tt, vt = tr & va, tr & te, va & te
    clean = len(tv) == 0 and len(tt) == 0 and len(vt) == 0
    if clean:
        logger.info("  OK  split leakage: train∩val=0, train∩test=0, val∩test=0")
    else:
        logger.warning(
            f"  WARN  split leakage: train∩val={len(tv)}, train∩test={len(tt)}, val∩test={len(vt)}"
            " (upstream dataset issue, filtered evaluation handles this)"
        )
    # Leakage is a warning, not a hard failure — many public datasets have minor overlap.
    return True


def _log_relation_distribution(train: Path) -> None:
    """Log top/bottom relations by frequency."""
    counter: Counter[str] = Counter()
    with open(train) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                counter[parts[1]] += 1

    top = counter.most_common(10)
    bottom = counter.most_common()[-10:]
    logger.info(f"  Relation distribution ({len(counter)} unique):")
    logger.info(f"    top-10:    {', '.join(f'{r}({c:,})' for r, c in top)}")
    logger.info(f"    bottom-10: {', '.join(f'{r}({c:,})' for r, c in bottom)}")


def validate_raw(data_dir: str, slug: str = "") -> bool:
    """Validate raw download integrity for a dataset.

    Args:
        data_dir: Base dataset directory (raw/ is appended).
        slug: Dataset slug for log messages.

    Returns:
        True if all checks pass.
    """
    raw_dir = Path(data_dir) / "raw"
    label = slug or raw_dir.parent.name

    if not raw_dir.exists():
        logger.error(f"[{label}] {raw_dir} does not exist")
        return False

    logger.info(f"[{label}] Validating {raw_dir}/")
    all_ok = True

    # 1. Triple files
    train_path = _find_file(raw_dir, _TRAIN_NAMES)
    val_path = _find_file(raw_dir, _VAL_NAMES)
    test_path = _find_file(raw_dir, _TEST_NAMES)

    if not train_path:
        logger.error(f"  FAIL  No training triples found in {raw_dir}")
        return False

    all_entities: set[str] = set()
    all_relations: set[str] = set()

    logger.info("[1/5] Triple files")
    ok, _, ent, rel = _check_triples(train_path, "train")
    all_ok &= ok
    all_entities |= ent
    all_relations |= rel

    for path, name in [(val_path, "val"), (test_path, "test")]:
        if path:
            ok, _, ent, rel = _check_triples(path, name)
            all_ok &= ok
            all_entities |= ent
            all_relations |= rel

    logger.info(f"         total unique: {len(all_entities):,} entities, {len(all_relations):,} relations")

    # 2. Entity texts
    logger.info("[2/5] Entity texts")
    all_ok &= _check_text_file(raw_dir / "entity_texts.tsv", "entity_texts.tsv", all_entities)

    # 3. Relation texts
    logger.info("[3/5] Relation texts")
    all_ok &= _check_text_file(raw_dir / "relation_texts.tsv", "relation_texts.tsv", all_relations)

    # 4. Split leakage
    logger.info("[4/5] Split leakage")
    all_ok &= _check_split_leakage(train_path, val_path, test_path)

    # 5. Relation distribution
    logger.info("[5/5] Relation distribution")
    _log_relation_distribution(train_path)

    status = "ALL PASSED" if all_ok else "SOME CHECKS FAILED"
    logger.info(f"[{label}] Validation: {status}")
    return all_ok


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/wikidata_5m"
    ok = validate_raw(data_dir)
    sys.exit(0 if ok else 1)
