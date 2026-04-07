"""Preprocessing: ID mapping and text embedding precomputation.

Reads raw/ data (string-ID triples, entity/relation texts) and produces
processed/ outputs (int tensors, embedding vectors).

Two stages:
1. Build entity2id / relation2id mappings, convert triples to int tensors
2. Precompute text embeddings for entities and relations

Additionally provides `build_mini_wikidata_5m` for creating a small
engineering validation subset from the full WikiData5M download.
"""

import logging
import random
from collections import defaultdict
from pathlib import Path

import torch

from riemannfm.data.pipeline.embed import (
    RiemannFMTextEmbedder,
    embedding_filename,
    encoder_slug,
)

logger = logging.getLogger(__name__)

# ─── Triple file name conventions ────────────────────────────────────────────

_SPLIT_NAMES: dict[str, list[str]] = {
    "train": ["train_triples.txt", "train.txt"],
    "val": ["val_triples.txt", "valid.txt"],
    "test": ["test_triples.txt", "test.txt"],
}


def _find_split_file(raw_dir: Path, split: str) -> Path | None:
    """Find the triple file for a given split."""
    for name in _SPLIT_NAMES.get(split, []):
        path = raw_dir / name
        if path.exists():
            return path
    return None


# ─── Stage 1: ID mappings ────────────────────────────────────────────────────


def build_id_mappings(
    raw_dir: Path,
    processed_dir: Path,
    force: bool = False,
) -> tuple[dict[str, int], dict[str, int]]:
    """Build entity2id and relation2id from raw triples, save int tensors.

    Reads all triple files (train/val/test), collects unique entity and
    relation string IDs, assigns consecutive integer IDs, and saves:
    - processed/entity2id.tsv
    - processed/relation2id.tsv
    - processed/{split}_triples.pt (int tensors)

    Args:
        raw_dir: Path to raw/ directory with triple files.
        processed_dir: Path to processed/ directory for output.
        force: Rebuild even if files exist.

    Returns:
        (entity2id, relation2id) dictionaries.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    entity2id_path = processed_dir / "entity2id.tsv"
    relation2id_path = processed_dir / "relation2id.tsv"

    # Check if already built
    if entity2id_path.exists() and relation2id_path.exists() and not force:
        entity2id = _load_id_mapping(entity2id_path)
        relation2id = _load_id_mapping(relation2id_path)

        # Verify triples exist too
        all_exist = all(
            (processed_dir / f"{split}_triples.pt").exists()
            for split in _SPLIT_NAMES
        )
        if all_exist:
            logger.info(f"ID mappings already exist ({len(entity2id)} entities, {len(relation2id)} relations), skipping.")
            return entity2id, relation2id

    # Collect all entities and relations from all splits
    logger.info("Building ID mappings from raw triples...")
    entities: set[str] = set()
    relations: set[str] = set()
    split_triples: dict[str, list[tuple[str, str, str]]] = {}

    for split in _SPLIT_NAMES:
        path = _find_split_file(raw_dir, split)
        if path is None:
            logger.warning(f"Split file not found for '{split}' in {raw_dir}")
            split_triples[split] = []
            continue

        triples: list[tuple[str, str, str]] = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    h, r, t = parts
                    entities.add(h)
                    entities.add(t)
                    relations.add(r)
                    triples.append((h, r, t))

        split_triples[split] = triples
        logger.info(f"  {split}: {len(triples)} triples")

    # Assign IDs (sorted for determinism)
    entity2id = {e: i for i, e in enumerate(sorted(entities))}
    relation2id = {r: i for i, r in enumerate(sorted(relations))}
    logger.info(f"  Total: {len(entity2id)} entities, {len(relation2id)} relations")

    # Save mappings
    _save_id_mapping(entity2id, entity2id_path)
    _save_id_mapping(relation2id, relation2id_path)

    # Convert and save int triples
    for split, triples in split_triples.items():
        if not triples:
            continue
        int_triples = torch.tensor(
            [[entity2id[h], relation2id[r], entity2id[t]] for h, r, t in triples],
            dtype=torch.long,
        )
        pt_path = processed_dir / f"{split}_triples.pt"
        torch.save(int_triples, pt_path)
        logger.info(f"  Saved {split}_triples.pt ({int_triples.shape})")

    return entity2id, relation2id


def _save_id_mapping(mapping: dict[str, int], path: Path) -> None:
    """Save string→int mapping as TSV."""
    with open(path, "w") as f:
        for key, idx in sorted(mapping.items(), key=lambda x: x[1]):
            f.write(f"{key}\t{idx}\n")


def _load_id_mapping(path: Path) -> dict[str, int]:
    """Load string→int mapping from TSV."""
    mapping: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                mapping[parts[0]] = int(parts[1])
    return mapping


# ─── Stage 2: Text embeddings ────────────────────────────────────────────────


def precompute_embeddings(
    raw_dir: Path,
    processed_dir: Path,
    embedder: RiemannFMTextEmbedder,
    encoder_key: str,
    dim: int,
    force: bool = False,
) -> None:
    """Precompute text embeddings for entities and relations.

    Texts are loaded in the order defined by entity2id.tsv / relation2id.tsv
    so that row i of the embedding tensor corresponds to integer ID i.

    Saves to:
        processed/text_embeddings/entity_emb_{key}_{dim}.pt
        processed/text_embeddings/relation_emb_{key}_{dim}.pt

    Args:
        raw_dir: Path to raw/ directory.
        processed_dir: Path to processed/ directory.
        embedder: Text embedder instance.
        encoder_key: Short encoder slug for filenames.
        dim: Embedding dimension for filenames.
        force: Re-encode even if files exist.
    """
    emb_dir = processed_dir / "text_embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    # Entity embeddings
    entity_texts_path = raw_dir / "entity_texts.tsv"
    entity_id_path = processed_dir / "entity2id.tsv"
    entity_emb_path = emb_dir / embedding_filename(encoder_key, dim, prefix="entity")

    if entity_texts_path.exists() and entity_id_path.exists():
        if entity_emb_path.exists() and not force:
            logger.info(f"Entity embeddings already exist: {entity_emb_path.name}")
        else:
            texts = _load_texts_ordered(entity_texts_path, entity_id_path)
            logger.info(f"Encoding {len(texts)} entity texts with {encoder_key}...")
            embedder.embed(texts, output_path=entity_emb_path)
    else:
        logger.warning(f"entity_texts.tsv or entity2id.tsv not found in {raw_dir}")

    # Relation embeddings
    relation_texts_path = raw_dir / "relation_texts.tsv"
    relation_id_path = processed_dir / "relation2id.tsv"
    relation_emb_path = emb_dir / embedding_filename(encoder_key, dim, prefix="relation")

    if relation_texts_path.exists() and relation_id_path.exists():
        if relation_emb_path.exists() and not force:
            logger.info(f"Relation embeddings already exist: {relation_emb_path.name}")
        else:
            texts = _load_texts_ordered(relation_texts_path, relation_id_path)
            logger.info(f"Encoding {len(texts)} relation texts with {encoder_key}...")
            embedder.embed(texts, output_path=relation_emb_path)
    else:
        logger.warning(f"relation_texts.tsv or relation2id.tsv not found in {raw_dir}")


def _load_texts_ordered(texts_path: Path, id_path: Path) -> list[str]:
    """Load texts ordered by integer IDs from an id mapping file.

    Reads ``texts_path`` (id<TAB>text) into a dict, then orders by the
    integer IDs in ``id_path`` (string_id<TAB>int_id). Only IDs present
    in the mapping are included; extras in texts_path are dropped.

    Args:
        texts_path: Path to TSV with ``string_id<TAB>text``.
        id_path: Path to TSV with ``string_id<TAB>int_id``.

    Returns:
        List of texts where index i corresponds to integer ID i.
    """
    # Build string_id -> text lookup
    id_to_text: dict[str, str] = {}
    with open(texts_path) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                id_to_text[parts[0]] = parts[1]

    # Load id mapping: string_id -> int_id
    id_mapping: list[tuple[str, int]] = []
    with open(id_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                id_mapping.append((parts[0], int(parts[1])))

    # Sort by int_id and build ordered text list
    id_mapping.sort(key=lambda x: x[1])
    n = len(id_mapping)
    texts = [""] * n
    missing = 0
    for string_id, int_id in id_mapping:
        if string_id in id_to_text:
            texts[int_id] = id_to_text[string_id]
        else:
            missing += 1

    if missing > 0:
        logger.warning(
            f"{missing}/{n} IDs in {id_path.name} have no text in {texts_path.name}"
        )
    else:
        logger.info(f"Loaded {n} texts aligned with {id_path.name}")

    return texts


# ─── Mini dataset builder ───────────────────────────────────────────────────


def build_mini_wikidata_5m(
    source_dir: str = "data/wikidata_5m",
    output_dir: str = "data/wikidata_5m_mini",
    max_entities: int = 1000,
    target_triples: int = 5000,
    seed: int = 42,
    force: bool = False,
) -> Path:
    """Build a small engineering validation subset from full WikiData5M.

    Uses relation-aware sampling to ensure diverse relation coverage:
    1. Sample triples from every relation type for coverage
    2. Fill entity budget with additional triples
    3. Densify by adding triples within the sampled entity set
    4. Split into train/val/test (80/10/10)

    Args:
        source_dir: Path to full WikiData5M dataset (with raw/ subdirectory).
        output_dir: Path for the mini dataset output.
        max_entities: Target number of entities (~1K).
        target_triples: Target number of triples (~5K).
        seed: Random seed for deterministic sampling.
        force: Rebuild even if output already exists.

    Returns:
        Path to the mini dataset raw/ directory.
    """
    from riemannfm.data.pipeline.validate import validate_raw

    source_raw = Path(source_dir) / "raw"
    output_raw = Path(output_dir) / "raw"

    if (output_raw / "train_triples.txt").exists() and not force:
        logger.info(f"Mini dataset already exists at {output_raw}, skipping. Use force=True to rebuild.")
        return output_raw

    if not source_raw.exists():
        raise FileNotFoundError(
            f"Source data not found at {source_raw}. "
            "Run `make download ARGS=\"data=wikidata_5m\"` first."
        )

    random.seed(seed)
    output_raw.mkdir(parents=True, exist_ok=True)

    # ── Load all triples from all splits ────────────────────────────────────
    all_triples: list[tuple[str, str, str]] = []
    for split in ("train", "val", "test"):
        path = _find_split_file(source_raw, split)
        if path is None:
            continue
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    all_triples.append((parts[0], parts[1], parts[2]))

    logger.info(f"Loaded {len(all_triples):,} total triples from {source_raw}")

    # ── Phase 1: Relation coverage ──────────────────────────────────────────
    rel_to_triples: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for h, r, t in all_triples:
        rel_to_triples[r].append((h, r, t))

    num_relations = len(rel_to_triples)
    per_rel = max(3, target_triples // num_relations)
    logger.info(f"  {num_relations} relation types, sampling {per_rel} triples per relation")

    selected: list[tuple[str, str, str]] = []
    selected_set: set[tuple[str, str, str]] = set()
    entities: set[str] = set()

    for _rel, triples in rel_to_triples.items():
        k = min(per_rel, len(triples))
        sampled = random.sample(triples, k)
        for triple in sampled:
            if triple not in selected_set:
                selected.append(triple)
                selected_set.add(triple)
                entities.add(triple[0])
                entities.add(triple[2])

    logger.info(f"  Phase 1 (relation coverage): {len(selected):,} triples, {len(entities):,} entities")

    # ── Phase 2: Fill entity budget ─────────────────────────────────────────
    if len(entities) < max_entities:
        remaining = [t for t in all_triples if t not in selected_set]
        random.shuffle(remaining)
        for triple in remaining:
            if len(entities) >= max_entities:
                break
            if triple not in selected_set:
                selected.append(triple)
                selected_set.add(triple)
                entities.add(triple[0])
                entities.add(triple[2])

        logger.info(f"  Phase 2 (entity budget): {len(selected):,} triples, {len(entities):,} entities")

    # ── Phase 3: Densify ────────────────────────────────────────────────────
    densified = 0
    for h, r, t in all_triples:
        if (h, r, t) not in selected_set and h in entities and t in entities:
            selected.append((h, r, t))
            selected_set.add((h, r, t))
            densified += 1

    logger.info(f"  Phase 3 (densify): +{densified:,} triples → {len(selected):,} total")

    # ── Split into train/val/test (80/10/10) ────────────────────────────────
    random.shuffle(selected)
    n = len(selected)
    n_val = max(1, n // 10)
    n_test = max(1, n // 10)
    n_train = n - n_val - n_test

    splits = {
        "train": selected[:n_train],
        "val": selected[n_train : n_train + n_val],
        "test": selected[n_train + n_val :],
    }

    for split_name, triples in splits.items():
        out_path = output_raw / f"{split_name}_triples.txt"
        with open(out_path, "w") as f:
            for h, r, t in triples:
                f.write(f"{h}\t{r}\t{t}\n")
        logger.info(f"  {split_name}: {len(triples):,} triples → {out_path.name}")

    # ── Filter text files ───────────────────────────────────────────────────
    relations = {r for _, r, _ in selected}

    entity_texts_src = source_raw / "entity_texts.tsv"
    if entity_texts_src.exists():
        entity_texts_dst = output_raw / "entity_texts.tsv"
        _filter_text_file(entity_texts_src, entity_texts_dst, entities)
        logger.info(f"  Filtered entity_texts.tsv: {len(entities):,} entities")

    relation_texts_src = source_raw / "relation_texts.tsv"
    if relation_texts_src.exists():
        relation_texts_dst = output_raw / "relation_texts.tsv"
        _filter_text_file(relation_texts_src, relation_texts_dst, relations)
        logger.info(f"  Filtered relation_texts.tsv: {len(relations):,} relations")

    # ── Validate ────────────────────────────────────────────────────────────
    logger.info("Validating mini dataset...")
    if not validate_raw(output_dir, slug="wikidata_5m_mini"):
        logger.warning("Mini dataset validation had warnings — check logs above.")

    logger.info(
        f"Mini dataset built: {len(entities):,} entities, "
        f"{len(relations):,} relations, {len(selected):,} triples → {output_raw}"
    )
    return output_raw


def _filter_text_file(src: Path, dst: Path, keep_ids: set[str]) -> None:
    """Filter a TSV text file to only include IDs in keep_ids."""
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            parts = line.split("\t", 1)
            if parts and parts[0] in keep_ids:
                fout.write(line)


# ─── Full pipeline ───────────────────────────────────────────────────────────


def run_preprocess(
    data_dir: str,
    embedding_provider: str,
    model_name: str,
    output_dim: int,
    batch_size: int = 256,
    max_length: int = 512,
    pooling: str = "mean",
    api_base: str | None = None,
    api_key: str | None = None,
    force: bool = False,
) -> None:
    """Run the full preprocess pipeline for a single dataset.

    Args:
        data_dir: Base dataset directory (with raw/ subdirectory).
        embedding_provider: "huggingface", "ollama", "openai", or "none".
        model_name: Model name for the embedder.
        output_dim: Embedding output dimension.
        batch_size: Encoding batch size.
        max_length: Max token length (huggingface).
        pooling: Pooling strategy (huggingface).
        api_base: API base URL (ollama/openai).
        api_key: API key (openai).
        force: Re-run even if outputs exist.
    """
    raw_dir = Path(data_dir) / "raw"
    processed_dir = Path(data_dir) / "processed"

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_dir}. "
            "Run `make download` first."
        )

    # Stage 1: ID mappings + int triples
    build_id_mappings(raw_dir, processed_dir, force=force)

    # Stage 2: Text embeddings (skip if none)
    if embedding_provider == "none":
        logger.info("embedding_provider=none, skipping text embeddings.")
        return

    key = encoder_slug(model_name)
    embedder = RiemannFMTextEmbedder(
        embedding_provider=embedding_provider,
        model_name=model_name,
        output_dim=output_dim,
        api_base=api_base,
        api_key=api_key,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,
    )
    precompute_embeddings(
        raw_dir, processed_dir,
        embedder=embedder,
        encoder_key=key,
        dim=output_dim,
        force=force,
    )
