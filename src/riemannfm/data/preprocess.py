"""Preprocessing: ID mapping and text embedding precomputation.

Reads raw/ data (string-ID triples, entity/relation texts) and produces
processed/ outputs (int tensors, embedding vectors).

Two stages:
1. Build entity2id / relation2id mappings, convert triples to int tensors
2. Precompute text embeddings for entities and relations
"""

import logging
from pathlib import Path

import torch

from riemannfm.data.embed import (
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

    Reads raw/entity_texts.tsv and raw/relation_texts.tsv,
    encodes with the given embedder, and saves to
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
    entity_emb_path = emb_dir / embedding_filename(encoder_key, dim, prefix="entity")

    if entity_texts_path.exists():
        if entity_emb_path.exists() and not force:
            logger.info(f"Entity embeddings already exist: {entity_emb_path.name}")
        else:
            texts = _load_texts(entity_texts_path)
            logger.info(f"Encoding {len(texts)} entity texts with {encoder_key}...")
            embedder.embed(texts, output_path=entity_emb_path)
    else:
        logger.warning(f"entity_texts.tsv not found in {raw_dir}")

    # Relation embeddings
    relation_texts_path = raw_dir / "relation_texts.tsv"
    relation_emb_path = emb_dir / embedding_filename(encoder_key, dim, prefix="relation")

    if relation_texts_path.exists():
        if relation_emb_path.exists() and not force:
            logger.info(f"Relation embeddings already exist: {relation_emb_path.name}")
        else:
            texts = _load_texts(relation_texts_path)
            logger.info(f"Encoding {len(texts)} relation texts with {encoder_key}...")
            embedder.embed(texts, output_path=relation_emb_path)
    else:
        logger.warning(f"relation_texts.tsv not found in {raw_dir}")


def _load_texts(path: Path) -> list[str]:
    """Load texts from a TSV file (id<TAB>text)."""
    texts: list[str] = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                texts.append(parts[1])
            elif len(parts) == 1 and parts[0]:
                texts.append(parts[0])
    return texts


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
