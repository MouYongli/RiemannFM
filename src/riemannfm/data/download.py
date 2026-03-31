"""Dataset download, text extraction, and embedding precomputation.

Provides a three-stage pipeline for preparing KG datasets:

1. Download raw graph structure (triples, entity/relation mappings)
2. Extract entity text descriptions
3. Precompute text embeddings with a specified encoder

Each stage is idempotent — skipped if output files already exist.
"""

import json
import logging
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm

logger = logging.getLogger(__name__)

# ─── Encoder slug mapping ────────────────────────────────────────────────────

_SLUG_MAP: dict[str, str] = {
    "sentence-transformers/all-MiniLM-L6-v2": "sbert",
    "xlm-roberta-large": "xlm_roberta",
    "Alibaba-NLP/Qwen3-Embedding-8B": "qwen3_embed",
}

_SLUG_TO_MODEL: dict[str, str] = {v: k for k, v in _SLUG_MAP.items()}

ENCODER_DIMS: dict[str, int] = {
    "sbert": 384,
    "xlm_roberta": 1024,
    "qwen3_embed": 4096,
}


def encoder_slug(text_encoder: str) -> str:
    """Derive a filesystem-safe short key from a model name or config key.

    Args:
        text_encoder: HuggingFace model name or short slug.

    Returns:
        Short key like "sbert", "xlm_roberta", "qwen3_embed".
    """
    if text_encoder in _SLUG_MAP:
        return _SLUG_MAP[text_encoder]
    if text_encoder in _SLUG_TO_MODEL:
        return text_encoder
    return text_encoder.split("/")[-1].replace("-", "_").lower()


def encoder_model_name(text_encoder: str) -> str:
    """Resolve a short slug or model name to the full HuggingFace model name.

    Args:
        text_encoder: Short slug or full model name.

    Returns:
        Full HuggingFace model name.
    """
    if text_encoder in _SLUG_TO_MODEL:
        return _SLUG_TO_MODEL[text_encoder]
    return text_encoder


def embedding_filename(encoder_key: str, dim: int) -> str:
    """Build the deterministic embedding filename.

    Args:
        encoder_key: Short encoder slug.
        dim: Embedding dimension.

    Returns:
        Filename like "entity_emb_sbert_384.pt".
    """
    return f"entity_emb_{encoder_key}_{dim}.pt"


# ─── Download-specific registry ──────────────────────────────────────────────
# Only metadata that does NOT belong in Hydra data configs:
# download URLs, archive format, text description URLs.
# data_dir, text_source, num_relations etc. come from configs/data/*.yaml.


@dataclass
class DownloadMeta:
    """Download-specific metadata for a KG dataset."""

    download_url: str | None
    download_format: str
    text_url: str | None


_DOWNLOAD_REGISTRY: dict[str, DownloadMeta] = {
    "wikidata_5m": DownloadMeta(
        download_url=None,
        download_format="huggingface",
        text_url=None,
    ),
    "fb15k_237": DownloadMeta(
        download_url="https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237.tar.gz",
        download_format="tar.gz",
        text_url="https://raw.githubusercontent.com/wangbo9719/StAR_KGC/main/data/FB15k-237/entity2textlong.txt",
    ),
    "wn18rr": DownloadMeta(
        download_url="https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR.tar.gz",
        download_format="tar.gz",
        text_url=None,
    ),
    "codex_l": DownloadMeta(
        download_url=None,
        download_format="codex",
        text_url="https://raw.githubusercontent.com/tsafavi/codex/master/data/entities/en/entities.json",
    ),
    "wiki27k": DownloadMeta(
        download_url=None,
        download_format="zip",
        text_url=None,
    ),
    "yago3_10": DownloadMeta(
        download_url=None,
        download_format="tar.gz",
        text_url=None,
    ),
}

ALL_DATASETS = list(_DOWNLOAD_REGISTRY.keys())

# CoDEx split URLs
_CODEX_BASE = "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-l"
_CODEX_SPLIT_URLS: dict[str, str] = {
    "train": f"{_CODEX_BASE}/train.txt",
    "valid": f"{_CODEX_BASE}/valid.txt",
    "test": f"{_CODEX_BASE}/test.txt",
}


def get_download_meta(slug: str) -> DownloadMeta:
    """Look up download metadata from the registry.

    Args:
        slug: Dataset slug (e.g. "fb15k_237").

    Returns:
        DownloadMeta for the requested dataset.

    Raises:
        ValueError: If slug is not in the registry.
    """
    if slug not in _DOWNLOAD_REGISTRY:
        raise ValueError(f"Unknown dataset slug: {slug}. Available: {ALL_DATASETS}")
    return _DOWNLOAD_REGISTRY[slug]


# ─── Stage 1: Download graph structure ────────────────────────────────────────


def download_graph(
    slug: str,
    data_dir: str,
    force: bool = False,
) -> Path:
    """Stage 1: Download raw graph structure files.

    Downloads triples and entity/relation mappings into data_dir.
    Skips if train.txt (or train_triples.txt for wikidata) already exists.

    Args:
        slug: Dataset slug for registry lookup.
        data_dir: Target directory path.
        force: Re-download even if files exist.

    Returns:
        Path to the data directory.
    """
    meta = get_download_meta(slug)
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    has_triples = (
        (path / "train.txt").exists()
        or (path / "train_triples.txt").exists()
    )
    if has_triples and not force:
        logger.info(f"[{slug}] Graph files already exist, skipping download.")
        return path

    if meta.download_format == "tar.gz":
        _download_tar_gz(slug, meta, path)
    elif meta.download_format == "huggingface":
        _download_huggingface(slug, path)
    elif meta.download_format == "codex":
        _download_codex(slug, path)
    elif meta.download_format == "zip":
        _download_zip(slug, meta, path)
    else:
        logger.warning(f"[{slug}] Unknown download format: {meta.download_format}")

    return path


def _download_tar_gz(slug: str, meta: DownloadMeta, data_dir: Path) -> None:
    """Download and extract a tar.gz archive."""
    if meta.download_url is None:
        logger.warning(
            f"[{slug}] No download URL configured. "
            f"Please manually place graph files in {data_dir}/"
        )
        return

    logger.info(f"[{slug}] Downloading from {meta.download_url}...")
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "dataset.tar.gz"
        urlretrieve(meta.download_url, str(archive_path))
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(tmpdir)

        # Find extracted directory and copy files
        for extracted_dir in Path(tmpdir).iterdir():
            if extracted_dir.is_dir() and extracted_dir.name != "__MACOSX":
                for f in extracted_dir.iterdir():
                    if f.is_file():
                        shutil.copy2(f, data_dir / f.name)
                break

    logger.info(f"[{slug}] Graph files downloaded to {data_dir}/")


def _download_huggingface(slug: str, data_dir: Path) -> None:
    """Download WikiData5M from HuggingFace datasets library."""
    try:
        from datasets import load_dataset

        logger.info(f"[{slug}] Downloading WikiData5M from HuggingFace...")
        ds = load_dataset("wikidata5m", trust_remote_code=True)

        # Save triples per split
        for split_name, hf_split in [("train", "train"), ("val", "validation"), ("test", "test")]:
            if hf_split not in ds:
                continue
            split_data = ds[hf_split]
            triples_path = data_dir / f"{split_name}_triples.txt"
            with open(triples_path, "w") as f:
                for row in tqdm(split_data, desc=f"Writing {split_name} triples"):
                    h, r, t = row["head"], row["relation"], row["tail"]
                    f.write(f"{h}\t{r}\t{t}\n")

        # Save entity info if available
        if "entities" in ds:
            entities = {}
            for row in tqdm(ds["entities"], desc="Writing entities"):
                entities[row["id"]] = {
                    "label": row.get("label", ""),
                    "description": row.get("description", ""),
                }
            with open(data_dir / "entities.json", "w") as f:
                json.dump(entities, f, ensure_ascii=False)

        logger.info(f"[{slug}] WikiData5M downloaded to {data_dir}/")

    except Exception as e:
        logger.warning(
            f"[{slug}] HuggingFace download failed: {e}\n"
            f"Please manually download WikiData5M and place files in {data_dir}/\n"
            f"Expected files: train_triples.txt, val_triples.txt, test_triples.txt, entities.json\n"
            f"Source: https://deepgraphlearning.github.io/project/wikidata5m"
        )


def _download_codex(slug: str, data_dir: Path) -> None:
    """Download CoDEx-L dataset from GitHub."""
    logger.info(f"[{slug}] Downloading CoDEx-L splits...")
    for split_name, url in _CODEX_SPLIT_URLS.items():
        target = data_dir / f"{split_name}.txt"
        logger.info(f"  Downloading {split_name}...")
        urlretrieve(url, str(target))

    # Download entity/relation mappings
    entities_url = f"{_CODEX_BASE}/entities.dict"
    relations_url = f"{_CODEX_BASE}/relations.dict"
    for url, fname in [(entities_url, "entities.dict"), (relations_url, "relations.dict")]:
        target = data_dir / fname
        try:
            urlretrieve(url, str(target))
        except Exception:
            logger.warning(f"[{slug}] Could not download {fname}, may need manual setup.")

    logger.info(f"[{slug}] CoDEx-L downloaded to {data_dir}/")


def _download_zip(slug: str, meta: DownloadMeta, data_dir: Path) -> None:
    """Download and extract a zip archive."""
    if meta.download_url is None:
        logger.warning(
            f"[{slug}] No download URL configured. "
            f"Please manually place graph files in {data_dir}/"
        )
        return

    logger.info(f"[{slug}] Downloading from {meta.download_url}...")
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "dataset.zip"
        urlretrieve(meta.download_url, str(archive_path))
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(tmpdir)

        for extracted in Path(tmpdir).iterdir():
            if extracted.is_dir() and extracted.name != "__MACOSX":
                for f in extracted.iterdir():
                    if f.is_file():
                        shutil.copy2(f, data_dir / f.name)
                break

    logger.info(f"[{slug}] Graph files downloaded to {data_dir}/")


# ─── Stage 2: Extract entity texts ───────────────────────────────────────────


def extract_entity_texts(
    slug: str,
    data_dir: str,
    text_source: str,
    force: bool = False,
) -> Path:
    """Stage 2: Extract or download entity text descriptions.

    Produces entity_texts.tsv with columns: entity_id<TAB>text

    Args:
        slug: Dataset slug for registry lookup (text_url).
        data_dir: Dataset directory path.
        text_source: Text source strategy name.
        force: Re-extract even if file exists.

    Returns:
        Path to entity_texts.tsv.
    """
    meta = get_download_meta(slug)
    path = Path(data_dir)
    output_path = path / "entity_texts.tsv"

    if output_path.exists() and not force:
        logger.info(f"[{slug}] entity_texts.tsv already exists, skipping.")
        return output_path

    path.mkdir(parents=True, exist_ok=True)

    if text_source == "wordnet_defs":
        _extract_texts_wordnet(path, output_path)
    elif text_source == "wikipedia_mapping":
        _extract_texts_wikipedia_mapping(path, output_path, meta.text_url)
    elif text_source == "wikidata_description":
        _extract_texts_wikidata_description(path, output_path, meta.text_url)
    elif text_source == "wikipedia":
        _extract_texts_wikipedia(path, output_path)
    else:
        logger.warning(f"[{slug}] Unknown text source: {text_source}")

    return output_path


def _extract_texts_wordnet(data_dir: Path, output_path: Path) -> None:
    """Extract texts from WordNet synset definitions for WN18RR."""
    import nltk

    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import wordnet as wn

    # Load entity mapping
    entity2id = _load_entity_mapping(data_dir)
    if not entity2id:
        logger.warning("[wn18rr] No entities.dict found, cannot extract texts.")
        return

    logger.info(f"[wn18rr] Extracting WordNet definitions for {len(entity2id)} entities...")
    with open(output_path, "w") as f:
        for entity_name in tqdm(entity2id, desc="Extracting WordNet texts"):
            # WN18RR entity names are synset IDs like "__ablaze.s.01"
            # Convert to NLTK format: ablaze.s.01
            synset_name = entity_name.strip("_").lstrip("_")
            # Remove leading underscores common in WN18RR format
            while synset_name.startswith("_"):
                synset_name = synset_name[1:]

            text = ""
            try:
                synset = wn.synset(synset_name)
                definition = synset.definition()
                lemmas = ", ".join(lem.name().replace("_", " ") for lem in synset.lemmas())
                text = f"{lemmas}: {definition}"
            except Exception:
                # Fallback: use entity name as text
                text = entity_name.replace("_", " ").strip()

            f.write(f"{entity_name}\t{text}\n")

    logger.info(f"[wn18rr] Wrote {len(entity2id)} entity texts to {output_path}")


def _extract_texts_wikipedia_mapping(
    data_dir: Path, output_path: Path, text_url: str | None
) -> None:
    """Extract texts via entity-to-text mapping file (FB15k-237, YAGO3-10)."""
    if text_url is None:
        logger.warning(
            f"No text URL configured for {data_dir.name}. "
            f"Please manually create {output_path}"
        )
        return

    logger.info(f"[{data_dir.name}] Downloading entity texts from {text_url}...")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        urlretrieve(text_url, str(tmp_path))

        # Parse entity2textlong.txt format: entity_id \t text
        entity2id = _load_entity_mapping(data_dir)
        with open(tmp_path) as fin, open(output_path, "w") as fout:
            for line in fin:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    entity_id, text = parts
                    # Keep only entities that exist in our mapping (if mapping exists)
                    if not entity2id or entity_id in entity2id:
                        fout.write(f"{entity_id}\t{text}\n")

        logger.info(f"[{data_dir.name}] Entity texts written to {output_path}")
    finally:
        tmp_path.unlink(missing_ok=True)


def _extract_texts_wikidata_description(
    data_dir: Path, output_path: Path, text_url: str | None
) -> None:
    """Extract texts from WikiData entity descriptions (CoDEx-L, Wiki27K)."""
    if text_url is None:
        logger.warning(
            f"No text URL configured for {data_dir.name}. "
            f"Please manually create {output_path}"
        )
        return

    logger.info(f"[{data_dir.name}] Downloading entity descriptions from {text_url}...")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        urlretrieve(text_url, str(tmp_path))

        with open(tmp_path) as fin:
            entities_data = json.load(fin)

        with open(output_path, "w") as fout:
            for entity_id, info_dict in entities_data.items():
                if isinstance(info_dict, dict):
                    label = info_dict.get("label", "")
                    description = info_dict.get("description", "")
                    text = f"{label}: {description}" if description else label
                else:
                    text = str(info_dict)
                fout.write(f"{entity_id}\t{text}\n")

        logger.info(f"[{data_dir.name}] Entity texts written to {output_path}")
    finally:
        tmp_path.unlink(missing_ok=True)


def _extract_texts_wikipedia(data_dir: Path, output_path: Path) -> None:
    """Extract texts from WikiData5M entities (labels + descriptions)."""
    entities_path = data_dir / "entities.json"
    if entities_path.exists():
        logger.info("[wikidata_5m] Extracting texts from entities.json...")
        with open(entities_path) as f:
            entities = json.load(f)

        with open(output_path, "w") as fout:
            for entity_id, info_dict in tqdm(entities.items(), desc="Extracting texts"):
                label = info_dict.get("label", "")
                description = info_dict.get("description", "")
                text = f"{label}: {description}" if description else label
                fout.write(f"{entity_id}\t{text}\n")

        logger.info(f"[wikidata_5m] Wrote {len(entities)} entity texts to {output_path}")
    else:
        # Try HuggingFace as fallback
        try:
            from datasets import load_dataset

            logger.info("[wikidata_5m] Loading entity texts from HuggingFace...")
            ds = load_dataset("wikidata5m", split="entities", trust_remote_code=True)
            with open(output_path, "w") as fout:
                for row in tqdm(ds, desc="Writing entity texts"):
                    eid = row["id"]
                    label = row.get("label", "")
                    desc = row.get("description", "")
                    text = f"{label}: {desc}" if desc else label
                    fout.write(f"{eid}\t{text}\n")
            logger.info(f"[wikidata_5m] Entity texts written to {output_path}")
        except Exception as e:
            logger.warning(
                f"[wikidata_5m] Could not extract texts: {e}\n"
                f"Please manually create {output_path}"
            )


def _load_entity_mapping(data_dir: Path) -> dict[str, int]:
    """Load entity name → ID mapping from entities.dict."""
    entity_file = data_dir / "entities.dict"
    entity2id: dict[str, int] = {}
    if entity_file.exists():
        with open(entity_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    idx, name = int(parts[0]), parts[1]
                    entity2id[name] = idx
    return entity2id


# ─── Stage 3: Precompute text embeddings ─────────────────────────────────────


def precompute_embeddings(
    slug: str,
    data_dir: str,
    text_encoder: str,
    dim_text_emb: int,
    batch_size: int = 256,
    device: str = "cuda",
    force: bool = False,
) -> Path:
    """Stage 3: Precompute text embeddings for all entities.

    Reads entity_texts.tsv, encodes with the specified text encoder,
    and saves to text_embeddings/entity_emb_{encoder}_{dim}.pt

    Args:
        slug: Dataset slug (for logging).
        data_dir: Dataset directory path.
        text_encoder: Encoder slug or HuggingFace model name.
        dim_text_emb: Output embedding dimension.
        batch_size: Encoding batch size.
        device: Torch device for encoding.
        force: Re-encode even if file exists.

    Returns:
        Path to the saved .pt file.
    """
    path = Path(data_dir)
    emb_dir = path / "text_embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    key = encoder_slug(text_encoder)
    emb_path = emb_dir / embedding_filename(key, dim_text_emb)

    if emb_path.exists() and not force:
        logger.info(f"[{slug}] Embeddings {emb_path.name} already exist, skipping.")
        return emb_path

    texts_path = path / "entity_texts.tsv"
    if not texts_path.exists():
        logger.warning(
            f"[{slug}] entity_texts.tsv not found. "
            "Run text extraction (stage 2) first."
        )
        return emb_path

    # Load entity texts
    entity_ids: list[str] = []
    texts: list[str] = []
    with open(texts_path) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                entity_ids.append(parts[0])
                texts.append(parts[1])
            elif len(parts) == 1 and parts[0]:
                entity_ids.append(parts[0])
                texts.append(parts[0])

    if not texts:
        logger.warning(f"[{slug}] No entity texts found.")
        return emb_path

    logger.info(
        f"[{slug}] Encoding {len(texts)} entities with {key} "
        f"(dim={dim_text_emb})..."
    )

    # Use RiemannFMTextFeatureExtractor for encoding
    from riemannfm.data.text_features import RiemannFMTextFeatureExtractor

    model_name = encoder_model_name(text_encoder)
    extractor = RiemannFMTextFeatureExtractor(
        model_name=model_name,
        batch_size=batch_size,
    )

    embeddings = extractor.extract_embeddings(texts, output_path=emb_path)

    # Verify dimension matches expectation
    if embeddings.shape[1] != dim_text_emb:
        logger.warning(
            f"[{slug}] Expected dim={dim_text_emb}, got {embeddings.shape[1]}. "
            f"Saving with actual dimension."
        )
        # Rename to actual dimension
        actual_path = emb_dir / embedding_filename(key, embeddings.shape[1])
        if actual_path != emb_path:
            emb_path.rename(actual_path)
            emb_path = actual_path

    # Save entity ID → index mapping alongside embeddings
    mapping_path = emb_dir / f"entity_ids_{key}_{dim_text_emb}.json"
    with open(mapping_path, "w") as f:
        json.dump(entity_ids, f)

    logger.info(
        f"[{slug}] Saved embeddings ({embeddings.shape}) to {emb_path}"
    )
    return emb_path


# ─── Pipeline orchestration ──────────────────────────────────────────────────


def verify_data_ready(
    data_dir: str,
    text_encoder: str | None = None,
    dim_text_emb: int = 0,
) -> bool:
    """Check if a dataset directory has required files for training.

    Args:
        data_dir: Path to dataset directory.
        text_encoder: Optional encoder to verify embeddings exist.
        dim_text_emb: Embedding dimension to verify.

    Returns:
        True if required files are present.
    """
    path = Path(data_dir)
    has_triples = (
        (path / "train.txt").exists()
        or (path / "train_triples.txt").exists()
    )
    if not has_triples:
        return False

    # If text encoder specified, check embeddings exist
    if text_encoder and dim_text_emb > 0:
        key = encoder_slug(text_encoder)
        emb_path = path / "text_embeddings" / embedding_filename(key, dim_text_emb)
        if not emb_path.exists():
            logger.warning(
                f"Graph files found but text embeddings missing: {emb_path}\n"
                f"Run: make download ARGS=\"data=... text_encoder=...\""
            )

    return True


def run_pipeline(
    slug: str,
    data_dir: str,
    text_source: str,
    text_encoder: str = "sbert",
    dim_text_emb: int | None = None,
    batch_size: int = 256,
    device: str = "cuda",
    force: bool = False,
) -> None:
    """Run the full three-stage pipeline for a single dataset.

    Args:
        slug: Dataset slug (for registry lookup and logging).
        data_dir: Dataset directory path (from Hydra config).
        text_source: Text source strategy (from Hydra config).
        text_encoder: Encoder slug or model name.
        dim_text_emb: Embedding dimension (auto-derived if None).
        batch_size: Encoding batch size.
        device: Torch device.
        force: Re-download/re-encode all stages.
    """
    key = encoder_slug(text_encoder)

    if dim_text_emb is None:
        dim_text_emb = ENCODER_DIMS.get(key, 384)

    logger.info(f"{'=' * 60}")
    logger.info(f"Processing {slug} (encoder={key}, dim={dim_text_emb})")
    logger.info(f"{'=' * 60}")

    # Stage 1: Download graph
    download_graph(slug, data_dir, force=force)

    # Stage 2: Extract entity texts
    extract_entity_texts(slug, data_dir, text_source, force=force)

    # Stage 3: Precompute embeddings (skip if text_encoder is "none")
    if text_encoder != "none" and key != "none":
        precompute_embeddings(
            slug,
            data_dir,
            text_encoder=text_encoder,
            dim_text_emb=dim_text_emb,
            batch_size=batch_size,
            device=device,
            force=force,
        )
    else:
        logger.info(f"[{slug}] text_encoder=none, skipping embedding precomputation.")

    logger.info(f"[{slug}] Pipeline complete.")
