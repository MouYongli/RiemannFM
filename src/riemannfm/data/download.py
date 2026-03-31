"""Dataset download and text extraction.

Provides a two-stage pipeline for downloading raw KG data:

1. Download graph structure (triples, entity/relation mappings) → raw/
2. Extract entity text descriptions → raw/entity_texts.tsv

Each stage is idempotent — skipped if output files already exist.
Text embedding precomputation is handled separately by the preprocess CLI.
"""

import gzip
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


@dataclass
class DownloadMeta:
    """Download-specific metadata for a KG dataset."""

    download_url: str | None
    download_format: str
    text_url: str | None


_DOWNLOAD_REGISTRY: dict[str, DownloadMeta] = {
    "wikidata_5m": DownloadMeta(
        download_url="https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1",
        download_format="wikidata5m",
        text_url="https://www.dropbox.com/s/7jp4ib8zo3i6m10/wikidata5m_text.txt.gz?dl=1",
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


# ─── Stage 1: Download graph structure → raw/ ────────────────────────────────


def download_graph(
    slug: str,
    data_dir: str,
    force: bool = False,
) -> Path:
    """Stage 1: Download raw graph structure files.

    Downloads triples and entity/relation mappings into {data_dir}/raw/.
    Skips if raw/train.txt (or raw/train_triples.txt) already exists.

    Args:
        slug: Dataset slug for registry lookup.
        data_dir: Base dataset directory (raw/ is appended).
        force: Re-download even if files exist.

    Returns:
        Path to the raw/ directory.
    """
    meta = get_download_meta(slug)
    raw_dir = Path(data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    has_triples = (
        (raw_dir / "train.txt").exists()
        or (raw_dir / "train_triples.txt").exists()
    )
    if has_triples and not force:
        logger.info(f"[{slug}] Graph files already exist in raw/, skipping download.")
        return raw_dir

    if meta.download_format == "tar.gz":
        _download_tar_gz(slug, meta, raw_dir)
    elif meta.download_format == "wikidata5m":
        _download_wikidata5m(slug, meta, raw_dir)
    elif meta.download_format == "codex":
        _download_codex(slug, raw_dir)
    elif meta.download_format == "zip":
        _download_zip(slug, meta, raw_dir)
    else:
        logger.warning(f"[{slug}] Unknown download format: {meta.download_format}")

    return raw_dir


def _download_tar_gz(slug: str, meta: DownloadMeta, raw_dir: Path) -> None:
    """Download and extract a tar.gz archive."""
    if meta.download_url is None:
        logger.warning(
            f"[{slug}] No download URL configured. "
            f"Please manually place graph files in {raw_dir}/"
        )
        return

    logger.info(f"[{slug}] Downloading from {meta.download_url}...")
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "dataset.tar.gz"
        urlretrieve(meta.download_url, str(archive_path))
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(tmpdir)

        for extracted_dir in Path(tmpdir).iterdir():
            if extracted_dir.is_dir() and extracted_dir.name != "__MACOSX":
                for f in extracted_dir.iterdir():
                    if f.is_file():
                        shutil.copy2(f, raw_dir / f.name)
                break

    logger.info(f"[{slug}] Graph files downloaded to {raw_dir}/")


def _download_wikidata5m(slug: str, meta: DownloadMeta, raw_dir: Path) -> None:
    """Download WikiData5M transductive split from Dropbox.

    Downloads the tar.gz archive containing train.txt, valid.txt, test.txt
    and renames them to the project-standard naming convention.
    """
    if meta.download_url is None:
        logger.warning(f"[{slug}] No download URL configured.")
        return

    logger.info(f"[{slug}] Downloading WikiData5M transductive split...")
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "wikidata5m_transductive.tar.gz"
        urlretrieve(meta.download_url, str(archive_path))
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(tmpdir)

        # Archive contains: wikidata5m_transductive_{train,valid,test}.txt
        rename_map = {
            "wikidata5m_transductive_train.txt": "train_triples.txt",
            "wikidata5m_transductive_valid.txt": "val_triples.txt",
            "wikidata5m_transductive_test.txt": "test_triples.txt",
        }
        for src_name, dst_name in rename_map.items():
            candidates = list(Path(tmpdir).rglob(src_name))
            if candidates:
                shutil.copy2(candidates[0], raw_dir / dst_name)
                logger.info(f"  {src_name} → {dst_name}")
            else:
                logger.warning(f"  {src_name} not found in archive")

    logger.info(f"[{slug}] Graph files downloaded to {raw_dir}/")


def _download_codex(slug: str, raw_dir: Path) -> None:
    """Download CoDEx-L dataset from GitHub."""
    logger.info(f"[{slug}] Downloading CoDEx-L splits...")
    for split_name, url in _CODEX_SPLIT_URLS.items():
        target = raw_dir / f"{split_name}.txt"
        logger.info(f"  Downloading {split_name}...")
        urlretrieve(url, str(target))

    entities_url = f"{_CODEX_BASE}/entities.dict"
    relations_url = f"{_CODEX_BASE}/relations.dict"
    for url, fname in [(entities_url, "entities.dict"), (relations_url, "relations.dict")]:
        target = raw_dir / fname
        try:
            urlretrieve(url, str(target))
        except Exception:
            logger.warning(f"[{slug}] Could not download {fname}, may need manual setup.")

    logger.info(f"[{slug}] CoDEx-L downloaded to {raw_dir}/")


def _download_zip(slug: str, meta: DownloadMeta, raw_dir: Path) -> None:
    """Download and extract a zip archive."""
    if meta.download_url is None:
        logger.warning(
            f"[{slug}] No download URL configured. "
            f"Please manually place graph files in {raw_dir}/"
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
                        shutil.copy2(f, raw_dir / f.name)
                break

    logger.info(f"[{slug}] Graph files downloaded to {raw_dir}/")


# ─── Stage 2: Extract entity texts → raw/entity_texts.tsv ────────────────────


def extract_entity_texts(
    slug: str,
    data_dir: str,
    text_source: str,
    force: bool = False,
) -> Path:
    """Stage 2: Extract or download entity text descriptions.

    Produces raw/entity_texts.tsv with columns: entity_id<TAB>text

    Args:
        slug: Dataset slug for registry lookup (text_url).
        data_dir: Base dataset directory (raw/ is appended).
        text_source: Text source strategy name.
        force: Re-extract even if file exists.

    Returns:
        Path to entity_texts.tsv.
    """
    meta = get_download_meta(slug)
    raw_dir = Path(data_dir) / "raw"
    output_path = raw_dir / "entity_texts.tsv"

    if output_path.exists() and not force:
        logger.info(f"[{slug}] entity_texts.tsv already exists, skipping.")
        return output_path

    raw_dir.mkdir(parents=True, exist_ok=True)

    if text_source == "wordnet_defs":
        _extract_texts_wordnet(raw_dir, output_path)
    elif text_source == "wikipedia_mapping":
        _extract_texts_wikipedia_mapping(raw_dir, output_path, meta.text_url)
    elif text_source == "wikidata_description":
        _extract_texts_wikidata_description(raw_dir, output_path, meta.text_url)
    elif text_source == "wikipedia":
        _extract_texts_wikipedia(raw_dir, output_path)
    else:
        logger.warning(f"[{slug}] Unknown text source: {text_source}")

    return output_path


def _extract_texts_wordnet(raw_dir: Path, output_path: Path) -> None:
    """Extract texts from WordNet synset definitions for WN18RR."""
    import nltk

    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import wordnet as wn

    entity2id = _load_entity_mapping(raw_dir)
    if not entity2id:
        logger.warning("[wn18rr] No entities.dict found, cannot extract texts.")
        return

    logger.info(f"[wn18rr] Extracting WordNet definitions for {len(entity2id)} entities...")
    with open(output_path, "w") as f:
        for entity_name in tqdm(entity2id, desc="Extracting WordNet texts"):
            synset_name = entity_name.strip("_").lstrip("_")
            while synset_name.startswith("_"):
                synset_name = synset_name[1:]

            text = ""
            try:
                synset = wn.synset(synset_name)
                definition = synset.definition()
                lemmas = ", ".join(lem.name().replace("_", " ") for lem in synset.lemmas())
                text = f"{lemmas}: {definition}"
            except Exception:
                text = entity_name.replace("_", " ").strip()

            f.write(f"{entity_name}\t{text}\n")

    logger.info(f"[wn18rr] Wrote {len(entity2id)} entity texts to {output_path}")


def _extract_texts_wikipedia_mapping(
    raw_dir: Path, output_path: Path, text_url: str | None
) -> None:
    """Extract texts via entity-to-text mapping file (FB15k-237, YAGO3-10)."""
    if text_url is None:
        logger.warning(
            f"No text URL configured for {raw_dir.parent.name}. "
            f"Please manually create {output_path}"
        )
        return

    logger.info(f"[{raw_dir.parent.name}] Downloading entity texts from {text_url}...")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        urlretrieve(text_url, str(tmp_path))

        entity2id = _load_entity_mapping(raw_dir)
        with open(tmp_path) as fin, open(output_path, "w") as fout:
            for line in fin:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    entity_id, text = parts
                    if not entity2id or entity_id in entity2id:
                        fout.write(f"{entity_id}\t{text}\n")

        logger.info(f"[{raw_dir.parent.name}] Entity texts written to {output_path}")
    finally:
        tmp_path.unlink(missing_ok=True)


def _extract_texts_wikidata_description(
    raw_dir: Path, output_path: Path, text_url: str | None
) -> None:
    """Extract texts from WikiData entity descriptions (CoDEx-L, Wiki27K)."""
    if text_url is None:
        logger.warning(
            f"No text URL configured for {raw_dir.parent.name}. "
            f"Please manually create {output_path}"
        )
        return

    logger.info(f"[{raw_dir.parent.name}] Downloading entity descriptions from {text_url}...")
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

        logger.info(f"[{raw_dir.parent.name}] Entity texts written to {output_path}")
    finally:
        tmp_path.unlink(missing_ok=True)


def _extract_texts_wikipedia(raw_dir: Path, output_path: Path) -> None:
    """Extract texts from WikiData5M entities.

    Prefers entities.json (if present from a prior download).
    Otherwise downloads the official wikidata5m_text.txt.gz corpus.
    """
    entities_path = raw_dir / "entities.json"
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
        return

    meta = get_download_meta("wikidata_5m")
    if meta.text_url is None:
        logger.warning(f"[wikidata_5m] No text URL configured. Please create {output_path} manually.")
        return

    logger.info("[wikidata_5m] Downloading text corpus from Dropbox...")
    with tempfile.TemporaryDirectory() as tmpdir:
        gz_path = Path(tmpdir) / "wikidata5m_text.txt.gz"
        urlretrieve(meta.text_url, str(gz_path))

        count = 0
        with gzip.open(gz_path, "rt", encoding="utf-8") as fin, open(output_path, "w") as fout:
            for line in tqdm(fin, desc="Extracting entity texts"):
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    entity_id, text = parts
                    # Use first sentence as description (full text is very long)
                    first_sent = text.split(". ")[0].rstrip(".")
                    fout.write(f"{entity_id}\t{first_sent}\n")
                    count += 1

        logger.info(f"[wikidata_5m] Wrote {count} entity texts to {output_path}")


def _load_entity_mapping(raw_dir: Path) -> dict[str, int]:
    """Load entity name → ID mapping from entities.dict."""
    entity_file = raw_dir / "entities.dict"
    entity2id: dict[str, int] = {}
    if entity_file.exists():
        with open(entity_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    idx, name = int(parts[0]), parts[1]
                    entity2id[name] = idx
    return entity2id


# ─── Pipeline orchestration ──────────────────────────────────────────────────


def verify_data_ready(data_dir: str) -> bool:
    """Check if raw graph files exist for training.

    Args:
        data_dir: Base dataset directory (checks raw/ subdirectory).

    Returns:
        True if raw graph files are present.
    """
    raw = Path(data_dir) / "raw"
    return (
        (raw / "train.txt").exists()
        or (raw / "train_triples.txt").exists()
    )


def run_pipeline(
    slug: str,
    data_dir: str,
    text_source: str,
    force: bool = False,
) -> None:
    """Run the download pipeline: graph structure + entity texts.

    Args:
        slug: Dataset slug (for registry lookup and logging).
        data_dir: Base dataset directory (from Hydra config).
        text_source: Text source strategy (from Hydra config).
        force: Re-download even if files exist.
    """
    logger.info(f"{'=' * 60}")
    logger.info(f"Downloading {slug}")
    logger.info(f"{'=' * 60}")

    download_graph(slug, data_dir, force=force)
    extract_entity_texts(slug, data_dir, text_source, force=force)

    logger.info(f"[{slug}] Download complete.")
