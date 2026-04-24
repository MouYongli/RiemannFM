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

# ─── Download-specific registry ──────────────────────────────────────────────


@dataclass
class DownloadMeta:
    """Download-specific metadata for a KG dataset."""

    download_url: str | None
    download_format: str
    text_url: str | None
    alias_url: str | None = None


_DOWNLOAD_REGISTRY: dict[str, DownloadMeta] = {
    "wikidata_5m": DownloadMeta(
        download_url="https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1",
        download_format="wikidata5m",
        text_url="https://www.dropbox.com/s/7jp4ib8zo3i6m10/wikidata5m_text.txt.gz?dl=1",
        alias_url="https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1",
    ),
    "fb15k_237": DownloadMeta(
        download_url="https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237",
        download_format="github_files",
        text_url="https://raw.githubusercontent.com/wangbo9719/StAR_KGC/master/StAR/data/FB15k-237/entity2textlong.txt",
    ),
    "wn18rr": DownloadMeta(
        download_url="https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/original",
        download_format="github_files",
        text_url=None,
    ),
    "codex_l": DownloadMeta(
        download_url=None,
        download_format="codex",
        text_url="https://raw.githubusercontent.com/tsafavi/codex/master/data/entities/en/entities.json",
    ),
    # Wiki27K: from THU-KEG/PKGC (Lv et al., ACL Findings 2022).
    # Google Drive hosts the full dataset zip containing Wiki27K/ and FB15K-237-N/.
    "wiki27k": DownloadMeta(
        download_url="https://drive.google.com/uc?id=1waE2QeVepwntuTDYNncBa6y94EPDn69C",
        download_format="wiki27k",
        text_url=None,
    ),
    "yago3_10": DownloadMeta(
        download_url="https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10",
        download_format="github_files",
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

    has_triples = (raw_dir / "train.txt").exists() or (raw_dir / "train_triples.txt").exists()
    if has_triples and not force:
        logger.info(f"[{slug}] Graph files already exist in raw/, skipping download.")
        return raw_dir

    if meta.download_format == "wikidata5m":
        _download_wikidata5m(slug, meta, raw_dir)
    elif meta.download_format == "codex":
        _download_codex(slug, raw_dir)
    elif meta.download_format == "github_files":
        _download_github_files(slug, meta, raw_dir)
    elif meta.download_format == "wiki27k":
        _download_wiki27k(slug, meta, raw_dir)
    else:
        logger.warning(f"[{slug}] Unknown download format: {meta.download_format}")

    return raw_dir


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
            tar.extractall(tmpdir, filter="data")

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


def _download_github_files(slug: str, meta: DownloadMeta, raw_dir: Path) -> None:
    """Download individual files from a GitHub raw directory URL.

    Expects ``meta.download_url`` to be a raw.githubusercontent.com directory base,
    e.g. ``https://raw.githubusercontent.com/owner/repo/branch/path/to/dir``.
    Downloads ``train.txt``, ``valid.txt``, ``test.txt`` and optional dict files.
    """
    if meta.download_url is None:
        logger.warning(f"[{slug}] No download URL configured. Please manually place graph files in {raw_dir}/")
        return

    base_url = meta.download_url.rstrip("/")
    files_to_try = [
        "train.txt",
        "valid.txt",
        "test.txt",
        "entities.dict",
        "relations.dict",
        "entity2wikidata.json",
    ]

    logger.info(f"[{slug}] Downloading files from {base_url}/...")
    for fname in files_to_try:
        url = f"{base_url}/{fname}"
        dest = raw_dir / fname
        try:
            urlretrieve(url, str(dest))
            logger.info(f"[{slug}]   ✓ {fname}")
        except Exception as exc:
            if fname in ("train.txt", "valid.txt", "test.txt"):
                logger.warning(
                    f"[{slug}]   ✗ {fname} failed to download: {exc!r} "
                    "(required file — check network / upstream availability)",
                    exc_info=True,
                )
            else:
                # Optional files (dicts, json): log at debug for diagnosis
                # without cluttering normal output.
                logger.debug(f"[{slug}]   ✗ {fname} (optional) skipped: {exc!r}")

    logger.info(f"[{slug}] Graph files downloaded to {raw_dir}/")


def _download_wiki27k(slug: str, meta: DownloadMeta, raw_dir: Path) -> None:
    """Download Wiki27K from Google Drive via gdown.

    The PKGC dataset zip contains ``dataset/Wiki27K/{train,valid,test}.txt``
    plus ``entity2label.txt``, ``entity2definition.txt``, ``relation2label.json``.
    Triples and text metadata are extracted; entity/relation texts are built
    from label + definition files.
    """
    if meta.download_url is None:
        logger.warning(f"[{slug}] No download URL. Place files in {raw_dir}/ manually.")
        return

    try:
        import gdown
    except ImportError:
        logger.warning(f"[{slug}] gdown not installed. Run: pip install gdown")
        return

    logger.info(f"[{slug}] Downloading PKGC dataset from Google Drive...")
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "dataset.zip"
        gdown.download(meta.download_url, str(zip_path), quiet=False)

        with zipfile.ZipFile(zip_path, "r") as zf:
            prefix = "dataset/Wiki27K/"
            raw_dir_resolved = raw_dir.resolve()
            for member in zf.namelist():
                if not (member.startswith(prefix) and not member.endswith("/")):
                    continue
                fname = member[len(prefix) :]
                target = (raw_dir / fname).resolve()
                if raw_dir_resolved not in target.parents and target != raw_dir_resolved:
                    logger.warning(f"[{slug}] Skipping unsafe archive member: {member}")
                    continue
                with zf.open(member) as fin, open(target, "wb") as fout:
                    shutil.copyfileobj(fin, fout)

    # Build entity_texts.tsv from label + definition.
    _build_wiki27k_entity_texts(raw_dir)
    # Build relation_texts.tsv from relation2label.json.
    _build_wiki27k_relation_texts(raw_dir)

    logger.info(f"[{slug}] Graph files downloaded to {raw_dir}/")


def _build_wiki27k_entity_texts(raw_dir: Path) -> None:
    """Build entity_texts.tsv from entity2label.txt and entity2definition.txt."""
    labels: dict[str, str] = {}
    label_path = raw_dir / "entity2label.txt"
    if label_path.exists():
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    labels[parts[0]] = parts[1]

    defs: dict[str, str] = {}
    def_path = raw_dir / "entity2definition.txt"
    if def_path.exists():
        with open(def_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    defs[parts[0]] = parts[1]

    entities: set[str] = set()
    for name in ("train.txt", "valid.txt", "test.txt"):
        path = raw_dir / name
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        entities.add(parts[0])
                        entities.add(parts[2])

    output = raw_dir / "entity_texts.tsv"
    with open(output, "w", encoding="utf-8") as f:
        for eid in sorted(entities):
            label = labels.get(eid, eid)
            defn = defs.get(eid, "")
            text = f"{label}: {defn}" if defn else label
            f.write(f"{eid}\t{text}\n")

    logger.info(f"[wiki27k] Wrote {len(entities)} entity texts to {output}")


def _build_wiki27k_relation_texts(raw_dir: Path) -> None:
    """Build relation_texts.tsv from relation2label.json."""
    label_path = raw_dir / "relation2label.json"
    rel_labels: dict[str, str] = {}
    if label_path.exists():
        with open(label_path, encoding="utf-8") as f:
            rel_labels = json.load(f)

    relations: set[str] = set()
    for name in ("train.txt", "valid.txt", "test.txt"):
        path = raw_dir / name
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        relations.add(parts[1])

    output = raw_dir / "relation_texts.tsv"
    with open(output, "w", encoding="utf-8") as f:
        for rid in sorted(relations):
            text = rel_labels.get(rid, rid)
            f.write(f"{rid}\t{text}\n")

    logger.info(f"[wiki27k] Wrote {len(relations)} relation texts to {output}")


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
    elif text_source == "entity_name":
        _extract_texts_entity_name(raw_dir, output_path)
    else:
        logger.warning(f"[{slug}] Unknown text source: {text_source}")

    return output_path


def _extract_texts_wordnet(raw_dir: Path, output_path: Path) -> None:
    """Extract texts from WordNet synset definitions for WN18RR.

    Supports two entity formats:
    - Synset names (e.g. ``land_reform.n.01``) from ``entities.dict``
    - Numeric offsets (e.g. ``00260881``) from ``original/`` triples,
      resolved via a parallel ``text/`` download from the same repo.
    """
    import nltk

    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import wordnet as wn

    entity2id = _load_entity_mapping(raw_dir)

    if not entity2id:
        # No entities.dict — entities are numeric offsets.
        # Build offset→synset mapping from the text/ triples variant.
        entity2id, offset_to_synset = _build_wn18rr_offset_mapping(raw_dir)
        if not entity2id:
            logger.warning("[wn18rr] Cannot resolve entity offsets, cannot extract texts.")
            return
    else:
        offset_to_synset = None

    logger.info(f"[wn18rr] Extracting WordNet definitions for {len(entity2id)} entities...")
    with open(output_path, "w", encoding="utf-8") as f:
        for entity_name in tqdm(entity2id, desc="Extracting WordNet texts"):
            # Resolve synset name from offset if needed.
            if offset_to_synset is not None:
                synset_name = offset_to_synset.get(entity_name, entity_name)
            else:
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
                text = synset_name.replace("_", " ").replace(".", " ").strip()

            f.write(f"{entity_name}\t{text}\n")

    logger.info(f"[wn18rr] Wrote {len(entity2id)} entity texts to {output_path}")


def _build_wn18rr_offset_mapping(
    raw_dir: Path,
) -> tuple[dict[str, int], dict[str, str]]:
    """Build offset→synset mapping for WN18RR by downloading text/ triples.

    The ``original/`` triples use numeric offsets (e.g. ``00260881``),
    while the ``text/`` triples use synset names (e.g. ``land_reform.n.01``).
    Both variants share the same triple ordering, enabling a 1-to-1 mapping.

    Returns:
        (entity2id, offset_to_synset) — entity2id maps offset strings to
        dummy integer IDs; offset_to_synset maps offset to synset name.
    """
    _WN_TEXT_BASE = "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/text"

    logger.info("[wn18rr] Building offset→synset mapping from text/ triples...")

    offset_to_synset: dict[str, str] = {}
    offsets: set[str] = set()

    # Collect all offsets from the original triples.
    for split in ("train.txt", "valid.txt", "test.txt"):
        path = raw_dir / split
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    offsets.add(parts[0])
                    offsets.add(parts[2])

    # Download text triples and pair with original triples.
    for split in ("train.txt", "valid.txt", "test.txt"):
        orig_path = raw_dir / split
        if not orig_path.exists():
            continue
        url = f"{_WN_TEXT_BASE}/{split}"
        try:
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as tmp:
                tmp_path = Path(tmp.name)
            urlretrieve(url, str(tmp_path))

            with open(orig_path, encoding="utf-8") as f_orig, open(tmp_path, encoding="utf-8") as f_text:
                for orig_line, text_line in zip(f_orig, f_text):
                    orig_parts = orig_line.strip().split("\t")
                    text_parts = text_line.strip().split("\t")
                    if len(orig_parts) >= 3 and len(text_parts) >= 3:
                        offset_to_synset[orig_parts[0]] = text_parts[0]
                        offset_to_synset[orig_parts[2]] = text_parts[2]
        except Exception as e:
            logger.warning(f"[wn18rr] Failed to download text/{split}: {e}")
        finally:
            tmp_path.unlink(missing_ok=True)

    if not offset_to_synset:
        return {}, {}

    logger.info(f"[wn18rr] Mapped {len(offset_to_synset)} offsets to synset names.")
    entity2id = {offset: i for i, offset in enumerate(sorted(offsets))}
    return entity2id, offset_to_synset


def _extract_texts_wikipedia_mapping(raw_dir: Path, output_path: Path, text_url: str | None) -> None:
    """Extract texts via entity-to-text mapping file (FB15k-237, YAGO3-10)."""
    if text_url is None:
        logger.warning(f"No text URL configured for {raw_dir.parent.name}. Please manually create {output_path}")
        return

    logger.info(f"[{raw_dir.parent.name}] Downloading entity texts from {text_url}...")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        urlretrieve(text_url, str(tmp_path))

        entity2id = _load_entity_mapping(raw_dir)
        with open(tmp_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    entity_id, text = parts
                    if not entity2id or entity_id in entity2id:
                        fout.write(f"{entity_id}\t{text}\n")

        logger.info(f"[{raw_dir.parent.name}] Entity texts written to {output_path}")
    finally:
        tmp_path.unlink(missing_ok=True)


def _extract_texts_wikidata_description(raw_dir: Path, output_path: Path, text_url: str | None) -> None:
    """Extract texts from WikiData entity descriptions (CoDEx-L, Wiki27K)."""
    if text_url is None:
        logger.warning(f"No text URL configured for {raw_dir.parent.name}. Please manually create {output_path}")
        return

    logger.info(f"[{raw_dir.parent.name}] Downloading entity descriptions from {text_url}...")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        urlretrieve(text_url, str(tmp_path))

        with open(tmp_path, encoding="utf-8") as fin:
            entities_data = json.load(fin)

        with open(output_path, "w", encoding="utf-8") as fout:
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
        with open(entities_path, encoding="utf-8") as f:
            entities = json.load(f)

        with open(output_path, "w", encoding="utf-8") as fout:
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
        with gzip.open(gz_path, "rt", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
            for line in tqdm(fin, desc="Extracting entity texts"):
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    entity_id, text = parts
                    first_sent = text.split(". ")[0].strip().rstrip(".")
                    if not first_sent:
                        # Text starts with "..." or is empty — use entity ID
                        first_sent = entity_id
                    fout.write(f"{entity_id}\t{first_sent}\n")
                    count += 1
                elif len(parts) == 1 and parts[0]:
                    # Entity with no Wikipedia text — use entity ID
                    fout.write(f"{parts[0]}\t{parts[0]}\n")
                    count += 1

        logger.info(f"[wikidata_5m] Wrote {count} entity texts to {output_path}")


def _extract_texts_entity_name(raw_dir: Path, output_path: Path) -> None:
    """Generate entity texts by cleaning up entity name strings.

    Converts entity identifiers to human-readable text:
      - ``Albert_Einstein`` → ``Albert Einstein``
      - ``/m/012345`` → ``m 012345``

    Used for datasets (e.g. YAGO3-10) where no external text source is available.
    """
    entity2id = _load_entity_mapping(raw_dir)
    if not entity2id:
        # Fall back to collecting entity IDs from triple files.
        entities: set[str] = set()
        for name in ("train.txt", "train_triples.txt"):
            path = raw_dir / name
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 3:
                            entities.add(parts[0])
                            entities.add(parts[2])
                break
        entity_names = sorted(entities)
    else:
        entity_names = list(entity2id.keys())

    logger.info(f"[entity_name] Generating text for {len(entity_names)} entities from names...")

    with open(output_path, "w", encoding="utf-8") as f:
        for name in tqdm(entity_names, desc="Cleaning entity names"):
            cleaned = name.replace("_", " ").replace("/", " ").strip()
            # Remove leading dots or special chars.
            cleaned = cleaned.lstrip(". ")
            if not cleaned:
                cleaned = name
            f.write(f"{name}\t{cleaned}\n")

    logger.info(f"[entity_name] Wrote {len(entity_names)} entity texts to {output_path}")


def _load_entity_mapping(raw_dir: Path) -> dict[str, int]:
    """Load entity name → ID mapping from entities.dict."""
    entity_file = raw_dir / "entities.dict"
    entity2id: dict[str, int] = {}
    if entity_file.exists():
        with open(entity_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    idx, name = int(parts[0]), parts[1]
                    entity2id[name] = idx
    return entity2id


# ─── Stage 3: Extract relation texts → raw/relation_texts.tsv ──────────────


def extract_relation_texts(
    slug: str,
    data_dir: str,
    force: bool = False,
) -> Path:
    """Stage 3: Extract or generate relation text descriptions.

    Produces raw/relation_texts.tsv with columns: relation_id<TAB>text

    Args:
        slug: Dataset slug for registry lookup.
        data_dir: Base dataset directory (raw/ is appended).
        force: Re-extract even if file exists.

    Returns:
        Path to relation_texts.tsv.
    """
    raw_dir = Path(data_dir) / "raw"
    output_path = raw_dir / "relation_texts.tsv"

    if output_path.exists() and not force:
        logger.info(f"[{slug}] relation_texts.tsv already exists, skipping.")
        return output_path

    raw_dir.mkdir(parents=True, exist_ok=True)

    if slug == "wikidata_5m":
        _extract_relation_texts_wikidata5m(raw_dir, output_path)
    elif slug == "wn18rr":
        _extract_relation_texts_from_triples(raw_dir, output_path, style="wordnet")
    elif slug in ("fb15k_237", "yago3_10"):
        _extract_relation_texts_from_triples(raw_dir, output_path, style="freebase")
    elif slug == "codex_l":
        _extract_relation_texts_codex(raw_dir, output_path)
    else:
        _extract_relation_texts_from_triples(raw_dir, output_path, style="generic")

    return output_path


def _extract_relation_texts_wikidata5m(raw_dir: Path, output_path: Path) -> None:
    """Extract relation texts from WikiData5M alias archive.

    Downloads wikidata5m_alias.tar.gz and parses wikidata5m_relation.txt
    which has format: P-ID<TAB>alias1<TAB>alias2<TAB>...
    Uses first alias as label, remaining as description.
    """
    meta = get_download_meta("wikidata_5m")
    if meta.alias_url is None:
        logger.warning("[wikidata_5m] No alias URL configured.")
        return

    logger.info("[wikidata_5m] Downloading alias archive for relation texts...")
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "wikidata5m_alias.tar.gz"
        urlretrieve(meta.alias_url, str(archive_path))
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(tmpdir)

        rel_files = list(Path(tmpdir).rglob("wikidata5m_relation.txt"))
        if not rel_files:
            logger.warning("[wikidata_5m] wikidata5m_relation.txt not found in alias archive.")
            return

        # Parse alias file
        known: dict[str, str] = {}
        with open(rel_files[0], encoding="utf-8") as fin:
            for line in fin:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    rel_id = parts[0]
                    label = parts[1]
                    aliases = ", ".join(parts[2:]) if len(parts) > 2 else ""
                    known[rel_id] = f"{label}: {aliases}" if aliases else label

        # Collect all relations from triples to fill gaps
        all_rels: set[str] = set()
        for fname in ("train_triples.txt", "val_triples.txt", "test_triples.txt"):
            fpath = raw_dir / fname
            if fpath.exists():
                with open(fpath, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) == 3:
                            all_rels.add(parts[1])

        count = 0
        with open(output_path, "w", encoding="utf-8") as fout:
            for rel_id in sorted(all_rels | known.keys()):
                text = known.get(rel_id, rel_id)
                fout.write(f"{rel_id}\t{text}\n")
                count += 1

    logger.info(f"[wikidata_5m] Wrote {count} relation texts to {output_path}")


def _extract_relation_texts_from_triples(raw_dir: Path, output_path: Path, style: str) -> None:
    """Extract relation texts by collecting unique relations from triple files.

    Converts relation IDs to human-readable text based on naming style:
    - wordnet: '_hypernym' → 'hypernym'
    - freebase: '/film/film/genre' → 'film film genre'
    - generic: pass through as-is
    """
    relations: set[str] = set()
    for fname in ("train.txt", "train_triples.txt"):
        fpath = raw_dir / fname
        if fpath.exists():
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        relations.add(parts[1])

    if not relations:
        logger.warning(f"No relations found in {raw_dir}")
        return

    count = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for rel in sorted(relations):
            if style == "wordnet":
                text = rel.lstrip("_").replace("_", " ")
            elif style == "freebase":
                text = rel.strip("/").replace("/", " ").replace("_", " ")
            else:
                text = rel.replace("_", " ")
            fout.write(f"{rel}\t{text}\n")
            count += 1

    logger.info(f"Wrote {count} relation texts to {output_path}")


def _extract_relation_texts_codex(raw_dir: Path, output_path: Path) -> None:
    """Extract relation texts for CoDEx-L from relations.dict.

    CoDEx relations.dict has format: index<TAB>relation_label
    """
    rel_file = raw_dir / "relations.dict"
    if not rel_file.exists():
        logger.warning(f"[codex_l] {rel_file} not found, falling back to triple extraction.")
        _extract_relation_texts_from_triples(raw_dir, output_path, style="generic")
        return

    count = 0
    with open(rel_file, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                _, rel_name = parts
                text = rel_name.replace("_", " ")
                fout.write(f"{rel_name}\t{text}\n")
                count += 1

    logger.info(f"[codex_l] Wrote {count} relation texts to {output_path}")


# ─── Pipeline orchestration ──────────────────────────────────────────────────


def verify_data_ready(data_dir: str) -> bool:
    """Check if raw graph files exist for training.

    Args:
        data_dir: Base dataset directory (checks raw/ subdirectory).

    Returns:
        True if raw graph files are present.
    """
    raw = Path(data_dir) / "raw"
    return (raw / "train.txt").exists() or (raw / "train_triples.txt").exists()


def run_pipeline(
    slug: str,
    data_dir: str,
    text_source: str,
    force: bool = False,
) -> None:
    """Run the full download pipeline for a dataset.

    Stages (each idempotent — skipped when outputs already exist unless
    ``force=True``):

    1. ``download_graph``: fetch triples and entity/relation mappings → ``raw/``.
    2. ``extract_entity_texts``: build ``raw/entity_texts.tsv`` according to
       the dataset's ``text_source`` strategy (Wikipedia first-paragraph,
       WordNet definitions, WikiData labels, etc.).
    3. ``extract_relation_texts``: build ``raw/relation_texts.tsv`` from the
       dataset's relation inventory.
    4. Optional post-download validation via
       ``riemannfm.data.pipeline.validate.validate_raw`` when that module is
       present (it lands with the preprocess Issue); skipped otherwise.

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
    extract_relation_texts(slug, data_dir, force=force)

    # validate_raw lives in riemannfm.data.pipeline.validate which is not yet
    # on main (arrives with the preprocess Issue). Invoke it only if present
    # so end-to-end download works in isolation.
    try:
        from riemannfm.data.pipeline.validate import validate_raw
    except ImportError:
        logger.info(f"[{slug}] validate_raw not available yet; skipping post-download validation.")
    else:
        if not validate_raw(data_dir, slug=slug):
            raise RuntimeError(f"[{slug}] Validation failed — check logs above.")

    logger.info(f"[{slug}] Download complete.")
