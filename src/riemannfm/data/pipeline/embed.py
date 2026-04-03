"""Multi-backend text embedding for KG entities and relations.

Supports three embedding providers:
- huggingface: Local transformer models (SBERT)
- ollama: Ollama server via OpenAI-compatible /v1/embeddings endpoint (Qwen3, Nomic)
- openai: OpenAI API via /v1/embeddings endpoint

Results are cached to disk. Embedding files are named
entity_emb_{encoder}_{dim}.pt for ablation coexistence.
"""

import logging
import os
import random
import time
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ─── Encoder slug mapping ────────────────────────────────────────────────────

_SLUG_MAP: dict[str, str] = {
    "sentence-transformers/all-MiniLM-L6-v2": "sbert",
    "all-minilm:l6-v2": "sbert",
    "qwen3-embedding": "qwen3",
    "qwen3-embedding:8b": "qwen3",
    "nomic-embed-text": "nomic",
}

_SLUG_TO_MODEL: dict[str, str] = {
    "sbert": "sentence-transformers/all-MiniLM-L6-v2",
    "qwen3": "qwen3-embedding:8b",
    "nomic": "nomic-embed-text",
}

ENCODER_DIMS: dict[str, int] = {
    "sbert": 384,
    "qwen3": 768,
    "nomic": 768,
}


def encoder_slug(text_encoder: str) -> str:
    """Derive a filesystem-safe short key from a model name or config key.

    Args:
        text_encoder: Model name, HuggingFace path, or short slug.

    Returns:
        Short key like "sbert", "nomic", "qwen3".
    """
    if text_encoder in _SLUG_MAP:
        return _SLUG_MAP[text_encoder]
    if text_encoder in _SLUG_TO_MODEL:
        return text_encoder
    return text_encoder.split("/")[-1].replace("-", "_").lower()


def encoder_model_name(text_encoder: str) -> str:
    """Resolve a short slug to the full model name.

    Args:
        text_encoder: Short slug or full model name.

    Returns:
        Full model name.
    """
    if text_encoder in _SLUG_TO_MODEL:
        return _SLUG_TO_MODEL[text_encoder]
    return text_encoder


def embedding_filename(encoder_key: str, dim: int, prefix: str = "entity") -> str:
    """Build the deterministic embedding filename.

    Args:
        encoder_key: Short encoder slug.
        dim: Embedding dimension.
        prefix: File prefix ("entity" or "relation").

    Returns:
        Filename like "entity_emb_sbert_384.pt".
    """
    return f"{prefix}_emb_{encoder_key}_{dim}.pt"


# ─── Multi-backend embedder ──────────────────────────────────────────────────


class RiemannFMTextEmbedder:
    """Encode texts into embeddings via HuggingFace, Ollama, or OpenAI.

    Args:
        embedding_provider: Backend to use ("huggingface", "ollama", "openai").
        model_name: Model identifier (HF path or API model name).
        output_dim: Expected output embedding dimension.
        api_base: API base URL for ollama/openai backends.
        api_key: API key for openai backend (default: reads OPENAI_API_KEY env).
        batch_size: Batch size for encoding.
        max_length: Maximum token length (huggingface only).
        pooling: Pooling strategy for huggingface ("mean" or "last_token").
        max_workers: Max concurrent requests for API backends (default: 8).
    """

    def __init__(
        self,
        embedding_provider: str = "huggingface",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dim: int = 384,
        api_base: str | None = None,
        api_key: str | None = None,
        batch_size: int = 256,
        max_length: int = 512,
        pooling: str = "mean",
        max_workers: int = 8,
    ):
        self.embedding_provider = embedding_provider
        self.model_name = model_name
        self.output_dim = output_dim
        self.api_base = api_base
        self.api_key = api_key
        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling = pooling
        self.max_workers = max_workers

        # Lazy-loaded HuggingFace model
        self._hf_tokenizer: object | None = None
        self._hf_model: torch.nn.Module | None = None

    def embed(
        self,
        texts: list[str],
        output_path: Path | None = None,
        shard_size: int = 10000,
    ) -> Tensor:
        """Encode texts into embeddings with checkpoint support.

        Large datasets are processed in shards and saved to a checkpoint
        directory next to ``output_path``.  If the process is interrupted,
        re-running will skip already-completed shards and resume from the
        first missing one.

        A persistent log file (``embed_{encoder}.log``) is written next to
        ``output_path`` so that errors, retries, and progress can be reviewed
        after the run completes.

        Args:
            texts: List of text strings to encode.
            output_path: If provided, save/load embeddings at this path.
            shard_size: Number of texts per checkpoint shard (default 10 000).

        Returns:
            Embeddings tensor, shape (len(texts), output_dim).
        """
        # ── Attach a persistent file logger for this run ──────────────────
        file_handler: logging.FileHandler | None = None
        if output_path is not None:
            slug = encoder_slug(self.model_name)
            log_path = output_path.parent / f"embed_{slug}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
            )
            logging.getLogger("riemannfm.data.pipeline.embed").addHandler(file_handler)
            logger.info(f"Embedding log → {log_path}")

        try:
            return self._embed_with_checkpoints(texts, output_path, shard_size)
        finally:
            if file_handler is not None:
                logging.getLogger("riemannfm.data.pipeline.embed").removeHandler(
                    file_handler
                )
                file_handler.close()

    def _embed_with_checkpoints(
        self,
        texts: list[str],
        output_path: Path | None,
        shard_size: int,
    ) -> Tensor:
        """Inner implementation of embed() with checkpoint logic."""
        if output_path is not None and output_path.exists():
            cached: Tensor = torch.load(output_path, map_location="cpu", weights_only=True)
            logger.info(f"Loaded cached embeddings ({cached.shape}) from {output_path}")
            return cached

        # For small datasets or no output_path, run without checkpointing
        if output_path is None or len(texts) <= shard_size:
            result = self._embed_batch(texts)
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(result, output_path)
                logger.info(f"Saved embeddings ({result.shape}) to {output_path}")
            return result

        # ── Checkpointed path for large datasets ───────────────────────────
        ckpt_dir = output_path.parent / f"_checkpoints_{output_path.stem}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        n_shards = (len(texts) + shard_size - 1) // shard_size
        logger.info(
            f"Embedding {len(texts):,} texts in {n_shards} shards "
            f"(shard_size={shard_size:,}), checkpoints → {ckpt_dir}"
        )

        # Count already-completed shards for progress bar initial position
        done_shards = sum(
            1
            for si in range(n_shards)
            if (ckpt_dir / f"shard_{si:05d}.pt").exists()
        )
        if done_shards > 0:
            logger.info(f"Resuming: {done_shards}/{n_shards} shards already done")

        shard_pbar = tqdm(
            range(n_shards),
            desc="Shards",
            initial=done_shards,
            total=n_shards,
            unit="shard",
        )
        for shard_idx in shard_pbar:
            shard_path = ckpt_dir / f"shard_{shard_idx:05d}.pt"
            if shard_path.exists():
                continue

            start = shard_idx * shard_size
            end = min(start + shard_size, len(texts))
            shard_pbar.set_postfix_str(f"texts [{start:,}..{end:,})")
            shard_emb = self._embed_batch(texts[start:end])
            # Atomic write: tmp → rename, prevents corrupt checkpoint on crash
            tmp_path = shard_path.with_suffix(".pt.tmp")
            torch.save(shard_emb, tmp_path)
            os.replace(tmp_path, shard_path)
        shard_pbar.close()

        # ── Concatenate all shards into final output ───────────────────────
        logger.info("Concatenating shards...")
        all_shards = []
        for shard_idx in range(n_shards):
            shard_path = ckpt_dir / f"shard_{shard_idx:05d}.pt"
            all_shards.append(torch.load(shard_path, map_location="cpu", weights_only=True))

        result = torch.cat(all_shards, dim=0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, output_path)
        logger.info(f"Saved embeddings ({result.shape}) to {output_path}")

        # Clean up checkpoint directory
        for f in ckpt_dir.iterdir():
            f.unlink()
        ckpt_dir.rmdir()
        logger.info(f"Cleaned up checkpoint directory: {ckpt_dir}")

        return result

    def _embed_batch(self, texts: list[str]) -> Tensor:
        """Dispatch to the correct backend for a batch of texts."""
        if self.embedding_provider == "huggingface":
            return self._embed_huggingface(texts)
        if self.embedding_provider in ("ollama", "openai"):
            return self._embed_openai_compat(texts)
        if self.embedding_provider == "none":
            return torch.zeros(len(texts), self.output_dim)
        raise ValueError(f"Unknown embedding_provider: {self.embedding_provider}")

    @torch.no_grad()
    def _embed_huggingface(self, texts: list[str]) -> Tensor:
        """Encode texts using a local HuggingFace transformer model."""
        self._load_hf_model()
        assert self._hf_model is not None
        assert self._hf_tokenizer is not None

        all_embeds = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding (HF)"):
            batch_texts = texts[i : i + self.batch_size]
            inputs = self._hf_tokenizer(  # type: ignore[operator]
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            outputs = self._hf_model(**inputs)  # type: ignore[operator]

            if self.pooling == "last_token":
                # Use the last non-padding token
                seq_len = inputs["attention_mask"].sum(dim=1) - 1
                embeds = outputs.last_hidden_state[
                    torch.arange(len(batch_texts)), seq_len
                ]
            else:
                # Mean pooling (default)
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                embeds = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            all_embeds.append(embeds.cpu())

        return torch.cat(all_embeds, dim=0)

    def _embed_openai_compat(self, texts: list[str]) -> Tensor:
        """Encode texts via OpenAI-compatible /v1/embeddings API.

        Works with both OpenAI and Ollama (which exposes /v1/embeddings).
        Sends concurrent requests (max_workers) with per-batch retry
        and exponential backoff for robustness on long-running jobs.

        Logs a summary at the end with success/retry/failure counts and
        text index ranges of any failed batches.
        """
        import concurrent.futures

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package required for ollama/openai embedding_provider. "
                "Install with: pip install openai"
            ) from e

        api_base = self.api_base
        if self.embedding_provider == "ollama" and api_base is None:
            api_base = "http://localhost:11434"

        # Ollama's OpenAI-compat endpoint is at /v1
        if api_base and not api_base.endswith("/v1"):
            api_base = f"{api_base}/v1"

        client = OpenAI(
            base_url=api_base,
            api_key=self.api_key or "ollama",  # Ollama ignores api_key
            max_retries=5,
            timeout=120.0,
        )

        # Build batch index → texts mapping
        batches: list[tuple[int, list[str]]] = []
        for i in range(0, len(texts), self.batch_size):
            batches.append((i, texts[i : i + self.batch_size]))

        max_attempts = 5

        # ── Tracking counters ─────────────────────────────────────────────
        retried_batches: set[int] = set()  # batches that failed >= 1 attempt

        def _encode_batch(batch_idx: int, batch_texts: list[str]) -> Tensor:
            """Encode a single batch with exponential backoff retry."""
            for attempt in range(max_attempts):
                try:
                    response = client.embeddings.create(
                        model=self.model_name,
                        input=batch_texts,
                    )
                    if attempt > 0:
                        retried_batches.add(batch_idx)
                        logger.info(
                            f"Batch {batch_idx} succeeded on attempt {attempt + 1}"
                        )
                    return torch.stack([torch.tensor(item.embedding) for item in response.data])
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Batch {batch_idx} (texts [{batch_idx * self.batch_size}..]) "
                        f"failed (attempt {attempt + 1}/{max_attempts}): "
                        f"{e!r} — retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
            raise RuntimeError("unreachable")  # for type checker

        # Concurrent requests for throughput
        results: list[Tensor | None] = [None] * len(batches)
        failed: list[tuple[int, list[str]]] = []
        pbar = tqdm(total=len(batches), desc="Encoding (API)")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(_encode_batch, idx, batch_texts): idx
                for idx, (_, batch_texts) in enumerate(batches)
            }
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(
                        f"Batch {idx} (texts [{idx * self.batch_size}..]) "
                        f"failed after {max_attempts} attempts: {e!r}"
                    )
                    failed.append((idx, batches[idx][1]))
                pbar.update(1)
        pbar.close()

        # Sequential retry for batches that failed all concurrent attempts
        seq_recovered: list[int] = []
        permanently_failed: list[int] = []
        if failed:
            logger.warning(f"Retrying {len(failed)} failed batch(es) sequentially...")
            for idx, batch_texts in failed:
                delay = 10.0
                for retry in range(max_attempts):
                    try:
                        results[idx] = _encode_batch(idx, batch_texts)
                        logger.info(f"Batch {idx} succeeded on sequential retry {retry + 1}")
                        seq_recovered.append(idx)
                        break
                    except Exception as e:
                        if retry == max_attempts - 1:
                            permanently_failed.append(idx)
                            raise RuntimeError(
                                f"Batch {idx} (texts [{idx * self.batch_size}..]) "
                                f"failed permanently after all retries: {e!r}"
                            ) from e
                        logger.warning(f"Sequential retry {retry + 1} for batch {idx}: {e!r}")
                        time.sleep(delay)
                        delay *= 2

        # ── Summary ───────────────────────────────────────────────────────
        n_total = len(batches)
        n_failed_concurrent = len(failed)
        n_recovered = len(seq_recovered)
        n_permanent = len(permanently_failed)
        n_retried_inline = len(retried_batches - {idx for idx, _ in failed})
        n_first_try = n_total - n_retried_inline - n_failed_concurrent

        summary_lines = [
            f"Embedding summary: {n_total} batches, "
            f"{len(texts):,} texts, batch_size={self.batch_size}",
            f"  first-try success : {n_first_try}",
            f"  retried (in-line) : {n_retried_inline}",
            f"  failed→sequential : {n_failed_concurrent}",
            f"    recovered       : {n_recovered}",
            f"    permanent fail  : {n_permanent}",
        ]
        if permanently_failed:
            ranges = [
                f"[{idx * self.batch_size}..{min((idx + 1) * self.batch_size, len(texts))})"
                for idx in permanently_failed
            ]
            summary_lines.append(f"  failed text ranges: {', '.join(ranges)}")

        summary = "\n".join(summary_lines)
        logger.info(summary)

        return torch.cat([r for r in results if r is not None], dim=0)

    def _load_hf_model(self):
        """Lazy-load HuggingFace model and tokenizer."""
        if self._hf_model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading HuggingFace model: {self.model_name}")
        self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._hf_model = AutoModel.from_pretrained(self.model_name)
        self._hf_model.eval()
        if torch.cuda.is_available():
            self._hf_model = self._hf_model.cuda()
