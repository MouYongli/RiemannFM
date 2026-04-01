"""Multi-backend text embedding for KG entities and relations.

Supports three embedding providers:
- huggingface: Local transformer models (SBERT, XLM-RoBERTa, etc.)
- ollama: Ollama server via OpenAI-compatible /v1/embeddings endpoint
- openai: OpenAI API via /v1/embeddings endpoint

Results are cached to disk. Embedding files are named
entity_emb_{encoder}_{dim}.pt for ablation coexistence.
"""

import logging
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ─── Encoder slug mapping ────────────────────────────────────────────────────

_SLUG_MAP: dict[str, str] = {
    "sentence-transformers/all-MiniLM-L6-v2": "sbert",
    "xlm-roberta-large": "xlm_roberta",
    "qwen3-embedding": "qwen3",
    "qwen3-embedding:8b": "qwen3",
    "nomic-embed-text": "nomic",
}

_SLUG_TO_MODEL: dict[str, str] = {
    "sbert": "sentence-transformers/all-MiniLM-L6-v2",
    "xlm_roberta": "xlm-roberta-large",
    "qwen3": "qwen3-embedding",
    "nomic": "nomic-embed-text",
}

ENCODER_DIMS: dict[str, int] = {
    "sbert": 384,
    "xlm_roberta": 1024,
    "qwen3": 4096,
    "nomic": 768,
}


def encoder_slug(text_encoder: str) -> str:
    """Derive a filesystem-safe short key from a model name or config key.

    Args:
        text_encoder: Model name, HuggingFace path, or short slug.

    Returns:
        Short key like "sbert", "xlm_roberta", "qwen3_embed".
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
    ):
        self.embedding_provider = embedding_provider
        self.model_name = model_name
        self.output_dim = output_dim
        self.api_base = api_base
        self.api_key = api_key
        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling = pooling

        # Lazy-loaded HuggingFace model
        self._hf_tokenizer: object | None = None
        self._hf_model: torch.nn.Module | None = None

    def embed(
        self,
        texts: list[str],
        output_path: Path | None = None,
    ) -> Tensor:
        """Encode texts into embeddings.

        Args:
            texts: List of text strings to encode.
            output_path: If provided, save/load embeddings at this path.

        Returns:
            Embeddings tensor, shape (len(texts), output_dim).
        """
        if output_path is not None and output_path.exists():
            cached: Tensor = torch.load(output_path, map_location="cpu", weights_only=True)
            logger.info(f"Loaded cached embeddings ({cached.shape}) from {output_path}")
            return cached

        if self.embedding_provider == "huggingface":
            result = self._embed_huggingface(texts)
        elif self.embedding_provider in ("ollama", "openai"):
            result = self._embed_openai_compat(texts)
        elif self.embedding_provider == "none":
            result = torch.zeros(len(texts), self.output_dim)
        else:
            raise ValueError(f"Unknown embedding_provider: {self.embedding_provider}")

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(result, output_path)
            logger.info(f"Saved embeddings ({result.shape}) to {output_path}")

        return result

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
        """
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
        )

        all_embeds = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding (API)"):
            batch_texts = texts[i : i + self.batch_size]
            response = client.embeddings.create(
                model=self.model_name,
                input=batch_texts,
            )
            batch_embeds = [torch.tensor(item.embedding) for item in response.data]
            all_embeds.extend(batch_embeds)

        return torch.stack(all_embeds, dim=0)

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
