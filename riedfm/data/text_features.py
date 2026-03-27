"""Text feature extraction and caching for entities and relations.

Preprocesses WikiData entity labels and descriptions into text embeddings
using a frozen text encoder. Results are cached to disk for efficiency.
"""

import json
from pathlib import Path

import torch
from torch import Tensor


class TextFeatureExtractor:
    """Extracts and caches text embeddings for KG entities and relations.

    Args:
        model_name: HuggingFace model name for the text encoder.
        cache_dir: Directory to cache extracted features.
        max_length: Maximum token length for text inputs.
        batch_size: Batch size for encoding.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-large",
        cache_dir: str = "data/text_cache",
        max_length: int = 128,
        batch_size: int = 256,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.max_length = max_length
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        if torch.cuda.is_available():
            self._model = self._model.cuda()

    @torch.no_grad()
    def extract_embeddings(
        self,
        texts: list[str],
        entity_ids: list[str] | None = None,
    ) -> Tensor:
        """Extract text embeddings for a list of texts.

        Args:
            texts: List of text strings (label + description).
            entity_ids: Optional entity IDs for cache lookup.

        Returns:
            Embeddings tensor, shape (len(texts), hidden_dim).
        """
        # Check cache
        if entity_ids is not None:
            cache_path = self.cache_dir / f"embeddings_{hash(tuple(entity_ids)) % 1000000}.pt"
            if cache_path.exists():
                return torch.load(cache_path, map_location="cpu", weights_only=True)

        self._load_model()

        all_embeds = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            inputs = self._tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            outputs = self._model(**inputs)
            # Mean pooling
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            embeds = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_embeds.append(embeds.cpu())

        result = torch.cat(all_embeds, dim=0)

        # Save to cache
        if entity_ids is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(result, cache_path)

        return result

    def tokenize_batch(
        self,
        texts: list[str],
    ) -> tuple[Tensor, Tensor]:
        """Tokenize texts without encoding (for on-the-fly encoding during training).

        Args:
            texts: List of text strings.

        Returns:
            (input_ids, attention_mask) tensors.
        """
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        inputs = self._tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return inputs["input_ids"], inputs["attention_mask"]
