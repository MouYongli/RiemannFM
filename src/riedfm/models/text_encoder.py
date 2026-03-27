"""Frozen text encoder wrapper for XLM-R or mE5.

Loads a pretrained multilingual text encoder, freezes all parameters,
and provides a projection layer from text hidden dim to model hidden dim.
"""

import torch
import torch.nn as nn
from torch import Tensor


class RieDFMTextEncoder(nn.Module):
    """Frozen multilingual text encoder with a learnable projection.

    Wraps a HuggingFace transformer model (e.g., xlm-roberta-large or
    intfloat/multilingual-e5-large) with all parameters frozen. Only the
    projection layer from text_dim to model_dim is trainable.

    Args:
        model_name: HuggingFace model name/path.
        model_dim: Target dimension for projection.
        pooling: Pooling strategy - "mean" or "cls".
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-large",
        model_dim: int = 768,
        pooling: str = "mean",
    ):
        super().__init__()
        self.model_name = model_name
        self.pooling = pooling

        # Lazy loading to avoid importing transformers at module level
        self._encoder: nn.Module | None = None
        self._text_dim: int | None = None
        self._model_dim = model_dim

        # Projection will be initialized on first forward pass
        self.projection: nn.Linear | None = None

    def _load_encoder(self, device: torch.device):
        """Lazy-load the text encoder on first use."""
        if self._encoder is not None:
            return

        from transformers import AutoModel

        self._encoder = AutoModel.from_pretrained(self.model_name)
        self._encoder = self._encoder.to(device)
        self._encoder.eval()

        # Freeze all parameters
        for param in self._encoder.parameters():
            param.requires_grad = False

        self._text_dim = int(self._encoder.config.hidden_size)  # type: ignore[arg-type, union-attr]

        # Initialize projection layer
        self.projection = nn.Linear(self._text_dim, self._model_dim).to(device)

    @property
    def text_dim(self) -> int:
        """Output dimension of the raw text encoder."""
        if self._text_dim is None:
            # Infer from model config without loading
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.model_name)
            return int(config.hidden_size)
        return self._text_dim

    @torch.no_grad()
    def encode_raw(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Get raw text encoder output (no projection).

        Args:
            input_ids: Token IDs, shape (B, T).
            attention_mask: Attention mask, shape (B, T).

        Returns:
            Token-level embeddings, shape (B, T, text_dim).
        """
        self._load_encoder(input_ids.device)
        assert self._encoder is not None
        outputs = self._encoder(input_ids=input_ids, attention_mask=attention_mask)
        return Tensor(outputs.last_hidden_state)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        return_tokens: bool = False,
    ) -> Tensor:
        """Encode text and project to model dimension.

        Args:
            input_ids: Token IDs, shape (B, T).
            attention_mask: Attention mask, shape (B, T).
            return_tokens: If True, return token-level embeddings (B, T, model_dim).
                          If False, return pooled embeddings (B, model_dim).

        Returns:
            Projected text embeddings.
        """
        raw = self.encode_raw(input_ids, attention_mask)  # (B, T, text_dim)
        assert self.projection is not None
        projected = self.projection(raw)  # (B, T, model_dim)

        if return_tokens:
            return Tensor(projected)

        # Pool
        if self.pooling == "cls":
            return Tensor(projected[:, 0])
        elif self.pooling == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).float()
            return Tensor((projected * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1))
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
