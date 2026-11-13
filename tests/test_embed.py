"""Unit tests for embed.py.

These tests target branches of ``_embed_huggingface`` that no shipped
embedding config currently exercises (e.g. ``pooling='last_token'``
against a huggingface-served decoder model). We mock the tokenizer
and model so no weights are downloaded.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from riemannfm.data.pipeline.embed import RiemannFMTextEmbedder


class _FakeTokenizerOut(dict):
    """Dict subclass so ``{k: v.cuda() ...}`` and ``self._hf_model(**inputs)`` both work."""


def _make_fake_tokenizer(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    def tokenizer(*_args: Any, **_kwargs: Any) -> _FakeTokenizerOut:
        return _FakeTokenizerOut(input_ids=input_ids, attention_mask=attention_mask)

    return tokenizer


def _make_fake_model(last_hidden_state: torch.Tensor):
    def model(**_inputs: Any) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=last_hidden_state)

    return model


@pytest.fixture
def force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_embed_huggingface_last_token(force_cpu: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the last_token pooling branch picks hidden[seq_len - 1]."""
    # Two sequences (lengths 3 and 2) padded to length 3. Hidden size 4.
    hidden = torch.arange(24, dtype=torch.float).reshape(2, 3, 4)
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    input_ids = torch.zeros(2, 3, dtype=torch.long)

    def fake_load(self: RiemannFMTextEmbedder) -> None:
        self._hf_tokenizer = _make_fake_tokenizer(input_ids, attention_mask)
        self._hf_model = _make_fake_model(hidden)

    monkeypatch.setattr(RiemannFMTextEmbedder, "_load_hf_model", fake_load)

    embedder = RiemannFMTextEmbedder(
        embedding_provider="huggingface",
        model_name="fake",
        output_dim=4,
        batch_size=2,
        pooling="last_token",
    )
    result = embedder._embed_huggingface(["text_a", "text_b"])

    # text_a: length 3 → seq_len=2 → hidden[0, 2] = [8, 9, 10, 11]
    # text_b: length 2 → seq_len=1 → hidden[1, 1] = [16, 17, 18, 19]
    expected = torch.tensor([[8.0, 9.0, 10.0, 11.0], [16.0, 17.0, 18.0, 19.0]])
    torch.testing.assert_close(result, expected)


def test_embed_huggingface_mean_pooling(force_cpu: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanity check: mean-pooling branch averages over non-padding tokens."""
    hidden = torch.arange(24, dtype=torch.float).reshape(2, 3, 4)
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    input_ids = torch.zeros(2, 3, dtype=torch.long)

    def fake_load(self: RiemannFMTextEmbedder) -> None:
        self._hf_tokenizer = _make_fake_tokenizer(input_ids, attention_mask)
        self._hf_model = _make_fake_model(hidden)

    monkeypatch.setattr(RiemannFMTextEmbedder, "_load_hf_model", fake_load)

    embedder = RiemannFMTextEmbedder(
        embedding_provider="huggingface",
        model_name="fake",
        output_dim=4,
        batch_size=2,
        pooling="mean",
    )
    result = embedder._embed_huggingface(["text_a", "text_b"])

    # text_a: mean of rows 0,1,2 → hidden[0].mean(dim=0) = [4, 5, 6, 7]
    # text_b: mean of rows 0,1 only (mask 1,1,0) → hidden[1, :2].mean(dim=0) = [14, 15, 16, 17]
    expected = torch.tensor([[4.0, 5.0, 6.0, 7.0], [14.0, 15.0, 16.0, 17.0]])
    torch.testing.assert_close(result, expected)
