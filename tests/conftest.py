"""Shared pytest fixtures for RiemannFM tests."""

import pytest
import torch


@pytest.fixture(autouse=True)
def seed():
    """Fix random seed for reproducibility."""
    torch.manual_seed(42)


@pytest.fixture
def device():
    """Return available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Small batch size for fast tests."""
    return 4
