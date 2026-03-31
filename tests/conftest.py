"""Shared pytest fixtures for RiemannFM tests."""

import pytest
import torch

from riemannfm.manifolds.product import RiemannFMProductManifold

ATOL = 1e-4
RTOL = 1e-4


@pytest.fixture(autouse=True)
def seed():
    """Fix random seed for reproducibility."""
    torch.manual_seed(42)


@pytest.fixture
def device():
    """Return available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def product_manifold():
    """Small product manifold H^5 x S^4 x E^4 for fast testing."""
    return RiemannFMProductManifold(dim_hyperbolic=4, dim_spherical=4, dim_euclidean=4)


@pytest.fixture
def batch_size():
    """Small batch size for fast tests."""
    return 4
