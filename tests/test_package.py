"""Smoke tests for the riemannfm package itself."""

import riemannfm


def test_import() -> None:
    assert riemannfm is not None


def test_version() -> None:
    assert isinstance(riemannfm.__version__, str)
    assert riemannfm.__version__ == "0.1.0"
