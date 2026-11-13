"""Tests for the masked discrete edge flow (spec §8)."""

from __future__ import annotations

import math

import pytest
import torch

from riemannfm.flow.discrete_flow import discrete_interpolation, schedule_alpha

B = 8
N = 6
K = 4


class TestScheduleAlpha:
    def test_linear_endpoints(self) -> None:
        assert schedule_alpha(torch.tensor(0.0), kind="linear").item() == 0.0
        assert schedule_alpha(torch.tensor(1.0), kind="linear").item() == 1.0

    def test_cosine_endpoints(self) -> None:
        assert schedule_alpha(torch.tensor(0.0), kind="cosine").item() == pytest.approx(0.0, abs=1e-6)
        assert schedule_alpha(torch.tensor(1.0), kind="cosine").item() == pytest.approx(1.0, abs=1e-6)

    def test_concave_endpoints(self) -> None:
        assert schedule_alpha(torch.tensor(0.0), kind="concave").item() == 0.0
        assert schedule_alpha(torch.tensor(1.0), kind="concave").item() == 1.0

    def test_cosine_matches_formula(self) -> None:
        t = torch.tensor(0.3)
        expected = 1.0 - math.cos(math.pi * 0.3 / 2.0)
        assert schedule_alpha(t, kind="cosine").item() == pytest.approx(expected, abs=1e-6)

    def test_monotonic(self) -> None:
        t = torch.linspace(0.0, 1.0, 100)
        for kind in ("linear", "cosine", "concave"):
            a = schedule_alpha(t, kind=kind)
            assert (a[1:] >= a[:-1] - 1e-6).all()

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown schedule"):
            schedule_alpha(torch.tensor(0.5), kind="wat")


class TestDiscreteInterpolation:
    def test_t0_fully_masked(self) -> None:
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        t = torch.zeros(B)
        E_t, mu_t = discrete_interpolation(E_1, t, schedule="cosine")
        assert (mu_t == 1.0).all()
        assert (E_t == 0.0).all()

    def test_t1_fully_decoded(self) -> None:
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        t = torch.ones(B)
        E_t, mu_t = discrete_interpolation(E_1, t, schedule="cosine")
        assert (mu_t == 0.0).all()
        assert torch.allclose(E_t, E_1)

    def test_shapes(self) -> None:
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        t = torch.rand(B)
        E_t, mu_t = discrete_interpolation(E_1, t, schedule="cosine")
        assert E_t.shape == (B, N, N, K)
        assert mu_t.shape == (B, N, N)

    def test_mu_is_binary(self) -> None:
        E_1 = torch.randint(0, 2, (B, N, N, K)).float()
        t = torch.full((B,), 0.5)
        _E_t, mu_t = discrete_interpolation(E_1, t, schedule="cosine")
        assert ((mu_t == 0) | (mu_t == 1)).all()

    def test_zeros_at_masked_positions(self) -> None:
        E_1 = torch.ones(B, N, N, K)
        t = torch.full((B,), 0.5)
        E_t, mu_t = discrete_interpolation(E_1, t, schedule="linear")
        # Wherever μ = 1, E_t should be 0 across all K relations.
        masked = mu_t.unsqueeze(-1).expand_as(E_t) == 1.0
        assert (E_t[masked] == 0.0).all()
        # Wherever μ = 0, E_t should equal E_1.
        kept = mu_t.unsqueeze(-1).expand_as(E_t) == 0.0
        assert (E_t[kept] == 1.0).all()

    def test_mask_prob_matches_schedule(self) -> None:
        """μ ~ Bernoulli(1 - α(t)): empirical mean should match 1 - α(t)."""
        torch.manual_seed(0)
        E_1 = torch.zeros(256, 32, 32, K)
        t_val = 0.3
        t = torch.full((256,), t_val)
        _E_t, mu_t = discrete_interpolation(E_1, t, schedule="cosine")
        empirical = mu_t.mean().item()
        expected = 1.0 - (1.0 - math.cos(math.pi * t_val / 2.0))
        assert abs(empirical - expected) < 0.01, f"empirical {empirical}, expected {expected}"
