"""Tests for the canonical weight contract (quantcortex/portfolio/base.py)."""

from __future__ import annotations

import numpy as np
import pytest

from quantcortex.portfolio.base import (
    PortfolioMode,
    WeightContractViolationError,
    enforce_weight_contract,
)
from quantcortex.portfolio.equal_weight import EqualWeight


def test_equal_weight_sums_to_one(returns):
    opt = EqualWeight()
    w = opt.optimize(returns)
    assert isinstance(w, np.ndarray)
    assert w.dtype == np.float64
    assert w.shape == (returns.shape[1],)
    assert w.sum() == pytest.approx(1.0)
    # every asset gets an identical share
    assert np.allclose(w, 1.0 / returns.shape[1])


def test_equal_weight_from_n_assets():
    w = EqualWeight().optimize(None, n_assets=5)
    assert w.shape == (5,)
    assert w.sum() == pytest.approx(1.0)


def test_violation_when_sum_is_1_1():
    bad = np.array([0.6, 0.5], dtype=np.float64)  # sums to 1.1
    with pytest.raises(WeightContractViolationError):
        enforce_weight_contract(bad, mode=PortfolioMode.LONG_ONLY)


def test_violation_when_weight_below_minus_one():
    # sums to 1.0 but one weight is below -1.0
    bad = np.array([-1.5, 2.5, 0.0], dtype=np.float64)
    with pytest.raises(WeightContractViolationError):
        enforce_weight_contract(bad, mode=PortfolioMode.LONG_ONLY)


def test_violation_when_weight_above_one():
    bad = np.array([1.5, -0.5], dtype=np.float64)  # sums to 1.0 but 1.5 > 1
    with pytest.raises(WeightContractViolationError):
        enforce_weight_contract(bad, mode=PortfolioMode.LONG_ONLY)


def test_market_neutral_requires_zero_sum():
    good = np.array([0.5, -0.5], dtype=np.float64)
    out = enforce_weight_contract(good, mode=PortfolioMode.MARKET_NEUTRAL)
    assert out.sum() == pytest.approx(0.0)

    bad = np.array([0.5, 0.5], dtype=np.float64)  # sums to 1.0, not 0.0
    with pytest.raises(WeightContractViolationError):
        enforce_weight_contract(bad, mode=PortfolioMode.MARKET_NEUTRAL)


def test_non_finite_rejected():
    with pytest.raises(WeightContractViolationError):
        enforce_weight_contract(np.array([np.nan, 1.0]))


def test_empty_rejected():
    with pytest.raises(WeightContractViolationError):
        enforce_weight_contract(np.array([], dtype=np.float64))
