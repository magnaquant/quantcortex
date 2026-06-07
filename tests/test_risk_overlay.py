"""Tests for the risk overlay layer (circuit breaker + vol targeting)."""

from __future__ import annotations

import numpy as np
import pytest

from risk.circuit_breaker import CircuitBreaker, compute_drawdown
from risk.vol_targeting import VolTargeting, realized_portfolio_vol


# --------------------------------------------------------------------------- #
# circuit breaker
# --------------------------------------------------------------------------- #
def test_circuit_breaker_zeroes_weights_on_drawdown(equity_curve):
    weights = np.array([0.4, 0.6])
    breaker = CircuitBreaker(max_drawdown=0.15)
    # equity_curve fixture draws down ~25% > 15% threshold
    assert compute_drawdown(equity_curve) > 0.15
    out = breaker.apply(weights, equity_curve)
    assert np.allclose(out, 0.0)
    assert breaker.is_tripped


def test_circuit_breaker_passes_weights_when_within_limit():
    weights = np.array([0.5, 0.5])
    breaker = CircuitBreaker(max_drawdown=0.15)
    rising = np.linspace(100.0, 120.0, 50)  # no drawdown
    out = breaker.apply(weights, rising)
    assert np.allclose(out, weights)
    assert not breaker.is_tripped


def test_circuit_breaker_explicit_drawdown():
    breaker = CircuitBreaker(max_drawdown=0.10)
    out = breaker.apply(np.array([1.0]), current_drawdown=0.2)
    assert np.allclose(out, 0.0)


def test_circuit_breaker_hysteresis():
    breaker = CircuitBreaker(max_drawdown=0.20, reset_drawdown=0.05)
    w = np.array([0.5, 0.5])
    breaker.apply(w, current_drawdown=0.25)  # trip
    assert breaker.is_tripped
    # still in moderate drawdown -> stay flat
    out = breaker.apply(w, current_drawdown=0.10)
    assert np.allclose(out, 0.0)
    # recovered below reset -> re-engage
    out = breaker.apply(w, current_drawdown=0.02)
    assert np.allclose(out, w)


# --------------------------------------------------------------------------- #
# vol targeting
# --------------------------------------------------------------------------- #
def test_vol_targeting_scales_down_when_vol_above_target():
    rng = np.random.default_rng(0)
    weights = np.array([0.5, 0.5])
    # high realised vol relative to a 10% target
    high_vol_returns = rng.normal(0.0, 0.03, size=(252, 2))
    realized = realized_portfolio_vol(weights, high_vol_returns)
    assert realized > 0.10

    vt = VolTargeting(target_vol=0.10, max_leverage=1.0)
    out = vt.apply(weights, high_vol_returns)

    assert np.abs(out).sum() < np.abs(weights).sum()  # de-levered
    assert vt.last_scale < 1.0


def test_vol_targeting_uses_explicit_realized_vol():
    weights = np.array([0.5, 0.5])
    vt = VolTargeting(target_vol=0.10, max_leverage=1.0)
    out = vt.apply(weights, realized_vol=0.20)  # 2x target -> 0.5x scale
    assert np.allclose(out, weights * 0.5)
    assert vt.last_scale == pytest.approx(0.5)


def test_vol_targeting_can_lever_up_within_cap():
    weights = np.array([0.5, 0.5])
    vt = VolTargeting(target_vol=0.10, max_leverage=2.0)
    out = vt.apply(weights, realized_vol=0.05)  # half target -> 2x scale
    assert vt.last_scale == pytest.approx(2.0)
    assert np.allclose(out, weights * 2.0)
