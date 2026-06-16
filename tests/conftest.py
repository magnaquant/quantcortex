"""Shared pytest fixtures for the quantcortex test-suite.

These fixtures synthesise small, deterministic market datasets so the tests run
fast and offline (no network, no API keys).  All randomness is seeded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def tickers() -> list[str]:
    return ["AAA", "BBB", "CCC", "DDD"]


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=300)


@pytest.fixture
def prices(dates, tickers, rng) -> pd.DataFrame:
    """Geometric-Brownian-motion price panel (dates x tickers)."""
    n = len(dates)
    drift, vol = 0.0003, 0.01
    rets = rng.normal(drift, vol, size=(n, len(tickers)))
    px = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(px, index=dates, columns=tickers)


@pytest.fixture
def returns(prices) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).dropna()


@pytest.fixture
def equity_curve() -> pd.Series:
    """An equity curve that rises then suffers a deep (~25%) drawdown."""
    up = np.linspace(100.0, 130.0, 120)
    down = np.linspace(130.0, 97.5, 60)  # ~25% peak-to-trough
    return pd.Series(np.concatenate([up, down]))
