"""Shared cross-sectional helpers for the classical factor modules.

Private module: these functions were previously copy-pasted into each of the
classical factor modules (momentum, value, quality, low-vol). They are
consolidated here so the implementations cannot drift apart. Each factor class
still exposes the helpers it always had (e.g.
``MomentumFactor.cross_sectional_zscore``) by delegating to this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "cross_sectional_zscore",
    "cross_sectional_rank",
    "validate_prices",
    "to_returns",
    "rolling_cov",
]


def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """Row-wise (cross-sectional) z-score, robust to missing values."""
    mean = panel.mean(axis=1, skipna=True)
    std = panel.std(axis=1, skipna=True, ddof=0)
    std = std.replace(0.0, np.nan)
    return panel.sub(mean, axis=0).div(std, axis=0)


def cross_sectional_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """Row-wise cross-sectional rank scaled to ``[0, 1]``."""
    return panel.rank(axis=1, method="average", pct=True, na_option="keep")


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a date x symbol price panel."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if prices.empty or len(prices.columns) == 0:
        raise ValueError("prices must contain at least one row and one symbol")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must use a DatetimeIndex")
    if prices.index.hasnans or prices.index.has_duplicates:
        raise ValueError("prices index must contain unique, valid timestamps")
    if prices.columns.has_duplicates:
        raise ValueError("prices columns must be unique")
    if any(not isinstance(col, str) or not col.strip() for col in prices.columns):
        raise ValueError("prices columns must be non-empty symbol strings")

    normalized = prices.astype(float).copy()
    normalized.columns = pd.Index([column.strip() for column in prices.columns])
    if normalized.columns.has_duplicates:
        raise ValueError("prices columns must remain unique after whitespace trimming")
    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_convert("UTC").tz_localize(None)
    values = normalized.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("prices must not contain infinite values")
    if (normalized.notna() & (normalized <= 0.0)).any(axis=None):
        raise ValueError("observed prices must be strictly positive")
    return normalized.sort_index()


def to_returns(
    series: pd.Series, market_is_returns: bool | None = None
) -> pd.Series:
    """Convert a price level to simple returns; pass through if already returns.

    Parameters
    ----------
    series:
        Market price level or return series.
    market_is_returns:
        If ``True``, ``series`` is used as a return series as-is; if ``False``
        it is treated as a price level and converted via ``pct_change``. The
        value must be explicit because prices below one and positive return
        streaks cannot be distinguished reliably from their values alone.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("market series must use a DatetimeIndex")
    if series.index.hasnans or series.index.has_duplicates:
        raise ValueError("market series index must contain unique, valid timestamps")
    if not series.index.is_monotonic_increasing:
        raise ValueError("market series dates must be sorted in increasing order")
    if market_is_returns is None:
        raise ValueError("market_is_returns must be explicitly True or False")
    s = series.astype(float).copy()
    if np.isinf(s.to_numpy(dtype=float)).any():
        raise ValueError("market series must not contain infinite values")
    if market_is_returns is True:
        return s
    if market_is_returns is False:
        observed = s.dropna()
        if (observed <= 0.0).any():
            raise ValueError("market price levels must be strictly positive")
        return s.pct_change(fill_method=None)
    raise TypeError("market_is_returns must be a boolean")


def rolling_cov(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Trailing population covariance on pairwise-complete observations."""
    if isinstance(window, (bool, np.bool_)) or int(window) != window or window <= 1:
        raise ValueError("window must be an integer greater than 1")
    return x.rolling(window, min_periods=window).cov(y, ddof=0)
