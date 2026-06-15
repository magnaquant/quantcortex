"""Shared cross-sectional helpers for the classical factor modules.

Private module: these functions were previously copy-pasted into each of the
classical factor modules (momentum, value, quality, low-vol). They are
consolidated here so the implementations cannot drift apart. Each factor class
still exposes the helpers it always had (e.g.
``MomentumFactor.cross_sectional_zscore``) by delegating to this module.
"""

from __future__ import annotations

import warnings

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
    """Validate and normalize a date x symbol price panel (sorted, float)."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if not prices.index.is_monotonic_increasing:
        prices = prices.sort_index()
    return prices.astype(float)


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
        it is treated as a price level and converted via ``pct_change``. If
        ``None`` (default) a heuristic is used: a strictly positive series
        whose median exceeds 1.0 is treated as a price level. When the
        heuristic is ambiguous (all-positive but median <= 1.0) the series is
        passed through as returns and a warning is emitted; pass
        ``market_is_returns`` explicitly to silence it.
    """
    s = series.astype(float)
    if market_is_returns is True:
        return s
    if market_is_returns is False:
        return s.pct_change()
    # Heuristic: a return series is centered near zero with small magnitude.
    positive = s.dropna()
    if not positive.empty and (positive > 0).all():
        if positive.median() > 1.0:
            return s.pct_change()
        warnings.warn(
            "to_returns: market series is all-positive with median <= 1.0; "
            "ambiguous whether it is a price level or a return series. "
            "Treating it as returns; pass market_is_returns=True/False to be "
            "explicit.",
            UserWarning,
            stacklevel=2,
        )
    return s


def rolling_cov(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Strictly-trailing rolling covariance of two aligned series."""
    xm = x.rolling(window, min_periods=window).mean()
    ym = y.rolling(window, min_periods=window).mean()
    return (x * y).rolling(window, min_periods=window).mean() - xm * ym
