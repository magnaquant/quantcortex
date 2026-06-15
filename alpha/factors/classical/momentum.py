"""Cross-sectional price momentum factor.

Implements the classic 12-1 month momentum factor (Jegadeesh & Titman, 1993;
Asness et al., 2013): the cumulative return measured over a long lookback
window while skipping the most recent ``gap`` trading days to avoid the
well-documented short-term reversal effect.

All computations are strictly causal. A factor value reported on date ``t`` is
constructed only from prices observed on or before ``t`` (in fact, on or before
``t - gap``), so it may be acted upon at the close of ``t`` without look-ahead.

Cross-sectional normalization and price/return helpers are shared with the
other classical factor modules via the private
:mod:`alpha.factors.classical._cross_section` module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.factors.classical._cross_section import (
    cross_sectional_rank,
    cross_sectional_zscore,
    rolling_cov,
    to_returns,
    validate_prices,
)


class MomentumFactor:
    """Cross-sectional 12-1 month price momentum.

    The factor for symbol ``i`` on date ``t`` is the total return realized
    between ``t - lookback`` and ``t - gap``::

        momentum_i(t) = price_i(t - gap) / price_i(t - lookback) - 1

    Excluding the most recent ``gap`` days (typically one month) skips the
    short-term reversal window. A higher score means stronger past performance,
    which is the more attractive end of the momentum factor.

    Parameters
    ----------
    lookback:
        Total formation window in trading days (default 252, i.e. ~12 months).
    gap:
        Number of most-recent trading days to skip (default 21, i.e. ~1 month).
    """

    def __init__(self, lookback: int = 252, gap: int = 21) -> None:
        if lookback <= 0:
            raise ValueError("lookback must be a positive integer")
        if gap < 0:
            raise ValueError("gap must be non-negative")
        if gap >= lookback:
            raise ValueError("gap must be strictly smaller than lookback")
        self.lookback = int(lookback)
        self.gap = int(gap)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------
    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute the momentum factor panel.

        Parameters
        ----------
        prices:
            Adjusted close prices indexed by date (ascending), with one column
            per symbol.

        Returns
        -------
        pandas.DataFrame
            Factor panel indexed by the same dates as ``prices`` with the same
            columns. Higher values are more attractive. The first ``lookback``
            rows are ``NaN`` because insufficient history exists.
        """
        prices = self._validate_prices(prices)

        # price(t - gap): the numerator, lagged so the most recent `gap` days
        # are excluded from the formation window. shift is strictly backward in
        # time, so no future information leaks in.
        recent = prices.shift(self.gap)
        # price(t - lookback): the denominator / start of the formation window.
        base = prices.shift(self.lookback)

        with np.errstate(divide="ignore", invalid="ignore"):
            momentum = recent.divide(base) - 1.0

        # Guard against non-positive base prices producing spurious infinities.
        momentum = momentum.replace([np.inf, -np.inf], np.nan)
        return momentum

    def residual_momentum(
        self,
        prices: pd.DataFrame,
        market: pd.Series,
        window: int | None = None,
        market_is_returns: bool | None = None,
    ) -> pd.DataFrame:
        """Momentum of CAPM residual returns.

        Residual (idiosyncratic) momentum (Blitz, Huij & Martens, 2011) removes
        the market component from each stock's returns before measuring
        momentum, producing a cleaner, lower-turnover signal.

        For each symbol a rolling-window CAPM regression of stock excess return
        on market return is estimated using a strictly trailing window; the
        residual return series is then accumulated over the same 12-1 formation
        window as :meth:`compute`.

        Parameters
        ----------
        prices:
            Adjusted close prices indexed by date, columns are symbols.
        market:
            Market (benchmark) price level or return-proxy series indexed by the
            same dates. If it looks like a price level (all positive, low
            relative variation) it is converted to simple returns; if it already
            looks like a return series it is used as-is.
        window:
            Rolling window (in trading days) for the beta estimation. Defaults to
            ``lookback``.
        market_is_returns:
            Explicitly declare whether ``market`` is a return series (``True``)
            or a price level (``False``). ``None`` (default) keeps the
            heuristic detection described above.

        Returns
        -------
        pandas.DataFrame
            Residual-momentum factor panel, strictly causal, higher = better.
        """
        prices = self._validate_prices(prices)
        if not isinstance(market, pd.Series):
            raise TypeError("market must be a pandas Series")
        window = int(window) if window is not None else self.lookback
        if window <= 1:
            raise ValueError("window must be greater than 1")

        stock_ret = prices.pct_change()
        market = market.reindex(prices.index)
        market_ret = self._to_returns(market, market_is_returns)

        # Rolling-window CAPM residuals computed in a strictly causal manner:
        # beta_t is estimated from returns up to and including t, and applied to
        # the contemporaneous return to form the residual at t. The residual at
        # t therefore uses no information after t.
        x = market_ret
        x_mean = x.rolling(window, min_periods=window).mean()
        # ddof=0 to match the population covariance from _rolling_cov; mixing
        # ddof=1 variance with ddof=0 covariance biases beta by (w-1)/w.
        x_var = x.rolling(window, min_periods=window).var(ddof=0)

        residuals = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        for col in prices.columns:
            y = stock_ret[col]
            cov = self._rolling_cov(x, y, window)
            beta = cov.divide(x_var)
            alpha = y.rolling(window, min_periods=window).mean() - beta * x_mean
            residuals[col] = y - (alpha + beta * x)

        # Accumulate residual log-like contributions over the 12-1 window.
        # Use cumulative sum of residual returns (approx. cumulative residual
        # log-return) over [t-lookback, t-gap].
        cum = residuals.cumsum()
        res_mom = cum.shift(self.gap) - cum.shift(self.lookback)
        res_mom = res_mom.replace([np.inf, -np.inf], np.nan)
        return res_mom

    # ------------------------------------------------------------------
    # Cross-sectional normalization (shared via _cross_section)
    # ------------------------------------------------------------------
    cross_sectional_zscore = staticmethod(cross_sectional_zscore)
    rank = staticmethod(cross_sectional_rank)

    # ------------------------------------------------------------------
    # Internal helpers (shared via _cross_section)
    # ------------------------------------------------------------------
    _validate_prices = staticmethod(validate_prices)
    _to_returns = staticmethod(to_returns)
    _rolling_cov = staticmethod(rolling_cov)
