"""Cross-sectional low-volatility factor.

Captures the low-volatility / low-beta anomaly (Frazzini & Pedersen, 2014):
low-risk stocks have historically earned higher risk-adjusted returns than
high-risk stocks. Because a higher factor score must mean *more attractive*,
this factor returns the **negative** of trailing realized volatility (and,
optionally, the negative of CAPM beta), so low-risk names score high.

All statistics are strictly causal: the value reported on date ``t`` uses only
returns observed strictly before ``t`` (the most recent return entering the
window is the one from ``t-1`` to ``t``... shifted out), so it can be acted
upon at ``t`` without look-ahead.

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

# Trading days per year, used to annualize realized volatility.
_TRADING_DAYS = 252.0


class LowVolFactor:
    """Cross-sectional low-volatility (and low-beta) factor.

    Parameters
    ----------
    window:
        Lookback window in trading days for realized volatility (default 63,
        i.e. ~3 months).
    beta_window:
        Lookback window in trading days for rolling CAPM beta (default 252,
        i.e. ~12 months).
    """

    def __init__(self, window: int = 63, beta_window: int = 252) -> None:
        if window <= 1:
            raise ValueError("window must be greater than 1")
        if beta_window <= 1:
            raise ValueError("beta_window must be greater than 1")
        self.window = int(window)
        self.beta_window = int(beta_window)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------
    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Negative annualized trailing realized volatility (higher = lower risk).

        Parameters
        ----------
        prices:
            Adjusted close prices indexed by date, columns are symbols.

        Returns
        -------
        pandas.DataFrame
            Factor panel indexed like ``prices``. Values equal
            ``-annualized_volatility`` so that low-volatility names score high.
            Strictly causal.
        """
        vol = self.realized_volatility(prices)
        return -vol

    def realized_volatility(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Annualized trailing realized volatility of daily returns.

        The rolling standard deviation is computed over the window ending at the
        prior bar and then shifted by one so that the value reported at ``t``
        excludes the return realized into ``t`` itself, keeping it strictly
        causal for a signal acted on at ``t``.
        """
        prices = self._validate_prices(prices)
        returns = prices.pct_change()
        daily_vol = returns.rolling(self.window, min_periods=self.window).std(ddof=0)
        # Shift by one bar: the volatility usable at t must not include the
        # return measured from t-1 to t (which is only known at t's close).
        daily_vol = daily_vol.shift(1)
        annualized = daily_vol * np.sqrt(_TRADING_DAYS)
        return annualized.replace([np.inf, -np.inf], np.nan)

    def beta(
        self,
        prices: pd.DataFrame,
        market: pd.Series,
        window: int | None = None,
        market_is_returns: bool | None = None,
    ) -> pd.DataFrame:
        """Rolling CAPM beta of each symbol versus the market.

        Beta is estimated as ``cov(stock, market) / var(market)`` over a
        trailing window, then shifted by one bar so the value at ``t`` excludes
        the return into ``t`` and is therefore strictly causal.

        Parameters
        ----------
        prices:
            Adjusted close prices indexed by date, columns are symbols.
        market:
            Market price level (or return series) indexed by the same dates.
        window:
            Rolling window in trading days. Defaults to ``beta_window``.
        market_is_returns:
            Explicitly declare whether ``market`` is a return series (``True``)
            or a price level (``False``). ``None`` (default) uses a heuristic.

        Returns
        -------
        pandas.DataFrame
            Rolling beta panel indexed like ``prices``.
        """
        prices = self._validate_prices(prices)
        if not isinstance(market, pd.Series):
            raise TypeError("market must be a pandas Series")
        window = int(window) if window is not None else self.beta_window
        if window <= 1:
            raise ValueError("window must be greater than 1")

        stock_ret = prices.pct_change()
        market_ret = self._to_returns(market.reindex(prices.index), market_is_returns)
        market_var = market_ret.rolling(window, min_periods=window).var(ddof=0)

        betas = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        for col in prices.columns:
            cov = self._rolling_cov(stock_ret[col], market_ret, window)
            betas[col] = cov.divide(market_var)

        betas = betas.replace([np.inf, -np.inf], np.nan)
        # Strict causality: exclude the contemporaneous bar.
        return betas.shift(1)

    def composite(
        self,
        prices: pd.DataFrame,
        market: pd.Series,
        market_is_returns: bool | None = None,
    ) -> pd.DataFrame:
        """Composite low-risk score combining negative vol and negative beta.

        Both ``-realized_volatility`` and ``-beta`` are converted to
        cross-sectional z-scores and equal-weighted, so low-volatility,
        low-beta names receive the highest composite score. Strictly causal.

        Parameters
        ----------
        prices:
            Adjusted close prices indexed by date, columns are symbols.
        market:
            Market price level (or return series) indexed by the same dates.
        market_is_returns:
            Explicitly declare whether ``market`` is a return series (``True``)
            or a price level (``False``). ``None`` (default) uses a heuristic.

        Returns
        -------
        pandas.DataFrame
            Composite low-risk factor panel, higher = more attractive.
        """
        prices = self._validate_prices(prices)
        neg_vol = -self.realized_volatility(prices)
        neg_beta = -self.beta(prices, market, market_is_returns=market_is_returns)

        z_vol = self.cross_sectional_zscore(neg_vol)
        z_beta = self.cross_sectional_zscore(neg_beta)

        composite = pd.concat([z_vol, z_beta]).groupby(level=0).mean()
        composite = composite.reindex(index=prices.index, columns=prices.columns)
        return composite

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
