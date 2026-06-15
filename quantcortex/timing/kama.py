"""Kaufman Adaptive Moving Average (KAMA) trend-following timing overlay.

KAMA (Perry Kaufman) is a moving average whose smoothing constant adapts to the
"efficiency" of recent price action.  When price trends cleanly the average
tracks closely (fast); when price chops sideways the average lags heavily
(slow), filtering out noise.

Definitions (all strictly causal - value at ``t`` uses only data up to ``t``):

* **Efficiency Ratio**
  ``ER_t = |P_t - P_{t-er_window}| / sum_{i=t-er_window+1..t} |P_i - P_{i-1}|``
  (net directional move over total path length).
* **Smoothing Constant**
  ``SC_t = ( ER_t * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1) )^2``
* **KAMA recursion**
  ``KAMA_t = KAMA_{t-1} + SC_t * (P_t - KAMA_{t-1})``

The overlay gates exposure on price-vs-KAMA: below the line is treated as a
down-trend and the book is scaled flat; at/above the line exposure passes
through.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import pandas as pd

from quantcortex.portfolio.base import enforce_exposure_contract

__all__ = ["KAMA"]


class KAMA:
    """Kaufman Adaptive Moving Average and its trend-gate overlay.

    Parameters
    ----------
    er_window:
        Lookback for the efficiency ratio (the "change" and "volatility" window).
    fast:
        Fast EMA period bounding the smoothing constant from above.
    slow:
        Slow EMA period bounding the smoothing constant from below.
    """

    def __init__(self, er_window: int = 10, fast: int = 2, slow: int = 30) -> None:
        if er_window < 1:
            raise ValueError("er_window must be >= 1")
        if fast < 1 or slow < 1:
            raise ValueError("fast and slow must be >= 1")
        if fast >= slow:
            raise ValueError("fast period must be smaller than slow period")
        self.er_window = int(er_window)
        self.fast = int(fast)
        self.slow = int(slow)
        self._fast_sc = 2.0 / (self.fast + 1.0)
        self._slow_sc = 2.0 / (self.slow + 1.0)

    # ------------------------------------------------------------------ #
    def compute(self, prices: pd.Series) -> pd.Series:
        """Compute the KAMA series for ``prices`` (strictly causal).

        Returns a Series aligned to ``prices``; entries before enough history
        exists for the efficiency ratio are ``NaN``.
        """
        series = self._as_series(prices)
        p = series.to_numpy(dtype=np.float64)
        n = p.size

        kama = np.full(n, np.nan, dtype=np.float64)
        if n == 0:
            return pd.Series(kama, index=series.index, name="kama")

        # Efficiency ratio components.
        change = np.abs(p - self._shift(p, self.er_window))
        abs_diff = np.abs(np.diff(p, prepend=p[0]))
        # Rolling sum of |diff| over er_window (volatility / path length).
        volatility = (
            pd.Series(abs_diff).rolling(self.er_window, min_periods=self.er_window).sum().to_numpy()
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            er = np.where(volatility > 0, change / volatility, 0.0)
        sc = (er * (self._fast_sc - self._slow_sc) + self._slow_sc) ** 2

        # Seed KAMA at the first index where ER is defined (>= er_window).
        seed_idx = self.er_window
        if seed_idx >= n:
            # Not enough data for a single ER point: return all-NaN.
            return pd.Series(kama, index=series.index, name="kama")

        kama[seed_idx] = p[seed_idx]
        for t in range(seed_idx + 1, n):
            sc_t = sc[t]
            if not np.isfinite(sc_t):
                sc_t = self._slow_sc ** 2
            kama[t] = kama[t - 1] + sc_t * (p[t] - kama[t - 1])

        return pd.Series(kama, index=series.index, name="kama")

    def trend_signal(self, prices: pd.Series) -> pd.Series:
        """Binary trend signal in {0, 1}: 1 when price > KAMA, else 0.

        ``NaN`` KAMA values (insufficient history) map to 0 (no trend / flat).
        """
        series = self._as_series(prices)
        kama = self.compute(series)
        signal = (series.to_numpy(dtype=np.float64) > kama.to_numpy(dtype=np.float64)).astype(int)
        # Where KAMA is undefined, force flat (0).
        signal = np.where(np.isnan(kama.to_numpy()), 0, signal)
        return pd.Series(signal, index=series.index, name="kama_trend")

    # ------------------------------------------------------------------ #
    def apply(
        self,
        weights: np.ndarray,
        prices: Union[pd.Series, pd.DataFrame, np.ndarray, Any] = None,
    ) -> np.ndarray:
        """Gate ``weights`` on the latest price-vs-KAMA trend.

        * ``prices`` a single Series / 1-D array: whole-vector gating - if the
          latest price is at/above KAMA the weights pass through, otherwise they
          are scaled flat (to zero).
        * ``prices`` a DataFrame / 2-D array (one column per asset): per-asset
          gating - each asset's weight is kept only if that asset's latest price
          is at/above its KAMA, else zeroed.
        * a ``StrategyContext``-like object exposing ``.prices``.

        Result is validated via :func:`enforce_exposure_contract`.
        """
        w = np.asarray(weights, dtype=np.float64).ravel()
        gates = self._latest_gates(prices, n_assets=w.size)
        gated = w * gates

        input_gross = float(np.abs(w).sum())
        max_gross = max(1.0, input_gross) + 1e-9
        return enforce_exposure_contract(
            gated, max_gross=max_gross, name=type(self).__name__
        )

    # ------------------------------------------------------------------ #
    def _latest_gates(
        self,
        prices: Union[pd.Series, pd.DataFrame, np.ndarray, Any],
        n_assets: int,
    ) -> np.ndarray:
        """Compute the latest {0,1} gate per asset (broadcast if single series)."""
        prices = self._coerce_prices(prices)

        if isinstance(prices, pd.DataFrame):
            gates = np.array(
                [int(self.trend_signal(prices[col]).iloc[-1]) for col in prices.columns],
                dtype=np.float64,
            )
            if gates.size != n_assets:
                raise ValueError(
                    f"prices has {gates.size} columns but weights has {n_assets}"
                )
            return gates

        # Single price series -> one gate, broadcast to all assets.
        latest = int(self.trend_signal(prices).iloc[-1])
        return np.full(n_assets, float(latest), dtype=np.float64)

    @staticmethod
    def _coerce_prices(
        prices: Union[pd.Series, pd.DataFrame, np.ndarray, Any],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Normalise ``prices`` (incl. context objects) to a Series/DataFrame."""
        if isinstance(prices, (pd.Series, pd.DataFrame)):
            return prices
        if isinstance(prices, np.ndarray):
            return pd.DataFrame(prices) if prices.ndim == 2 else pd.Series(prices.ravel())
        if isinstance(prices, (list, tuple)):
            arr = np.asarray(prices, dtype=np.float64)
            return pd.DataFrame(arr) if arr.ndim == 2 else pd.Series(arr.ravel())

        ctx_prices = getattr(prices, "prices", None)
        if ctx_prices is not None:
            return KAMA._coerce_prices(ctx_prices)
        raise TypeError(
            "KAMA.apply requires prices (Series/DataFrame/array) or a context "
            "object exposing .prices"
        )

    @staticmethod
    def _as_series(prices: Union[pd.Series, np.ndarray]) -> pd.Series:
        """Coerce a 1-D price input to a float64 Series."""
        if isinstance(prices, pd.Series):
            return prices.astype(np.float64)
        arr = np.asarray(prices, dtype=np.float64).ravel()
        return pd.Series(arr)

    @staticmethod
    def _shift(arr: np.ndarray, k: int) -> np.ndarray:
        """Shift ``arr`` forward by ``k`` (NaN-filled head); causal helper."""
        out = np.full_like(arr, np.nan, dtype=np.float64)
        if k < arr.size:
            out[k:] = arr[:-k] if k > 0 else arr
        return out
