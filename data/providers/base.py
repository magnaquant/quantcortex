"""Abstract data provider interface.

Every concrete provider (yfinance, Polygon, Alpaca, CCXT, FRED, FMP) implements
this ABC so the rest of the platform can swap data sources without code changes.
Providers normalise their raw payloads to the platform's canonical shapes:

* **OHLCV** — a ``pandas.DataFrame`` per symbol, ``DatetimeIndex`` (UTC-naive,
  ascending), columns ``[open, high, low, close, adj_close, volume]`` (float64).
* **Fundamentals** — a tidy ``DataFrame`` with at least the columns
  ``[symbol, period_end, announcement_date, field, value]``.  The presence of
  ``announcement_date`` is what lets ``pit_enforcer`` guarantee point-in-time
  correctness.
* **Macro** — a wide ``DataFrame`` indexed by date, one column per series id.

Network calls are made lazily *inside* methods, never at import time, so the
package imports cleanly without provider credentials or connectivity.
"""

from __future__ import annotations

import abc
from typing import Dict, Iterable, List, Optional, Sequence, Union

import pandas as pd

__all__ = ["DataProvider", "OHLCV_COLUMNS", "FUNDAMENTAL_COLUMNS"]

OHLCV_COLUMNS: List[str] = ["open", "high", "low", "close", "adj_close", "volume"]
FUNDAMENTAL_COLUMNS: List[str] = [
    "symbol",
    "period_end",
    "announcement_date",
    "field",
    "value",
]


def _as_symbol_list(symbols: Union[str, Iterable[str]]) -> List[str]:
    if isinstance(symbols, str):
        return [symbols]
    return list(symbols)


class DataProvider(abc.ABC):
    """Abstract base class for market & alternative data providers."""

    #: human-readable provider name
    name: str = "base"

    # ------------------------------------------------------------------ #
    # abstract interface
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def fetch_ohlcv(
        self,
        symbols: Union[str, Sequence[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
        timeframe: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Return ``{symbol: ohlcv_dataframe}`` for the requested window."""
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_fundamentals(
        self,
        symbols: Union[str, Sequence[str]],
        fields: Optional[Sequence[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return tidy fundamentals with PIT ``announcement_date`` column."""
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_macro(
        self,
        series: Union[str, Sequence[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return a wide DataFrame of macro series indexed by date."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # shared helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce an arbitrary OHLCV frame to the canonical schema."""
        out = df.copy()
        out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
        rename = {"adjclose": "adj_close", "adjusted_close": "adj_close"}
        out = out.rename(columns=rename)
        if "adj_close" not in out.columns and "close" in out.columns:
            out["adj_close"] = out["close"]
        for col in OHLCV_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA
        out = out[OHLCV_COLUMNS].astype("float64")
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index)
        out.index.name = "date"
        return out.sort_index()

    def get_prices(
        self,
        symbols: Union[str, Sequence[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
        field: str = "adj_close",
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """Convenience: wide price panel (dates x symbols) for ``field``."""
        data = self.fetch_ohlcv(symbols, start, end, timeframe)
        cols = {sym: df[field] for sym, df in data.items() if field in df}
        if not cols:
            return pd.DataFrame()
        panel = pd.DataFrame(cols).sort_index()
        return panel[_as_symbol_list(symbols)].reindex(
            columns=[s for s in _as_symbol_list(symbols) if s in panel.columns]
        )

    def get_returns(
        self,
        symbols: Union[str, Sequence[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Convenience: simple returns panel derived from prices."""
        return self.get_prices(symbols, start, end, **kwargs).pct_change().dropna(
            how="all"
        )
