"""FRED (Federal Reserve Economic Data) macro provider.

Fetches one or more economic time series from the St. Louis Fed via the
``fredapi`` library and assembles them into a wide, date-indexed DataFrame  -
the platform's canonical macro shape.  OHLCV and fundamentals are not served
here.

A small :attr:`COMMON_SERIES` catalogue maps friendly aliases to frequently
used FRED series ids for convenience.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from data.providers.base import DataProvider, _as_symbol_list

__all__ = ["FREDProvider"]


class FREDProvider(DataProvider):
    """:class:`DataProvider` backed by FRED via ``fredapi``."""

    name = "fred"

    #: curated friendly-name -> FRED series id catalogue
    COMMON_SERIES: Dict[str, str] = {
        "DGS10": "DGS10",  # 10-Year Treasury constant maturity yield
        "DGS2": "DGS2",  # 2-Year Treasury constant maturity yield
        "VIXCLS": "VIXCLS",  # CBOE Volatility Index
        "T10Y2Y": "T10Y2Y",  # 10Y minus 2Y Treasury spread
        "BAMLH0A0HYM2": "BAMLH0A0HYM2",  # ICE BofA US high-yield OAS
        "UNRATE": "UNRATE",  # Civilian unemployment rate
        "CPIAUCSL": "CPIAUCSL",  # CPI for all urban consumers
        "FEDFUNDS": "FEDFUNDS",  # Effective federal funds rate
        "GDP": "GDP",  # Gross domestic product
        "DTWEXBGS": "DTWEXBGS",  # Trade-weighted US dollar index (broad)
    }

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Store the FRED API key, falling back to ``FRED_API_KEY``."""
        self.api_key = api_key or os.environ.get("FRED_API_KEY")

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _client(self):
        """Lazily import ``fredapi`` and build an authenticated client."""
        if not self.api_key:
            raise ValueError(
                "FREDProvider requires an API key; pass api_key=... or set "
                "the FRED_API_KEY environment variable."
            )
        try:
            from fredapi import Fred
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "fredapi is required for FREDProvider; "
                "install it with `pip install fredapi`."
            ) from exc
        return Fred(api_key=self.api_key)

    # ------------------------------------------------------------------ #
    # Macro
    # ------------------------------------------------------------------ #
    def fetch_macro(
        self,
        series: Union[str, Sequence[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
        ffill: bool = False,
    ) -> pd.DataFrame:
        """Return a wide DataFrame of FRED series indexed by date.

        Parameters
        ----------
        series:
            One or more FRED series ids (friendly aliases in
            :attr:`COMMON_SERIES` are resolved automatically).
        start, end:
            Optional inclusive date bounds.
        ffill:
            When ``True``, reindex onto a common business-day calendar and
            forward-fill so lower-frequency series align with daily ones.
        """
        fred = self._client()

        columns: Dict[str, pd.Series] = {}
        for sid in _as_symbol_list(series):
            resolved = self.COMMON_SERIES.get(sid, sid)
            data = fred.get_series(
                resolved, observation_start=start, observation_end=end
            )
            s = pd.Series(data)
            s.index = pd.to_datetime(s.index)
            s = pd.to_numeric(s, errors="coerce")
            columns[sid] = s

        if not columns:
            empty = pd.DataFrame()
            empty.index.name = "date"
            return empty

        wide = pd.DataFrame(columns).sort_index()
        wide.index.name = "date"

        if ffill and not wide.empty:
            cal = pd.bdate_range(wide.index.min(), wide.index.max())
            wide = wide.reindex(cal).ffill()
            wide.index.name = "date"
        return wide

    # ------------------------------------------------------------------ #
    # OHLCV / Fundamentals
    # ------------------------------------------------------------------ #
    def fetch_ohlcv(
        self,
        symbols: Union[str, Sequence[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
        timeframe: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Not supported - FRED serves macro series only."""
        raise NotImplementedError(
            "FRED does not serve OHLCV; use a market-data provider "
            "(yfinance/Polygon/Alpaca/ccxt/FMP)."
        )

    def fetch_fundamentals(
        self,
        symbols: Union[str, Sequence[str]],
        fields: Optional[Sequence[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Not supported - FRED serves macro series only."""
        raise NotImplementedError(
            "FRED does not serve fundamentals; use FMP/Polygon."
        )

    def available_series(self) -> List[str]:
        """Return the friendly aliases available in :attr:`COMMON_SERIES`."""
        return sorted(self.COMMON_SERIES)
