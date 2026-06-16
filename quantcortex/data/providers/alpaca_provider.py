"""Alpaca market-data provider.

Serves OHLCV bars via the maintained ``alpaca-py`` SDK. Alpaca does not offer
company fundamentals or macro series, so those methods raise
:class:`NotImplementedError` pointing at the appropriate providers.

Credentials default to ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY``. An optional
market-data URL override uses ``ALPACA_DATA_URL``; the trading endpoint is not
reused for market data.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Optional, Sequence, Union

import pandas as pd

from quantcortex.data.providers.base import DataProvider, _as_symbol_list

__all__ = ["AlpacaProvider"]


class AlpacaProvider(DataProvider):
    """:class:`DataProvider` backed by Alpaca's market-data API."""

    name = "alpaca"

    #: map canonical timeframes to Alpaca unit/amount specifications
    _TIMEFRAME_MAP: Dict[str, tuple[int, str]] = {
        "1d": (1, "Day"),
        "1h": (1, "Hour"),
        "1wk": (1, "Week"),
        "1w": (1, "Week"),
        "1mo": (1, "Month"),
        "1m": (1, "Minute"),
        "5m": (5, "Minute"),
        "15m": (15, "Minute"),
        "30m": (30, "Minute"),
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        data_url: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
    ) -> None:
        """Store credentials and an optional market-data URL override.

        ``base_url`` remains as a deprecated compatibility alias for
        ``data_url``. Supplying both is rejected.
        """
        if data_url is not None and base_url is not None:
            raise ValueError("supply at most one of data_url and base_url")
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        self.data_url = data_url or base_url or os.environ.get("ALPACA_DATA_URL")
        if self.data_url is not None and (
            not isinstance(self.data_url, str) or not self.data_url.strip()
        ):
            raise ValueError("data_url must be a non-empty string")
        if self.data_url is not None:
            self.data_url = self.data_url.strip()

    @staticmethod
    def _load_sdk() -> dict[str, object]:
        try:
            from alpaca.data.enums import Adjustment
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "alpaca-py is required for AlpacaProvider; "
                "install it with `pip install alpaca-py`."
            ) from exc
        return {
            "Adjustment": Adjustment,
            "StockHistoricalDataClient": StockHistoricalDataClient,
            "StockBarsRequest": StockBarsRequest,
            "TimeFrame": TimeFrame,
            "TimeFrameUnit": TimeFrameUnit,
        }

    @staticmethod
    def _parse_bound(value: Optional[str], name: str) -> Optional[datetime]:
        if value is None:
            return None
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(parsed):
            raise ValueError(f"{name} must be a valid timestamp")
        return parsed.to_pydatetime()

    # ------------------------------------------------------------------ #
    # OHLCV
    # ------------------------------------------------------------------ #
    def fetch_ohlcv(
        self,
        symbols: Union[str, Sequence[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
        timeframe: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch bars per symbol via ``get_bars`` and normalise the schema."""
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "AlpacaProvider requires credentials; pass api_key/secret_key "
                "or set ALPACA_API_KEY and ALPACA_SECRET_KEY."
            )
        timeframe_spec = self._TIMEFRAME_MAP.get(timeframe)
        if timeframe_spec is None:
            raise ValueError(
                f"Unsupported timeframe {timeframe!r} for Alpaca; "
                f"choose one of {sorted(self._TIMEFRAME_MAP)}."
            )
        start_time = self._parse_bound(start, "start")
        end_time = self._parse_bound(end, "end")
        if start_time is not None and end_time is not None and start_time > end_time:
            raise ValueError("start must not be after end")
        sdk = self._load_sdk()
        unit = getattr(sdk["TimeFrameUnit"], timeframe_spec[1])
        alpaca_timeframe = sdk["TimeFrame"](timeframe_spec[0], unit)
        client = sdk["StockHistoricalDataClient"](
            api_key=self.api_key,
            secret_key=self.secret_key,
            url_override=self.data_url,
        )
        requested_symbols = _as_symbol_list(symbols)
        request = sdk["StockBarsRequest"](
            symbol_or_symbols=requested_symbols,
            timeframe=alpaca_timeframe,
            start=start_time,
            end=end_time,
            adjustment=sdk["Adjustment"].ALL,
        )
        bars = client.get_stock_bars(request)
        frame = bars.df if hasattr(bars, "df") else pd.DataFrame(bars)

        out: Dict[str, pd.DataFrame] = {}
        for symbol in requested_symbols:
            if isinstance(frame.index, pd.MultiIndex) and "symbol" in frame.index.names:
                try:
                    df = frame.xs(symbol, level="symbol").copy()
                except KeyError:
                    df = pd.DataFrame()
            elif "symbol" in frame.columns:
                df = frame[frame["symbol"] == symbol].drop(columns=["symbol"])
            elif len(requested_symbols) == 1:
                df = frame.copy()
            else:
                df = pd.DataFrame()
            if df is None or df.empty:
                out[symbol] = self._standardize_ohlcv(pd.DataFrame())
                continue
            out[symbol] = self._standardize_ohlcv(df)
        return out

    # ------------------------------------------------------------------ #
    # Fundamentals / Macro
    # ------------------------------------------------------------------ #
    def fetch_fundamentals(
        self,
        symbols: Union[str, Sequence[str]],
        fields: Optional[Sequence[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Not supported - Alpaca does not serve fundamentals."""
        raise NotImplementedError(
            "Alpaca does not serve fundamentals; use FMP/Polygon"
        )

    def fetch_macro(
        self,
        series: Union[str, Sequence[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Not supported - use FREDProvider for macro series."""
        raise NotImplementedError("Use FREDProvider for macro series")
