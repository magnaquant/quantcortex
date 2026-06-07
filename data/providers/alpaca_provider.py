"""Alpaca market-data provider.

Serves OHLCV bars via the ``alpaca-trade-api`` SDK.  Alpaca does not offer
company fundamentals or macro series, so those methods raise
:class:`NotImplementedError` pointing at the appropriate providers.

Credentials default to the ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY`` /
``ALPACA_BASE_URL`` environment variables and are only required when the
network method is actually called.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Union

import pandas as pd

from data.providers.base import DataProvider, _as_symbol_list

__all__ = ["AlpacaProvider"]


class AlpacaProvider(DataProvider):
    """:class:`DataProvider` backed by Alpaca's market-data API."""

    name = "alpaca"

    #: map canonical timeframes -> Alpaca ``TimeFrame`` string codes
    _TIMEFRAME_MAP: Dict[str, str] = {
        "1d": "1Day",
        "1h": "1Hour",
        "1wk": "1Week",
        "1w": "1Week",
        "1mo": "1Month",
        "1m": "1Min",
        "5m": "5Min",
        "15m": "15Min",
        "30m": "30Min",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """Store credentials, falling back to ``ALPACA_*`` env vars."""
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        self.base_url = (
            base_url
            or os.environ.get("ALPACA_BASE_URL")
            or "https://paper-api.alpaca.markets"
        )

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
        tf = self._TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            raise ValueError(
                f"Unsupported timeframe {timeframe!r} for Alpaca; "
                f"choose one of {sorted(self._TIMEFRAME_MAP)}."
            )
        try:
            import alpaca_trade_api as tradeapi
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "alpaca-trade-api is required for AlpacaProvider; "
                "install it with `pip install alpaca-trade-api`."
            ) from exc

        api = tradeapi.REST(
            key_id=self.api_key,
            secret_key=self.secret_key,
            base_url=self.base_url,
        )

        out: Dict[str, pd.DataFrame] = {}
        for symbol in _as_symbol_list(symbols):
            bars = api.get_bars(symbol, tf, start=start, end=end)
            df = bars.df if hasattr(bars, "df") else pd.DataFrame(bars)
            if df is None or df.empty:
                out[symbol] = self._standardize_ohlcv(pd.DataFrame())
                continue
            # Multi-symbol responses carry a 'symbol' column; keep this one.
            if "symbol" in df.columns:
                df = df[df["symbol"] == symbol].drop(columns=["symbol"])
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
        """Not supported — Alpaca does not serve fundamentals."""
        raise NotImplementedError(
            "Alpaca does not serve fundamentals; use FMP/Polygon"
        )

    def fetch_macro(
        self,
        series: Union[str, Sequence[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Not supported — use FREDProvider for macro series."""
        raise NotImplementedError("Use FREDProvider for macro series")
