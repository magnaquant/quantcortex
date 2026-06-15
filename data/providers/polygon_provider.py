"""Polygon.io data provider.

Serves OHLCV aggregates and tidy quarterly fundamentals from Polygon's vX
financials endpoint.  Macro series are not available; use
:class:`FREDProvider`.

The Polygon SDK is imported lazily inside the methods that need it, and the
API key is only required when a network method is actually invoked.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from data.providers.base import DataProvider, FUNDAMENTAL_COLUMNS, _as_symbol_list

__all__ = ["PolygonProvider"]


class PolygonProvider(DataProvider):
    """:class:`DataProvider` backed by the Polygon.io REST API."""

    name = "polygon"

    #: map canonical timeframes -> (multiplier, timespan) for aggregates
    _TIMESPAN_MAP: Dict[str, tuple] = {
        "1d": (1, "day"),
        "1h": (1, "hour"),
        "1wk": (1, "week"),
        "1w": (1, "week"),
        "1mo": (1, "month"),
        "1m": (1, "minute"),
        "5m": (5, "minute"),
        "15m": (15, "minute"),
        "30m": (30, "minute"),
    }

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Store credentials; fall back to ``POLYGON_API_KEY`` env var."""
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _client(self):
        """Lazily build and return an authenticated ``RESTClient``."""
        if not self.api_key:
            raise ValueError(
                "PolygonProvider requires an API key; pass api_key=... or set "
                "the POLYGON_API_KEY environment variable."
            )
        try:
            from polygon import RESTClient
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "polygon-api-client is required for PolygonProvider; "
                "install it with `pip install polygon-api-client`."
            ) from exc
        return RESTClient(self.api_key)

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
        """Fetch aggregate bars per symbol and normalise to canonical schema."""
        spec = self._TIMESPAN_MAP.get(timeframe)
        if spec is None:
            raise ValueError(
                f"Unsupported timeframe {timeframe!r} for Polygon; "
                f"choose one of {sorted(self._TIMESPAN_MAP)}."
            )
        multiplier, timespan = spec
        client = self._client()

        out: Dict[str, pd.DataFrame] = {}
        for symbol in _as_symbol_list(symbols):
            rows: List[dict] = []
            for agg in client.list_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=start,
                to=end,
                adjusted=True,
                limit=50000,
            ):
                rows.append(
                    {
                        "date": getattr(agg, "timestamp", None),
                        "open": getattr(agg, "open", None),
                        "high": getattr(agg, "high", None),
                        "low": getattr(agg, "low", None),
                        "close": getattr(agg, "close", None),
                        "volume": getattr(agg, "volume", None),
                    }
                )
            if not rows:
                out[symbol] = self._standardize_ohlcv(pd.DataFrame())
                continue
            df = pd.DataFrame(rows)
            # Polygon timestamps are epoch milliseconds.
            df["date"] = pd.to_datetime(df["date"], unit="ms")
            df = df.set_index("date")
            out[symbol] = self._standardize_ohlcv(df)
        return out

    # ------------------------------------------------------------------ #
    # Fundamentals
    # ------------------------------------------------------------------ #
    def fetch_fundamentals(
        self,
        symbols: Union[str, Sequence[str]],
        fields: Optional[Sequence[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return tidy quarterly fundamentals from Polygon's vX financials.

        ``period_end`` is taken from ``end_date``/``period_of_report`` and the
        PIT ``announcement_date`` from ``filing_date``.
        """
        client = self._client()
        wanted = {str(f) for f in fields} if fields else None

        frames: List[pd.DataFrame] = []
        for symbol in _as_symbol_list(symbols):
            for fin in client.vx.list_stock_financials(
                ticker=symbol,
                timeframe="quarterly",
                period_of_report_date_gte=start,
                period_of_report_date_lte=end,
                limit=100,
            ):
                tidy = self._melt_financial(symbol, fin, wanted)
                if not tidy.empty:
                    frames.append(tidy)

        if not frames:
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def _melt_financial(self, symbol: str, fin, wanted: Optional[set]) -> pd.DataFrame:
        """Flatten one Polygon financial report into tidy rows."""
        period_end = (
            getattr(fin, "end_date", None)
            or getattr(fin, "period_of_report", None)
        )
        announcement = getattr(fin, "filing_date", None) or period_end
        financials = getattr(fin, "financials", None)
        if financials is None or period_end is None:
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)

        records: List[dict] = []
        # ``financials`` groups statements (income_statement, balance_sheet, ...);
        # each maps field -> object with a numeric ``value`` attribute.
        for statement in vars(financials).values():
            if statement is None:
                continue
            items = statement if isinstance(statement, dict) else vars(statement)
            for field, datapoint in items.items():
                if wanted is not None and field not in wanted:
                    continue
                value = getattr(datapoint, "value", None)
                if value is None and isinstance(datapoint, dict):
                    value = datapoint.get("value")
                if value is None:
                    continue
                records.append({"field": field, "value": value})

        if not records:
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)

        tidy = pd.DataFrame(records)
        tidy["symbol"] = symbol
        tidy["period_end"] = pd.to_datetime(period_end)
        tidy["announcement_date"] = pd.to_datetime(announcement)
        tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce")
        tidy = tidy.dropna(subset=["value"])
        return tidy[FUNDAMENTAL_COLUMNS]

    # ------------------------------------------------------------------ #
    # Macro
    # ------------------------------------------------------------------ #
    def fetch_macro(
        self,
        series: Union[str, Sequence[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Not supported - use FREDProvider for macro series."""
        raise NotImplementedError("Use FREDProvider for macro series")
