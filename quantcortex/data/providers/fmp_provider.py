"""Financial Modeling Prep (FMP) data provider.

Serves OHLCV history, tidy quarterly fundamentals (income, balance sheet, and
cash flow), and macro economic indicators. HTTP is done with the standard library
(``urllib.request`` + ``json``) so no third-party HTTP client is required.

The API key defaults to the ``FMP_API_KEY`` environment variable and is only
required when a network method is actually called.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from quantcortex.data.providers.base import (
    FUNDAMENTAL_COLUMNS,
    DataProvider,
    _as_symbol_list,
    _canonical_fundamental_fields,
    _canonicalize_fundamental_records,
)

__all__ = ["FMPProvider"]


class FMPProvider(DataProvider):
    """:class:`DataProvider` backed by the Financial Modeling Prep REST API."""

    name = "fmp"

    #: base URL for the v3 REST API
    _BASE_URL = "https://financialmodelingprep.com/api/v3"

    #: network timeout in seconds for HTTP requests
    _TIMEOUT = 30

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Store the FMP API key, falling back to ``FMP_API_KEY``."""
        self.api_key = api_key or os.environ.get("FMP_API_KEY")

    # ------------------------------------------------------------------ #
    # HTTP helper
    # ------------------------------------------------------------------ #
    def _get(self, path: str, **params) -> Union[list, dict]:
        """GET ``/{path}`` with query params, returning parsed JSON.

        The API key is appended automatically; ``None`` params are dropped.
        """
        if not self.api_key:
            raise ValueError(
                "FMPProvider requires an API key; pass api_key=... or set "
                "the FMP_API_KEY environment variable."
            )
        query = {k: v for k, v in params.items() if v is not None}
        query["apikey"] = self.api_key
        url = f"{self._BASE_URL}/{path}?{urllib.parse.urlencode(query)}"
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("FMPProvider requires HTTPS without URL userinfo")
        request = urllib.request.Request(url, headers={"User-Agent": "quantcortex"})
        # The URL is constrained to HTTPS without URL userinfo immediately above.
        with urllib.request.urlopen(  # nosec B310
            request,
            timeout=self._TIMEOUT,
        ) as resp:
            payload = resp.read().decode("utf-8")
        return json.loads(payload)

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
        """Fetch daily history via ``historical-price-full`` and standardise.

        Only the ``1d`` timeframe is served by this endpoint.
        """
        if timeframe not in ("1d", "1day"):
            raise ValueError(
                f"FMPProvider.fetch_ohlcv only supports the '1d' timeframe, "
                f"got {timeframe!r}."
            )

        out: Dict[str, pd.DataFrame] = {}
        for symbol in _as_symbol_list(symbols):
            payload = self._get(
                f"historical-price-full/{symbol}", **{"from": start, "to": end}
            )
            historical = (
                payload.get("historical", [])
                if isinstance(payload, dict)
                else payload
            )
            if not historical:
                out[symbol] = self._standardize_ohlcv(pd.DataFrame())
                continue
            df = pd.DataFrame(historical)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            # FMP names the split/dividend-adjusted column 'adjClose'.
            if "adjclose" in df.columns:
                df = df.rename(columns={"adjclose": "adj_close"})
            if "adjClose" in df.columns:
                df = df.rename(columns={"adjClose": "adj_close"})
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
        """Return tidy quarterly fundamentals with canonical field names.

        ``period_end`` is the statement ``date``; the PIT ``announcement_date``
        is the ``fillingDate`` (FMP's spelling), ``filingDate``, or
        ``acceptedDate``. Rows without a public-availability date are omitted.
        """
        wanted = _canonical_fundamental_fields(fields)
        meta_keys = {
            "date",
            "symbol",
            "reportedCurrency",
            "cik",
            "fillingDate",
            "filingDate",
            "acceptedDate",
            "calendarYear",
            "period",
            "link",
            "finalLink",
        }

        frames: List[pd.DataFrame] = []
        for symbol in _as_symbol_list(symbols):
            for statement in (
                "income-statement",
                "balance-sheet-statement",
                "cash-flow-statement",
            ):
                rows = self._get(
                    f"{statement}/{symbol}", period="quarter", limit=120
                )
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    tidy = self._melt_row(symbol, row, wanted, meta_keys)
                    if not tidy.empty:
                        frames.append(tidy)

        if not frames:
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)

        result = pd.concat(frames, ignore_index=True)
        result = result.drop_duplicates(
            ["symbol", "period_end", "announcement_date", "field"], keep="first"
        )
        if start is not None:
            result = result[
                result["period_end"]
                >= pd.to_datetime(start, errors="raise", utc=True).tz_localize(None)
            ]
        if end is not None:
            result = result[
                result["period_end"]
                <= pd.to_datetime(end, errors="raise", utc=True).tz_localize(None)
            ]
        return result.reset_index(drop=True)

    def _melt_row(
        self, symbol: str, row: dict, wanted: Optional[set], meta_keys: set
    ) -> pd.DataFrame:
        """Flatten one FMP statement row into tidy fundamental rows."""
        period_end = row.get("date")
        if period_end is None:
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
        announcement = (
            row.get("fillingDate")
            or row.get("filingDate")
            or row.get("acceptedDate")
        )
        if announcement is None:
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
        period_ts = pd.to_datetime(period_end, errors="coerce", utc=True)
        announcement_ts = pd.to_datetime(announcement, errors="coerce", utc=True)
        if (
            pd.isna(period_ts)
            or pd.isna(announcement_ts)
            or announcement_ts < period_ts
        ):
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
        period_ts = period_ts.tz_localize(None)
        announcement_ts = announcement_ts.tz_localize(None)

        records = _canonicalize_fundamental_records(
            ((field, value) for field, value in row.items() if field not in meta_keys),
            wanted,
        )

        if not records:
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)

        tidy = pd.DataFrame(records)
        tidy["symbol"] = symbol
        tidy["period_end"] = period_ts
        tidy["announcement_date"] = announcement_ts
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
        """Return a wide DataFrame of FMP economic indicators indexed by date."""
        columns: Dict[str, pd.Series] = {}
        for name in _as_symbol_list(series):
            payload = self._get(
                "economic", name=name, **{"from": start, "to": end}
            )
            if not isinstance(payload, list) or not payload:
                continue
            df = pd.DataFrame(payload)
            if "date" not in df.columns or "value" not in df.columns:
                continue
            df["date"] = pd.to_datetime(df["date"])
            s = pd.to_numeric(df.set_index("date")["value"], errors="coerce")
            columns[name] = s.sort_index()

        if not columns:
            empty = pd.DataFrame()
            empty.index.name = "date"
            return empty

        wide = pd.DataFrame(columns).sort_index()
        wide.index.name = "date"
        return wide
