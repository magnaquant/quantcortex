"""Abstract data provider interface.

Every concrete provider (yfinance, Polygon, Alpaca, CCXT, FRED, FMP) implements
this ABC so the rest of the platform can swap data sources without code changes.
Providers normalise their raw payloads to the platform's canonical shapes:

* **OHLCV** - a ``pandas.DataFrame`` per symbol, ``DatetimeIndex`` (UTC-naive,
  ascending), columns ``[open, high, low, close, adj_close, volume]`` (float64).
* **Fundamentals** - a tidy ``DataFrame`` with at least the columns
  ``[symbol, period_end, announcement_date, field, value]``.  The presence of
  ``announcement_date`` is what lets ``pit_enforcer`` guarantee point-in-time
  correctness.
* **Macro** - a wide ``DataFrame`` indexed by date, one column per series id.

Network calls are made lazily *inside* methods, never at import time, so the
package imports cleanly without provider credentials or connectivity.
"""

from __future__ import annotations

import abc
import re
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = [
    "DataProvider",
    "OHLCV_COLUMNS",
    "FUNDAMENTAL_COLUMNS",
    "canonical_fundamental_field",
]

OHLCV_COLUMNS: List[str] = ["open", "high", "low", "close", "adj_close", "volume"]
FUNDAMENTAL_COLUMNS: List[str] = [
    "symbol",
    "period_end",
    "announcement_date",
    "field",
    "value",
]

_FUNDAMENTAL_FIELD_ALIASES = {
    "earnings": "net_income",
    "net_income": "net_income",
    "net_income_loss": "net_income",
    "net_income_common_stockholders": "net_income",
    "net_income_applicable_to_common_shares": "net_income",
    "book_value": "book_value",
    "equity": "book_value",
    "stockholders_equity": "book_value",
    "total_equity_gross_minority_interest": "book_value",
    "total_stockholders_equity": "book_value",
    "ebitda": "ebitda",
    "normalized_ebitda": "ebitda",
    "enterprise_value": "enterprise_value",
    "shares_outstanding": "shares_outstanding",
    "common_stock_shares_outstanding": "shares_outstanding",
    "ordinary_shares_number": "shares_outstanding",
    "share_issued": "shares_outstanding",
    # Period-average income-statement share counts are not interchangeable with
    # point-in-time shares outstanding for market-cap construction.
    "weighted_average_shares": "weighted_average_shares",
    "weighted_average_shares_outstanding": "weighted_average_shares",
    "weighted_average_shs_out": "weighted_average_shares",
    "diluted_average_shares": "diluted_weighted_average_shares",
    "weighted_average_shares_diluted": "diluted_weighted_average_shares",
    "weighted_average_shs_out_dil": "diluted_weighted_average_shares",
    "market_cap": "market_cap",
    "market_capitalization": "market_cap",
    "gross_profit": "gross_profit",
    "revenue": "revenue",
    "revenues": "revenue",
    "total_revenue": "revenue",
    "operating_cashflow": "operating_cashflow",
    "operating_cash_flow": "operating_cashflow",
    "cash_flow_from_continuing_operating_activities": "operating_cashflow",
    "net_cash_flow_from_operating_activities": "operating_cashflow",
    "assets": "total_assets",
    "total_assets": "total_assets",
}

_FUNDAMENTAL_FIELD_PRIORITY = {
    "net_income_common_stockholders": 0,
    "net_income_applicable_to_common_shares": 0,
    "net_income": 1,
    "net_income_loss": 1,
    "stockholders_equity": 0,
    "total_stockholders_equity": 0,
    "book_value": 0,
    "equity": 1,
    "total_equity_gross_minority_interest": 2,
    "diluted_average_shares": 0,
    "weighted_average_shares_diluted": 1,
    "weighted_average_shs_out_dil": 2,
    "weighted_average_shares": 0,
    "weighted_average_shares_outstanding": 1,
    "weighted_average_shs_out": 2,
    "shares_outstanding": 0,
    "common_stock_shares_outstanding": 0,
    "ordinary_shares_number": 1,
    "share_issued": 2,
}


def _normalize_fundamental_field(field: str) -> str:
    if not isinstance(field, str) or not field.strip():
        raise ValueError("fundamental field names must be non-empty strings")
    snake_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", field.strip())
    normalized = re.sub(r"[^a-z0-9]+", "_", snake_case.lower()).strip("_")
    return normalized


def canonical_fundamental_field(field: str) -> str:
    """Map a provider-native statement label to the canonical field name."""
    normalized = _normalize_fundamental_field(field)
    return _FUNDAMENTAL_FIELD_ALIASES.get(normalized, normalized)


def _canonical_fundamental_fields(
    fields: Optional[Sequence[str]],
) -> Optional[set[str]]:
    if fields is None:
        return None
    canonical = {canonical_fundamental_field(field) for field in fields}
    if not canonical:
        raise ValueError("fields must contain at least one field name")
    return canonical


def _canonicalize_fundamental_records(
    records: Iterable[tuple[str, object]], wanted: Optional[set[str]] = None
) -> List[dict]:
    """Normalize one report and choose one deterministic source per field."""
    selected: dict[str, tuple[int, int, float]] = {}
    for order, (raw_field, value) in enumerate(records):
        normalized = _normalize_fundamental_field(raw_field)
        field = _FUNDAMENTAL_FIELD_ALIASES.get(normalized, normalized)
        if wanted is not None and field not in wanted:
            continue
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric) or not np.isfinite(float(numeric)):
            continue
        candidate = (
            _FUNDAMENTAL_FIELD_PRIORITY.get(normalized, 100),
            order,
            float(numeric),
        )
        current = selected.get(field)
        if current is None or candidate[:2] < current[:2]:
            selected[field] = candidate
    return [
        {"field": field, "value": selected[field][2]}
        for field in sorted(selected)
    ]


def _as_symbol_list(symbols: Union[str, Iterable[str]]) -> List[str]:
    values = [symbols] if isinstance(symbols, str) else list(symbols)
    if not values or any(not isinstance(s, str) or not s.strip() for s in values):
        raise ValueError("symbols must contain non-empty strings")
    cleaned = [s.strip() for s in values]
    if len(cleaned) != len(set(cleaned)):
        raise ValueError("symbols must be unique")
    return cleaned


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
        if out.columns.has_duplicates:
            raise ValueError("OHLCV columns must be unique after normalization")
        if "adj_close" not in out.columns and "close" in out.columns:
            out["adj_close"] = out["close"]
        for col in OHLCV_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA
        out = out[OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce").astype("float64")
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index, errors="coerce", utc=True)
        elif out.index.tz is None:
            out.index = pd.to_datetime(out.index, errors="coerce", utc=True)
        if out.index.tz is not None:
            # Enforce the documented UTC-naive contract.
            out.index = out.index.tz_convert("UTC").tz_localize(None)
        if out.index.hasnans or out.index.has_duplicates:
            raise ValueError("OHLCV index must contain unique, valid timestamps")
        values = out.to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("OHLCV values must not be infinite")
        price_cols = ["open", "high", "low", "close", "adj_close"]
        observed_prices = out[price_cols]
        if (observed_prices.notna() & (observed_prices <= 0.0)).any(axis=None):
            raise ValueError("OHLCV prices must be positive when present")
        if (out["volume"].notna() & (out["volume"] < 0.0)).any():
            raise ValueError("OHLCV volume must be non-negative when present")
        observed_ohlc = out[["open", "high", "low", "close"]]
        row_max = observed_ohlc.max(axis=1, skipna=True)
        row_min = observed_ohlc.min(axis=1, skipna=True)
        if (out["high"].notna() & (out["high"] < row_max)).any():
            raise ValueError("OHLCV high is below another price field")
        if (out["low"].notna() & (out["low"] > row_min)).any():
            raise ValueError("OHLCV low is above another price field")
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
        if field not in OHLCV_COLUMNS:
            raise ValueError(f"unknown OHLCV field {field!r}")
        data = self.fetch_ohlcv(symbols, start, end, timeframe)
        cols = {sym: df[field] for sym, df in data.items() if field in df}
        if not cols:
            return pd.DataFrame()
        panel = pd.DataFrame(cols).sort_index()
        return panel.reindex(
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
        return self.get_prices(symbols, start, end, **kwargs).pct_change(
            fill_method=None
        ).dropna(how="all")
