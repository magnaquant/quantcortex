"""Yahoo Finance data provider (``yfinance``).

Serves OHLCV bars and a best-effort tidy view of quarterly fundamentals.
Yahoo and yfinance document usage restrictions, including personal-use terms;
callers are responsible for confirming that their use and redistribution are
permitted. See https://ranaroussi.github.io/yfinance/.
``yfinance`` does not expose exact SEC filing dates, so the point-in-time
``announcement_date`` is approximated as ``period_end + 45 calendar days`` - a
heuristic reporting-lag proxy, not an observed publication timestamp or a
guarantee against lookahead. Macro series are not available here; use
:class:`FREDProvider`.

All heavy/optional imports happen lazily inside the methods so the module
imports with only the core scientific stack present.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from quantcortex.data.providers.base import (
    FUNDAMENTAL_COLUMNS,
    DataProvider,
    _as_symbol_list,
    _canonical_fundamental_fields,
    _canonicalize_fundamental_records,
)

__all__ = ["YFinanceProvider"]


class YFinanceProvider(DataProvider):
    """:class:`DataProvider` backed by the ``yfinance`` library."""

    name = "yfinance"

    #: map canonical platform timeframes -> yfinance ``interval`` strings
    _INTERVAL_MAP: Dict[str, str] = {
        "1d": "1d",
        "1h": "1h",
        "1wk": "1wk",
        "1w": "1wk",
        "1mo": "1mo",
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
    }

    #: heuristic reporting lag (days) added because exact filing dates are absent.
    _REPORTING_LAG_DAYS: int = 45

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
        """Download OHLCV bars per symbol and normalise to the canonical schema.

        ``auto_adjust=False`` is used so both raw ``close`` and ``adj_close``
        survive the download.
        """
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - exercised only w/o dep
            raise ImportError(
                "yfinance is required for YFinanceProvider.fetch_ohlcv; "
                "install it with `pip install yfinance`."
            ) from exc

        interval = self._INTERVAL_MAP.get(timeframe)
        if interval is None:
            raise ValueError(
                f"Unsupported timeframe {timeframe!r} for yfinance; "
                f"choose one of {sorted(self._INTERVAL_MAP)}."
            )

        out: Dict[str, pd.DataFrame] = {}
        for symbol in _as_symbol_list(symbols):
            raw = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                actions=False,
                threads=False,
            )
            if raw is None or raw.empty:
                out[symbol] = self._standardize_ohlcv(pd.DataFrame())
                continue
            # Recent yfinance returns a MultiIndex (field, ticker); flatten it.
            if isinstance(raw.columns, pd.MultiIndex):
                raw = raw.droplevel(-1, axis=1)
            out[symbol] = self._standardize_ohlcv(raw)
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
        """Return canonical quarterly income, balance, and cash-flow data.

        ``announcement_date`` is set to ``period_end`` plus
        :attr:`_REPORTING_LAG_DAYS` calendar days because yfinance does not
        provide exact filing dates.
        """
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "yfinance is required for YFinanceProvider.fetch_fundamentals; "
                "install it with `pip install yfinance`."
            ) from exc

        wanted = _canonical_fundamental_fields(fields)
        frames: List[pd.DataFrame] = []
        for symbol in _as_symbol_list(symbols):
            ticker = yf.Ticker(symbol)
            for statement in (
                ticker.quarterly_financials,
                ticker.quarterly_balance_sheet,
                ticker.quarterly_cashflow,
            ):
                tidy = self._melt_statement(symbol, statement, wanted)
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

    def _melt_statement(
        self,
        symbol: str,
        statement: Optional[pd.DataFrame],
        wanted: Optional[set],
    ) -> pd.DataFrame:
        """Melt a yfinance statement (fields x period_end) into tidy rows."""
        if statement is None or statement.empty:
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)

        rows: List[pd.DataFrame] = []
        for period_end in statement.columns:
            period_ts = pd.to_datetime(period_end, errors="coerce", utc=True)
            if pd.isna(period_ts):
                continue
            period_ts = period_ts.tz_localize(None)
            records = _canonicalize_fundamental_records(
                zip(statement.index.astype(str), statement[period_end]), wanted
            )
            if not records:
                continue
            tidy = pd.DataFrame(records)
            tidy["symbol"] = symbol
            tidy["period_end"] = period_ts
            tidy["announcement_date"] = period_ts + pd.Timedelta(
                days=self._REPORTING_LAG_DAYS
            )
            rows.append(tidy[FUNDAMENTAL_COLUMNS])
        if not rows:
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
        return pd.concat(rows, ignore_index=True)

    # ------------------------------------------------------------------ #
    # Macro
    # ------------------------------------------------------------------ #
    def fetch_macro(
        self,
        series: Union[str, Sequence[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Not supported - yfinance does not serve macro series."""
        raise NotImplementedError("Use FREDProvider for macro series")
