"""Yahoo Finance data provider (``yfinance``).

Serves OHLCV bars and a best-effort tidy view of quarterly fundamentals.
``yfinance`` does not expose exact SEC filing dates, so the point-in-time
``announcement_date`` is approximated as ``period_end + 45 calendar days`` - a
conservative reporting-lag proxy (the SEC 10-Q deadline for large accelerated
filers is 40 days; smaller filers get 45).  Macro series are not available
here; use :class:`FREDProvider`.

All heavy/optional imports happen lazily inside the methods so the module
imports with only the core scientific stack present.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from data.providers.base import DataProvider, FUNDAMENTAL_COLUMNS, _as_symbol_list

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

    #: conservative reporting-lag (days) added to ``period_end`` because
    #: yfinance lacks exact filing dates.
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
        """Return tidy quarterly fundamentals from income & balance statements.

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

        wanted = {str(f) for f in fields} if fields else None
        frames: List[pd.DataFrame] = []
        for symbol in _as_symbol_list(symbols):
            ticker = yf.Ticker(symbol)
            for statement in (
                ticker.quarterly_financials,
                ticker.quarterly_balance_sheet,
            ):
                tidy = self._melt_statement(symbol, statement, wanted)
                if not tidy.empty:
                    frames.append(tidy)

        if not frames:
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)

        result = pd.concat(frames, ignore_index=True)
        if start is not None:
            result = result[result["period_end"] >= pd.Timestamp(start)]
        if end is not None:
            result = result[result["period_end"] <= pd.Timestamp(end)]
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

        df = statement.copy()
        df.index = df.index.astype(str)
        if wanted is not None:
            df = df.loc[df.index.intersection(wanted)]
            if df.empty:
                return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)

        df = df.reset_index().rename(columns={"index": "field"})
        tidy = df.melt(
            id_vars="field", var_name="period_end", value_name="value"
        )
        tidy["period_end"] = pd.to_datetime(tidy["period_end"])
        tidy["announcement_date"] = tidy["period_end"] + pd.Timedelta(
            days=self._REPORTING_LAG_DAYS
        )
        tidy["symbol"] = symbol
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
        """Not supported - yfinance does not serve macro series."""
        raise NotImplementedError("Use FREDProvider for macro series")
