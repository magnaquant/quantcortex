"""Crypto exchange OHLCV provider via ``ccxt``.

Fetches paginated OHLCV candles from any ccxt-supported exchange (Binance by
default).  Crypto markets have no notion of split/dividend adjustment, so
``adj_close`` is set equal to ``close`` by :meth:`_standardize_ohlcv`.
Fundamentals and macro series are not applicable to crypto.
"""

from __future__ import annotations

from typing import ClassVar, Dict, List, Optional, Sequence, Union

import pandas as pd

from quantcortex.data.providers.base import DataProvider, _as_symbol_list

__all__ = ["CCXTProvider"]


class CCXTProvider(DataProvider):
    """:class:`DataProvider` backed by a ccxt exchange connector."""

    name = "ccxt"

    #: map canonical timeframes -> ccxt timeframe strings
    _TIMEFRAME_MAP: ClassVar[Dict[str, str]] = {
        "1d": "1d",
        "1h": "1h",
        "1wk": "1w",
        "1w": "1w",
        "1mo": "1M",
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
    }

    #: candles requested per paginated call
    _PAGE_LIMIT: int = 1000

    def __init__(
        self,
        exchange: str = "binance",
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
    ) -> None:
        """Store the target exchange id and optional trading credentials."""
        self.exchange = exchange
        self.api_key = api_key
        self.secret = secret

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _exchange(self):
        """Lazily import ccxt and instantiate the configured exchange."""
        try:
            import ccxt
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ccxt is required for CCXTProvider; "
                "install it with `pip install ccxt`."
            ) from exc

        if not hasattr(ccxt, self.exchange):
            raise ValueError(f"Unknown ccxt exchange {self.exchange!r}.")
        config: Dict[str, object] = {"enableRateLimit": True}
        if self.api_key:
            config["apiKey"] = self.api_key
        if self.secret:
            config["secret"] = self.secret
        return getattr(ccxt, self.exchange)(config)

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
        """Fetch paginated OHLCV candles per symbol and standardise them.

        When both ``start`` and ``end`` are ``None`` the exchange returns its
        most recent ``_PAGE_LIMIT`` candles per symbol.  Supplying ``end``
        without ``start`` is rejected: ccxt pagination needs a ``since``
        cursor to walk a historical range.
        """
        tf = self._TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            raise ValueError(
                f"Unsupported timeframe {timeframe!r} for ccxt; "
                f"choose one of {sorted(self._TIMEFRAME_MAP)}."
            )
        if start is None and end is not None:
            raise ValueError(
                "historical queries require `start`; omit both start and end "
                "to fetch the most recent candles"
            )
        client = self._exchange()

        start_ts = pd.to_datetime(start, errors="coerce", utc=True)
        end_ts = pd.to_datetime(end, errors="coerce", utc=True)
        if start is not None and pd.isna(start_ts):
            raise ValueError("start must be a valid timestamp")
        if end is not None and pd.isna(end_ts):
            raise ValueError("end must be a valid timestamp")
        if start is not None and end is not None and start_ts > end_ts:
            raise ValueError("start must not be after end")
        since = int(start_ts.timestamp() * 1000) if start is not None else None
        end_ms = int(end_ts.timestamp() * 1000) if end is not None else None

        out: Dict[str, pd.DataFrame] = {}
        for symbol in _as_symbol_list(symbols):
            out[symbol] = self._standardize_ohlcv(
                self._fetch_symbol(client, symbol, tf, since, end_ms)
            )
        return out

    def _fetch_symbol(
        self,
        client,
        symbol: str,
        tf: str,
        since: Optional[int],
        end_ms: Optional[int],
    ) -> pd.DataFrame:
        """Page through ``fetch_ohlcv`` until exhausted or past ``end_ms``."""
        rows: List[list] = []
        cursor = since
        previous_last_ts: Optional[int] = None
        while True:
            batch = client.fetch_ohlcv(
                symbol, timeframe=tf, since=cursor, limit=self._PAGE_LIMIT
            )
            if not batch:
                break
            rows.extend(batch)
            try:
                last_ts = int(batch[-1][0])
            except (IndexError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError("ccxt returned a malformed OHLCV batch") from exc
            if previous_last_ts is not None and last_ts <= previous_last_ts:
                raise RuntimeError("ccxt pagination did not advance")
            previous_last_ts = last_ts
            # Advance past the final candle to avoid an infinite loop.
            cursor = last_ts + 1
            if len(batch) < self._PAGE_LIMIT:
                break
            if end_ms is not None and last_ts >= end_ms:
                break

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df = df.drop_duplicates(subset="timestamp")
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop(columns="timestamp").set_index("date")
        if end_ms is not None:
            df = df[df.index <= pd.Timestamp(end_ms, unit="ms")]
        return df

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
        """Not applicable - crypto assets have no company fundamentals."""
        raise NotImplementedError(
            "Fundamentals are not applicable to crypto markets; "
            "use FMP/Polygon for equities."
        )

    def fetch_macro(
        self,
        series: Union[str, Sequence[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Not applicable - use FREDProvider for macro series."""
        raise NotImplementedError("Use FREDProvider for macro series")
