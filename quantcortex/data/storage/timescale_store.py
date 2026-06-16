"""TimescaleDB-backed OHLCV store (optional dependency: SQLAlchemy + psycopg2).

TimescaleDB is a PostgreSQL extension that turns ordinary tables into
time-partitioned *hypertables*, making large OHLCV histories fast to query.
This store wraps that workflow behind a small typed API.

SQLAlchemy and psycopg2 are optional; they are imported lazily inside
:meth:`connect`.  If they are missing, :meth:`connect` raises ``ImportError``
with an install hint.  Nothing here touches the network at import time.
"""

from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["TimescaleStore"]

# A SQL table identifier cannot be passed as a bind parameter, so wherever a
# table name is interpolated into a statement we validate it against this
# pattern first (optionally schema-qualified, e.g. ``market.ohlcv``).  This
# closes the only string-interpolation surface in the module; all VALUES
# (symbol, timestamps) are already passed as bound ``:param`` placeholders.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


def _safe_table(table: str) -> str:
    """Return ``table`` if it is a valid SQL identifier, else raise ValueError."""
    if not isinstance(table, str) or not _IDENTIFIER_RE.match(table):
        raise ValueError(
            f"invalid table identifier {table!r}: must match "
            f"[schema.]name with only letters, digits and underscores"
        )
    return table


def _prepare_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize and validate one symbol's OHLCV batch for persistence."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "time" not in out.columns:
        out = out.reset_index()
        first = out.columns[0]
        if first != "time":
            out = out.rename(columns={first: "time"})
    if "close" not in out.columns:
        raise ValueError("OHLCV writes require a close column")
    out["time"] = pd.to_datetime(out["time"], errors="coerce", utc=True)
    if out["time"].isna().any():
        raise ValueError("OHLCV timestamps must be valid")
    if out["time"].duplicated().any():
        raise ValueError("OHLCV batch contains duplicate timestamps")

    numeric = [
        col
        for col in ("open", "high", "low", "close", "adj_close", "volume")
        if col in out.columns
    ]
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce")
    values = out[numeric].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("OHLCV numeric fields must not contain infinities")
    if out["close"].isna().any():
        raise ValueError("close prices must be present on every OHLCV row")
    for col in ("open", "high", "low", "close", "adj_close"):
        if col in out.columns and (out[col].dropna() <= 0.0).any():
            raise ValueError(f"{col} prices must be positive when present")
    if "volume" in out.columns and (out["volume"].dropna() < 0.0).any():
        raise ValueError("volume must be non-negative when present")
    if {"high", "low"} <= set(out.columns):
        invalid = out[["high", "low"]].dropna()
        if (invalid["high"] < invalid["low"]).any():
            raise ValueError("high must be greater than or equal to low")
    for reference in ("open", "close"):
        if {"high", reference} <= set(out.columns):
            invalid = out[["high", reference]].dropna()
            if (invalid["high"] < invalid[reference]).any():
                raise ValueError(f"high must be at least {reference}")
        if {"low", reference} <= set(out.columns):
            invalid = out[["low", reference]].dropna()
            if (invalid["low"] > invalid[reference]).any():
                raise ValueError(f"low must be at most {reference}")

    out["symbol"] = symbol.strip()
    keep = [
        "time",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    return out[[col for col in keep if col in out.columns]].sort_values("time")


class TimescaleStore:
    """Read/write OHLCV bars in a TimescaleDB hypertable.

    Parameters
    ----------
    url:
        SQLAlchemy database URL.  Falls back to the ``TIMESCALE_URL`` environment
        variable when not supplied.
    """

    def __init__(self, url: Optional[str] = None) -> None:
        self.url = url or os.environ.get("TIMESCALE_URL")
        self._engine = None

    def connect(self):
        """Create (and cache) the SQLAlchemy engine.

        Raises
        ------
        ImportError
            If SQLAlchemy is not installed.
        ValueError
            If no database URL was provided or found in the environment.
        """
        if self._engine is not None:
            return self._engine
        try:
            import sqlalchemy  # lazy optional import
        except ImportError as exc:  # pragma: no cover - exercised only without deps
            raise ImportError("pip install sqlalchemy psycopg2-binary") from exc

        if not self.url:
            raise ValueError(
                "no database URL: pass url=... or set the TIMESCALE_URL env var"
            )
        self._engine = sqlalchemy.create_engine(self.url)
        return self._engine

    def create_ohlcv_hypertable(self, table: str = "ohlcv") -> None:
        """Create the OHLCV table and register it as a TimescaleDB hypertable."""
        import sqlalchemy  # lazy optional import

        table = _safe_table(table)
        engine = self.connect()
        create_sql = sqlalchemy.text(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                time        TIMESTAMPTZ      NOT NULL,
                symbol      TEXT             NOT NULL,
                open        DOUBLE PRECISION,
                high        DOUBLE PRECISION,
                low         DOUBLE PRECISION,
                close       DOUBLE PRECISION,
                adj_close   DOUBLE PRECISION,
                volume      DOUBLE PRECISION,
                PRIMARY KEY (symbol, time)
            );
            """
        )
        hypertable_sql = sqlalchemy.text(
            f"SELECT create_hypertable('{table}', 'time', "
            f"if_not_exists => TRUE, migrate_data => TRUE);"
        )
        add_adj_close_sql = sqlalchemy.text(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS adj_close DOUBLE PRECISION;"
        )
        with engine.begin() as conn:
            conn.execute(create_sql)
            conn.execute(add_adj_close_sql)
            conn.execute(hypertable_sql)

    def write_ohlcv(
        self, df: pd.DataFrame, symbol: str, table: str = "ohlcv"
    ) -> int:
        """Insert OHLCV bars for ``symbol`` into ``table``.

        The DataFrame is expected to be indexed (or carry a ``time`` column) by
        timestamp with ``open/high/low/close/volume`` columns.  Returns the
        number of rows written.
        """
        table = _safe_table(table)
        out = _prepare_ohlcv(df, symbol)
        if out.empty:
            return 0
        import sqlalchemy  # lazy optional import

        engine = self.connect()
        cols = list(out.columns)
        placeholders = ", ".join(f":{col}" for col in cols)
        updates = [col for col in cols if col not in {"time", "symbol"}]
        conflict = (
            "DO UPDATE SET "
            + ", ".join(f"{col} = EXCLUDED.{col}" for col in updates)
            if updates
            else "DO NOTHING"
        )
        statement = sqlalchemy.text(
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT (symbol, time) {conflict}"
        )
        records = out.astype(object).where(pd.notna(out), None).to_dict("records")
        with engine.begin() as conn:
            conn.execute(statement, records)
        return len(out)

    def read_ohlcv(
        self,
        symbol: str,
        start=None,
        end=None,
        table: str = "ohlcv",
    ) -> pd.DataFrame:
        """Read OHLCV bars for ``symbol`` within an optional ``[start, end]``.

        Returns a DataFrame indexed by ``time`` and ordered ascending.
        """
        table = _safe_table(table)
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        symbol = symbol.strip()
        start_ts = self._coerce_bound(start, "start")
        end_ts = self._coerce_bound(end, "end")
        if start_ts is not None and end_ts is not None and start_ts > end_ts:
            raise ValueError("start must be before or equal to end")
        import sqlalchemy  # lazy optional import

        engine = self.connect()
        clauses = ["symbol = :symbol"]
        params: dict = {"symbol": symbol}
        if start_ts is not None:
            clauses.append("time >= :start")
            params["start"] = start_ts.to_pydatetime()
        if end_ts is not None:
            clauses.append("time <= :end")
            params["end"] = end_ts.to_pydatetime()
        where = " AND ".join(clauses)
        query = sqlalchemy.text(
            f"SELECT * FROM {table} WHERE {where} ORDER BY time ASC"
        )
        with engine.connect() as conn:
            frame = pd.read_sql(query, conn, params=params, parse_dates=["time"])
        if "time" in frame.columns:
            frame = frame.set_index("time")
        return frame

    @staticmethod
    def _coerce_bound(value, name: str) -> Optional[pd.Timestamp]:
        if value is None:
            return None
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a valid timestamp") from exc
        if pd.isna(timestamp):
            raise ValueError(f"{name} must be a valid timestamp")
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp
