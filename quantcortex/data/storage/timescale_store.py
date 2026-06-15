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
                volume      DOUBLE PRECISION,
                PRIMARY KEY (symbol, time)
            );
            """
        )
        hypertable_sql = sqlalchemy.text(
            f"SELECT create_hypertable('{table}', 'time', "
            f"if_not_exists => TRUE, migrate_data => TRUE);"
        )
        with engine.begin() as conn:
            conn.execute(create_sql)
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
        engine = self.connect()
        out = df.copy()
        if "time" not in out.columns:
            out = out.reset_index()
            # Normalize the index column name to 'time'.
            first = out.columns[0]
            if first != "time":
                out = out.rename(columns={first: "time"})
        out["symbol"] = symbol
        out["time"] = pd.to_datetime(out["time"])

        keep = ["time", "symbol", "open", "high", "low", "close", "volume"]
        cols = [c for c in keep if c in out.columns]
        out = out[cols]
        out.to_sql(table, engine, if_exists="append", index=False, method="multi")
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
        import sqlalchemy  # lazy optional import

        table = _safe_table(table)
        engine = self.connect()
        clauses = ["symbol = :symbol"]
        params: dict = {"symbol": symbol}
        if start is not None:
            clauses.append("time >= :start")
            params["start"] = pd.Timestamp(start).to_pydatetime()
        if end is not None:
            clauses.append("time <= :end")
            params["end"] = pd.Timestamp(end).to_pydatetime()
        where = " AND ".join(clauses)
        query = sqlalchemy.text(
            f"SELECT * FROM {table} WHERE {where} ORDER BY time ASC"
        )
        with engine.connect() as conn:
            frame = pd.read_sql(query, conn, params=params, parse_dates=["time"])
        if "time" in frame.columns:
            frame = frame.set_index("time")
        return frame
