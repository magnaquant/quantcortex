"""Redis-backed cache with a transparent in-process fallback.

Redis is an optional dependency.  When it is unavailable (or unreachable) this
cache logs a warning once and falls back to an in-process dictionary with manual
TTL expiry tracked via :func:`time.monotonic`.  The fallback means cache calls
always work offline — convenient for tests and local development — at the cost
of not being shared across processes.

DataFrames are serialized to Parquet bytes via PyArrow so they round-trip with
dtypes intact and stay compact.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = ["RedisCache"]

logger = logging.getLogger(__name__)


def _df_to_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to Parquet bytes (index preserved)."""
    table = pa.Table.from_pandas(df, preserve_index=True)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def _bytes_to_df(blob: bytes) -> pd.DataFrame:
    """Deserialize Parquet bytes back into a DataFrame."""
    buf = io.BytesIO(blob)
    table = pq.read_table(buf)
    return table.to_pandas()


class RedisCache:
    """Key/value cache for bytes, scalars, and DataFrames.

    Parameters
    ----------
    url:
        Redis connection URL (e.g. ``redis://localhost:6379/0``).  When omitted,
        or when the ``redis`` package / server is unavailable, the in-process
        fallback is used.
    default_ttl:
        Default time-to-live in seconds for entries written without an explicit
        ``ttl``.
    namespace:
        Prefix applied to every key to avoid collisions across applications.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        default_ttl: int = 3600,
        namespace: str = "qc",
    ) -> None:
        self.url = url
        self.default_ttl = default_ttl
        self.namespace = namespace
        self._client = None
        self._fallback: dict[str, tuple[bytes, Optional[float]]] = {}
        self._using_fallback = False
        self._init_client()

    def _init_client(self) -> None:
        try:
            import redis  # lazy optional import

            if self.url:
                client = redis.Redis.from_url(self.url)
            else:
                client = redis.Redis()
            client.ping()  # force a connection check
            self._client = client
        except Exception as exc:
            logger.warning(
                "redis unavailable (%s); using in-process dict cache fallback", exc
            )
            self._client = None
            self._using_fallback = True

    @property
    def using_fallback(self) -> bool:
        """``True`` when the in-process dict fallback is active."""
        return self._using_fallback

    def _key(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    # -- fallback helpers -------------------------------------------------

    def _fallback_expired(self, full_key: str) -> bool:
        entry = self._fallback.get(full_key)
        if entry is None:
            return True
        _, expiry = entry
        if expiry is not None and time.monotonic() >= expiry:
            self._fallback.pop(full_key, None)
            return True
        return False

    # -- raw bytes/scalar API ---------------------------------------------

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value (``bytes``, ``str``, or numeric) under ``key``."""
        full = self._key(key)
        ttl = self.default_ttl if ttl is None else ttl
        if isinstance(value, bytes):
            blob = value
        elif isinstance(value, str):
            blob = value.encode("utf-8")
        else:
            blob = repr(value).encode("utf-8")

        if self._client is not None:
            if ttl and ttl > 0:
                self._client.set(full, blob, ex=ttl)
            else:
                self._client.set(full, blob)
        else:
            expiry = time.monotonic() + ttl if ttl and ttl > 0 else None
            self._fallback[full] = (blob, expiry)

    def get(self, key: str) -> Optional[bytes]:
        """Return the raw bytes stored under ``key``, or ``None``."""
        full = self._key(key)
        if self._client is not None:
            return self._client.get(full)
        if self._fallback_expired(full):
            return None
        return self._fallback[full][0]

    def delete(self, key: str) -> None:
        """Remove ``key`` from the cache (no error if absent)."""
        full = self._key(key)
        if self._client is not None:
            self._client.delete(full)
        else:
            self._fallback.pop(full, None)

    def exists(self, key: str) -> bool:
        """Return ``True`` if ``key`` is present and unexpired."""
        full = self._key(key)
        if self._client is not None:
            return bool(self._client.exists(full))
        return not self._fallback_expired(full)

    # -- DataFrame API ----------------------------------------------------

    def set_df(self, key: str, df: pd.DataFrame, ttl: Optional[int] = None) -> None:
        """Serialize and store a DataFrame under ``key``."""
        self.set(key, _df_to_bytes(df), ttl=ttl)

    def get_df(self, key: str) -> Optional[pd.DataFrame]:
        """Return the DataFrame stored under ``key``, or ``None`` if absent."""
        blob = self.get(key)
        if blob is None:
            return None
        return _bytes_to_df(blob)
