"""Redis-backed cache with an explicit in-process fallback.

Redis is optional. With no URL, the cache uses an in-process dictionary with
manual TTL expiry. If an explicit Redis URL cannot be reached, fallback occurs
only when ``allow_fallback=True``. Runtime Redis failures raise instead of
silently changing consistency semantics.

DataFrames are serialized to Parquet bytes via PyArrow so they round-trip with
dtypes intact and stay compact.
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from typing import Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = ["RedisCache", "RedisCacheError"]

logger = logging.getLogger(__name__)


class RedisCacheError(RuntimeError):
    """Raised when an explicitly configured Redis operation fails."""


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
        Redis connection URL (e.g. ``redis://localhost:6379/0``). When omitted,
        the in-process fallback is used. A configured but unavailable Redis URL
        fails closed unless ``allow_fallback=True`` is explicit.
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
        allow_fallback: bool = False,
    ) -> None:
        if isinstance(default_ttl, bool) or int(default_ttl) != default_ttl:
            raise ValueError("default_ttl must be a non-negative integer")
        if default_ttl < 0:
            raise ValueError("default_ttl must be a non-negative integer")
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be a non-empty string")
        if not isinstance(allow_fallback, bool):
            raise TypeError("allow_fallback must be a boolean")
        self.url = url
        self.default_ttl = int(default_ttl)
        self.namespace = namespace.strip()
        self.allow_fallback = allow_fallback
        self._client = None
        self._fallback: dict[str, tuple[bytes, Optional[float]]] = {}
        self._using_fallback = False
        self._lock = threading.RLock()
        self._init_client()

    def _init_client(self) -> None:
        if not self.url:
            self._using_fallback = True
            return
        try:
            import redis  # lazy optional import

            client = redis.Redis.from_url(self.url)
            client.ping()  # force a connection check
            self._client = client
        except Exception as exc:
            if not self.allow_fallback:
                raise RedisCacheError(
                    f"could not connect to configured Redis URL: {exc}"
                ) from exc
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
        if not isinstance(key, str) or not key.strip():
            raise ValueError("cache key must be a non-empty string")
        return f"{self.namespace}:{key.strip()}"

    # -- fallback helpers -------------------------------------------------

    def _fallback_expired(self, full_key: str) -> bool:
        with self._lock:
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
        """Store ``value`` under ``key``.

        ``bytes`` are stored verbatim; ``str``, numbers, ``dict`` and ``list``
        values are JSON-encoded so they round-trip through :meth:`get`.  Any
        other type (e.g. ``Timestamp``, ``set``) raises ``TypeError`` rather
        than being silently degraded to a string - use :meth:`set_df` for
        DataFrames.  A one-byte type tag distinguishes raw bytes from JSON so
        DataFrame Parquet bytes (stored via :meth:`set_df`) survive intact.
        """
        full = self._key(key)
        ttl = self.default_ttl if ttl is None else ttl
        if isinstance(ttl, bool) or int(ttl) != ttl or ttl < 0:
            raise ValueError("ttl must be a non-negative integer or None")
        ttl = int(ttl)
        if isinstance(value, bytes):
            blob = b"B" + value
        else:
            blob = b"J" + json.dumps(value, allow_nan=False).encode("utf-8")

        if self._client is not None:
            try:
                if ttl > 0:
                    self._client.set(full, blob, ex=ttl)
                else:
                    self._client.set(full, blob)
            except Exception as exc:
                raise RedisCacheError(f"Redis SET failed for {full!r}") from exc
        else:
            expiry = time.monotonic() + ttl if ttl > 0 else None
            with self._lock:
                self._fallback[full] = (blob, expiry)

    def get(self, key: str) -> Any:
        """Return the value stored under ``key`` (decoded), or ``None``.

        ``bytes`` values come back as ``bytes``; JSON values come back as the
        original Python object.
        """
        full = self._key(key)
        if self._client is not None:
            try:
                raw = self._client.get(full)
            except Exception as exc:
                raise RedisCacheError(f"Redis GET failed for {full!r}") from exc
        elif self._fallback_expired(full):
            raw = None
        else:
            with self._lock:
                raw = self._fallback[full][0]

        if raw is None:
            return None
        tag, body = raw[:1], raw[1:]
        if tag == b"B":
            return body
        if tag == b"J":
            try:
                return json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise RedisCacheError(f"corrupt JSON cache value for {full!r}") from exc
        return raw  # legacy/untagged value

    def delete(self, key: str) -> None:
        """Remove ``key`` from the cache (no error if absent)."""
        full = self._key(key)
        if self._client is not None:
            try:
                self._client.delete(full)
            except Exception as exc:
                raise RedisCacheError(f"Redis DELETE failed for {full!r}") from exc
        else:
            with self._lock:
                self._fallback.pop(full, None)

    def exists(self, key: str) -> bool:
        """Return ``True`` if ``key`` is present and unexpired."""
        full = self._key(key)
        if self._client is not None:
            try:
                return bool(self._client.exists(full))
            except Exception as exc:
                raise RedisCacheError(f"Redis EXISTS failed for {full!r}") from exc
        return not self._fallback_expired(full)

    # -- DataFrame API ----------------------------------------------------

    def set_df(self, key: str, df: pd.DataFrame, ttl: Optional[int] = None) -> None:
        """Serialize and store a DataFrame under ``key``."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self.set(key, _df_to_bytes(df), ttl=ttl)

    def get_df(self, key: str) -> Optional[pd.DataFrame]:
        """Return the DataFrame stored under ``key``, or ``None`` if absent."""
        blob = self.get(key)
        if blob is None:
            return None
        return _bytes_to_df(blob)
