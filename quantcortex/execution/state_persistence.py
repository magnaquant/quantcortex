"""Durable persistence of execution state across restarts.

A trading process must survive crashes and restarts without losing track of its
open orders and positions.  :class:`StatePersistence` provides a small key/value
store for exactly that, backed by **Redis** when it is available and reachable,
and transparently falling back to an on-disk **JSON file** store otherwise - so
the execution layer always has somewhere durable to write, even offline or in
local development where Redis is not installed.

All values are JSON-serialised.  :class:`~execution.order_manager.Order`
dataclasses and the ``OrderStatus`` / ``OrderSide`` / ``OrderType`` enums are
converted to plain dicts on save and rehydrated into ``Order`` instances on
load, so callers round-trip rich objects without bespoke serialisation.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from quantcortex.execution.brokers.base import Position
from quantcortex.execution.order_manager import Order, OrderStatus

__all__ = ["StatePersistence"]


class StatePersistence:
    """Redis-backed state store with a JSON-file fallback.

    Parameters
    ----------
    url:
        Redis connection URL (e.g. ``redis://localhost:6379/0``).  Falls back to
        the ``REDIS_URL`` environment variable.  If Redis cannot be imported or
        the connection fails, the store silently switches to the file backend.
    namespace:
        Key prefix isolating this store's keys from others sharing the backend.
    """

    POSITIONS_KEY = "positions"
    ORDERS_KEY = "orders"

    def __init__(self, url: Optional[str] = None, namespace: str = "qc") -> None:
        self.namespace = namespace
        self.url = url or os.environ.get("REDIS_URL")
        self._redis = None
        self._file_path = self._resolve_file_path()
        self._connect_redis()

    # ------------------------------------------------------------------ #
    # backend selection
    # ------------------------------------------------------------------ #
    @property
    def backend(self) -> str:
        """Which backend is active: ``"redis"`` or ``"file"``."""
        return "redis" if self._redis is not None else "file"

    def _connect_redis(self) -> None:
        """Try to connect to Redis; on any failure stay on the file backend."""
        if self.url is None:
            return
        try:
            import redis  # noqa: F401  (lazy optional dependency)
        except ImportError:
            self._redis = None
            return
        try:
            client = redis.Redis.from_url(self.url, decode_responses=True)
            client.ping()
            self._redis = client
        except Exception:
            # Unreachable / auth failure / etc. -> fall back to file store.
            self._redis = None

    def _resolve_file_path(self) -> str:
        path = os.environ.get("QC_STATE_PATH")
        if path:
            return path
        return os.path.join(
            tempfile.gettempdir(), f"quantcortex_state_{self.namespace}.json"
        )

    def _key(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    # ------------------------------------------------------------------ #
    # generic key/value API
    # ------------------------------------------------------------------ #
    def save_state(self, key: str, obj: Any) -> None:
        """Persist ``obj`` (JSON-serialised) under ``key``."""
        payload = json.dumps(obj, default=_json_default)
        if self._redis is not None:
            self._redis.set(self._key(key), payload)
        else:
            store = self._read_file()
            store[key] = json.loads(payload)
            self._write_file(store)

    def load_state(self, key: str, default: Any = None) -> Any:
        """Load and JSON-decode the value stored under ``key``."""
        if self._redis is not None:
            raw = self._redis.get(self._key(key))
            return json.loads(raw) if raw is not None else default
        store = self._read_file()
        return store.get(key, default)

    def delete_state(self, key: str) -> None:
        """Remove ``key`` from the store (no-op if absent)."""
        if self._redis is not None:
            self._redis.delete(self._key(key))
        else:
            store = self._read_file()
            if key in store:
                del store[key]
                self._write_file(store)

    # ------------------------------------------------------------------ #
    # positions
    # ------------------------------------------------------------------ #
    def save_positions(self, positions: Dict[str, Position]) -> None:
        """Persist a ``symbol -> Position`` mapping."""
        serialised = {
            symbol: _serialise(pos) for symbol, pos in positions.items()
        }
        self.save_state(self.POSITIONS_KEY, serialised)

    def load_positions(self) -> Dict[str, Position]:
        """Load positions back into a ``symbol -> Position`` mapping."""
        raw = self.load_state(self.POSITIONS_KEY, default={}) or {}
        out: Dict[str, Position] = {}
        for symbol, data in raw.items():
            if isinstance(data, dict):
                out[symbol] = Position(
                    symbol=data.get("symbol", symbol),
                    quantity=float(data.get("quantity", 0.0)),
                    avg_price=float(data.get("avg_price", 0.0)),
                    market_price=float(data.get("market_price", 0.0)),
                )
            else:
                # A bare scalar quantity was persisted (symbol -> qty mapping).
                out[symbol] = Position(symbol=symbol, quantity=float(data))
        return out

    # ------------------------------------------------------------------ #
    # orders
    # ------------------------------------------------------------------ #
    def save_orders(self, orders) -> None:
        """Persist an iterable of :class:`Order` objects (or order dicts)."""
        serialised = [_serialise(o) for o in orders]
        self.save_state(self.ORDERS_KEY, serialised)

    def load_orders(self) -> List[Order]:
        """Load orders back into a list of :class:`Order` instances."""
        raw = self.load_state(self.ORDERS_KEY, default=[]) or []
        return [_deserialise_order(item) for item in raw]

    # ------------------------------------------------------------------ #
    # file backend helpers
    # ------------------------------------------------------------------ #
    def _read_file(self) -> Dict[str, Any]:
        try:
            with open(self._file_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_file(self, store: Dict[str, Any]) -> None:
        # Atomic write: dump to a temp file in the same dir, then rename.
        directory = os.path.dirname(self._file_path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(store, fh, default=_json_default)
            os.replace(tmp, self._file_path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise


# ---------------------------------------------------------------------- #
# (de)serialisation helpers
# ---------------------------------------------------------------------- #
def _json_default(obj: Any) -> Any:
    """``json.dumps`` fallback for enums and dataclasses."""
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        return _serialise(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _serialise(obj: Any) -> Any:
    """Convert a dataclass / enum / mapping into a JSON-safe structure."""
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialise(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    return obj


def _deserialise_order(data: Dict[str, Any]) -> Order:
    """Rehydrate an :class:`Order` from its serialised dict form."""
    order = Order(
        order_id=data["order_id"],
        symbol=data["symbol"],
        side=data["side"],
        quantity=float(data["quantity"]),
        order_type=data.get("order_type", "MARKET"),
        limit_price=data.get("limit_price"),
    )
    # Restore mutable lifecycle fields that the constructor does not take.
    order.filled_quantity = float(data.get("filled_quantity", 0.0))
    order.avg_fill_price = data.get("avg_fill_price")
    order.reject_reason = data.get("reject_reason")
    status = data.get("status", OrderStatus.NEW.value)
    order.status = OrderStatus(status)
    history = data.get("history")
    if history:
        order.history = [OrderStatus(s) for s in history]
    return order
