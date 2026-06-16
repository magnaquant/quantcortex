"""Durable persistence of execution state across restarts.

A trading process must survive crashes and restarts without losing track of its
open orders and positions.  :class:`StatePersistence` provides a small key/value
store for exactly that, backed by **Redis** when explicitly configured or an
on-disk **JSON file** otherwise. An unreachable configured Redis instance fails
closed unless the caller explicitly allows file fallback.

All values are JSON-serialised.  :class:`~quantcortex.execution.order_manager.Order`
dataclasses and the ``OrderStatus`` / ``OrderSide`` / ``OrderType`` enums are
converted to plain dicts on save and rehydrated into ``Order`` instances on
load, so callers round-trip rich objects without bespoke serialisation.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from quantcortex.execution.brokers.base import BrokerError, Position
from quantcortex.execution.order_manager import Order, OrderError, OrderStatus

__all__ = ["ExecutionSnapshot", "StatePersistence", "StatePersistenceError"]

_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class StatePersistenceError(RuntimeError):
    """Raised when execution state cannot be read or written safely."""


@dataclass
class ExecutionSnapshot:
    """One coherent persisted view of execution and reconciliation state."""

    schema_version: int
    revision: int
    updated_at: str
    positions: Dict[str, Position]
    orders: List[Order]
    intents: Dict[str, Any]
    metadata: Dict[str, Any]


class StatePersistence:
    """Redis-backed state store with a JSON-file fallback.

    Parameters
    ----------
    url:
        Redis connection URL (e.g. ``redis://localhost:6379/0``). Falls back to
        the ``REDIS_URL`` environment variable. With no URL, the file backend is
        used.
    namespace:
        Key prefix isolating this store's keys from others sharing the backend.
    file_path:
        Explicit JSON path for the file backend. Takes precedence over
        ``QC_STATE_PATH`` and is useful for isolated tests or demonstrations.
        When supplied without an explicit ``url``, it also disables the
        ambient ``REDIS_URL`` fallback.
    """

    POSITIONS_KEY = "positions"
    ORDERS_KEY = "orders"
    EXECUTION_SNAPSHOT_KEY = "execution_snapshot"
    EXECUTION_SNAPSHOT_SCHEMA = 1

    def __init__(
        self,
        url: Optional[str] = None,
        namespace: str = "qc",
        *,
        allow_file_fallback: bool = False,
        file_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if not isinstance(namespace, str) or not _NAMESPACE_RE.fullmatch(namespace):
            raise ValueError("namespace must contain only letters, digits, _, ., or -")
        self.namespace = namespace
        if url is not None:
            self.url = url
        elif file_path is not None:
            self.url = None
        else:
            self.url = os.environ.get("REDIS_URL")
        if not isinstance(allow_file_fallback, bool):
            raise TypeError("allow_file_fallback must be a boolean")
        self.allow_file_fallback = allow_file_fallback
        self._redis = None
        self._file_path = self._resolve_file_path(file_path)
        self._thread_lock = threading.RLock()
        self._connect_redis()

    # ------------------------------------------------------------------ #
    # backend selection
    # ------------------------------------------------------------------ #
    @property
    def backend(self) -> str:
        """Which backend is active: ``"redis"`` or ``"file"``."""
        return "redis" if self._redis is not None else "file"

    def _connect_redis(self) -> None:
        """Connect to configured Redis without silently changing durability."""
        if self.url is None:
            return
        try:
            import redis  # noqa: F401  (lazy optional dependency)
        except ImportError as exc:
            if self.allow_file_fallback:
                self._redis = None
                return
            raise StatePersistenceError(
                "REDIS_URL is configured but the redis package is unavailable"
            ) from exc
        try:
            client = redis.Redis.from_url(self.url, decode_responses=True)
            client.ping()
            self._redis = client
        except Exception as exc:
            if self.allow_file_fallback:
                self._redis = None
                return
            raise StatePersistenceError(
                "could not connect to configured Redis state backend"
            ) from exc

    def _resolve_file_path(
        self, file_path: str | os.PathLike[str] | None
    ) -> str:
        path = os.fspath(file_path) if file_path is not None else None
        if path is not None and not path.strip():
            raise ValueError("file_path must not be empty")
        path = path or os.environ.get("QC_STATE_PATH")
        if path:
            return str(Path(path).expanduser().resolve())
        state_home = Path(
            os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")
        )
        return str(
            (state_home / "quantcortex" / f"state_{self.namespace}.json").resolve()
        )

    def _state_key(self, key: str) -> str:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("state key must be a non-empty string")
        return key.strip()

    def _key(self, key: str) -> str:
        return f"{self.namespace}:{self._state_key(key)}"

    # ------------------------------------------------------------------ #
    # generic key/value API
    # ------------------------------------------------------------------ #
    def save_state(self, key: str, obj: Any) -> None:
        """Persist ``obj`` (JSON-serialised) under ``key``."""
        state_key = self._state_key(key)
        redis_key = self._key(state_key)
        payload = json.dumps(obj, default=_json_default, allow_nan=False)
        if self._redis is not None:
            try:
                self._redis.set(redis_key, payload)
            except Exception as exc:
                raise StatePersistenceError("Redis state write failed") from exc
        else:
            try:
                with self._file_lock(exclusive=True):
                    store = self._read_file()
                    store[state_key] = json.loads(payload)
                    self._write_file(store)
            except StatePersistenceError:
                raise
            except Exception as exc:
                raise StatePersistenceError("file state write failed") from exc

    def load_state(self, key: str, default: Any = None) -> Any:
        """Load and JSON-decode the value stored under ``key``."""
        state_key = self._state_key(key)
        redis_key = self._key(state_key)
        if self._redis is not None:
            try:
                raw = self._redis.get(redis_key)
                return json.loads(raw) if raw is not None else default
            except (json.JSONDecodeError, TypeError) as exc:
                raise StatePersistenceError("Redis state contains invalid JSON") from exc
            except Exception as exc:
                raise StatePersistenceError("Redis state read failed") from exc
        try:
            with self._file_lock(exclusive=False):
                store = self._read_file()
                return store.get(state_key, default)
        except StatePersistenceError:
            raise
        except Exception as exc:
            raise StatePersistenceError("file state read failed") from exc

    def delete_state(self, key: str) -> None:
        """Remove ``key`` from the store (no-op if absent)."""
        state_key = self._state_key(key)
        redis_key = self._key(state_key)
        if self._redis is not None:
            try:
                self._redis.delete(redis_key)
            except Exception as exc:
                raise StatePersistenceError("Redis state delete failed") from exc
        else:
            try:
                with self._file_lock(exclusive=True):
                    store = self._read_file()
                    if state_key in store:
                        del store[state_key]
                        self._write_file(store)
            except StatePersistenceError:
                raise
            except Exception as exc:
                raise StatePersistenceError("file state delete failed") from exc

    # ------------------------------------------------------------------ #
    # positions
    # ------------------------------------------------------------------ #
    def save_positions(self, positions: Dict[str, Position | float]) -> None:
        """Persist a ``symbol -> Position`` or signed-quantity mapping."""
        self.save_state(self.POSITIONS_KEY, _serialise_positions(positions))

    def load_positions(self) -> Dict[str, Position]:
        """Load positions back into a ``symbol -> Position`` mapping."""
        return _deserialise_positions(
            self.load_state(self.POSITIONS_KEY, default={}) or {}
        )

    # ------------------------------------------------------------------ #
    # orders
    # ------------------------------------------------------------------ #
    def save_orders(self, orders) -> None:
        """Persist an iterable of :class:`Order` objects (or order dicts)."""
        self.save_state(self.ORDERS_KEY, _serialise_orders(orders))

    def load_orders(self) -> List[Order]:
        """Load orders back into a list of :class:`Order` instances."""
        return _deserialise_orders(self.load_state(self.ORDERS_KEY, default=[]) or [])

    # ------------------------------------------------------------------ #
    # coherent execution snapshot
    # ------------------------------------------------------------------ #
    def load_execution_snapshot(self) -> Optional[ExecutionSnapshot]:
        """Load the current versioned execution snapshot, if one exists."""
        raw = self.load_state(self.EXECUTION_SNAPSHOT_KEY)
        return None if raw is None else _deserialise_snapshot(raw)

    def save_execution_snapshot(
        self,
        *,
        positions: Dict[str, Position | float],
        orders,
        intents: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expected_revision: Optional[int] = None,
    ) -> ExecutionSnapshot:
        """Atomically replace execution state with optimistic concurrency.

        The first write uses ``expected_revision=None``. Every subsequent write
        must supply the revision returned by :meth:`load_execution_snapshot` or
        the prior save. A stale writer fails instead of silently losing state.
        """
        if expected_revision is not None and (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 1
        ):
            raise StatePersistenceError("expected_revision must be a positive integer")
        serialised_positions = _serialise_positions(positions)
        serialised_orders = _serialise_orders(orders)
        intents = {} if intents is None else intents
        metadata = {} if metadata is None else metadata
        if not isinstance(intents, dict) or not isinstance(metadata, dict):
            raise StatePersistenceError("snapshot intents and metadata must be dictionaries")
        try:
            safe_intents = json.loads(
                json.dumps(intents, default=_json_default, allow_nan=False)
            )
            safe_metadata = json.loads(
                json.dumps(metadata, default=_json_default, allow_nan=False)
            )
        except (TypeError, ValueError) as exc:
            raise StatePersistenceError("snapshot metadata is not JSON-safe") from exc

        state_key = self.EXECUTION_SNAPSHOT_KEY
        redis_key = self._key(state_key)

        def build(current_raw: Any) -> dict[str, Any]:
            current_revision = _snapshot_revision(current_raw)
            _assert_snapshot_revision(current_revision, expected_revision)
            return {
                "schema_version": self.EXECUTION_SNAPSHOT_SCHEMA,
                "revision": current_revision + 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "positions": serialised_positions,
                "orders": serialised_orders,
                "intents": safe_intents,
                "metadata": safe_metadata,
            }

        if self._redis is not None:
            try:
                with self._redis.pipeline() as pipeline:
                    pipeline.watch(redis_key)
                    raw_text = pipeline.get(redis_key)
                    current_raw = json.loads(raw_text) if raw_text is not None else None
                    payload = build(current_raw)
                    pipeline.multi()
                    pipeline.set(
                        redis_key,
                        json.dumps(payload, allow_nan=False),
                    )
                    pipeline.execute()
            except StatePersistenceError:
                raise
            except (json.JSONDecodeError, TypeError) as exc:
                raise StatePersistenceError(
                    "Redis execution snapshot contains invalid JSON"
                ) from exc
            except Exception as exc:
                if type(exc).__name__ == "WatchError":
                    raise StatePersistenceError(
                        "execution snapshot changed concurrently"
                    ) from exc
                raise StatePersistenceError(
                    "Redis execution snapshot write failed"
                ) from exc
        else:
            try:
                with self._file_lock(exclusive=True):
                    store = self._read_file()
                    payload = build(store.get(state_key))
                    store[state_key] = payload
                    self._write_file(store)
            except StatePersistenceError:
                raise
            except Exception as exc:
                raise StatePersistenceError(
                    "file execution snapshot write failed"
                ) from exc
        return _deserialise_snapshot(payload)

    # ------------------------------------------------------------------ #
    # file backend helpers
    # ------------------------------------------------------------------ #
    def _read_file(self) -> Dict[str, Any]:
        try:
            with open(self._file_path, "r", encoding="utf-8") as fh:
                store = json.load(fh)
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as exc:
            raise StatePersistenceError(
                f"could not read execution state {self._file_path!r}"
            ) from exc
        if not isinstance(store, dict):
            raise StatePersistenceError("execution state root must be a JSON object")
        return store

    def _write_file(self, store: Dict[str, Any]) -> None:
        # Atomic write: dump to a temp file in the same dir, then rename.
        directory = os.path.dirname(self._file_path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(store, fh, default=_json_default, allow_nan=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self._file_path)
            try:
                dir_fd = os.open(directory, os.O_RDONLY)
            except OSError:  # pragma: no cover - platform/filesystem dependent
                dir_fd = None
            if dir_fd is not None:
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    @contextmanager
    def _file_lock(self, *, exclusive: bool):
        """Serialize file-backend operations across threads and POSIX processes."""
        lock_path = f"{self._file_path}.lock"
        directory = os.path.dirname(lock_path) or "."
        os.makedirs(directory, exist_ok=True)
        with self._thread_lock, open(lock_path, "a", encoding="utf-8") as lock_file:
            try:
                import fcntl
            except ImportError:  # pragma: no cover - Windows fallback
                fcntl = None
            if fcntl is not None:
                mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
                fcntl.flock(lock_file.fileno(), mode)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


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


def _serialise_positions(
    positions: Dict[str, Position | float],
) -> Dict[str, Any]:
    if not isinstance(positions, dict):
        raise StatePersistenceError("positions must be a dictionary")
    serialised: Dict[str, Any] = {}
    for symbol, position in positions.items():
        if not isinstance(symbol, str) or not symbol.strip():
            raise StatePersistenceError("position keys must be non-empty strings")
        key = symbol.strip()
        if key in serialised:
            raise StatePersistenceError(f"duplicate position symbol {key!r}")
        if isinstance(position, Position):
            if position.symbol != key:
                raise StatePersistenceError(
                    f"position key {key!r} does not match payload symbol "
                    f"{position.symbol!r}"
                )
            serialised[key] = _serialise(position)
        else:
            if isinstance(position, bool):
                raise StatePersistenceError("position quantities must be numeric")
            try:
                validated = Position(symbol=key, quantity=position)
            except (BrokerError, TypeError, ValueError) as exc:
                raise StatePersistenceError("position quantities are invalid") from exc
            serialised[key] = validated.quantity
    return serialised


def _deserialise_positions(raw: Any) -> Dict[str, Position]:
    if not isinstance(raw, dict):
        raise StatePersistenceError("persisted positions must be a JSON object")
    out: Dict[str, Position] = {}
    try:
        for symbol, data in raw.items():
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("invalid position symbol")
            if isinstance(data, dict):
                out[symbol] = Position(
                    symbol=data.get("symbol", symbol),
                    quantity=data.get("quantity", 0.0),
                    avg_price=data.get("avg_price", 0.0),
                    market_price=data.get("market_price", 0.0),
                )
            else:
                out[symbol] = Position(symbol=symbol, quantity=data)
    except (BrokerError, TypeError, ValueError) as exc:
        raise StatePersistenceError("persisted positions are invalid") from exc
    return out


def _serialise_orders(orders) -> List[dict[str, Any]]:
    if isinstance(orders, (str, bytes, dict)):
        raise StatePersistenceError("orders must be an iterable of order objects")
    try:
        iterator = iter(orders)
    except TypeError as exc:
        raise StatePersistenceError("orders must be iterable") from exc
    serialised: List[dict[str, Any]] = []
    seen_ids: set[str] = set()
    try:
        for order in iterator:
            validated = order if isinstance(order, Order) else _deserialise_order(order)
            if not isinstance(validated, Order):
                raise StatePersistenceError(
                    "orders must contain Order instances or order dictionaries"
                )
            validated.validate()
            if validated.order_id in seen_ids:
                raise StatePersistenceError(
                    f"duplicate persisted order id {validated.order_id!r}"
                )
            seen_ids.add(validated.order_id)
            serialised.append(_serialise(validated))
    except StatePersistenceError:
        raise
    except (KeyError, OrderError, TypeError, ValueError) as exc:
        raise StatePersistenceError("orders contain invalid state") from exc
    return serialised


def _deserialise_orders(raw: Any) -> List[Order]:
    if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
        raise StatePersistenceError("persisted orders must be a list of objects")
    try:
        orders = [_deserialise_order(item) for item in raw]
    except (KeyError, OrderError, TypeError, ValueError) as exc:
        raise StatePersistenceError("persisted orders are invalid") from exc
    ids = [order.order_id for order in orders]
    if len(ids) != len(set(ids)):
        raise StatePersistenceError("persisted orders contain duplicate ids")
    return orders


def _snapshot_revision(raw: Any) -> int:
    if raw is None:
        return 0
    if not isinstance(raw, dict):
        raise StatePersistenceError("persisted execution snapshot must be an object")
    revision = raw.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise StatePersistenceError("persisted execution snapshot has invalid revision")
    return revision


def _assert_snapshot_revision(
    current_revision: int,
    expected_revision: Optional[int],
) -> None:
    if current_revision == 0:
        if expected_revision is not None:
            raise StatePersistenceError(
                "execution snapshot revision conflict: no snapshot exists"
            )
        return
    if expected_revision is None:
        raise StatePersistenceError(
            "expected_revision is required when replacing an execution snapshot"
        )
    if expected_revision != current_revision:
        raise StatePersistenceError(
            "execution snapshot revision conflict: "
            f"expected {expected_revision}, found {current_revision}"
        )


def _deserialise_snapshot(raw: Any) -> ExecutionSnapshot:
    if not isinstance(raw, dict):
        raise StatePersistenceError("persisted execution snapshot must be an object")
    schema_version = raw.get("schema_version")
    if schema_version != StatePersistence.EXECUTION_SNAPSHOT_SCHEMA:
        raise StatePersistenceError(
            f"unsupported execution snapshot schema {schema_version!r}"
        )
    revision = _snapshot_revision(raw)
    updated_at = raw.get("updated_at")
    if not isinstance(updated_at, str):
        raise StatePersistenceError("execution snapshot updated_at must be a string")
    try:
        parsed_updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise StatePersistenceError(
            "execution snapshot updated_at is invalid"
        ) from exc
    if parsed_updated_at.tzinfo is None:
        raise StatePersistenceError("execution snapshot updated_at must include timezone")
    intents = raw.get("intents", {})
    metadata = raw.get("metadata", {})
    if not isinstance(intents, dict) or not isinstance(metadata, dict):
        raise StatePersistenceError(
            "execution snapshot intents and metadata must be objects"
        )
    return ExecutionSnapshot(
        schema_version=schema_version,
        revision=revision,
        updated_at=updated_at,
        positions=_deserialise_positions(raw.get("positions", {})),
        orders=_deserialise_orders(raw.get("orders", [])),
        intents=intents,
        metadata=metadata,
    )


def _deserialise_order(data: Dict[str, Any]) -> Order:
    """Rehydrate an :class:`Order` from its serialised dict form."""
    order = Order(
        order_id=data["order_id"],
        symbol=data["symbol"],
        side=data["side"],
        quantity=data["quantity"],
        order_type=data.get("order_type", "MARKET"),
        limit_price=data.get("limit_price"),
    )
    # Restore mutable lifecycle fields that the constructor does not take.
    order.filled_quantity = _coerce_persisted_number(
        data.get("filled_quantity", 0.0), "filled_quantity"
    )
    avg_fill_price = data.get("avg_fill_price")
    order.avg_fill_price = (
        None
        if avg_fill_price is None
        else _coerce_persisted_number(avg_fill_price, "avg_fill_price")
    )
    order.reject_reason = data.get("reject_reason")
    status = data.get("status", OrderStatus.NEW.value)
    order.status = OrderStatus(status)
    history = data.get("history")
    if history:
        order.history = [OrderStatus(s) for s in history]
    order.validate()
    return order


def _coerce_persisted_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"persisted {field_name} must be numeric, not boolean")
    return float(value)
