"""Order lifecycle management.

The :class:`OrderManager` owns the canonical order state machine that every
broker adapter feeds into.  Keeping the state machine here (rather than in each
broker) means a single, testable definition of *legal* lifecycle transitions::

    NEW --submit--> SUBMITTED --fill--> PARTIALLY_FILLED --fill--> FILLED
      |                 |                      |
      +--cancel/reject--+------cancel----------+

Illegal transitions (e.g. filling a cancelled order, or registering two orders
with the same id) raise immediately - a malformed order must never reach a live
venue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

__all__ = [
    "OrderStatus",
    "OrderSide",
    "OrderType",
    "Order",
    "validate_order_request",
    "OrderManager",
    "OrderError",
    "DuplicateOrderError",
    "UnknownOrderError",
    "InvalidOrderTransitionError",
]


class OrderError(Exception):
    """Base class for order-management errors."""


class DuplicateOrderError(OrderError):
    """Raised when an order id is registered more than once."""


class UnknownOrderError(OrderError):
    """Raised when operating on an order id that does not exist."""


class InvalidOrderTransitionError(OrderError):
    """Raised when an illegal lifecycle transition is attempted."""


class OrderStatus(str, Enum):
    NEW = "NEW"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_STATES


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


_TERMINAL_STATES = {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}

# Legal state transitions for the order lifecycle.
_VALID_TRANSITIONS: Dict[OrderStatus, set] = {
    OrderStatus.NEW: {
        OrderStatus.SUBMITTED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
    },
    OrderStatus.SUBMITTED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
    },
    OrderStatus.PARTIALLY_FILLED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.FILLED: set(),
    OrderStatus.CANCELLED: set(),
    OrderStatus.REJECTED: set(),
}


def _coerce_order_number(value, field_name: str) -> float:
    if isinstance(value, bool):
        raise OrderError(f"Order {field_name} must be numeric, not boolean.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise OrderError(f"Order {field_name} is invalid: {exc}") from exc


@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    reject_reason: Optional[str] = None
    history: List[OrderStatus] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.order_id, str):
            raise OrderError("Order id must be a string.")
        if not isinstance(self.symbol, str):
            raise OrderError("Order symbol must be a string.")
        self.order_id = self.order_id.strip()
        self.symbol = self.symbol.strip()
        try:
            self.side = OrderSide(self.side)
            self.order_type = OrderType(self.order_type)
            self.status = OrderStatus(self.status)
            self.history = [OrderStatus(item) for item in self.history]
        except (TypeError, ValueError) as exc:
            raise OrderError(f"Invalid order enum value: {exc}") from exc
        self.quantity = _coerce_order_number(self.quantity, "quantity")
        self.filled_quantity = _coerce_order_number(
            self.filled_quantity, "filled_quantity"
        )
        if self.limit_price is not None:
            self.limit_price = _coerce_order_number(self.limit_price, "limit_price")
        if self.avg_fill_price is not None:
            self.avg_fill_price = _coerce_order_number(
                self.avg_fill_price, "avg_fill_price"
            )
        if not self.history:
            self.history.append(self.status)
        self.validate()

    def validate(self) -> None:
        """Validate static fields and the current cumulative fill state."""
        if not str(self.order_id).strip():
            raise OrderError("Order id must be non-empty.")
        if not str(self.symbol).strip():
            raise OrderError("Order symbol must be non-empty.")
        if not math.isfinite(self.quantity) or self.quantity <= 0.0:
            raise OrderError("Order quantity must be finite and positive.")
        if self.order_type is OrderType.LIMIT:
            if self.limit_price is None:
                raise OrderError("Limit orders require a limit_price.")
            if not math.isfinite(self.limit_price) or self.limit_price <= 0.0:
                raise OrderError("Limit price must be finite and positive.")
        elif self.limit_price is not None and (
            not math.isfinite(self.limit_price) or self.limit_price <= 0.0
        ):
            raise OrderError("limit_price must be finite and positive when provided.")
        if (
            not math.isfinite(self.filled_quantity)
            or self.filled_quantity < 0.0
            or self.filled_quantity > self.quantity + 1e-9
        ):
            raise OrderError("filled_quantity must be finite and within order quantity.")
        if self.avg_fill_price is not None and (
            not math.isfinite(self.avg_fill_price) or self.avg_fill_price <= 0.0
        ):
            raise OrderError("avg_fill_price must be finite and positive.")
        if self.status is OrderStatus.FILLED and not math.isclose(
            self.filled_quantity, self.quantity, rel_tol=0.0, abs_tol=1e-9
        ):
            raise OrderError("FILLED orders must report the full filled quantity.")
        if self.status is OrderStatus.PARTIALLY_FILLED and not (
            0.0 < self.filled_quantity < self.quantity - 1e-9
        ):
            raise OrderError(
                "PARTIALLY_FILLED orders must report a positive partial fill."
            )
        if not self.history or self.history[-1] is not self.status:
            raise OrderError("Order history must end at the current status.")
        for previous, current in zip(self.history, self.history[1:]):
            if current not in _VALID_TRANSITIONS[previous]:
                raise OrderError(
                    f"Order history contains illegal transition "
                    f"{previous.value} -> {current.value}."
                )

    @property
    def remaining_quantity(self) -> float:
        return max(self.quantity - self.filled_quantity, 0.0)

    @property
    def is_active(self) -> bool:
        return not self.status.is_terminal


def validate_order_request(
    symbol: str,
    side,
    quantity: float,
    order_type=OrderType.MARKET,
    limit_price: Optional[float] = None,
) -> Order:
    """Validate and normalize an order request before any broker call."""
    return Order(
        order_id="preflight",
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
    )


class OrderManager:
    """Tracks orders and enforces legal lifecycle transitions."""

    def __init__(self) -> None:
        self._orders: Dict[str, Order] = {}

    @staticmethod
    def _order_key(order_id) -> str:
        if not isinstance(order_id, str):
            raise OrderError("Order id must be a string.")
        key = order_id.strip()
        if not key:
            raise OrderError("Order id must be non-empty.")
        return key

    # ------------------------------------------------------------------ #
    # registry
    # ------------------------------------------------------------------ #
    def create_order(
        self,
        order_id: str,
        symbol: str,
        side,
        quantity: float,
        order_type=OrderType.MARKET,
        limit_price: Optional[float] = None,
    ) -> Order:
        key = self._order_key(order_id)
        if key in self._orders:
            raise DuplicateOrderError(f"Order id {key!r} already exists.")
        order = Order(
            order_id=key,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
        )
        self._orders[key] = order
        return order

    def register(self, order: Order) -> Order:
        """Register an order returned by a broker adapter."""
        if not isinstance(order, Order):
            raise OrderError("register requires an Order instance")
        order.validate()
        key = self._order_key(order.order_id)
        if key in self._orders:
            raise DuplicateOrderError(f"Order id {key!r} already exists.")
        order.order_id = key
        self._orders[key] = order
        return order

    def get(self, order_id: str) -> Order:
        key = self._order_key(order_id)
        try:
            return self._orders[key]
        except KeyError as exc:
            raise UnknownOrderError(f"Unknown order id {key!r}.") from exc

    @property
    def orders(self) -> List[Order]:
        return list(self._orders.values())

    @property
    def active_orders(self) -> List[Order]:
        return [o for o in self._orders.values() if o.is_active]

    # ------------------------------------------------------------------ #
    # transitions
    # ------------------------------------------------------------------ #
    def _transition(self, order: Order, new_status: OrderStatus) -> None:
        if new_status not in _VALID_TRANSITIONS[order.status]:
            raise InvalidOrderTransitionError(
                f"Illegal transition for order {order.order_id!r}: "
                f"{order.status.value} -> {new_status.value}"
            )
        order.status = new_status
        order.history.append(new_status)

    def submit(self, order_id: str) -> Order:
        order = self.get(order_id)
        self._transition(order, OrderStatus.SUBMITTED)
        return order

    def fill(
        self,
        order_id: str,
        filled_quantity: Optional[float] = None,
        fill_price: Optional[float] = None,
    ) -> Order:
        """Record a (partial or complete) fill.

        ``filled_quantity`` is the *incremental* quantity filled by this event;
        omitting it fills the entire remaining quantity.
        """
        order = self.get(order_id)
        increment = (
            order.remaining_quantity
            if filled_quantity is None
            else _coerce_order_number(filled_quantity, "fill quantity")
        )
        if not math.isfinite(increment) or increment <= 0:
            raise OrderError("Fill quantity must be finite and positive.")
        if fill_price is not None:
            fill_price = _coerce_order_number(fill_price, "fill price")
            if not math.isfinite(fill_price) or fill_price <= 0.0:
                raise OrderError("Fill price must be finite and positive.")
        if increment - order.remaining_quantity > 1e-9:
            raise OrderError(
                f"Fill {increment} exceeds remaining {order.remaining_quantity}."
            )

        # Validate the prospective lifecycle transition BEFORE mutating the
        # book, so an illegal fill (e.g. on a NEW or CANCELLED order) cannot
        # corrupt filled_quantity / avg_fill_price and then raise.
        target_status = (
            OrderStatus.FILLED
            if order.remaining_quantity - increment <= 1e-9
            else OrderStatus.PARTIALLY_FILLED
        )
        if target_status not in _VALID_TRANSITIONS[order.status]:
            raise InvalidOrderTransitionError(
                f"Illegal transition for order {order.order_id!r}: "
                f"{order.status.value} -> {target_status.value}"
            )

        # Volume-weighted average fill price across partial fills.
        if fill_price is not None:
            prior_value = (order.avg_fill_price or 0.0) * order.filled_quantity
            new_value = prior_value + fill_price * increment
            order.filled_quantity += increment
            order.avg_fill_price = new_value / order.filled_quantity
        else:
            order.filled_quantity += increment

        self._transition(order, target_status)
        order.validate()
        return order

    def cancel(self, order_id: str) -> Order:
        order = self.get(order_id)
        self._transition(order, OrderStatus.CANCELLED)
        return order

    def reject(self, order_id: str, reason: str = "") -> Order:
        order = self.get(order_id)
        self._transition(order, OrderStatus.REJECTED)
        order.reject_reason = reason
        return order
