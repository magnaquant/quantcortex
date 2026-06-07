"""Order lifecycle management.

The :class:`OrderManager` owns the canonical order state machine that every
broker adapter feeds into.  Keeping the state machine here (rather than in each
broker) means a single, testable definition of *legal* lifecycle transitions::

    NEW ──submit──▶ SUBMITTED ──fill──▶ PARTIALLY_FILLED ──fill──▶ FILLED
      │                 │                      │
      └──cancel/reject──┴──────cancel──────────┘

Illegal transitions (e.g. filling a cancelled order, or registering two orders
with the same id) raise immediately — a malformed order must never reach a live
venue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

__all__ = [
    "OrderStatus",
    "OrderSide",
    "OrderType",
    "Order",
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
        self.side = OrderSide(self.side)
        self.order_type = OrderType(self.order_type)
        self.status = OrderStatus(self.status)
        if self.quantity <= 0:
            raise OrderError("Order quantity must be positive.")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise OrderError("Limit orders require a limit_price.")
        if not self.history:
            self.history.append(self.status)

    @property
    def remaining_quantity(self) -> float:
        return max(self.quantity - self.filled_quantity, 0.0)

    @property
    def is_active(self) -> bool:
        return not self.status.is_terminal


class OrderManager:
    """Tracks orders and enforces legal lifecycle transitions."""

    def __init__(self) -> None:
        self._orders: Dict[str, Order] = {}

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
        if order_id in self._orders:
            raise DuplicateOrderError(f"Order id {order_id!r} already exists.")
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=float(quantity),
            order_type=order_type,
            limit_price=limit_price,
        )
        self._orders[order_id] = order
        return order

    def get(self, order_id: str) -> Order:
        try:
            return self._orders[order_id]
        except KeyError as exc:
            raise UnknownOrderError(f"Unknown order id {order_id!r}.") from exc

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
            else float(filled_quantity)
        )
        if increment <= 0:
            raise OrderError("Fill quantity must be positive.")
        if increment - order.remaining_quantity > 1e-9:
            raise OrderError(
                f"Fill {increment} exceeds remaining {order.remaining_quantity}."
            )

        # Volume-weighted average fill price across partial fills.
        if fill_price is not None:
            prior_value = (order.avg_fill_price or 0.0) * order.filled_quantity
            new_value = prior_value + fill_price * increment
            order.filled_quantity += increment
            order.avg_fill_price = new_value / order.filled_quantity
        else:
            order.filled_quantity += increment

        if order.remaining_quantity <= 1e-9:
            self._transition(order, OrderStatus.FILLED)
        else:
            self._transition(order, OrderStatus.PARTIALLY_FILLED)
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
