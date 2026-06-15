"""Abstract broker interface.

Concrete adapters (Alpaca, Interactive Brokers, CCXT) implement this ABC so the
execution layer can route the same order objects to paper, live equities, or
crypto venues unchanged.  Heavy/optional broker SDKs are imported lazily inside
each adapter - importing this module never requires them.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from quantcortex.execution.order_manager import Order, OrderSide, OrderType

__all__ = ["Broker", "Position", "AccountInfo", "BrokerError"]


class BrokerError(Exception):
    """Raised on broker connectivity / submission failures."""


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float = 0.0
    market_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.market_price - self.avg_price) * self.quantity


@dataclass
class AccountInfo:
    cash: float = 0.0
    equity: float = 0.0
    buying_power: float = 0.0
    currency: str = "USD"
    extra: Dict[str, float] = field(default_factory=dict)


class Broker(abc.ABC):
    """Abstract broker adapter."""

    name: str = "base"
    paper: bool = True

    # ----- connection lifecycle ----- #
    def connect(self) -> None:  # pragma: no cover - adapters override
        """Establish a session with the venue (no-op by default)."""

    def disconnect(self) -> None:  # pragma: no cover - adapters override
        """Tear down the session (no-op by default)."""

    def __enter__(self) -> "Broker":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    # ----- abstract trading API ----- #
    @abc.abstractmethod
    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
    ) -> Order:
        """Submit an order and return the resulting :class:`Order`."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_positions(self) -> List[Position]:
        """Return current open positions."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_account(self) -> AccountInfo:
        """Return account cash / equity / buying power."""
        raise NotImplementedError

    # ----- optional API with sensible defaults ----- #
    def cancel_order(self, broker_order_id: str) -> None:  # pragma: no cover
        raise NotImplementedError(f"{self.name} does not support cancel_order")

    def positions_as_dict(self) -> Dict[str, float]:
        """Symbol -> signed quantity convenience view."""
        return {p.symbol: p.quantity for p in self.get_positions()}
