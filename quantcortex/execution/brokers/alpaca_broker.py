"""Alpaca broker adapter.

Routes quantcortex orders to the `Alpaca <https://alpaca.markets>`_ commission-free
US-equities API for both paper and live trading.  The Alpaca SDK
(``alpaca-trade-api``) is an *optional* dependency and is imported lazily inside
:meth:`AlpacaBroker.connect` - importing this module never requires it.

Credentials are read from the constructor arguments, falling back to the
``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY`` / ``ALPACA_BASE_URL`` environment
variables.  When ``paper=True`` (the default) and no ``base_url`` is supplied the
adapter targets the paper-trading endpoint; ``paper=False`` targets live.
"""

from __future__ import annotations

import os
import uuid
from typing import List, Optional

from quantcortex.execution.brokers.base import (
    AccountInfo,
    Broker,
    BrokerError,
    Position,
)
from quantcortex.execution.order_manager import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)

__all__ = ["AlpacaBroker"]

# Alpaca's documented REST endpoints.
_PAPER_URL = "https://paper-api.alpaca.markets"
_LIVE_URL = "https://api.alpaca.markets"

# Map Alpaca's order status strings onto the canonical OrderStatus enum.
_ALPACA_STATUS_MAP = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.SUBMITTED,
    "accepted_for_bidding": OrderStatus.SUBMITTED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.CANCELLED,
    "done_for_day": OrderStatus.CANCELLED,
    "replaced": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.REJECTED,
    "stopped": OrderStatus.REJECTED,
}


class AlpacaBroker(Broker):
    """Broker adapter for the Alpaca equities API (paper + live)."""

    name = "alpaca"

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        paper: bool = True,
    ) -> None:
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        self.paper = bool(paper)
        # Explicit base_url > env override > paper/live default.
        self.base_url = (
            base_url
            or os.environ.get("ALPACA_BASE_URL")
            or (_PAPER_URL if self.paper else _LIVE_URL)
        )
        self._api = None  # populated by connect()

    # ------------------------------------------------------------------ #
    # connection lifecycle
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        """Lazily import the Alpaca SDK and build the REST client."""
        try:
            import alpaca_trade_api as tradeapi
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "AlpacaBroker requires the 'alpaca-trade-api' package. "
                "Install it with: pip install alpaca-trade-api"
            ) from exc

        if not self.api_key or not self.secret_key:
            raise BrokerError(
                "Alpaca credentials missing: set ALPACA_API_KEY / "
                "ALPACA_SECRET_KEY or pass api_key / secret_key."
            )
        try:
            self._api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url,
            )
            # Cheap sanity round-trip to validate credentials/connectivity.
            self._api.get_account()
        except Exception as exc:  # pragma: no cover - network dependent
            raise BrokerError(f"Failed to connect to Alpaca: {exc}") from exc

    def disconnect(self) -> None:
        """Drop the REST client (Alpaca is stateless HTTP, so just release it)."""
        self._api = None

    def _require_api(self):
        if self._api is None:
            raise BrokerError("AlpacaBroker is not connected; call connect() first.")
        return self._api

    # ------------------------------------------------------------------ #
    # trading API
    # ------------------------------------------------------------------ #
    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
    ) -> Order:
        """Submit an order to Alpaca and return a populated :class:`Order`."""
        api = self._require_api()
        side = OrderSide(side)
        order_type = OrderType(order_type)
        if order_type is OrderType.LIMIT and limit_price is None:
            raise BrokerError("Limit orders require a limit_price.")

        try:
            raw = api.submit_order(
                symbol=symbol,
                qty=quantity,
                side="buy" if side is OrderSide.BUY else "sell",
                type="market" if order_type is OrderType.MARKET else "limit",
                time_in_force="day",
                limit_price=limit_price if order_type is OrderType.LIMIT else None,
            )
        except Exception as exc:
            raise BrokerError(f"Alpaca submit_order failed: {exc}") from exc

        return self._to_order(raw, symbol, side, quantity, order_type, limit_price)

    def _to_order(
        self,
        raw,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType,
        limit_price: Optional[float],
    ) -> Order:
        """Translate an Alpaca order object into a quantcortex :class:`Order`."""
        order_id = str(getattr(raw, "id", None) or getattr(raw, "client_order_id", None)
                       or uuid.uuid4())
        status = _ALPACA_STATUS_MAP.get(
            str(getattr(raw, "status", "")).lower(), OrderStatus.SUBMITTED
        )
        filled_qty = self._to_float(getattr(raw, "filled_qty", 0.0)) or 0.0
        avg_price = self._to_float(getattr(raw, "filled_avg_price", None))

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=float(quantity),
            order_type=order_type,
            limit_price=limit_price,
            status=OrderStatus.SUBMITTED,
        )
        # Reflect any fills already reported on the submission response.
        order.filled_quantity = filled_qty
        order.avg_fill_price = avg_price
        order.status = status
        if status not in order.history:
            order.history.append(status)
        return order

    @staticmethod
    def _to_float(value) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def cancel_order(self, broker_order_id: str) -> None:
        api = self._require_api()
        try:
            api.cancel_order(broker_order_id)
        except Exception as exc:
            raise BrokerError(f"Alpaca cancel_order failed: {exc}") from exc

    def get_positions(self) -> List[Position]:
        api = self._require_api()
        try:
            raw_positions = api.list_positions()
        except Exception as exc:
            raise BrokerError(f"Alpaca list_positions failed: {exc}") from exc

        positions: List[Position] = []
        for p in raw_positions:
            positions.append(
                Position(
                    symbol=str(getattr(p, "symbol", "")),
                    quantity=self._to_float(getattr(p, "qty", 0.0)) or 0.0,
                    avg_price=self._to_float(getattr(p, "avg_entry_price", 0.0)) or 0.0,
                    market_price=self._to_float(getattr(p, "current_price", 0.0)) or 0.0,
                )
            )
        return positions

    def get_account(self) -> AccountInfo:
        api = self._require_api()
        try:
            acct = api.get_account()
        except Exception as exc:
            raise BrokerError(f"Alpaca get_account failed: {exc}") from exc

        return AccountInfo(
            cash=self._to_float(getattr(acct, "cash", 0.0)) or 0.0,
            equity=self._to_float(getattr(acct, "equity", 0.0)) or 0.0,
            buying_power=self._to_float(getattr(acct, "buying_power", 0.0)) or 0.0,
            currency=str(getattr(acct, "currency", "USD")),
            extra={
                "portfolio_value": self._to_float(
                    getattr(acct, "portfolio_value", 0.0)
                )
                or 0.0,
            },
        )
