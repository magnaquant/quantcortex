"""Interactive Brokers adapter via ``ib_insync``.

Routes quantcortex orders to Interactive Brokers through a running TWS or IB
Gateway instance.  The ``ib_insync`` package is an *optional* dependency,
imported lazily inside :meth:`IBBroker.connect` - importing this module never
requires it.

Connection parameters are read from the constructor, falling back to the
``IB_HOST`` / ``IB_PORT`` / ``IB_CLIENT_ID`` environment variables.  Typical
defaults are host ``127.0.0.1``, port ``7497`` (TWS paper) / ``7496`` (TWS live)
/ ``4002`` (Gateway paper) / ``4001`` (Gateway live), and client id ``1``.

The adapter's API usage has been verified against ``ib_insync`` 0.9.86
(IB.connect / positions / accountSummary / qualifyContracts / placeOrder and the
Stock / MarketOrder / LimitOrder constructors).  Caveat: ``ib_insync`` (via
``eventkit``) grabs an implicit asyncio event loop at import, which Python 3.14
removed; use it under Python 3.11-3.13, or create a loop first with
``asyncio.set_event_loop(asyncio.new_event_loop())``.
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
from quantcortex.execution.order_manager import Order, OrderSide, OrderStatus, OrderType

__all__ = ["IBBroker"]

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 7497  # TWS paper-trading port
_DEFAULT_CLIENT_ID = 1


class IBBroker(Broker):
    """Broker adapter for Interactive Brokers (TWS / IB Gateway)."""

    name = "interactive_brokers"

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
    ) -> None:
        self.host = host or os.environ.get("IB_HOST") or _DEFAULT_HOST
        self.port = int(port or os.environ.get("IB_PORT") or _DEFAULT_PORT)
        self.client_id = int(
            client_id
            if client_id is not None
            else os.environ.get("IB_CLIENT_ID", _DEFAULT_CLIENT_ID)
        )
        # IB does not expose a clean "paper vs live" flag over the API; it is a
        # property of which port/gateway you connect to.  Paper ports are the
        # 7497 / 4002 pair.
        self.paper = self.port in (7497, 4002)
        self._ib = None  # populated by connect()
        self._sdk = None  # cached ib_insync module symbols

    # ------------------------------------------------------------------ #
    # connection lifecycle
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        """Lazily import ``ib_insync`` and connect to TWS / IB Gateway."""
        try:
            from ib_insync import IB, LimitOrder, MarketOrder, Stock
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "IBBroker requires the 'ib_insync' package. "
                "Install it with: pip install ib_insync"
            ) from exc

        self._sdk = {
            "IB": IB,
            "Stock": Stock,
            "MarketOrder": MarketOrder,
            "LimitOrder": LimitOrder,
        }
        try:
            ib = IB()
            ib.connect(self.host, self.port, clientId=self.client_id)
            self._ib = ib
        except Exception as exc:  # pragma: no cover - network dependent
            raise BrokerError(
                f"Failed to connect to Interactive Brokers at "
                f"{self.host}:{self.port} (clientId={self.client_id}): {exc}"
            ) from exc

    def disconnect(self) -> None:
        if self._ib is not None:
            try:
                self._ib.disconnect()
            except Exception:  # pragma: no cover - best effort teardown
                pass
        self._ib = None

    def _require_ib(self):
        if self._ib is None:
            raise BrokerError("IBBroker is not connected; call connect() first.")
        return self._ib

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
        """Build a Stock contract + order, place it, and return an Order."""
        ib = self._require_ib()
        side = OrderSide(side)
        order_type = OrderType(order_type)
        if order_type is OrderType.LIMIT and limit_price is None:
            raise BrokerError("Limit orders require a limit_price.")

        action = "BUY" if side is OrderSide.BUY else "SELL"
        Stock = self._sdk["Stock"]
        try:
            contract = Stock(symbol, "SMART", "USD")
            ib.qualifyContracts(contract)
            if order_type is OrderType.MARKET:
                ib_order = self._sdk["MarketOrder"](action, quantity)
            else:
                ib_order = self._sdk["LimitOrder"](action, quantity, limit_price)
            trade = ib.placeOrder(contract, ib_order)
        except Exception as exc:
            raise BrokerError(f"IB placeOrder failed: {exc}") from exc

        return self._to_order(trade, symbol, side, quantity, order_type, limit_price)

    def _to_order(
        self,
        trade,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType,
        limit_price: Optional[float],
    ) -> Order:
        """Translate an ``ib_insync`` Trade into a quantcortex :class:`Order`."""
        order_id = None
        ib_order = getattr(trade, "order", None)
        if ib_order is not None:
            oid = getattr(ib_order, "orderId", None) or getattr(
                ib_order, "permId", None
            )
            if oid:
                order_id = str(oid)
        order_id = order_id or str(uuid.uuid4())

        status_obj = getattr(trade, "orderStatus", None)
        filled = float(getattr(status_obj, "filled", 0.0) or 0.0)
        avg_price = getattr(status_obj, "avgFillPrice", None)
        avg_price = float(avg_price) if avg_price else None

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=float(quantity),
            order_type=order_type,
            limit_price=limit_price,
            status=OrderStatus.SUBMITTED,
        )
        order.filled_quantity = filled
        order.avg_fill_price = avg_price
        if filled >= float(quantity) - 1e-9 and filled > 0:
            order.status = OrderStatus.FILLED
            order.history.append(OrderStatus.FILLED)
        elif filled > 0:
            order.status = OrderStatus.PARTIALLY_FILLED
            order.history.append(OrderStatus.PARTIALLY_FILLED)
        return order

    def cancel_order(self, broker_order_id: str) -> None:
        ib = self._require_ib()
        try:
            for trade in ib.openTrades():
                oid = getattr(trade.order, "orderId", None)
                if str(oid) == str(broker_order_id):
                    ib.cancelOrder(trade.order)
                    return
            raise BrokerError(f"No open IB order with id {broker_order_id!r}.")
        except BrokerError:
            raise
        except Exception as exc:
            raise BrokerError(f"IB cancelOrder failed: {exc}") from exc

    def get_positions(self) -> List[Position]:
        ib = self._require_ib()
        try:
            raw_positions = ib.positions()
        except Exception as exc:
            raise BrokerError(f"IB positions() failed: {exc}") from exc

        positions: List[Position] = []
        for p in raw_positions:
            contract = getattr(p, "contract", None)
            symbol = str(getattr(contract, "symbol", "")) if contract else ""
            positions.append(
                Position(
                    symbol=symbol,
                    quantity=float(getattr(p, "position", 0.0) or 0.0),
                    avg_price=float(getattr(p, "avgCost", 0.0) or 0.0),
                )
            )
        return positions

    def get_account(self) -> AccountInfo:
        ib = self._require_ib()
        try:
            summary = ib.accountSummary()
        except Exception as exc:
            raise BrokerError(f"IB accountSummary() failed: {exc}") from exc

        values = {}
        currency = "USD"
        for row in summary:
            tag = getattr(row, "tag", None)
            val = getattr(row, "value", None)
            if tag is None:
                continue
            cur = getattr(row, "currency", None)
            if cur:
                currency = cur
            try:
                values[tag] = float(val)
            except (TypeError, ValueError):
                values[tag] = val

        def _num(tag: str) -> float:
            v = values.get(tag, 0.0)
            return float(v) if isinstance(v, (int, float)) else 0.0

        return AccountInfo(
            cash=_num("TotalCashValue"),
            equity=_num("NetLiquidation"),
            buying_power=_num("BuyingPower"),
            currency=currency,
            extra={
                "available_funds": _num("AvailableFunds"),
                "gross_position_value": _num("GrossPositionValue"),
            },
        )
