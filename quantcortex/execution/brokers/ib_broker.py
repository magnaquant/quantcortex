"""Interactive Brokers adapter via ``ib_insync``.

Routes quantcortex orders to Interactive Brokers through a running TWS or IB
Gateway instance.  The ``ib_insync`` package is an *optional* dependency,
imported lazily inside :meth:`IBBroker.connect` - importing this module never
requires it.

Connection parameters are read from the constructor, falling back to the
``IB_HOST`` / ``IB_PORT`` / ``IB_CLIENT_ID`` environment variables. Typical
ports are ``7497`` (TWS paper), ``7496`` (TWS live), ``4002`` (Gateway paper),
and ``4001`` (Gateway live). The adapter defaults to ``paper=True`` on
``127.0.0.1:7497`` with client id ``1`` and rejects a known live port unless
the caller explicitly sets ``paper=False``.

The adapter's API usage has been verified against ``ib_insync`` 0.9.86
(IB.connect / positions / accountSummary / qualifyContracts / placeOrder and the
Stock / MarketOrder / LimitOrder constructors). The optional SDK is not part of
the core CI matrix; verify its runtime compatibility in the deployment
environment before enabling broker connectivity.
"""

from __future__ import annotations

import math
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
    validate_order_request,
)

__all__ = ["IBBroker"]

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 7497  # TWS paper-trading port
_DEFAULT_CLIENT_ID = 1

_IB_STATUS_MAP = {
    "apipending": OrderStatus.SUBMITTED,
    "pendingsubmit": OrderStatus.SUBMITTED,
    "pendingcancel": OrderStatus.SUBMITTED,
    "presubmitted": OrderStatus.SUBMITTED,
    "submitted": OrderStatus.SUBMITTED,
    "filled": OrderStatus.FILLED,
    "apicancelled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "inactive": OrderStatus.REJECTED,
}


class IBBroker(Broker):
    """Broker adapter for Interactive Brokers (TWS / IB Gateway)."""

    name = "interactive_brokers"

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
        paper: bool = True,
        account: Optional[str] = None,
    ) -> None:
        if not isinstance(paper, bool):
            raise TypeError("paper must be a boolean")
        resolved_host = host or os.environ.get("IB_HOST") or _DEFAULT_HOST
        if not isinstance(resolved_host, str) or not resolved_host.strip():
            raise ValueError("host must be a non-empty string")
        resolved_port = port if port is not None else os.environ.get("IB_PORT", _DEFAULT_PORT)
        resolved_client_id = (
            client_id
            if client_id is not None
            else os.environ.get("IB_CLIENT_ID", _DEFAULT_CLIENT_ID)
        )
        try:
            if isinstance(resolved_port, bool) or int(resolved_port) != float(resolved_port):
                raise ValueError
            if isinstance(resolved_client_id, bool) or int(resolved_client_id) != float(resolved_client_id):
                raise ValueError
            self.port = int(resolved_port)
            self.client_id = int(resolved_client_id)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("port and client_id must be integers") from exc
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be in [1, 65535]")
        if self.client_id < 0:
            raise ValueError("client_id must be non-negative")
        if paper and self.port in (7496, 4001):
            raise BrokerError(
                f"paper=True refuses known Interactive Brokers live port {self.port}"
            )
        if not paper and self.port in (7497, 4002):
            raise BrokerError(
                f"paper=False refuses known Interactive Brokers paper port {self.port}"
            )
        self.host = resolved_host.strip()
        self.paper = paper
        resolved_account = account or os.environ.get("IB_ACCOUNT")
        if resolved_account is not None and (
            not isinstance(resolved_account, str) or not resolved_account.strip()
        ):
            raise ValueError("account must be a non-empty string")
        self.account = resolved_account.strip() if resolved_account is not None else None
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
        request = validate_order_request(
            symbol, side, quantity, order_type, limit_price
        )
        symbol = request.symbol
        side = request.side
        quantity = request.quantity
        order_type = request.order_type
        limit_price = request.limit_price
        ib = self._require_ib()

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
        raw_status = str(getattr(status_obj, "status", "")).strip().lower()
        try:
            status = _IB_STATUS_MAP[raw_status]
        except KeyError as exc:
            raise BrokerError(
                f"Unknown Interactive Brokers order status {raw_status!r}."
            ) from exc
        try:
            filled = float(getattr(status_obj, "filled", 0.0) or 0.0)
        except (TypeError, ValueError) as exc:
            raise BrokerError("Invalid filled quantity from Interactive Brokers.") from exc
        avg_price = getattr(status_obj, "avgFillPrice", None)
        try:
            avg_price = float(avg_price) if avg_price else None
        except (TypeError, ValueError) as exc:
            raise BrokerError("Invalid average fill price from Interactive Brokers.") from exc

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
        if status is OrderStatus.SUBMITTED and filled > 0.0:
            status = (
                OrderStatus.FILLED
                if filled >= float(quantity) - 1e-9
                else OrderStatus.PARTIALLY_FILLED
            )
        order.status = status
        if status not in order.history:
            order.history.append(status)
        order.validate()
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

        raw_positions = self._select_account_rows(raw_positions, "positions")
        positions: List[Position] = []
        seen_symbols: set[str] = set()
        for p in raw_positions:
            contract = getattr(p, "contract", None)
            symbol = str(getattr(contract, "symbol", "")) if contract else ""
            if symbol in seen_symbols:
                raise BrokerError(
                    f"IB positions contain duplicate symbol {symbol!r}; the "
                    "symbol-only Position model cannot represent both contracts"
                )
            seen_symbols.add(symbol)
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

        rows = self._select_account_rows(summary, "account summary")

        def _num(tag: str, *, required: bool = False) -> tuple[float, object | None]:
            candidates = [row for row in rows if getattr(row, "tag", None) == tag]
            base = [
                row
                for row in candidates
                if str(getattr(row, "currency", "")).upper() == "BASE"
            ]
            selected = base or candidates
            if not selected:
                if required:
                    raise BrokerError(f"IB account summary is missing required tag {tag!r}")
                return 0.0, None
            parsed: list[tuple[float, object]] = []
            for row in selected:
                try:
                    value = float(getattr(row, "value", None))
                except (TypeError, ValueError) as exc:
                    raise BrokerError(
                        f"IB account summary tag {tag!r} is not numeric"
                    ) from exc
                if not math.isfinite(value):
                    raise BrokerError(
                        f"IB account summary tag {tag!r} is non-finite"
                    )
                parsed.append((value, row))
            unique_values = {value for value, _ in parsed}
            if len(parsed) > 1 and len(unique_values) > 1:
                raise BrokerError(
                    f"IB account summary tag {tag!r} is ambiguous across currencies"
                )
            return parsed[0]

        equity, equity_row = _num("NetLiquidation", required=True)
        cash, _ = _num("TotalCashValue", required=True)
        buying_power, _ = _num("BuyingPower", required=True)
        currency = str(getattr(equity_row, "currency", None) or "USD")
        available_funds, _ = _num("AvailableFunds")
        gross_position_value, _ = _num("GrossPositionValue")

        return AccountInfo(
            cash=cash,
            equity=equity,
            buying_power=buying_power,
            currency=currency,
            extra={
                "account": self.account or "",
                "available_funds": available_funds,
                "gross_position_value": gross_position_value,
            },
        )

    def _select_account_rows(self, rows, label: str):
        rows = list(rows)
        account_ids = {
            str(getattr(row, "account", "")).strip()
            for row in rows
            if str(getattr(row, "account", "")).strip()
        }
        if self.account is not None:
            selected = [
                row
                for row in rows
                if str(getattr(row, "account", "")).strip() == self.account
            ]
            if not selected:
                raise BrokerError(
                    f"IB {label} contains no rows for account {self.account!r}"
                )
            return selected
        if len(account_ids) > 1:
            raise BrokerError(
                f"IB {label} spans multiple accounts {sorted(account_ids)}; "
                "configure account=... or IB_ACCOUNT"
            )
        return rows
