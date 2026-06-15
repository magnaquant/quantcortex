"""CCXT crypto-exchange adapter.

Routes quantcortex orders to any of the 100+ crypto exchanges supported by the
`ccxt <https://github.com/ccxt/ccxt>`_ library (Binance, Coinbase, Kraken, ...).
``ccxt`` is an *optional* dependency, imported lazily inside
:meth:`CCXTBroker.connect` - importing this module never requires it.

When ``paper=True`` (the default) the adapter calls ``set_sandbox_mode(True)`` if
the chosen exchange exposes a testnet/sandbox, so paper trading routes to the
exchange's test environment rather than live funds.
"""

from __future__ import annotations

import os
import uuid
from typing import List, Optional

from execution.brokers.base import AccountInfo, Broker, BrokerError, Position
from execution.order_manager import Order, OrderSide, OrderStatus, OrderType

__all__ = ["CCXTBroker"]

# CCXT order-status strings -> canonical OrderStatus.
_CCXT_STATUS_MAP = {
    "open": OrderStatus.SUBMITTED,
    "closed": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
}


class CCXTBroker(Broker):
    """Broker adapter for crypto exchanges via the unified CCXT API."""

    name = "ccxt"

    def __init__(
        self,
        exchange: str = "binance",
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        paper: bool = True,
    ) -> None:
        self.exchange_id = exchange
        self.api_key = api_key or os.environ.get("CCXT_API_KEY")
        self.secret = secret or os.environ.get("CCXT_SECRET")
        self.paper = bool(paper)
        self._exchange = None  # populated by connect()

    # ------------------------------------------------------------------ #
    # connection lifecycle
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        """Lazily import ccxt and instantiate the configured exchange."""
        try:
            import ccxt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "CCXTBroker requires the 'ccxt' package. "
                "Install it with: pip install ccxt"
            ) from exc

        try:
            exchange_cls = getattr(ccxt, self.exchange_id)
        except AttributeError as exc:
            raise BrokerError(
                f"Unknown ccxt exchange {self.exchange_id!r}."
            ) from exc

        config = {"enableRateLimit": True}
        if self.api_key:
            config["apiKey"] = self.api_key
        if self.secret:
            config["secret"] = self.secret

        try:
            exchange = exchange_cls(config)
            if self.paper and exchange.has.get("sandbox"):
                exchange.set_sandbox_mode(True)
            self._exchange = exchange
        except Exception as exc:  # pragma: no cover - network/config dependent
            raise BrokerError(
                f"Failed to initialise ccxt exchange {self.exchange_id!r}: {exc}"
            ) from exc

    def disconnect(self) -> None:
        self._exchange = None

    def _require_exchange(self):
        if self._exchange is None:
            raise BrokerError("CCXTBroker is not connected; call connect() first.")
        return self._exchange

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
        """Place an order via ``create_order`` and return an :class:`Order`."""
        exchange = self._require_exchange()
        side = OrderSide(side)
        order_type = OrderType(order_type)
        if order_type is OrderType.LIMIT and limit_price is None:
            raise BrokerError("Limit orders require a limit_price.")

        ccxt_type = "market" if order_type is OrderType.MARKET else "limit"
        ccxt_side = "buy" if side is OrderSide.BUY else "sell"
        try:
            raw = exchange.create_order(
                symbol,
                ccxt_type,
                ccxt_side,
                quantity,
                limit_price if order_type is OrderType.LIMIT else None,
            )
        except Exception as exc:
            raise BrokerError(f"ccxt create_order failed: {exc}") from exc

        return self._to_order(raw, symbol, side, quantity, order_type, limit_price)

    def _to_order(
        self,
        raw: dict,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType,
        limit_price: Optional[float],
    ) -> Order:
        """Translate a ccxt order dict into a quantcortex :class:`Order`."""
        raw = raw or {}
        order_id = str(raw.get("id") or uuid.uuid4())
        status = _CCXT_STATUS_MAP.get(
            str(raw.get("status", "")).lower(), OrderStatus.SUBMITTED
        )
        filled = self._to_float(raw.get("filled")) or 0.0
        avg_price = self._to_float(raw.get("average")) or self._to_float(
            raw.get("price")
        )

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

    def cancel_order(self, broker_order_id: str, symbol: Optional[str] = None) -> None:
        exchange = self._require_exchange()
        try:
            exchange.cancel_order(broker_order_id, symbol)
        except Exception as exc:
            raise BrokerError(f"ccxt cancel_order failed: {exc}") from exc

    def get_positions(self) -> List[Position]:
        """Derive positions from non-zero balances in :meth:`fetch_balance`."""
        exchange = self._require_exchange()
        try:
            balance = exchange.fetch_balance()
        except Exception as exc:
            raise BrokerError(f"ccxt fetch_balance failed: {exc}") from exc

        totals = balance.get("total", {}) or {}
        positions: List[Position] = []
        for currency, amount in totals.items():
            amt = self._to_float(amount) or 0.0
            if abs(amt) > 0.0:
                positions.append(Position(symbol=str(currency), quantity=amt))
        return positions

    def get_account(self) -> AccountInfo:
        """Summarise account funds from :meth:`fetch_balance`.

        Cash is reported in the exchange's quote stablecoin (USDT) if present,
        otherwise USD; equity/buying-power are best-effort from the same field.
        """
        exchange = self._require_exchange()
        try:
            balance = exchange.fetch_balance()
        except Exception as exc:
            raise BrokerError(f"ccxt fetch_balance failed: {exc}") from exc

        free = balance.get("free", {}) or {}
        total = balance.get("total", {}) or {}

        # Pick a sensible "cash" currency: prefer common quote stablecoins.
        currency = "USD"
        cash = 0.0
        for candidate in ("USDT", "USD", "USDC", "BUSD"):
            if candidate in free:
                currency = candidate
                cash = self._to_float(free.get(candidate)) or 0.0
                break

        equity = self._to_float(total.get(currency)) or cash

        return AccountInfo(
            cash=cash,
            equity=equity,
            buying_power=cash,
            currency=currency,
            extra={"exchange": self.exchange_id},
        )
