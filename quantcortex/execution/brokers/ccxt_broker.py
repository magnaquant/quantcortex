"""CCXT crypto-exchange adapter.

Routes quantcortex orders to crypto exchanges supported by the
`ccxt <https://github.com/ccxt/ccxt>`_ library (Binance, Coinbase, Kraken, ...).
``ccxt`` is an *optional* dependency, imported lazily inside
:meth:`CCXTBroker.connect` - importing this module never requires it.

When ``paper=True`` (the default) the adapter requires an exchange sandbox and
calls ``set_sandbox_mode(True)``. It refuses to connect if the exchange cannot
prove that paper orders are isolated from live funds.
"""

from __future__ import annotations

import math
import os
import uuid
from typing import List, Mapping, Optional

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
        account_currency: Optional[str] = None,
    ) -> None:
        if not isinstance(exchange, str) or not exchange.strip():
            raise ValueError("exchange must be a non-empty string")
        if not isinstance(paper, bool):
            raise TypeError("paper must be a boolean")
        self.exchange_id = exchange.strip()
        self.api_key = api_key or os.environ.get("CCXT_API_KEY")
        self.secret = secret or os.environ.get("CCXT_SECRET")
        self.paper = paper
        resolved_currency = account_currency or os.environ.get(
            "CCXT_ACCOUNT_CURRENCY"
        )
        if resolved_currency is not None and (
            not isinstance(resolved_currency, str) or not resolved_currency.strip()
        ):
            raise ValueError("account_currency must be a non-empty string")
        self.account_currency = (
            resolved_currency.strip().upper()
            if resolved_currency is not None
            else None
        )
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
            if self.paper:
                if not exchange.has.get("sandbox"):
                    raise BrokerError(
                        f"ccxt exchange {self.exchange_id!r} has no sandbox; "
                        "refusing paper mode on a live endpoint"
                    )
                exchange.set_sandbox_mode(True)
            self._exchange = exchange
        except BrokerError:
            raise
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
        request = validate_order_request(
            symbol, side, quantity, order_type, limit_price
        )
        symbol = request.symbol
        side = request.side
        quantity = request.quantity
        order_type = request.order_type
        limit_price = request.limit_price
        exchange = self._require_exchange()

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
        raw_status = str(raw.get("status", "")).strip().lower()
        try:
            status = _CCXT_STATUS_MAP[raw_status]
        except KeyError as exc:
            raise BrokerError(f"Unknown ccxt order status {raw_status!r}.") from exc
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
        order.validate()
        return order

    @staticmethod
    def _to_float(value) -> Optional[float]:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            raise BrokerError("Boolean values are not valid ccxt numeric fields.")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise BrokerError(f"Invalid numeric value from ccxt: {value!r}.") from exc
        if not math.isfinite(parsed):
            raise BrokerError(f"Non-finite numeric value from ccxt: {value!r}.")
        return parsed

    def cancel_order(self, broker_order_id: str, symbol: Optional[str] = None) -> None:
        exchange = self._require_exchange()
        try:
            exchange.cancel_order(broker_order_id, symbol)
        except Exception as exc:
            raise BrokerError(f"ccxt cancel_order failed: {exc}") from exc

    def get_positions(self) -> List[Position]:
        """Return non-zero asset balances.

        Symbols are asset codes (for example ``BTC``), not tradable pair names.
        Callers that target pair symbols must map balances explicitly.
        """
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
        """Return a complete single-currency cash account summary.

        CCXT balances do not provide a standardized total-equity valuation across
        assets. To avoid understating risk, this method fails when any non-zero
        balance exists outside the selected account currency. Multi-asset crypto
        accounts require an explicit, venue-aware valuation layer.
        """
        exchange = self._require_exchange()
        try:
            balance = exchange.fetch_balance()
        except Exception as exc:
            raise BrokerError(f"ccxt fetch_balance failed: {exc}") from exc

        free = balance.get("free", {}) or {}
        total = balance.get("total", {}) or {}
        if not isinstance(free, Mapping) or not isinstance(total, Mapping):
            raise BrokerError("ccxt fetch_balance returned malformed balance maps")

        currency = self.account_currency
        if currency is None:
            for candidate in ("USDT", "USD", "USDC", "BUSD"):
                if candidate in free or candidate in total:
                    currency = candidate
                    break
        if currency is None:
            raise BrokerError(
                "cannot infer a CCXT account currency; pass account_currency explicitly"
            )

        nonzero_assets = []
        for asset, raw_amount in total.items():
            amount = self._to_float(raw_amount)
            if amount is None:
                raise BrokerError(f"ccxt total balance for {asset!r} is missing")
            if str(asset).upper() != currency and abs(amount) > 0.0:
                nonzero_assets.append(str(asset))
        if nonzero_assets:
            raise BrokerError(
                "cannot compute complete CCXT account equity with non-zero "
                f"unvalued assets: {sorted(nonzero_assets)}"
            )

        cash_raw = free.get(currency, 0.0)
        equity_raw = total.get(currency, cash_raw)
        cash = self._to_float(cash_raw)
        equity = self._to_float(equity_raw)
        if cash is None or equity is None:
            raise BrokerError(f"ccxt returned an invalid {currency} balance")

        return AccountInfo(
            cash=cash,
            equity=equity,
            buying_power=cash,
            currency=currency,
            extra={
                "exchange": self.exchange_id,
                "valuation_scope": "single_currency_cash_only",
            },
        )
