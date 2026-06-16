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

import math
import os
from typing import List, Optional
from urllib.parse import urlparse

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

__all__ = ["AlpacaBroker", "is_alpaca_live_endpoint", "is_alpaca_paper_endpoint"]

# Alpaca's documented REST endpoints.
_PAPER_URL = "https://paper-api.alpaca.markets"
_LIVE_URL = "https://api.alpaca.markets"


def _is_alpaca_endpoint(base_url: str, hostname: str) -> bool:
    if not isinstance(base_url, str):
        return False
    parsed = urlparse(base_url)
    try:
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and parsed.hostname == hostname
        and port in (None, 443)
        and parsed.username is None
        and parsed.password is None
        and parsed.path in ("", "/")
        and not parsed.query
        and not parsed.fragment
    )


def is_alpaca_paper_endpoint(base_url: str) -> bool:
    """Return whether ``base_url`` is Alpaca's official HTTPS paper host."""
    return _is_alpaca_endpoint(base_url, "paper-api.alpaca.markets")


def is_alpaca_live_endpoint(base_url: str) -> bool:
    """Return whether ``base_url`` is Alpaca's official HTTPS live host."""
    return _is_alpaca_endpoint(base_url, "api.alpaca.markets")

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
        if not isinstance(paper, bool):
            raise TypeError("paper must be a boolean")
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        self.paper = paper
        # Explicit base_url > env override > paper/live default.
        self.base_url = (
            base_url
            or os.environ.get("ALPACA_BASE_URL")
            or (_PAPER_URL if self.paper else _LIVE_URL)
        )
        if self.paper and not is_alpaca_paper_endpoint(self.base_url):
            raise BrokerError(
                f"paper=True requires Alpaca's paper endpoint, got {self.base_url!r}"
            )
        if not self.paper and not is_alpaca_live_endpoint(self.base_url):
            raise BrokerError(
                f"paper=False requires Alpaca's live endpoint, got {self.base_url!r}"
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
        *,
        client_order_id: Optional[str] = None,
    ) -> Order:
        """Submit an order to Alpaca and return a populated :class:`Order`."""
        request = validate_order_request(
            symbol, side, quantity, order_type, limit_price
        )
        symbol = request.symbol
        side = request.side
        quantity = request.quantity
        order_type = request.order_type
        limit_price = request.limit_price
        client_order_id = self._validate_client_order_id(client_order_id)
        api = self._require_api()

        try:
            raw = api.submit_order(
                symbol=symbol,
                qty=quantity,
                side="buy" if side is OrderSide.BUY else "sell",
                type="market" if order_type is OrderType.MARKET else "limit",
                time_in_force="day",
                limit_price=limit_price if order_type is OrderType.LIMIT else None,
                client_order_id=client_order_id,
            )
        except Exception as exc:
            raise BrokerError(f"Alpaca submit_order failed: {exc}") from exc

        if client_order_id is not None:
            returned_client_id = str(
                getattr(raw, "client_order_id", "")
            ).strip()
            if returned_client_id != client_order_id:
                raise BrokerError(
                    "Alpaca submit response did not echo the requested "
                    "client_order_id."
                )

        return self._to_order(raw, symbol, side, quantity, order_type, limit_price)

    @staticmethod
    def _validate_client_order_id(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise BrokerError("Alpaca client_order_id must be a string.")
        value = value.strip()
        if not value:
            raise BrokerError("Alpaca client_order_id must be non-empty.")
        return value

    def _order_from_raw(self, raw) -> Order:
        """Translate a complete Alpaca order response without caller context."""
        symbol = str(getattr(raw, "symbol", "")).strip()
        if not symbol:
            raise BrokerError("Alpaca order response is missing a symbol.")

        raw_side = str(getattr(raw, "side", "")).strip().lower()
        if raw_side == "buy":
            side = OrderSide.BUY
        elif raw_side == "sell":
            side = OrderSide.SELL
        else:
            raise BrokerError(f"Unknown Alpaca order side {raw_side!r}.")

        raw_type = str(
            getattr(raw, "type", None) or getattr(raw, "order_type", "")
        ).strip().lower()
        if raw_type == "market":
            order_type = OrderType.MARKET
        elif raw_type == "limit":
            order_type = OrderType.LIMIT
        else:
            raise BrokerError(f"Unsupported Alpaca order type {raw_type!r}.")

        quantity = self._to_float(getattr(raw, "qty", None))
        if quantity is None:
            raise BrokerError("Alpaca order response is missing quantity.")
        limit_price = self._to_float(getattr(raw, "limit_price", None))
        return self._to_order(
            raw,
            symbol,
            side,
            quantity,
            order_type,
            limit_price,
        )

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
        order_id = str(getattr(raw, "id", "")).strip()
        if not order_id:
            raise BrokerError("Alpaca order response is missing its broker order id.")
        raw_status = str(getattr(raw, "status", "")).strip().lower()
        try:
            status = _ALPACA_STATUS_MAP[raw_status]
        except KeyError as exc:
            raise BrokerError(f"Unknown Alpaca order status {raw_status!r}.") from exc
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
        order.validate()
        return order

    @staticmethod
    def _to_float(value) -> Optional[float]:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            raise BrokerError(f"Boolean numeric value from Alpaca: {value!r}.")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise BrokerError(f"Invalid numeric value from Alpaca: {value!r}.") from exc
        if not math.isfinite(parsed):
            raise BrokerError(f"Non-finite numeric value from Alpaca: {value!r}.")
        return parsed

    @staticmethod
    def _is_not_found_error(exc: Exception) -> bool:
        values = [getattr(exc, "status_code", None), getattr(exc, "code", None)]
        response = getattr(exc, "response", None)
        values.append(getattr(response, "status_code", None))
        for value in values:
            try:
                if int(value) == 404:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    def find_order_by_client_order_id(self, client_order_id: str) -> Optional[Order]:
        """Return an order by client id, or ``None`` only for a definite 404."""
        client_order_id = self._validate_client_order_id(client_order_id)
        api = self._require_api()
        try:
            raw = api.get_order_by_client_order_id(client_order_id)
        except Exception as exc:
            if self._is_not_found_error(exc):
                return None
            raise BrokerError(
                f"Alpaca order lookup failed for {client_order_id!r}: {exc}"
            ) from exc
        returned_client_id = str(getattr(raw, "client_order_id", "")).strip()
        if returned_client_id != client_order_id:
            raise BrokerError(
                "Alpaca client-order lookup returned a different client_order_id."
            )
        return self._order_from_raw(raw)

    def get_open_orders(self) -> List[Order]:
        """Return all open orders, failing if the venue result may be truncated."""
        api = self._require_api()
        try:
            raw_orders = list(
                api.list_orders(status="open", limit=500, direction="asc")
            )
        except Exception as exc:
            raise BrokerError(f"Alpaca list_orders failed: {exc}") from exc
        if len(raw_orders) >= 500:
            raise BrokerError(
                "Alpaca returned 500 open orders; reconciliation may be truncated."
            )
        return [self._order_from_raw(raw) for raw in raw_orders]

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
