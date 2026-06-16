"""Real-time position tracking and weight-to-order translation.

The :class:`PositionManager` maintains an in-memory book of :class:`Position`
objects, updated either from filled :class:`Order` objects (the live path) or
from raw ``(symbol, qty_delta, price)`` deltas (the backtest / manual path).  It
also converts a *target weight* allocation - the output of the portfolio layer  -
into a concrete list of order intents given current holdings and capital.

Positions use a running volume-weighted average price (VWAP) so that
``unrealized_pnl`` and ``avg_price`` remain correct across incremental fills and
across position flips (long -> short).
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Union

import pandas as pd

from quantcortex.execution.brokers.base import Position
from quantcortex.execution.order_manager import Order, OrderSide

__all__ = ["PositionManager"]

# Type alias for the two accepted price/weight container shapes.
PriceLike = Union[Mapping[str, float], "pd.Series"]


class PositionManager:
    """Tracks live positions and turns target weights into order intents."""

    def __init__(self) -> None:
        self._positions: Dict[str, Position] = {}
        # Broker updates generally report cumulative filled quantity and VWAP.
        # Track what has already been applied so repeated updates are idempotent.
        self._applied_fills: Dict[str, tuple[float, float]] = {}

    # ------------------------------------------------------------------ #
    # state access
    # ------------------------------------------------------------------ #
    @property
    def positions(self) -> Dict[str, Position]:
        """Mapping of symbol -> :class:`Position` (only non-flat books)."""
        return dict(self._positions)

    def get_position(self, symbol: str) -> Position:
        """Return the current position for ``symbol`` (flat if untracked)."""
        symbol = self._symbol_key(symbol)
        return self._positions.get(symbol, Position(symbol=symbol, quantity=0.0))

    # ------------------------------------------------------------------ #
    # updates
    # ------------------------------------------------------------------ #
    def update(self, symbol: str, qty_delta: float, price: float) -> Position:
        """Apply a signed quantity delta at ``price`` and return the position.

        ``qty_delta`` is positive for buys, negative for sells.  The average
        price is updated as a VWAP when increasing exposure; when reducing or
        closing, the average price is preserved; when flipping sign, the new
        side's average price is reset to ``price``.
        """
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be non-empty")
        symbol = symbol.strip()
        qty_delta_value = self._to_float(qty_delta)
        price_value = self._to_float(price)
        if qty_delta_value is None:
            raise ValueError("qty_delta must be finite and numeric")
        if price_value is None:
            raise ValueError("price must be finite and numeric")
        qty_delta = qty_delta_value
        price = price_value
        if not math.isfinite(qty_delta):
            raise ValueError("qty_delta must be finite")
        if not math.isfinite(price) or price <= 0.0:
            raise ValueError("price must be finite and positive")
        pos = self._positions.get(symbol, Position(symbol=symbol, quantity=0.0))

        old_qty = pos.quantity
        new_qty = old_qty + qty_delta

        if old_qty == 0.0 or (old_qty > 0) == (qty_delta > 0):
            # Opening or increasing exposure on the same side -> blend VWAP.
            if new_qty != 0.0:
                pos.avg_price = (
                    abs(old_qty) * pos.avg_price + abs(qty_delta) * price
                ) / abs(new_qty)
        elif (old_qty > 0) != (new_qty > 0) and new_qty != 0.0:
            # Reduced through zero and flipped sign -> reset basis to fill price.
            pos.avg_price = price
        # Pure reduction (same side, smaller magnitude) keeps the old avg_price.

        pos.quantity = new_qty
        pos.market_price = price

        if abs(new_qty) < 1e-12:
            # Flat: drop from the book entirely.
            self._positions.pop(symbol, None)
            pos.quantity = 0.0
            return pos

        self._positions[symbol] = pos
        return pos

    def update_fill(self, order: Order) -> Position:
        """Apply a filled (or partially filled) :class:`Order` to the book.

        Broker updates are cumulative. Only the newly reported quantity and
        notional are applied, making repeated snapshots idempotent.
        """
        order.validate()
        filled = float(order.filled_quantity)
        prior_filled, prior_notional = self._applied_fills.get(
            order.order_id, (0.0, 0.0)
        )
        if filled < prior_filled - 1e-9:
            raise ValueError(
                f"cumulative fill for {order.order_id!r} decreased from "
                f"{prior_filled} to {filled}"
            )
        increment = filled - prior_filled
        if increment <= 1e-12:
            return self.get_position(order.symbol)
        price = order.avg_fill_price or order.limit_price
        if price is None:
            raise ValueError("filled orders require avg_fill_price or limit_price")
        cumulative_notional = filled * float(price)
        incremental_notional = cumulative_notional - prior_notional
        if incremental_notional <= 0.0 or not math.isfinite(incremental_notional):
            raise ValueError("cumulative fill notional must increase and remain finite")
        incremental_price = incremental_notional / increment
        signed = increment if order.side is OrderSide.BUY else -increment
        position = self.update(order.symbol, signed, incremental_price)
        self._applied_fills[order.order_id] = (filled, cumulative_notional)
        return position

    def mark(self, prices: PriceLike) -> None:
        """Update ``market_price`` on tracked positions from ``prices``."""
        prices = self._as_mapping(prices)
        for symbol, pos in self._positions.items():
            if symbol in prices:
                price = self._to_float(prices[symbol])
                if price is None or price <= 0.0:
                    raise ValueError(f"invalid mark price for {symbol!r}")
                pos.market_price = price

    # ------------------------------------------------------------------ #
    # valuation
    # ------------------------------------------------------------------ #
    def market_value(self, prices: PriceLike) -> float:
        """Signed total market value of the book at ``prices``."""
        prices = self._as_mapping(prices)
        total = 0.0
        for symbol, pos in self._positions.items():
            px = self._to_float(prices.get(symbol, pos.market_price))
            if px is None or px <= 0.0:
                raise ValueError(f"no valid price for position {symbol!r}")
            total += pos.quantity * px
        return total

    def net_exposure(self, prices: PriceLike) -> float:
        """Net signed exposure (longs minus shorts) at ``prices``.

        Identical to :meth:`market_value`; provided as a named alias because
        callers reason about *exposure* when sizing risk.
        """
        return self.market_value(prices)

    def gross_exposure(self, prices: PriceLike) -> float:
        """Gross exposure (sum of absolute position values) at ``prices``."""
        prices = self._as_mapping(prices)
        total = 0.0
        for symbol, pos in self._positions.items():
            px = self._to_float(prices.get(symbol, pos.market_price))
            if px is None or px <= 0.0:
                raise ValueError(f"no valid price for position {symbol!r}")
            total += abs(pos.quantity * px)
        return total

    # ------------------------------------------------------------------ #
    # weight -> order translation
    # ------------------------------------------------------------------ #
    def target_weights_to_orders(
        self,
        target_weights: Union[Mapping[str, float], "pd.Series"],
        prices: PriceLike,
        capital: float,
        current_positions: Optional[Mapping[str, float]] = None,
        min_trade_notional: float = 1.0,
        allow_fractional: bool = False,
    ) -> List[dict]:
        """Convert a target-weight allocation into a list of order intents.

        Parameters
        ----------
        target_weights:
            Mapping (or ``pd.Series``) of ``symbol -> target portfolio weight``.
            Weights are interpreted against ``capital`` (so ``0.25`` means hold
            25% of capital in that symbol's notional).
        prices:
            Mapping (or ``pd.Series``) of ``symbol -> price``. Every targeted or
            held symbol must have a finite positive price.
        capital:
            Total capital base the weights apply to.
        current_positions:
            Mapping of ``symbol -> current signed share quantity``.  Defaults to
            this manager's own tracked positions.
        min_trade_notional:
            Trades whose absolute notional (``|delta shares| * price``) is below
            this threshold are skipped, avoiding dust orders.
        allow_fractional:
            When ``True``, no rounding is applied and the raw float share
            delta is used (fractional venues, e.g. crypto).  When ``False``
            (default), target positions are rounded toward zero before the
            delta is computed.  This keeps whole-share orders inside the
            requested absolute exposure instead of letting nearest-share
            rounding exceed the portfolio risk budget.  Current positions
            must then also be whole-share quantities; use
            ``allow_fractional=True`` for fractional books.

        Returns
        -------
        list[dict]
            One intent per symbol that needs trading, each shaped
            ``{"symbol": str, "side": OrderSide, "quantity": float}`` with a
            positive ``quantity`` (whole-share unless ``allow_fractional``).
        """
        weights = self._symbol_mapping(target_weights, "target weights")
        price_map = self._symbol_mapping(prices, "prices")
        if not isinstance(allow_fractional, bool):
            raise TypeError("allow_fractional must be a boolean")
        capital_value = self._to_float(capital)
        min_trade_value = self._to_float(min_trade_notional)
        if capital_value is None or capital_value <= 0.0:
            raise ValueError("capital must be finite and positive")
        if min_trade_value is None or min_trade_value < 0.0:
            raise ValueError("min_trade_notional must be finite and non-negative")
        capital = capital_value
        min_trade_notional = min_trade_value
        for symbol, weight in weights.items():
            value = self._to_float(weight)
            if value is None:
                raise ValueError(f"target weight for {symbol!r} must be finite")
            weights[symbol] = value

        if current_positions is None:
            current = {s: p.quantity for s, p in self._positions.items()}
        else:
            current = {}
            normalized_positions = self._symbol_mapping(
                current_positions, "current positions"
            )
            for symbol, quantity in normalized_positions.items():
                value = self._to_float(quantity)
                if value is None:
                    raise ValueError(f"position quantity for {symbol!r} must be finite")
                current[symbol] = value

        # Universe is every symbol we either target or already hold, so that
        # dropped names get an explicit liquidation order.
        symbols = set(weights) | set(current)
        orders: List[dict] = []

        for symbol in sorted(symbols):
            price = self._to_float(price_map.get(symbol))
            if price is None or price <= 0.0:
                raise ValueError(f"no finite positive price for symbol {symbol!r}")

            target_w = self._to_float(weights.get(symbol)) or 0.0
            target_notional = target_w * capital
            target_shares = target_notional / price

            current_shares = self._to_float(current.get(symbol)) or 0.0
            if not allow_fractional:
                if not math.isclose(
                    current_shares, round(current_shares), abs_tol=1e-9
                ):
                    raise ValueError(
                        f"current position for {symbol!r} is fractional; "
                        "pass allow_fractional=True"
                    )
                target_shares = float(math.trunc(target_shares))
                current_shares = float(round(current_shares))
            delta = target_shares - current_shares
            if delta == 0:
                continue

            notional = abs(delta) * price
            if notional < min_trade_notional:
                continue

            orders.append(
                {
                    "symbol": symbol,
                    "side": OrderSide.BUY if delta > 0 else OrderSide.SELL,
                    "quantity": float(abs(delta)),
                }
            )
        return orders

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_mapping(obj: Union[Mapping, "pd.Series"]) -> Dict:
        if isinstance(obj, pd.Series):
            if obj.index.has_duplicates:
                raise ValueError("Series mappings must not contain duplicate symbols")
            return obj.to_dict()
        if obj is None:
            return {}
        if not isinstance(obj, Mapping):
            raise TypeError("expected a mapping or pandas Series")
        return dict(obj)

    @staticmethod
    def _symbol_key(symbol) -> str:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbols must be non-empty strings")
        return symbol.strip()

    @classmethod
    def _symbol_mapping(cls, obj, name: str) -> Dict:
        normalized: Dict = {}
        for symbol, value in cls._as_mapping(obj).items():
            key = cls._symbol_key(symbol)
            if key in normalized:
                raise ValueError(f"{name} contains duplicate symbol {key!r}")
            normalized[key] = value
        return normalized

    @staticmethod
    def _to_float(value) -> Optional[float]:
        if value is None or isinstance(value, bool):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None
