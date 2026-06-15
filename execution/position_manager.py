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

from typing import Dict, List, Mapping, Optional, Union

import pandas as pd

from execution.brokers.base import Position
from execution.order_manager import Order, OrderSide

__all__ = ["PositionManager"]

# Type alias for the two accepted price/weight container shapes.
PriceLike = Union[Mapping[str, float], "pd.Series"]


class PositionManager:
    """Tracks live positions and turns target weights into order intents."""

    def __init__(self) -> None:
        self._positions: Dict[str, Position] = {}

    # ------------------------------------------------------------------ #
    # state access
    # ------------------------------------------------------------------ #
    @property
    def positions(self) -> Dict[str, Position]:
        """Mapping of symbol -> :class:`Position` (only non-flat books)."""
        return dict(self._positions)

    def get_position(self, symbol: str) -> Position:
        """Return the current position for ``symbol`` (flat if untracked)."""
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
        qty_delta = float(qty_delta)
        price = float(price)
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

        Uses ``order.filled_quantity`` and ``order.avg_fill_price``; the sign is
        derived from ``order.side``.  Orders with no fill quantity are ignored.
        """
        filled = float(order.filled_quantity or 0.0)
        if filled <= 0.0:
            return self.get_position(order.symbol)
        price = order.avg_fill_price
        if price is None:
            price = order.limit_price if order.limit_price is not None else 0.0
        signed = filled if order.side is OrderSide.BUY else -filled
        return self.update(order.symbol, signed, float(price))

    def mark(self, prices: PriceLike) -> None:
        """Update ``market_price`` on tracked positions from ``prices``."""
        prices = self._as_mapping(prices)
        for symbol, pos in self._positions.items():
            if symbol in prices:
                pos.market_price = float(prices[symbol])

    # ------------------------------------------------------------------ #
    # valuation
    # ------------------------------------------------------------------ #
    def market_value(self, prices: PriceLike) -> float:
        """Signed total market value of the book at ``prices``."""
        prices = self._as_mapping(prices)
        total = 0.0
        for symbol, pos in self._positions.items():
            px = float(prices.get(symbol, pos.market_price))
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
            px = float(prices.get(symbol, pos.market_price))
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
            Mapping (or ``pd.Series``) of ``symbol -> price``.  A symbol without
            a positive price is skipped (it cannot be sized).
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
            (default), the delta is computed from the *unrounded* current and
            target share quantities and only the final delta share count is
            rounded to a whole number.  Note that this equities-style rounding
            can strand sub-half-share dust (a delta of magnitude < 0.5 rounds
            to no trade); venues that support fractional quantities should
            pass ``allow_fractional=True``.

        Returns
        -------
        list[dict]
            One intent per symbol that needs trading, each shaped
            ``{"symbol": str, "side": OrderSide, "quantity": float}`` with a
            positive ``quantity`` (whole-share unless ``allow_fractional``).
        """
        weights = self._as_mapping(target_weights)
        price_map = self._as_mapping(prices)

        if current_positions is None:
            current = {s: p.quantity for s, p in self._positions.items()}
        else:
            current = {s: float(q) for s, q in self._as_mapping(
                current_positions
            ).items()}

        # Universe is every symbol we either target or already hold, so that
        # dropped names get an explicit liquidation order.
        symbols = set(weights) | set(current)
        orders: List[dict] = []

        for symbol in sorted(symbols):
            price = self._to_float(price_map.get(symbol))
            if price is None or price <= 0.0:
                # Cannot size without a valid price (covers a held name with no
                # quote, too - we leave it untouched rather than guess).
                continue

            target_w = self._to_float(weights.get(symbol)) or 0.0
            target_notional = target_w * float(capital)
            target_shares = target_notional / price

            current_shares = self._to_float(current.get(symbol)) or 0.0
            # Delta from the UNrounded current and target share quantities;
            # only the final delta is rounded (whole-share venues).
            delta = target_shares - current_shares
            if not allow_fractional:
                delta = float(round(delta))
            if delta == 0:
                continue

            notional = abs(delta) * price
            if notional < float(min_trade_notional):
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
            return obj.to_dict()
        if obj is None:
            return {}
        return dict(obj)

    @staticmethod
    def _to_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
