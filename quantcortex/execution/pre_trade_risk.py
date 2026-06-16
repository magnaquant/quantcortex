"""Pre-trade risk checks - the last gate before live execution.

Every order intent and every target-weight vector produced by quantcortex must
clear :class:`PreTradeRiskCheck` *before* it is handed to a broker adapter.  This
is the final, authoritative safety gate: it re-validates the canonical weight
contract, enforces position-concentration and gross-exposure caps, and bounds
per-order and aggregate notional against the allowed symbol universe, and
reconstructs the post-trade book from current positions before submission.

The design philosophy mirrors the weight contract - fail loud and early.  Use
:meth:`PreTradeRiskCheck.assert_safe` on the hot path so that any violation
raises :class:`PreTradeRiskError` and the order(s) never reach the venue.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from quantcortex.execution.order_manager import OrderSide
from quantcortex.portfolio.base import PortfolioMode

__all__ = ["PreTradeRiskCheck", "PreTradeRiskError"]

PriceLike = Union[Mapping[str, float], "pd.Series"]


class PreTradeRiskError(Exception):
    """Raised by :meth:`PreTradeRiskCheck.assert_safe` on any violation."""

    def __init__(self, violations: List[str]) -> None:
        self.violations = list(violations)
        super().__init__(
            "Pre-trade risk check failed:\n  - " + "\n  - ".join(self.violations)
        )


class PreTradeRiskCheck:
    """Pre-flight risk limits applied before any order goes to a broker.

    Parameters
    ----------
    max_position_weight:
        Maximum absolute weight any single position may carry (default 0.20).
    max_gross:
        Maximum gross exposure (``sum |w|``) for a weight vector (default 1.0).
    max_notional:
        Optional cap on the total notional of a batch of orders.
    max_order_notional:
        Optional cap on the notional of any single order.
    allowed_symbols:
        Optional whitelist of tradable symbols; orders outside it are rejected.
    tolerance:
        Numerical slack applied to the cap comparisons.
    """

    def __init__(
        self,
        *,
        max_position_weight: float = 0.20,
        max_gross: float = 1.0,
        max_notional: Optional[float] = None,
        max_order_notional: Optional[float] = None,
        allowed_symbols: Optional[Iterable[str]] = None,
        tolerance: float = 1e-9,
    ) -> None:
        numeric_inputs = {
            "max_position_weight": max_position_weight,
            "max_gross": max_gross,
            "tolerance": tolerance,
        }
        for name, value in numeric_inputs.items():
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be numeric, not boolean")
        self.max_position_weight = float(max_position_weight)
        self.max_gross = float(max_gross)
        for name, value in (
            ("max_notional", max_notional),
            ("max_order_notional", max_order_notional),
        ):
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be numeric, not boolean")
        self.max_notional = None if max_notional is None else float(max_notional)
        self.max_order_notional = (
            None if max_order_notional is None else float(max_order_notional)
        )
        if isinstance(allowed_symbols, str):
            raise ValueError("allowed_symbols must be an iterable of symbols, not a string")
        self.allowed_symbols = None
        if allowed_symbols is not None:
            normalized = []
            for symbol in allowed_symbols:
                if not isinstance(symbol, str) or not symbol.strip():
                    raise ValueError(
                        "allowed_symbols must contain non-empty strings"
                    )
                normalized.append(symbol.strip())
            self.allowed_symbols = set(normalized)
        self.tolerance = float(tolerance)
        if (
            not np.isfinite(self.max_position_weight)
            or self.max_position_weight < 0.0
            or self.max_position_weight > 1.0
        ):
            raise ValueError("max_position_weight must be finite and in [0, 1]")
        if not np.isfinite(self.max_gross) or self.max_gross < 0.0:
            raise ValueError("max_gross must be finite and non-negative")
        for name, value in (
            ("max_notional", self.max_notional),
            ("max_order_notional", self.max_order_notional),
        ):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative")

    # ------------------------------------------------------------------ #
    # weight-level checks
    # ------------------------------------------------------------------ #
    def check_weights(
        self,
        weights: np.ndarray,
        mode: Union[PortfolioMode, str] = PortfolioMode.LONG_ONLY,
    ) -> Tuple[bool, List[str]]:
        """Validate a *post-overlay* target book. Returns ``(ok, violations)``.

        The pre-trade gate sits at the end of the pipeline, so it receives the
        weights *after* timing/risk overlays have scaled exposure.  It therefore
        validates the relaxed **exposure contract** (the same one
        :func:`quantcortex.portfolio.base.enforce_exposure_contract` enforces), not the
        strict ``sum == 1.0`` allocation contract: a de-risked or regime-gated
        long-only book legitimately sums to less than 1.0 with the remainder in
        cash, and a fully flat book (circuit breaker tripped) sums to 0.0.

        Checks: finite weights, each in ``[-1, 1]``, no short legs in a
        long-only book, dollar-neutrality for a market-neutral book, no single
        ``|weight|`` above ``max_position_weight``, and gross exposure
        (``sum |w|``) within ``max_gross``.
        """
        violations: List[str] = []
        try:
            mode = PortfolioMode.coerce(mode)
        except (TypeError, ValueError):
            return False, [f"invalid portfolio mode {mode!r}"]

        try:
            arr = np.asarray(weights, dtype=np.float64)
        except (TypeError, ValueError):
            return False, ["weights not coercible to a float64 array"]
        if arr.ndim != 1:
            return False, [f"weights must be 1-D, got shape {arr.shape}"]
        if arr.size == 0:
            return False, ["empty weight vector"]
        if not np.all(np.isfinite(arr)):
            bad = np.where(~np.isfinite(arr))[0].tolist()
            return False, [f"non-finite weights at indices {bad}"]

        # Per-asset box [-1, 1].
        box = np.where(np.abs(arr) > 1.0 + self.tolerance)[0]
        if box.size:
            violations.append(
                f"weights outside [-1, 1]: {{{', '.join(f'{int(i)}: {arr[i]:.4f}' for i in box)}}}"
            )

        # Long-only books must have no short legs; market-neutral must be ~flat.
        if mode is PortfolioMode.LONG_ONLY:
            shorts = np.where(arr < -self.tolerance)[0]
            if shorts.size:
                violations.append(
                    f"long-only book has short legs at indices {shorts.tolist()}"
                )
            total = float(arr.sum())
            if total > self.max_gross + self.tolerance:
                violations.append(
                    f"invested fraction {total:.6f} exceeds gross cap {self.max_gross}"
                )
        else:  # MARKET_NEUTRAL
            total = float(arr.sum())
            if abs(total) > max(self.tolerance, 1e-6):
                violations.append(
                    f"market-neutral book is not dollar-neutral (sum {total:+.6f})"
                )

        over = np.where(np.abs(arr) > self.max_position_weight + self.tolerance)[0]
        if over.size:
            offenders = {int(i): float(arr[i]) for i in over}
            violations.append(
                f"position weight cap {self.max_position_weight}: {offenders}"
            )

        gross = float(np.abs(arr).sum())
        if gross > self.max_gross + self.tolerance:
            violations.append(
                f"gross exposure {gross:.6f} exceeds cap {self.max_gross}"
            )

        return (not violations), violations

    # ------------------------------------------------------------------ #
    # order-level checks
    # ------------------------------------------------------------------ #
    def check_orders(
        self,
        orders: List[dict],
        prices: PriceLike,
        capital: Optional[float] = None,
    ) -> Tuple[bool, List[str]]:
        """Validate a batch of order intents. Returns ``(ok, violations)``.

        Each order is a dict with at least ``symbol`` and ``quantity`` (and
        usually ``side``).  Checks per-order notional against
        ``max_order_notional``, total notional against ``max_notional``, and
        symbol membership against ``allowed_symbols``.  ``capital`` is accepted
        for API symmetry and informational messages.
        """
        violations: List[str] = []
        try:
            price_map = self._normalize_symbol_mapping(prices, "prices")
        except (TypeError, ValueError) as exc:
            return False, [str(exc)]
        total_notional = 0.0
        if capital is not None:
            capital = self._to_float(capital)
            if capital is None or capital <= 0.0:
                violations.append("capital must be finite and positive when provided")

        for i, order in enumerate(orders):
            if not isinstance(order, Mapping):
                violations.append(f"order {i} is not a mapping")
                continue
            symbol = order.get("symbol")
            if not isinstance(symbol, str) or not symbol.strip():
                violations.append(f"order {i} has an empty or invalid symbol")
                continue
            symbol = symbol.strip()
            quantity = self._to_float(order.get("quantity"))
            if quantity is None or quantity <= 0.0:
                violations.append(
                    f"order {symbol!r} quantity must be finite and positive"
                )
                continue
            try:
                OrderSide(order.get("side"))
            except (TypeError, ValueError):
                violations.append(f"order {symbol!r} has invalid side")

            if self.allowed_symbols is not None and symbol not in self.allowed_symbols:
                violations.append(f"symbol {symbol!r} not in allowed_symbols")

            price = self._to_float(price_map.get(symbol))
            if price is None or price <= 0.0:
                violations.append(f"no usable price for symbol {symbol!r}")
                continue

            notional = abs(quantity) * price
            total_notional += notional

            if (
                self.max_order_notional is not None
                and notional > self.max_order_notional + self.tolerance
            ):
                violations.append(
                    f"order {symbol!r} notional {notional:.2f} exceeds "
                    f"per-order cap {self.max_order_notional}"
                )

        if (
            self.max_notional is not None
            and total_notional > self.max_notional + self.tolerance
        ):
            violations.append(
                f"total notional {total_notional:.2f} exceeds cap "
                f"{self.max_notional}"
            )

        return (not violations), violations

    def check_post_trade_positions(
        self,
        orders: List[dict],
        prices: PriceLike,
        capital: float,
        current_positions: Mapping[str, float],
        mode: Union[PortfolioMode, str] = PortfolioMode.LONG_ONLY,
    ) -> Tuple[bool, List[str]]:
        """Reconstruct and validate the post-trade exposure implied by a batch."""
        violations: List[str] = []
        capital_value = self._to_float(capital)
        if capital_value is None or capital_value <= 0.0:
            return False, ["post-trade capital must be finite and positive"]
        try:
            price_map = self._normalize_symbol_mapping(prices, "prices")
            quantities = self._normalize_symbol_mapping(
                current_positions, "current_positions"
            )
        except (TypeError, ValueError) as exc:
            return False, [str(exc)]

        normalized_quantities: dict[str, float] = {}
        for symbol, quantity in quantities.items():
            value = self._to_float(quantity)
            if value is None:
                violations.append(f"current position for {symbol!r} must be finite")
            else:
                normalized_quantities[symbol] = value

        for order in orders:
            if not isinstance(order, Mapping):
                continue
            symbol = order.get("symbol")
            quantity = self._to_float(order.get("quantity"))
            try:
                side = OrderSide(order.get("side"))
            except (TypeError, ValueError):
                violations.append("post-trade orders require a valid side")
                continue
            if (
                not isinstance(symbol, str)
                or not symbol.strip()
                or quantity is None
                or quantity <= 0.0
            ):
                violations.append("post-trade orders require valid symbols and positive quantities")
                continue
            symbol = symbol.strip()
            signed = quantity if side is OrderSide.BUY else -quantity
            normalized_quantities[symbol] = normalized_quantities.get(symbol, 0.0) + signed

        symbols = sorted(
            symbol
            for symbol, quantity in normalized_quantities.items()
            if abs(quantity) > self.tolerance
        )
        if not symbols:
            return (not violations), violations

        weights = []
        for symbol in symbols:
            price = self._to_float(price_map.get(symbol))
            if price is None or price <= 0.0:
                violations.append(f"no usable post-trade price for symbol {symbol!r}")
                continue
            weights.append(normalized_quantities[symbol] * price / capital_value)

        if len(weights) == len(symbols):
            ok, exposure_violations = self.check_weights(
                np.asarray(weights, dtype=np.float64), mode=mode
            )
            if not ok:
                violations.extend(
                    f"post-trade {message}" for message in exposure_violations
                )
        return (not violations), violations

    # ------------------------------------------------------------------ #
    # combined assertion (the live-execution gate)
    # ------------------------------------------------------------------ #
    def assert_safe(
        self,
        *,
        weights: Optional[np.ndarray] = None,
        mode: Union[PortfolioMode, str] = PortfolioMode.LONG_ONLY,
        orders: Optional[List[dict]] = None,
        prices: Optional[PriceLike] = None,
        capital: Optional[float] = None,
        current_positions: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Run all applicable checks and raise on any violation.

        This is the last gate before live execution: call it with whatever you
        are about to act on (a weight vector, a batch of orders, or both).  Any
        accumulated violation raises :class:`PreTradeRiskError`.
        """
        violations: List[str] = []

        if weights is not None:
            ok, w_violations = self.check_weights(weights, mode=mode)
            if not ok:
                violations.extend(w_violations)

        if orders is not None:
            if prices is None:
                violations.append("check_orders requires prices but none given")
            else:
                ok, o_violations = self.check_orders(orders, prices, capital)
                if not ok:
                    violations.extend(o_violations)
            if current_positions is None:
                violations.append(
                    "post-trade validation requires current_positions when orders are supplied"
                )
            elif prices is not None and capital is not None:
                ok, post_violations = self.check_post_trade_positions(
                    orders,
                    prices,
                    capital,
                    current_positions,
                    mode=mode,
                )
                if not ok:
                    violations.extend(post_violations)
            elif prices is not None:
                violations.append(
                    "post-trade validation requires capital when orders are supplied"
                )

        if violations:
            raise PreTradeRiskError(violations)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_mapping(obj) -> dict:
        if obj is None:
            return {}
        if isinstance(obj, pd.Series):
            if obj.index.has_duplicates:
                raise ValueError("Series mappings must not contain duplicate symbols")
            return obj.to_dict()
        if not isinstance(obj, Mapping):
            raise TypeError("expected a mapping or pandas Series")
        return dict(obj)

    @classmethod
    def _normalize_symbol_mapping(cls, obj, name: str) -> dict:
        normalized = {}
        for symbol, value in cls._as_mapping(obj).items():
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError(f"{name} must use non-empty string symbols")
            key = symbol.strip()
            if key in normalized:
                raise ValueError(f"{name} contains duplicate symbol {key!r}")
            normalized[key] = value
        return normalized

    @staticmethod
    def _to_float(value):
        if value is None or isinstance(value, (bool, np.bool_)):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if np.isfinite(result) else None
