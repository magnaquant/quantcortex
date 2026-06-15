"""Pre-trade risk checks - the last gate before live execution.

Every order intent and every target-weight vector produced by quantcortex must
clear :class:`PreTradeRiskCheck` *before* it is handed to a broker adapter.  This
is the final, authoritative safety gate: it re-validates the canonical weight
contract, enforces position-concentration and gross-exposure caps, and bounds
per-order and aggregate notional against the allowed symbol universe.

The design philosophy mirrors the weight contract - fail loud and early.  Use
:meth:`PreTradeRiskCheck.assert_safe` on the hot path so that any violation
raises :class:`PreTradeRiskError` and the order(s) never reach the venue.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from portfolio.base import (
    PortfolioMode,
    WeightContractViolationError,
    enforce_weight_contract,
)

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
        self.max_position_weight = float(max_position_weight)
        self.max_gross = float(max_gross)
        self.max_notional = (
            None if max_notional is None else float(max_notional)
        )
        self.max_order_notional = (
            None if max_order_notional is None else float(max_order_notional)
        )
        self.allowed_symbols = (
            None if allowed_symbols is None else set(allowed_symbols)
        )
        self.tolerance = float(tolerance)

    # ------------------------------------------------------------------ #
    # weight-level checks
    # ------------------------------------------------------------------ #
    def check_weights(
        self,
        weights: np.ndarray,
        mode: Union[PortfolioMode, str] = PortfolioMode.LONG_ONLY,
    ) -> Tuple[bool, List[str]]:
        """Validate a weight vector. Returns ``(ok, violations)``.

        Runs the canonical :func:`enforce_weight_contract` first (its failure is
        recorded as a violation rather than raised), then checks that no single
        absolute weight exceeds ``max_position_weight`` and that gross exposure
        does not exceed ``max_gross``.
        """
        violations: List[str] = []

        try:
            arr = enforce_weight_contract(weights, mode=mode)
        except WeightContractViolationError as exc:
            violations.append(f"weight contract: {exc}")
            # Best-effort coercion so the cap checks below still run when
            # possible; if even that fails, return early.
            try:
                arr = np.asarray(weights, dtype=np.float64).ravel()
            except (TypeError, ValueError):
                return False, violations

        abs_w = np.abs(arr)
        over = np.where(abs_w > self.max_position_weight + self.tolerance)[0]
        if over.size:
            offenders = {int(i): float(arr[i]) for i in over}
            violations.append(
                f"position weight cap {self.max_position_weight}: {offenders}"
            )

        gross = float(abs_w.sum())
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
        price_map = self._as_mapping(prices)
        total_notional = 0.0

        for order in orders:
            symbol = order.get("symbol")
            quantity = self._to_float(order.get("quantity")) or 0.0

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
            return obj.to_dict()
        return dict(obj)

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
