"""Drawdown circuit breaker - the platform's hard kill-switch.

When the strategy's equity curve falls more than ``max_drawdown`` below its
running peak, the breaker *trips* and forces the book flat (all weights zero)
until the drawdown recovers below a (lower) reset threshold.  This is the last
line of defence against a strategy that has decoupled from its backtested
behaviour: better to sit in cash than to keep bleeding.

A flat book sums to 0.0, which is why risk overlays validate against the
relaxed :func:`portfolio.base.enforce_exposure_contract` rather than the strict
fully-invested contract.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from portfolio.base import enforce_exposure_contract

__all__ = ["CircuitBreaker", "compute_drawdown"]


def compute_drawdown(equity_curve) -> float:
    """Return the *current* drawdown (>= 0) from an equity/NAV series.

    ``drawdown = 1 - equity[-1] / running_peak[-1]``.
    """
    eq = np.asarray(equity_curve, dtype=np.float64).ravel()
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    last_peak = peak[-1]
    if last_peak <= 0 or not np.isfinite(last_peak):
        return 0.0
    dd = 1.0 - eq[-1] / last_peak
    return float(max(dd, 0.0))


class CircuitBreaker:
    """Zero out weights once drawdown breaches ``max_drawdown``.

    Parameters
    ----------
    max_drawdown:
        Drawdown level (e.g. ``0.15`` = 15%) that trips the breaker.
    reset_drawdown:
        Once tripped, the breaker stays flat until drawdown recovers *below*
        this level (hysteresis to avoid whipsawing on/off at the boundary).
        Defaults to ``max_drawdown / 2``.
    """

    def __init__(
        self,
        max_drawdown: float = 0.15,
        reset_drawdown: Optional[float] = None,
    ) -> None:
        if not (0.0 < max_drawdown < 1.0):
            raise ValueError("max_drawdown must be in (0, 1).")
        self.max_drawdown = float(max_drawdown)
        self.reset_drawdown = (
            float(reset_drawdown)
            if reset_drawdown is not None
            else self.max_drawdown / 2.0
        )
        self._tripped = False

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    def reset(self) -> None:
        self._tripped = False

    def _update_state(self, drawdown: float) -> bool:
        """Advance the latch given the current drawdown; return tripped state."""
        if self._tripped:
            # Stay flat until we recover below the reset threshold.
            if drawdown <= self.reset_drawdown:
                self._tripped = False
        else:
            if drawdown >= self.max_drawdown:
                self._tripped = True
        return self._tripped

    def apply(
        self,
        weights,
        equity_curve=None,
        *,
        current_drawdown: Optional[float] = None,
    ) -> np.ndarray:
        """Return ``weights`` flat (zeros) if tripped, else unchanged.

        Provide either an ``equity_curve`` (drawdown is computed from it) or an
        explicit ``current_drawdown``.
        """
        w = np.asarray(weights, dtype=np.float64).ravel()

        if current_drawdown is None:
            if equity_curve is None:
                raise ValueError(
                    "CircuitBreaker.apply needs equity_curve or current_drawdown."
                )
            current_drawdown = compute_drawdown(equity_curve)

        tripped = self._update_state(float(current_drawdown))
        out = np.zeros_like(w) if tripped else w
        # Size the gross cap to the incoming book (like the sibling overlays):
        # an untripped pass-through of a levered / long-short book (gross > 1)
        # must not be rejected by the kill-switch's own validation.
        max_gross = max(1.0, float(np.abs(w).sum())) + 1e-9
        return enforce_exposure_contract(
            out, max_gross=max_gross, name="CircuitBreaker"
        )
