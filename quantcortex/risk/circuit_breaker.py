"""Drawdown circuit breaker - the platform's hard kill-switch.

When the strategy's equity curve falls more than ``max_drawdown`` below its
running peak, the breaker *trips* and forces the book flat (all weights zero).
By default it remains latched until :meth:`CircuitBreaker.reset` is called.
This is deliberate: a strategy that has been flattened generally cannot repair
its own drawdown, so an automatic drawdown-based reset is not a sound default.
An opt-in automatic reset is available when the supplied equity series can
recover independently (for example, an external mandate-level NAV).

A flat book sums to 0.0, which is why risk overlays validate against the
relaxed :func:`quantcortex.portfolio.base.enforce_exposure_contract` rather than the strict
fully-invested contract.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from quantcortex.portfolio.base import enforce_exposure_contract

__all__ = ["CircuitBreaker", "compute_drawdown"]


def compute_drawdown(equity_curve) -> float:
    """Return the *current* drawdown (>= 0) from an equity/NAV series.

    ``drawdown = 1 - equity[-1] / running_peak[-1]``.
    """
    eq = np.asarray(equity_curve, dtype=np.float64)
    if eq.ndim != 1:
        raise ValueError(f"equity_curve must be 1-D, got shape {eq.shape}")
    if eq.size == 0:
        raise ValueError("equity_curve must be non-empty")
    if not np.all(np.isfinite(eq)) or np.any(eq < 0.0):
        raise ValueError("equity_curve must contain finite, non-negative NAVs")
    peak = np.maximum.accumulate(eq)
    last_peak = peak[-1]
    if last_peak <= 0.0:
        raise ValueError("equity_curve must contain a positive NAV")
    dd = 1.0 - eq[-1] / last_peak
    return float(max(dd, 0.0))


class CircuitBreaker:
    """Zero out weights once drawdown breaches ``max_drawdown``.

    Parameters
    ----------
    max_drawdown:
        Drawdown level (e.g. ``0.15`` = 15%) that trips the breaker.
    reset_drawdown:
        With ``auto_reset=True``, the breaker stays flat until drawdown recovers
        to this level. Defaults to ``max_drawdown / 2``.
    auto_reset:
        Permit drawdown-based automatic re-entry. Defaults to ``False`` so a
        tripped strategy requires an explicit operator reset.
    """

    def __init__(
        self,
        max_drawdown: float = 0.15,
        reset_drawdown: Optional[float] = None,
        *,
        auto_reset: bool = False,
    ) -> None:
        if isinstance(max_drawdown, (bool, np.bool_)):
            raise TypeError("max_drawdown must be numeric, not boolean")
        if isinstance(reset_drawdown, (bool, np.bool_)):
            raise TypeError("reset_drawdown must be numeric, not boolean")
        if not np.isfinite(max_drawdown) or not (0.0 < max_drawdown < 1.0):
            raise ValueError("max_drawdown must be in (0, 1).")
        self.max_drawdown = float(max_drawdown)
        self.reset_drawdown = (
            float(reset_drawdown)
            if reset_drawdown is not None
            else self.max_drawdown / 2.0
        )
        if (
            not np.isfinite(self.reset_drawdown)
            or self.reset_drawdown < 0.0
            or self.reset_drawdown >= self.max_drawdown
        ):
            raise ValueError("reset_drawdown must be in [0, max_drawdown)")
        if not isinstance(auto_reset, (bool, np.bool_)):
            raise TypeError("auto_reset must be a boolean")
        self.auto_reset = bool(auto_reset)
        self._tripped = False

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    def reset(self) -> None:
        self._tripped = False

    def _update_state(self, drawdown: float) -> bool:
        """Advance the latch given the current drawdown; return tripped state."""
        if self._tripped:
            if self.auto_reset and drawdown <= self.reset_drawdown:
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
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w)):
            raise ValueError("weights must be a non-empty finite 1-D vector")

        if current_drawdown is None:
            if equity_curve is None:
                raise ValueError(
                    "CircuitBreaker.apply needs equity_curve or current_drawdown."
                )
            current_drawdown = compute_drawdown(equity_curve)

        current_drawdown = float(current_drawdown)
        if not np.isfinite(current_drawdown) or not 0.0 <= current_drawdown <= 1.0:
            raise ValueError("current_drawdown must be finite and in [0, 1]")
        tripped = self._update_state(current_drawdown)
        out = np.zeros_like(w) if tripped else w
        # Size the gross cap to the incoming book (like the sibling overlays):
        # an untripped pass-through of a levered / long-short book (gross > 1)
        # must not be rejected by the kill-switch's own validation.
        max_gross = max(1.0, float(np.abs(w).sum())) + 1e-9
        return enforce_exposure_contract(
            out, max_gross=max_gross, name="CircuitBreaker"
        )
