"""Inverse-volatility (VIX) exposure-scaling timing overlay.

:class:`VIXScaler` scales gross portfolio exposure inversely to the level of
implied volatility (the VIX index, or any comparable volatility series).  The
economic intuition is volatility targeting: when implied vol is elevated the
market is fragile and realised risk per unit of exposure is high, so we lean the
book down; when vol is subdued we are permitted to run closer to full exposure.

Scaling rule
------------
``scale = clip(target_vix / current_vix, floor, cap)``

* ``current_vix > target_vix`` (stress) -> ``scale < 1`` -> de-risk.
* ``current_vix < target_vix`` (calm)   -> ``scale > 1``, clipped at ``cap``.

The overlay is strictly causal: it consumes the *last* observed VIX value, which
is all an executing strategy could know at decision time.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import pandas as pd

from portfolio.base import enforce_exposure_contract

__all__ = ["VIXScaler"]


class VIXScaler:
    """Scale exposure by ``target_vix / current_vix`` within ``[floor, cap]``.

    Parameters
    ----------
    target_vix:
        The VIX level at which the strategy runs at its natural exposure (the
        point where the raw ratio equals 1.0).  Defaults to 20, roughly the
        long-run average of the VIX.
    floor:
        Minimum exposure multiplier.  Rationale: even in extreme stress we keep a
        small residual exposure rather than going fully flat, so the strategy can
        participate in any sharp mean-reversion rebound and so transaction costs
        from repeatedly toggling to zero are contained.  A non-zero floor also
        keeps the overlay distinct from a hard regime kill-switch.
    cap:
        Maximum exposure multiplier.  Rationale: capping at ``1.0`` (default)
        prevents the overlay from *levering up* into deceptively calm,
        low-vol markets where volatility can spike without warning ("picking up
        pennies in front of a steamroller").  Setting ``cap > 1.0`` is permitted
        for genuine vol-targeting strategies that want to add exposure in calm
        regimes, but then the downstream gross-exposure cap must accommodate it.
    """

    def __init__(
        self,
        target_vix: float = 20.0,
        floor: float = 0.3,
        cap: float = 1.0,
    ) -> None:
        if target_vix <= 0:
            raise ValueError("target_vix must be positive")
        if floor < 0:
            raise ValueError("floor must be non-negative")
        if cap < floor:
            raise ValueError("cap must be >= floor")
        self.target_vix = float(target_vix)
        self.floor = float(floor)
        self.cap = float(cap)

    # ------------------------------------------------------------------ #
    def compute_scale(self, vix: Union[float, pd.Series, np.ndarray]) -> float:
        """Return the clipped exposure multiplier for the latest VIX value."""
        current = self._latest_vix(vix)
        if current <= 0 or not np.isfinite(current):
            # Degenerate / missing reading -> retreat to the floor (de-risk).
            return self.floor
        raw = self.target_vix / current
        return float(np.clip(raw, self.floor, self.cap))

    def apply(
        self,
        weights: np.ndarray,
        vix: Any = None,
    ) -> np.ndarray:
        """Scale ``weights`` by the inverse-VIX multiplier.

        ``vix`` may be a float, a :class:`pandas.Series`/array (the last value is
        used), or a ``StrategyContext``-like object exposing ``.extra['vix']``.
        The result is validated via :func:`enforce_exposure_contract`.
        """
        w = np.asarray(weights, dtype=np.float64).ravel()
        scale = self.compute_scale(self._coerce_vix(vix))
        scaled = w * scale

        input_gross = float(np.abs(w).sum())
        max_gross = max(1.0, input_gross) + 1e-9
        # If cap > 1 the overlay can grow gross; widen the allowance accordingly.
        if self.cap > 1.0:
            max_gross = max(max_gross, input_gross * self.cap + 1e-9)
        return enforce_exposure_contract(
            scaled, max_gross=max_gross, name=type(self).__name__
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _latest_vix(vix: Union[float, pd.Series, np.ndarray]) -> float:
        """Extract the most recent scalar VIX value from a variety of inputs."""
        if isinstance(vix, pd.Series):
            if vix.empty:
                raise ValueError("vix series is empty")
            return float(vix.iloc[-1])
        arr = np.asarray(vix, dtype=np.float64).ravel()
        if arr.size == 0:
            raise ValueError("vix is empty")
        return float(arr[-1])

    def _coerce_vix(self, vix: Any) -> Union[float, pd.Series, np.ndarray]:
        """Resolve a context-like object to a VIX value if needed."""
        if isinstance(vix, (float, int, np.floating, np.integer, pd.Series, np.ndarray, list, tuple)):
            return vix  # already a usable value/sequence
        extra = getattr(vix, "extra", None)
        if isinstance(extra, dict) and "vix" in extra:
            return extra["vix"]
        raise TypeError(
            "VIXScaler.apply requires a float/Series vix or a context object "
            "exposing .extra['vix']"
        )
