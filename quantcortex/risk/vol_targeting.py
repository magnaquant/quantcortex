"""Volatility targeting overlay.

Scales a weight vector so the portfolio's *ex-ante* annualised volatility hits
a target (e.g. 10%).  When realised vol is above target the overlay levers
*down* (gross exposure < 1, remainder in cash); when below target it can lever
up to ``max_leverage``.  Vol targeting is the workhorse risk control behind the
platform's "max drawdown < 15%" objective.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from quantcortex.portfolio.base import enforce_exposure_contract

__all__ = ["VolTargeting", "realized_portfolio_vol"]

TRADING_DAYS = 252


def realized_portfolio_vol(
    weights,
    returns,
    *,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised volatility of the ``weights`` portfolio over ``returns``.

    ``returns`` may be a 2-D ``(T, n_assets)`` matrix of per-asset returns or a
    1-D ``(T,)`` series already representing portfolio returns.
    """
    w = np.asarray(weights, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w)):
        raise ValueError("weights must be a non-empty finite 1-D vector")
    if (
        isinstance(periods_per_year, (bool, np.bool_))
        or not isinstance(periods_per_year, (int, np.integer))
        or periods_per_year <= 0
    ):
        raise ValueError("periods_per_year must be a positive integer")
    if not np.all(np.isfinite(r)):
        raise ValueError("returns must contain only finite values")
    if r.ndim == 2:
        if r.shape[1] != w.size:
            raise ValueError("returns columns must match weights length")
        port = r @ w
    elif r.ndim == 1:
        port = r
    else:
        raise ValueError("returns must be a 1-D or 2-D array")
    if port.size < 2:
        raise ValueError("at least two return observations are required")
    return float(np.std(port, ddof=1) * np.sqrt(periods_per_year))


class VolTargeting:
    """Scale weights to hit ``target_vol`` annualised volatility."""

    def __init__(
        self,
        target_vol: float = 0.10,
        *,
        max_leverage: float = 1.0,
        min_scale: float = 0.0,
        periods_per_year: int = TRADING_DAYS,
        vol_floor: float = 1e-6,
    ) -> None:
        if any(
            isinstance(value, (bool, np.bool_))
            for value in (target_vol, max_leverage, min_scale, vol_floor)
        ):
            raise TypeError("volatility controls must be numeric, not boolean")
        if not np.isfinite(target_vol) or target_vol <= 0:
            raise ValueError("target_vol must be positive.")
        if not np.isfinite(max_leverage) or max_leverage <= 0:
            raise ValueError("max_leverage must be positive.")
        self.target_vol = float(target_vol)
        self.max_leverage = float(max_leverage)
        self.min_scale = float(min_scale)
        if (
            isinstance(periods_per_year, (bool, np.bool_))
            or not isinstance(periods_per_year, (int, np.integer))
            or periods_per_year <= 0
        ):
            raise ValueError("periods_per_year must be a positive integer")
        self.periods_per_year = int(periods_per_year)
        self.vol_floor = float(vol_floor)
        if (
            not np.isfinite(self.min_scale)
            or self.min_scale < 0.0
            or self.min_scale > self.max_leverage
        ):
            raise ValueError("min_scale must be in [0, max_leverage]")
        if not np.isfinite(self.vol_floor) or self.vol_floor <= 0.0:
            raise ValueError("vol_floor must be finite and positive")
        self.last_scale: Optional[float] = None
        self.last_realized_vol: Optional[float] = None

    def scale_factor(self, realized_vol: float) -> float:
        rv = float(realized_vol)
        if not np.isfinite(rv) or rv < 0.0:
            raise ValueError("realized_vol must be finite and non-negative")
        rv = max(rv, self.vol_floor)
        raw = self.target_vol / rv
        return float(np.clip(raw, self.min_scale, self.max_leverage))

    def apply(
        self,
        weights,
        returns=None,
        *,
        realized_vol: Optional[float] = None,
    ) -> np.ndarray:
        """Return ``weights`` scaled to the vol target.

        Provide either a ``returns`` history (per-asset matrix or portfolio
        series) from which realised vol is estimated, or an explicit
        ``realized_vol`` (already annualised).

        Notes
        -----
        * A non-finite realised vol or return input is rejected. Missing risk
          estimates must not silently pass exposure through unchanged.
        * The requested scalar scale is capped so that no element of the
          scaled book exceeds the ``[-1, 1]`` per-asset contract:
          ``effective_scale = min(scale, 1 / max|w_i|)``.  The capped scale is
          then applied *unclipped*, preserving the allocation proportions
          (per-asset clipping would silently distort them and make the
          realized gross differ from the prescribed scale).  ``last_scale``
          records the EFFECTIVE (possibly capped) scale.
        """
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w)):
            raise ValueError("weights must be a non-empty finite 1-D vector")

        if realized_vol is None:
            if returns is None:
                raise ValueError(
                    "VolTargeting.apply needs `returns` or `realized_vol`."
                )
            realized_vol = realized_portfolio_vol(
                w, returns, periods_per_year=self.periods_per_year
            )

        realized_vol = float(realized_vol)
        if not np.isfinite(realized_vol) or realized_vol < 0.0:
            raise ValueError("realized_vol must be finite and non-negative")
        if realized_vol <= self.vol_floor:
            # No measurable risk -> hold target exposure unchanged (no lever-up
            # into a degenerate estimate).
            scale = 1.0
        else:
            scale = self.scale_factor(realized_vol)

        # Cap the scalar so no element leaves the [-1, 1] per-asset contract;
        # the capped scale is applied unclipped to preserve proportions.
        in_gross = float(np.abs(w).sum())
        if in_gross > 0.0:
            scale = min(scale, self.max_leverage / in_gross)
        max_abs = float(np.max(np.abs(w)))
        if max_abs > 0.0:
            scale = min(scale, 1.0 / max_abs)

        self.last_realized_vol = realized_vol
        self.last_scale = float(scale)

        scaled = w * scale
        # Gross exposure may legitimately exceed 1.0 when levering up; size the
        # cap to the configured leverage / scaled gross so the contract still
        # guards typos.
        return enforce_exposure_contract(
            scaled, max_gross=self.max_leverage + 1e-9, name="VolTargeting"
        )
