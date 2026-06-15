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
    w = np.asarray(weights, dtype=np.float64).ravel()
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim == 2:
        port = r @ w
    else:
        port = r
    if port.size < 2:
        return 0.0
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
        if target_vol <= 0:
            raise ValueError("target_vol must be positive.")
        if max_leverage <= 0:
            raise ValueError("max_leverage must be positive.")
        self.target_vol = float(target_vol)
        self.max_leverage = float(max_leverage)
        self.min_scale = float(min_scale)
        self.periods_per_year = int(periods_per_year)
        self.vol_floor = float(vol_floor)
        self.last_scale: Optional[float] = None
        self.last_realized_vol: Optional[float] = None

    def scale_factor(self, realized_vol: float) -> float:
        rv = max(float(realized_vol), self.vol_floor)
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
        * A non-finite realised vol (NaN/inf, e.g. from NaN-containing
          returns) is treated as "no information": the overlay passes the
          weights through unchanged (``scale = 1.0``) rather than raising a
          misleading non-finite-weights contract error.
        * The requested scalar scale is capped so that no element of the
          scaled book exceeds the ``[-1, 1]`` per-asset contract:
          ``effective_scale = min(scale, 1 / max|w_i|)``.  The capped scale is
          then applied *unclipped*, preserving the allocation proportions
          (per-asset clipping would silently distort them and make the
          realized gross differ from the prescribed scale).  ``last_scale``
          records the EFFECTIVE (possibly capped) scale.
        """
        w = np.asarray(weights, dtype=np.float64).ravel()

        if realized_vol is None:
            if returns is None:
                raise ValueError(
                    "VolTargeting.apply needs `returns` or `realized_vol`."
                )
            realized_vol = realized_portfolio_vol(
                w, returns, periods_per_year=self.periods_per_year
            )

        realized_vol = float(realized_vol)
        if not np.isfinite(realized_vol):
            # No usable risk estimate -> pass-through (scale 1.0); see Notes.
            scale = 1.0
        elif realized_vol <= self.vol_floor:
            # No measurable risk -> hold target exposure unchanged (no lever-up
            # into a degenerate estimate).
            scale = 1.0
        else:
            scale = self.scale_factor(realized_vol)

        # Cap the scalar so no element leaves the [-1, 1] per-asset contract;
        # the capped scale is applied unclipped to preserve proportions.
        max_abs = float(np.max(np.abs(w))) if w.size else 0.0
        if max_abs > 0.0:
            scale = min(scale, 1.0 / max_abs)

        self.last_realized_vol = realized_vol
        self.last_scale = float(scale)

        scaled = w * scale
        # Gross exposure may legitimately exceed 1.0 when levering up; size the
        # cap to the configured leverage / scaled gross so the contract still
        # guards typos.
        in_gross = float(np.abs(w).sum())
        max_gross = max(self.max_leverage, in_gross, in_gross * scale) + 1e-9
        return enforce_exposure_contract(
            scaled, max_gross=max_gross, name="VolTargeting"
        )
