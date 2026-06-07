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

from portfolio.base import enforce_exposure_contract

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

        if realized_vol <= self.vol_floor:
            # No measurable risk -> hold target exposure unchanged (no lever-up
            # into a degenerate estimate).
            scale = 1.0
        else:
            scale = self.scale_factor(realized_vol)

        self.last_realized_vol = float(realized_vol)
        self.last_scale = float(scale)

        scaled = np.clip(w * scale, -1.0, 1.0)
        # Gross exposure may legitimately exceed 1.0 when levering up; size the
        # cap to the configured leverage so the contract still guards typos.
        max_gross = max(self.max_leverage, float(np.abs(w).sum())) + 1e-9
        return enforce_exposure_contract(
            scaled, max_gross=max_gross, name="VolTargeting"
        )
