"""VWAP participation execution model.

:class:`VWAPFill` fills orders around the bar's volume-weighted average price
(VWAP) and charges a slippage that grows with the order's **participation
rate** -- the fraction of the bar's traded volume that the order represents.
The intuition is liquidity-demand based: trading a small slice of the day's
volume executes close to VWAP, while trading a large slice walks the book and
pays progressively more.

Slippage is modelled as *linear in participation*::

    participation = |target_qty| / bar_volume          (capped at the
                                                        configured ``participation``)
    slippage_frac = slippage_coef * participation
    fill = vwap * (1 + sign(target_qty) * slippage_frac)

so buys fill above VWAP and sells below it, and a larger order (relative to bar
volume) gets a worse price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantcortex.backtest.execution_models.ideal_fill import ExecutionModel

__all__ = ["VWAPFill"]


def _bar_vwap(bar: "pd.Series") -> float:
    """Resolve a bar's VWAP, preferring an explicit ``vwap`` column.

    Falls back to the typical price ``(high + low + close) / 3`` when no
    ``vwap`` field is present, and finally to ``close`` if highs/lows are
    missing.
    """
    if "vwap" in bar.index and pd.notna(bar["vwap"]):
        return float(bar["vwap"])
    if "high" in bar.index and "low" in bar.index:
        return float((bar["high"] + bar["low"] + bar["close"]) / 3.0)
    return float(bar["close"])


class VWAPFill(ExecutionModel):
    """Fill around VWAP with participation-dependent (linear) slippage.

    Parameters
    ----------
    participation:
        Maximum assumed participation rate (fraction of bar volume).  The
        effective participation used for slippage is capped at this value, so
        that an order larger than ``participation * volume`` does not produce an
        unboundedly bad fill (it is presumed to be worked across the bar).
        Default ``0.1`` (10% of volume).
    slippage_coef:
        Linear coefficient mapping participation to a fractional price
        concession.  At full ``participation`` the slippage is
        ``slippage_coef * participation``.  Default ``0.1``.
    """

    def __init__(self, participation: float = 0.1, slippage_coef: float = 0.1) -> None:
        if not (0.0 < participation <= 1.0):
            raise ValueError("participation must be in (0, 1].")
        if slippage_coef < 0.0:
            raise ValueError("slippage_coef must be non-negative.")
        self.participation = float(participation)
        self.slippage_coef = float(slippage_coef)

    def fill(
        self,
        symbol: str,
        target_qty: float,
        bar: "pd.Series",
        **kw,
    ) -> float:
        """Return the participation-adjusted VWAP fill price."""
        vwap = _bar_vwap(bar)
        if target_qty == 0:
            return vwap

        volume = kw.get("volume")
        if volume is None and "volume" in bar.index:
            volume = bar["volume"]

        if volume is None or not np.isfinite(volume) or volume <= 0:
            # No volume information: assume full configured participation.
            participation = self.participation
        else:
            participation = min(abs(float(target_qty)) / float(volume), self.participation)

        slippage_frac = self.slippage_coef * participation
        direction = 1.0 if target_qty > 0 else -1.0
        return float(vwap * (1.0 + direction * slippage_frac))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"VWAPFill(participation={self.participation}, "
            f"slippage_coef={self.slippage_coef})"
        )
