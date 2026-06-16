"""Transaction cost model - mandatory in every backtest engine.

quantcortex treats transaction costs as a *first-class*, non-optional input:
backtest engines refuse to run without a cost model (see ``quantcortex/backtest/engines``).
The model below implements the three frictions that dominate realistic equity
execution:

* **commission** - broker/exchange fee, default 3 bps.
* **slippage** - adverse price movement between decision and fill, default 10
  bps.
* **transfer tax** - sell-side levy (e.g. SEC fee, stamp duty), default 0.

and one *liquidity* constraint:

* **volume cap** - a single rebalance may not trade more than ``volume_cap`` of
  a symbol's 20-day average daily (dollar) volume.  Oversized orders are
  truncated to the cap.

Cost arithmetic (weights are fractions of portfolio NAV)::

    position_change = weights_new - weights_prev
    buy_cost  = position_change.clip(lower=0)        * (commission + slippage)
    sell_cost = position_change.clip(upper=0).abs()  * (commission + slippage + tax)
    total_cost = buy_cost.sum() + sell_cost.sum()

Scope / limitations
-------------------
Slippage here is a *constant* per-unit-traded rate, independent of order size,
volatility, or spread.  That is a reasonable first-order model for liquid names
traded well within the ADV cap, but it understates the cost of large or urgent
orders.  For size- and volatility-dependent impact, use the Almgren-Chriss model
in ``quantcortex/backtest/execution_models/market_impact.py`` with the event-driven engine
(and set this model's ``slippage`` to 0 there to avoid double-counting).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["TransactionCostModel", "CostResult", "apply_costs"]

# Defaults from the platform spec.
DEFAULT_COMMISSION = 0.0003  # 3 bps
DEFAULT_SLIPPAGE = 0.0010  # 10 bps
DEFAULT_TAX = 0.0  # sell-side transfer tax / regulatory fee
DEFAULT_VOLUME_CAP = 0.10  # max 10% of 20-day ADV per symbol


@dataclass
class CostResult:
    """Structured result of applying the transaction cost model."""

    desired_change: np.ndarray  # weights_new - weights_prev (pre-cap)
    executed_change: np.ndarray  # change actually executed after ADV capping
    executed_weights: np.ndarray  # weights_prev + executed_change
    buy_cost: np.ndarray  # per-asset buy-side cost
    sell_cost: np.ndarray  # per-asset sell-side cost
    capped: np.ndarray  # per-asset bool: was the order truncated?
    total_cost: float  # scalar cost as a fraction of NAV
    net_return: Optional[float] = field(default=None)

    @property
    def per_asset_cost(self) -> np.ndarray:
        return self.buy_cost + self.sell_cost

    @property
    def turnover(self) -> float:
        """One-way turnover actually executed.

        This is the larger of buy and sell notional. It equals half-L1
        turnover for a fully invested rotation, while correctly reporting a
        100% cash-to-invested transition as 100% rather than 50%.
        """
        buys = float(np.clip(self.executed_change, 0.0, None).sum())
        sells = float(np.abs(np.clip(self.executed_change, None, 0.0)).sum())
        return max(buys, sells)

    @property
    def traded_notional(self) -> float:
        """Gross two-sided traded notional as a fraction of pre-trade NAV.

        Unlike one-way turnover, this sums buys and sells. It is therefore the
        denominator that reconciles directly to a symmetric per-dollar cost
        rate: ``total_cost == traded_notional * rate`` when buy and sell rates
        are equal.
        """
        return float(np.abs(self.executed_change).sum())


class TransactionCostModel:
    """Commission + slippage + tax cost model with an ADV liquidity cap."""

    def __init__(
        self,
        commission: float = DEFAULT_COMMISSION,
        slippage: float = DEFAULT_SLIPPAGE,
        tax: float = DEFAULT_TAX,
        volume_cap: float = DEFAULT_VOLUME_CAP,
    ) -> None:
        if any(
            isinstance(value, (bool, np.bool_))
            for value in (commission, slippage, tax, volume_cap)
        ):
            raise TypeError("cost rates and volume_cap must be numeric, not boolean")
        rates = np.asarray([commission, slippage, tax], dtype=np.float64)
        if not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
            raise ValueError("Cost rates must be finite and non-negative.")
        if commission + slippage >= 1.0 or commission + slippage + tax >= 1.0:
            raise ValueError("combined buy and sell cost rates must be below 100%")
        volume_cap = float(volume_cap)
        if not np.isfinite(volume_cap) or not (0.0 < volume_cap <= 1.0):
            raise ValueError("volume_cap must be in (0, 1].")
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.tax = float(tax)
        self.volume_cap = float(volume_cap)

    @property
    def buy_rate(self) -> float:
        return self.commission + self.slippage

    @property
    def sell_rate(self) -> float:
        return self.commission + self.slippage + self.tax

    def _dollar_adv(self, adv, prices) -> Optional[np.ndarray]:
        """Resolve ADV to *dollar* volume.

        If ``prices`` is supplied, ``adv`` is interpreted as *share* volume and
        converted; otherwise ``adv`` is taken to already be dollar volume.
        """
        if adv is None:
            return None
        adv_arr = np.asarray(adv, dtype=np.float64)
        if adv_arr.ndim != 1:
            raise ValueError(f"adv must be a 1-D vector, got shape {adv_arr.shape}")
        if not np.all(np.isfinite(adv_arr)) or np.any(adv_arr < 0.0):
            raise ValueError("adv must contain finite, non-negative values")
        if prices is not None:
            price_arr = np.asarray(prices, dtype=np.float64)
            if price_arr.shape != adv_arr.shape:
                raise ValueError(
                    f"price shape {price_arr.shape} != adv shape {adv_arr.shape}"
                )
            if not np.all(np.isfinite(price_arr)) or np.any(price_arr <= 0.0):
                raise ValueError("prices must contain finite, positive values")
            adv_arr = adv_arr * price_arr
        return adv_arr

    def apply_costs(
        self,
        weights_prev,
        weights_new,
        prices=None,
        adv=None,
        *,
        capital: float = 1.0,
        gross_returns=None,
        max_gross: Optional[float] = None,
    ) -> CostResult:
        """Apply the cost model to a single rebalance.

        Parameters
        ----------
        weights_prev, weights_new:
            Pre- and post-rebalance weight vectors (fractions of NAV).
        prices:
            Optional per-asset prices.  When given, ``adv`` is read as share
            volume and converted to dollar ADV.
        adv:
            Average daily volume per asset.  Dollar volume unless ``prices`` is
            also supplied (then share volume).  ``None`` disables the cap.
        capital:
            Portfolio NAV in the same currency as the (dollar) ADV.  Used to
            translate weight changes into traded notional for the cap.
        gross_returns:
            Optional per-asset (or scalar) returns realised over the period.
            When supplied, ``CostResult.net_return`` is populated with the
            portfolio return of ``executed_weights`` net of ``total_cost``.
        max_gross:
            Optional post-trade gross-exposure limit. If independent ADV caps
            would leave an over-gross intermediate book (for example, a sell
            is capped but its replacement buy is not), position-reducing legs
            execute first and exposure-increasing legs are scaled to the limit.

        Returns
        -------
        CostResult
        """
        w_prev = np.asarray(weights_prev, dtype=np.float64)
        w_new = np.asarray(weights_new, dtype=np.float64)
        if w_prev.ndim != 1 or w_new.ndim != 1 or w_prev.size == 0:
            raise ValueError("weights must be non-empty 1-D vectors")
        if w_prev.shape != w_new.shape:
            raise ValueError(
                f"weight shape mismatch: {w_prev.shape} vs {w_new.shape}"
            )
        if not np.all(np.isfinite(w_prev)) or not np.all(np.isfinite(w_new)):
            raise ValueError("weights must contain only finite values")
        if max_gross is not None:
            if isinstance(max_gross, (bool, np.bool_)):
                raise TypeError("max_gross must be numeric, not boolean")
            max_gross = float(max_gross)
            if not np.isfinite(max_gross) or max_gross <= 0.0:
                raise ValueError("max_gross must be finite and positive")
        if isinstance(capital, (bool, np.bool_)):
            raise TypeError("capital must be numeric, not boolean")
        capital = float(capital)
        if not np.isfinite(capital) or capital <= 0.0:
            raise ValueError("capital must be finite and positive")

        desired = w_new - w_prev
        dollar_adv = self._dollar_adv(adv, prices)

        if dollar_adv is not None:
            if dollar_adv.shape != desired.shape:
                raise ValueError(
                    f"adv shape {dollar_adv.shape} != weights {desired.shape}"
                )
            desired_notional = np.abs(desired) * capital
            max_notional = self.volume_cap * dollar_adv
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.where(
                    desired_notional > 0,
                    np.minimum(desired_notional, max_notional) / desired_notional,
                    1.0,
                )
            if not np.all(np.isfinite(scale)):
                raise ValueError("ADV cap produced a non-finite execution scale")
            scale = np.clip(scale, 0.0, 1.0)
            executed = desired * scale
        else:
            executed = desired.copy()

        if max_gross is not None:
            # Execute position-reducing legs first. Scaling the entire trade
            # vector can otherwise cancel a feasible sell merely because its
            # replacement buy is too large, preserving more risk than needed.
            reducing = np.zeros_like(executed)
            long_reduction = (w_prev > 0.0) & (executed < 0.0)
            short_reduction = (w_prev < 0.0) & (executed > 0.0)
            reducing[long_reduction] = np.maximum(
                executed[long_reduction], -w_prev[long_reduction]
            )
            reducing[short_reduction] = np.minimum(
                executed[short_reduction], -w_prev[short_reduction]
            )
            increasing = executed - reducing
            reduced_weights = w_prev + reducing
            reduced_gross = float(np.abs(reduced_weights).sum())

            if reduced_gross > max_gross + 1e-12:
                # ADV caps prevented enough liquidation to restore the limit.
                # Keep every feasible reduction and open no new exposure.
                executed = reducing
            else:
                candidate_gross = float(
                    np.abs(reduced_weights + increasing).sum()
                )
                if candidate_gross > max_gross + 1e-12:
                    # Gross is monotone along this path because `increasing`
                    # only adds exposure after reductions/crossings to zero.
                    low, high = 0.0, 1.0
                    for _ in range(64):
                        mid = (low + high) / 2.0
                        gross = float(
                            np.abs(reduced_weights + mid * increasing).sum()
                        )
                        if gross <= max_gross:
                            low = mid
                        else:
                            high = mid
                    increasing = increasing * low
                executed = reducing + increasing

        capped = ~np.isclose(executed, desired, rtol=1e-10, atol=1e-12)

        buy_cost = np.clip(executed, 0.0, None) * self.buy_rate
        sell_cost = np.abs(np.clip(executed, None, 0.0)) * self.sell_rate
        total_cost = float(buy_cost.sum() + sell_cost.sum())

        executed_weights = w_prev + executed

        net_return: Optional[float] = None
        if gross_returns is not None:
            gr = np.asarray(gross_returns, dtype=np.float64)
            if gr.ndim > 1 or (gr.ndim == 1 and gr.shape != executed_weights.shape):
                raise ValueError(
                    "gross_returns must be scalar or match the weight vector shape"
                )
            if not np.all(np.isfinite(gr)):
                raise ValueError("gross_returns must contain only finite values")
            gross_port = (
                float((executed_weights * gr).sum())
                if gr.ndim
                else float(executed_weights.sum() * gr)
            )
            net_return = gross_port - total_cost

        return CostResult(
            desired_change=desired,
            executed_change=executed,
            executed_weights=executed_weights,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
            capped=capped,
            total_cost=total_cost,
            net_return=net_return,
        )

    # Lightweight scalar helper used by vectorized engines.
    def cost_of_turnover(self, weights_prev, weights_new) -> float:
        """Return just the total cost fraction for a rebalance (no ADV cap)."""
        return self.apply_costs(weights_prev, weights_new).total_cost


def apply_costs(
    weights_prev,
    weights_new,
    prices=None,
    adv=None,
    *,
    model: Optional[TransactionCostModel] = None,
    **kwargs,
) -> CostResult:
    """Module-level convenience wrapper around :class:`TransactionCostModel`."""
    model = model or TransactionCostModel()
    return model.apply_costs(weights_prev, weights_new, prices, adv, **kwargs)
