"""Transaction cost model — mandatory in every backtest engine.

quantcortex treats transaction costs as a *first-class*, non-optional input:
backtest engines refuse to run without a cost model (see ``backtest/engines``).
The model below implements the three frictions that dominate realistic equity
execution:

* **commission** — broker/exchange fee, default 3 bps.
* **slippage** — adverse price movement between decision and fill, default 10
  bps.
* **transfer tax** — sell-side levy (e.g. SEC fee, stamp duty), default 0.

and one *liquidity* constraint:

* **volume cap** — a single rebalance may not trade more than ``volume_cap`` of
  a symbol's 20-day average daily (dollar) volume.  Oversized orders are
  truncated to the cap.

Cost arithmetic (weights are fractions of portfolio NAV)::

    position_change = weights_new - weights_prev
    buy_cost  = position_change.clip(lower=0)        * (commission + slippage)
    sell_cost = position_change.clip(upper=0).abs()  * (commission + slippage + tax)
    total_cost = buy_cost.sum() + sell_cost.sum()
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
        """One-way turnover actually executed (sum of |Δw|)/2."""
        return float(np.abs(self.executed_change).sum() / 2.0)


class TransactionCostModel:
    """Commission + slippage + tax cost model with an ADV liquidity cap."""

    def __init__(
        self,
        commission: float = DEFAULT_COMMISSION,
        slippage: float = DEFAULT_SLIPPAGE,
        tax: float = DEFAULT_TAX,
        volume_cap: float = DEFAULT_VOLUME_CAP,
    ) -> None:
        if commission < 0 or slippage < 0 or tax < 0:
            raise ValueError("Cost rates must be non-negative.")
        if not (0.0 < volume_cap <= 1.0):
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
        if prices is not None:
            adv_arr = adv_arr * np.asarray(prices, dtype=np.float64)
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

        Returns
        -------
        CostResult
        """
        w_prev = np.asarray(weights_prev, dtype=np.float64)
        w_new = np.asarray(weights_new, dtype=np.float64)
        if w_prev.shape != w_new.shape:
            raise ValueError(
                f"weight shape mismatch: {w_prev.shape} vs {w_new.shape}"
            )

        desired = w_new - w_prev
        dollar_adv = self._dollar_adv(adv, prices)

        if dollar_adv is not None:
            if dollar_adv.shape != desired.shape:
                raise ValueError(
                    f"adv shape {dollar_adv.shape} != weights {desired.shape}"
                )
            desired_notional = np.abs(desired) * float(capital)
            max_notional = self.volume_cap * dollar_adv
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.where(
                    desired_notional > 0,
                    np.minimum(desired_notional, max_notional) / desired_notional,
                    1.0,
                )
            scale = np.clip(np.nan_to_num(scale, nan=1.0), 0.0, 1.0)
            executed = desired * scale
            capped = scale < 1.0 - 1e-12
        else:
            executed = desired.copy()
            capped = np.zeros_like(desired, dtype=bool)

        buy_cost = np.clip(executed, 0.0, None) * self.buy_rate
        sell_cost = np.abs(np.clip(executed, None, 0.0)) * self.sell_rate
        total_cost = float(buy_cost.sum() + sell_cost.sum())

        executed_weights = w_prev + executed

        net_return: Optional[float] = None
        if gross_returns is not None:
            gr = np.asarray(gross_returns, dtype=np.float64)
            gross_port = float((executed_weights * gr).sum()) if gr.ndim else float(
                executed_weights.sum() * gr
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
