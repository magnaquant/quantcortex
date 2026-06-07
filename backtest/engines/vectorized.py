"""Vectorized portfolio backtest engine.

:class:`VectorizedBacktest` runs a target-weight strategy over a daily price
panel using fast, fully vectorized array arithmetic.  It is the workhorse for
research-scale sweeps where bar-by-bar fill modelling is unnecessary.

The engine is **strictly causal**: the weights decided at the close of day *t*
are applied to the asset return realised from *t* to *t+1* (weights are lagged
one bar before being multiplied by forward returns), so no information from the
future leaks into a period's return.

Transaction costs are **mandatory**: a :class:`TransactionCostModel` must be
supplied, and the constructor raises :class:`ValueError` if it is ``None``.
Costs are charged on every rebalance via ``cost_model.apply_costs`` and
subtracted from that period's gross return.

This module also defines :class:`BacktestResult`, the common result container
reused by the event-driven engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from backtest.costs.transaction_costs import TransactionCostModel

__all__ = ["BacktestResult", "VectorizedBacktest"]


@dataclass
class BacktestResult:
    """Container for the output of a backtest run.

    Attributes
    ----------
    returns:
        Net (after-cost) per-period portfolio returns.
    equity_curve:
        Cumulative NAV: ``(1 + returns).cumprod() * capital``.
    weights:
        Effective per-period portfolio weights (date x symbol).
    gross_returns:
        Per-period portfolio returns *before* transaction costs.
    costs:
        Per-period transaction cost (fraction of NAV).
    turnover:
        Per-period one-way turnover actually executed.
    metadata:
        Free-form dict of run parameters (capital, periods_per_year, ...).
    """

    returns: "pd.Series"
    equity_curve: "pd.Series"
    weights: "pd.DataFrame"
    gross_returns: "pd.Series"
    costs: "pd.Series"
    turnover: "pd.Series"
    metadata: dict = field(default_factory=dict)

    @property
    def total_return(self) -> float:
        """Total compounded net return over the whole backtest."""
        if len(self.returns) == 0:
            return 0.0
        return float((1.0 + self.returns).prod() - 1.0)

    def summary(self) -> dict:
        """Headline performance statistics.

        Returns a dict with ``total_return``, ``cagr``, ``ann_vol``,
        ``sharpe`` and ``max_drawdown``.
        """
        r = self.returns.dropna()
        ppy = float(self.metadata.get("periods_per_year", 252))
        n = len(r)
        if n == 0:
            return {
                "total_return": 0.0,
                "cagr": 0.0,
                "ann_vol": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
            }

        total_return = float((1.0 + r).prod() - 1.0)
        years = n / ppy
        if years > 0 and (1.0 + total_return) > 0:
            cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0)
        else:
            cagr = 0.0

        std = float(r.std(ddof=1)) if n > 1 else 0.0
        ann_vol = std * np.sqrt(ppy)
        mean = float(r.mean())
        sharpe = (mean / std) * np.sqrt(ppy) if std > 0 else 0.0

        eq = self.equity_curve.dropna()
        if len(eq) == 0:
            max_dd = 0.0
        else:
            running_max = eq.cummax()
            drawdown = eq / running_max - 1.0
            max_dd = float(drawdown.min())

        return {
            "total_return": total_return,
            "cagr": cagr,
            "ann_vol": ann_vol,
            "sharpe": float(sharpe),
            "max_drawdown": max_dd,
        }


class VectorizedBacktest:
    """Fast vectorized target-weight backtest with mandatory cost modelling.

    Parameters
    ----------
    cost_model:
        A :class:`TransactionCostModel`.  **Required** -- ``None`` raises
        :class:`ValueError` (transaction costs are non-optional in quantcortex).
    capital:
        Starting NAV.  Default ``1e6``.
    periods_per_year:
        Annualisation factor for summary statistics.  Default ``252``.
    """

    def __init__(
        self,
        cost_model: TransactionCostModel,
        *,
        capital: float = 1e6,
        periods_per_year: int = 252,
    ) -> None:
        if cost_model is None:
            raise ValueError(
                "a TransactionCostModel is required: transaction costs are "
                "mandatory in quantcortex backtests."
            )
        self.cost_model = cost_model
        self.capital = float(capital)
        self.periods_per_year = int(periods_per_year)

    def run(
        self,
        weights: "pd.DataFrame",
        prices: "pd.DataFrame",
        adv: Optional["pd.DataFrame"] = None,
    ) -> BacktestResult:
        """Run the backtest.

        Parameters
        ----------
        weights:
            Target weights (fractions of NAV) indexed by rebalance date and
            columns by symbol.  Need not be on the daily grid -- it is
            forward-filled onto the price index between rebalances.
        prices:
            Daily close prices (date x symbol).
        adv:
            Optional average daily volume panel (date x symbol) for the ADV
            liquidity cap.  ``None`` disables the cap.

        Returns
        -------
        BacktestResult
        """
        if prices.empty:
            raise ValueError("prices is empty.")

        prices = prices.sort_index()
        symbols = list(prices.columns)

        # Align target weights onto the price grid and forward-fill between
        # rebalances; assets without a target weight are flat (0).
        rebalance_dates = weights.index
        target = (
            weights.reindex(columns=symbols)
            .reindex(prices.index)
            .ffill()
            .fillna(0.0)
        )

        # Forward simple returns; return at row t is asset move t-1 -> t.
        asset_returns = prices.pct_change().fillna(0.0)

        # CAUSAL: weights decided at close of t apply to the t -> t+1 return.
        # Lag the target weights by one bar before multiplying by the return.
        lagged_weights = target.shift(1).fillna(0.0)
        gross = (lagged_weights * asset_returns).sum(axis=1)

        # Determine which dates are genuine rebalances (target changes) so we
        # charge costs only when the portfolio is actually re-traded.
        rebalance_mask = pd.Series(False, index=prices.index)
        for d in rebalance_dates:
            # snap each rebalance date to the first price bar on/after it
            pos = prices.index.searchsorted(d, side="left")
            if pos < len(prices.index):
                rebalance_mask.iloc[pos] = True
        # Always treat the first achievable target as a rebalance (build).
        first_nonzero = target.ne(0.0).any(axis=1)
        if first_nonzero.any():
            rebalance_mask.iloc[first_nonzero.values.argmax()] = True

        costs = pd.Series(0.0, index=prices.index)
        turnover = pd.Series(0.0, index=prices.index)
        effective_weights = pd.DataFrame(
            0.0, index=prices.index, columns=symbols
        )

        prev_w = np.zeros(len(symbols), dtype=np.float64)
        target_vals = target.to_numpy(dtype=np.float64)
        price_vals = prices.to_numpy(dtype=np.float64)
        adv_aligned = None
        if adv is not None:
            adv_aligned = (
                adv.reindex(columns=symbols).reindex(prices.index).ffill()
            ).to_numpy(dtype=np.float64)

        for i, dt in enumerate(prices.index):
            new_w = target_vals[i]
            if rebalance_mask.iloc[i]:
                row_adv = None if adv_aligned is None else adv_aligned[i]
                row_px = price_vals[i]
                result = self.cost_model.apply_costs(
                    prev_w,
                    new_w,
                    prices=row_px,
                    adv=row_adv,
                    capital=self.capital,
                )
                costs.iloc[i] = result.total_cost
                turnover.iloc[i] = result.turnover
                prev_w = result.executed_weights
            effective_weights.iloc[i] = prev_w

        net = gross - costs
        equity_curve = (1.0 + net).cumprod() * self.capital

        metadata = {
            "capital": self.capital,
            "periods_per_year": self.periods_per_year,
            "engine": "vectorized",
            "n_periods": int(len(net)),
            "n_rebalances": int(rebalance_mask.sum()),
        }

        return BacktestResult(
            returns=net,
            equity_curve=equity_curve,
            weights=effective_weights,
            gross_returns=gross,
            costs=costs,
            turnover=turnover,
            metadata=metadata,
        )
