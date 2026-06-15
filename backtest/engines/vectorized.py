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

    Modelling assumption
    --------------------
    Between rebalances the portfolio is implicitly **re-pegged to its executed
    target weights every bar at zero cost**: each period's gross return is
    ``sum(executed_weights * asset_returns)`` with the same weight vector for
    every bar of the segment, i.e. there is no drift of the holdings with
    relative price moves and no cost for the implied daily re-pegging trades.
    This differs from the event-driven engine
    (:class:`~backtest.engines.event_driven.EventDrivenBacktest`), which holds
    explicit share positions that drift between rebalances.

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
            columns by symbol.  Need not be on the daily grid -- each rebalance
            date is snapped to the first price bar on or after it (e.g. a
            weekend-dated rebalance executes on the next Monday bar).  When
            several rebalance dates snap to the same bar, the last stated
            target wins.
        prices:
            Daily close prices (date x symbol).
        adv:
            Optional average daily volume panel (date x symbol) for the ADV
            liquidity cap.  ``None`` disables the cap.  When the cap binds,
            the engine earns returns on the *executed* (capped) weights, and
            the cap notional is measured against the current running NAV.

        Returns
        -------
        BacktestResult
        """
        if prices.empty:
            raise ValueError("prices is empty.")

        prices = prices.sort_index()
        symbols = list(prices.columns)
        n = len(prices.index)
        n_sym = len(symbols)

        # SNAP each weight-panel date to the first price bar on/after it.
        # Dates past the last price bar are dropped; duplicate snaps keep the
        # LAST target stated for that bar.
        w = weights.sort_index().reindex(columns=symbols)
        if len(w.index) == 0:
            # Empty weights -> a fully flat (all-cash) backtest. Coerce the
            # index to the price dtype so searchsorted does not choke on an
            # empty int64 RangeIndex.
            w.index = prices.index[:0]
        pos = prices.index.searchsorted(w.index, side="left")
        keep = pos < n
        w = w.iloc[keep]
        w.index = prices.index[pos[keep]]
        w = w[~w.index.duplicated(keep="last")]
        # Forward-fill partially specified rows across rebalances, then treat
        # never-specified assets as flat (0).
        w = w.ffill().fillna(0.0)

        reb_positions = prices.index.get_indexer(w.index)
        target_rows = w.to_numpy(dtype=np.float64)

        # Forward simple returns; return at row t is asset move t-1 -> t.
        asset_returns = prices.pct_change().fillna(0.0)
        ret_vals = asset_returns.to_numpy(dtype=np.float64)
        price_vals = prices.to_numpy(dtype=np.float64)
        adv_aligned = None
        if adv is not None:
            adv_aligned = (
                adv.reindex(columns=symbols).reindex(prices.index).ffill()
            ).to_numpy(dtype=np.float64)

        gross_arr = np.zeros(n, dtype=np.float64)
        costs_arr = np.zeros(n, dtype=np.float64)
        turnover_arr = np.zeros(n, dtype=np.float64)
        eff = np.zeros((n, n_sym), dtype=np.float64)

        # Iterate rebalances chronologically, maintaining the EXECUTED weight
        # vector (what the cost model says was actually traded, e.g. after the
        # ADV cap) and the running NAV.  CAUSAL: weights executed at the close
        # of bar p earn returns from bar p+1 onward (one-period lag).
        current = np.zeros(n_sym, dtype=np.float64)
        nav = self.capital
        start = 0
        for k, p in enumerate(reb_positions):
            # Bars [start, p] earn returns on the previously executed weights.
            seg = ret_vals[start : p + 1] @ current
            gross_arr[start : p + 1] = seg
            if len(seg) > 1:
                nav *= float(np.prod(1.0 + seg[:-1]))
            # Mark-to-market NAV at the close of the rebalance bar (pre-cost):
            # this is the capital base for the ADV liquidity cap.
            nav_mtm = nav * (1.0 + float(seg[-1]))
            row_adv = None if adv_aligned is None else adv_aligned[p]
            result = self.cost_model.apply_costs(
                current,
                target_rows[k],
                prices=price_vals[p],
                adv=row_adv,
                capital=nav_mtm,
            )
            costs_arr[p] = result.total_cost
            turnover_arr[p] = result.turnover
            eff[start:p] = current
            nav *= 1.0 + float(seg[-1]) - result.total_cost
            current = np.asarray(result.executed_weights, dtype=np.float64)
            eff[p] = current
            start = p + 1
        if start < n:
            seg = ret_vals[start:] @ current
            gross_arr[start:] = seg
            eff[start:] = current

        gross = pd.Series(gross_arr, index=prices.index)
        costs = pd.Series(costs_arr, index=prices.index)
        turnover = pd.Series(turnover_arr, index=prices.index)
        effective_weights = pd.DataFrame(
            eff, index=prices.index, columns=symbols
        )

        net = gross - costs
        equity_curve = (1.0 + net).cumprod() * self.capital

        metadata = {
            "capital": self.capital,
            "periods_per_year": self.periods_per_year,
            "engine": "vectorized",
            "n_periods": int(len(net)),
            "n_rebalances": int(len(reb_positions)),
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
