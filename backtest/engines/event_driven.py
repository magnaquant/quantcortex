"""Event-driven (bar-by-bar) portfolio backtest engine.

:class:`EventDrivenBacktest` simulates the portfolio one bar at a time,
maintaining explicit **cash** and **share positions**.  On rebalance bars it
converts target weights into target share counts, prices the trades through an
:class:`~backtest.execution_models.ideal_fill.ExecutionModel` (so slippage /
market impact are modelled explicitly), charges transaction costs, and updates
positions.  Every bar is marked to market to build the equity curve.

Compared with :class:`~backtest.engines.vectorized.VectorizedBacktest` this
engine is **slower** but more faithful: it models the actual fill price of each
order (not just an aggregate cost), tracks cash drag, and naturally handles
position drift between rebalances.

It is **causal** -- trades execute using only the current bar's data (the fill
model may only read the current bar), and the resulting positions earn returns
from the current bar forward.

Transaction costs are **mandatory**: ``cost_model`` cannot be ``None``.  The
default execution model is the optimistic :class:`IdealFill`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from backtest.costs.transaction_costs import TransactionCostModel
from backtest.engines.vectorized import BacktestResult
from backtest.execution_models.ideal_fill import ExecutionModel, IdealFill

__all__ = ["EventDrivenBacktest"]


class EventDrivenBacktest:
    """Bar-by-bar backtest with explicit positions and fill modelling.

    Parameters
    ----------
    cost_model:
        A :class:`TransactionCostModel`.  **Required** -- ``None`` raises
        :class:`ValueError`.
    execution_model:
        How orders are priced.  Defaults to :class:`IdealFill` (fills at the
        bar close, zero slippage).
    capital:
        Starting cash NAV.  Default ``1e6``.
    periods_per_year:
        Annualisation factor for summary statistics.  Default ``252``.
    """

    def __init__(
        self,
        cost_model: TransactionCostModel,
        execution_model: Optional[ExecutionModel] = None,
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
        self.execution_model = execution_model if execution_model is not None else IdealFill()
        self.capital = float(capital)
        self.periods_per_year = int(periods_per_year)

    def run(
        self,
        weights: "pd.DataFrame",
        prices: "pd.DataFrame",
        adv: Optional["pd.DataFrame"] = None,
    ) -> BacktestResult:
        """Run the event-driven simulation.

        Parameters
        ----------
        weights:
            Target weights (date x symbol) at rebalance dates.  Forward-filled
            onto the price grid; rebalances occur on the first price bar on or
            after each weight date.
        prices:
            Daily close prices (date x symbol).  May also carry OHLCV columns
            in a MultiIndex/dict form for richer fill models, but a plain close
            panel is sufficient (the fill model receives a bar with ``close``).
        adv:
            Optional ADV panel (date x symbol) for the cost model's liquidity
            cap and as volume context for the execution model.

        Returns
        -------
        BacktestResult
        """
        if prices.empty:
            raise ValueError("prices is empty.")

        prices = prices.sort_index()
        symbols = list(prices.columns)
        n_sym = len(symbols)
        index = prices.index

        # Snap each rebalance date to the first price bar on/after it.
        rebalance_bars: set[int] = set()
        for d in weights.index:
            pos = index.searchsorted(d, side="left")
            if pos < len(index):
                rebalance_bars.add(int(pos))

        target = (
            weights.reindex(columns=symbols).reindex(index).ffill().fillna(0.0)
        )
        target_vals = target.to_numpy(dtype=np.float64)
        price_vals = prices.to_numpy(dtype=np.float64)
        adv_aligned = None
        if adv is not None:
            adv_aligned = (
                adv.reindex(columns=symbols).reindex(index).ffill()
            ).to_numpy(dtype=np.float64)

        cash = self.capital
        shares = np.zeros(n_sym, dtype=np.float64)

        equity = np.empty(len(index), dtype=np.float64)
        costs = np.zeros(len(index), dtype=np.float64)
        turnover = np.zeros(len(index), dtype=np.float64)
        eff_weights = np.zeros((len(index), n_sym), dtype=np.float64)

        for i, dt in enumerate(index):
            px = price_vals[i]
            # Pre-trade NAV (mark current holdings at this bar's close).
            nav = cash + float(np.dot(shares, px))

            if i in rebalance_bars and nav > 0:
                prev_w = (shares * px) / nav
                new_w = target_vals[i]

                # Cost model decides what can actually be executed (ADV cap).
                row_adv = None if adv_aligned is None else adv_aligned[i]
                cost_res = self.cost_model.apply_costs(
                    prev_w,
                    new_w,
                    prices=px,
                    adv=row_adv,
                    capital=nav,
                )
                target_shares = np.where(
                    px > 0, cost_res.executed_weights * nav / px, shares
                )
                delta_shares = target_shares - shares

                # Price each order through the execution model (slippage), then
                # settle cash at the realised fill price.
                trade_cash = 0.0
                for j in range(n_sym):
                    dq = delta_shares[j]
                    if dq == 0.0:
                        continue
                    bar = pd.Series({"close": px[j]}, name=dt)
                    if row_adv is not None:
                        bar["volume"] = row_adv[j]
                    fill_px = self.execution_model.fill(
                        symbols[j], float(dq), bar, adv=None if row_adv is None else row_adv[j]
                    )
                    trade_cash += dq * fill_px  # buys cost cash, sells add cash

                # Explicit commission/slippage/tax charge from the cost model.
                cost_cash = cost_res.total_cost * nav

                cash -= trade_cash
                cash -= cost_cash
                shares = target_shares

                costs[i] = cost_res.total_cost
                turnover[i] = cost_res.turnover

            post_nav = cash + float(np.dot(shares, px))
            equity[i] = post_nav
            if post_nav > 0:
                eff_weights[i] = (shares * px) / post_nav

        equity_curve = pd.Series(equity, index=index)
        net = equity_curve.pct_change().fillna(0.0)
        # First bar's return relative to starting capital.
        net.iloc[0] = equity[0] / self.capital - 1.0
        cost_series = pd.Series(costs, index=index)
        gross = net + cost_series  # net + cost back-out as an approximation

        metadata = {
            "capital": self.capital,
            "periods_per_year": self.periods_per_year,
            "engine": "event_driven",
            "execution_model": repr(self.execution_model),
            "n_periods": int(len(index)),
            "n_rebalances": len(rebalance_bars),
        }

        return BacktestResult(
            returns=net,
            equity_curve=equity_curve,
            weights=pd.DataFrame(eff_weights, index=index, columns=symbols),
            gross_returns=gross,
            costs=cost_series,
            turnover=pd.Series(turnover, index=index),
            metadata=metadata,
        )
