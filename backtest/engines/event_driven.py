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

import warnings
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

    Notes
    -----
    **Slippage double-count.**  Slippage can be modelled in two places: the
    cost model's ``slippage`` rate (an aggregate per-notional charge) and the
    execution model's fill-price perturbation.  If a non-:class:`IdealFill`
    execution model is supplied while ``cost_model.slippage > 0``, slippage is
    charged *twice*.  The engine emits a :class:`UserWarning` at construction
    in that case; set ``slippage=0`` in the cost model when using an explicit
    fill model.

    **Cost-aware sizing.**  Target share counts are sized off the investable
    NAV net of the *estimated* rebalance cost so the post-trade cash balance
    cannot go (materially) negative.  The estimate uses the cost model's
    charge for the full target trade, while the scaled-down trades actually
    executed cost slightly less, so a few dollars of residual cash typically
    remain; execution-model price impact on the fills is not part of the
    estimate and can still nudge cash slightly negative when impact is large.
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

        # Warn (once, at construction) about the slippage double-count: a
        # fill-perturbing execution model AND a positive cost-model slippage
        # rate both charge slippage on the same trades.
        if (
            not isinstance(self.execution_model, IdealFill)
            and getattr(self.cost_model, "slippage", 0.0) > 0
        ):
            warnings.warn(
                "EventDrivenBacktest: the supplied execution model perturbs "
                "fill prices AND the cost model charges slippage of "
                f"{self.cost_model.slippage} per unit notional -- slippage "
                "will be double-counted. Set slippage=0 in the cost model "
                "when using an explicit fill model.",
                UserWarning,
                stacklevel=2,
            )

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

        # SNAP each weight-panel date to the first price bar on/after it so
        # off-grid (e.g. weekend) rebalances execute on the next bar instead
        # of being silently dropped by reindex+ffill.  Dates past the last
        # bar are dropped; duplicate snaps keep the LAST target for that bar.
        snapped = weights.sort_index().reindex(columns=symbols)
        if len(snapped.index) == 0:
            # Empty weights -> a fully flat (all-cash) run. Coerce the index to
            # the price dtype so searchsorted does not choke on an empty
            # int64 RangeIndex.
            snapped.index = index[:0]
        pos = index.searchsorted(snapped.index, side="left")
        keep = pos < len(index)
        snapped = snapped.iloc[keep]
        snapped.index = index[pos[keep]]
        snapped = snapped[~snapped.index.duplicated(keep="last")]
        rebalance_bars: set[int] = {
            int(p) for p in index.get_indexer(snapped.index)
        }

        target = snapped.reindex(index).ffill().fillna(0.0)
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
                # Size targets off the investable NAV net of the estimated
                # rebalance cost so the post-trade cash cannot go negative
                # (no free implicit financing of the cost charge).  Residual
                # approximation: the estimate prices the full target trade,
                # while the scaled (slightly smaller) trades cost marginally
                # less, leaving a small positive cash remainder.
                est_cost_cash = cost_res.total_cost * nav
                investable = max(nav - est_cost_cash, 0.0)
                target_shares = np.where(
                    px > 0, cost_res.executed_weights * investable / px, shares
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
