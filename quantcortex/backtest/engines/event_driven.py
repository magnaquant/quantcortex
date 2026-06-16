"""Event-driven (bar-by-bar) portfolio backtest engine.

:class:`EventDrivenBacktest` simulates the portfolio one bar at a time,
maintaining explicit **cash** and **share positions**.  On rebalance bars it
converts target weights into target share counts, prices the trades through an
:class:`~quantcortex.backtest.execution_models.ideal_fill.ExecutionModel` (so slippage /
market impact are modelled explicitly), charges transaction costs, and updates
positions.  Every bar is marked to market to build the equity curve.

Compared with :class:`~quantcortex.backtest.engines.vectorized.VectorizedBacktest` this
engine is **slower** but more faithful: it models the actual fill price of each
order (not just an aggregate cost), tracks cash drag, and naturally handles
position drift between rebalances.

It is **causal** under the same convention as the vectorized engine: a target
dated on bar ``t`` is a close-of-bar decision and executes on the first bar
strictly after ``t``. The resulting position earns returns after that fill.

Transaction costs are **mandatory**: ``cost_model`` cannot be ``None``.  The
default execution model is the optimistic :class:`IdealFill`.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from quantcortex.backtest.costs.transaction_costs import TransactionCostModel
from quantcortex.backtest.engines.cash import align_cash_returns
from quantcortex.backtest.engines.vectorized import BacktestResult
from quantcortex.backtest.execution_models.ideal_fill import ExecutionModel, IdealFill
from quantcortex.portfolio.base import enforce_exposure_contract

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
    remain. Execution-model price impact is not part of the estimate; if it
    would require negative cash for an unlevered long-only target, the engine
    fails instead of silently financing the trade.
    """

    def __init__(
        self,
        cost_model: TransactionCostModel,
        execution_model: Optional[ExecutionModel] = None,
        *,
        capital: float = 1e6,
        periods_per_year: int = 252,
        max_gross: float = 1.0,
    ) -> None:
        if cost_model is None:
            raise ValueError(
                "a TransactionCostModel is required: transaction costs are "
                "mandatory in quantcortex backtests."
            )
        self.cost_model = cost_model
        self.execution_model = (
            execution_model if execution_model is not None else IdealFill()
        )
        if isinstance(capital, (bool, np.bool_)):
            raise TypeError("capital must be numeric, not boolean")
        if isinstance(max_gross, (bool, np.bool_)):
            raise TypeError("max_gross must be numeric, not boolean")
        self.capital = float(capital)
        if (
            isinstance(periods_per_year, (bool, np.bool_))
            or not isinstance(periods_per_year, (int, np.integer))
            or periods_per_year <= 0
        ):
            raise ValueError("periods_per_year must be a positive integer")
        self.periods_per_year = int(periods_per_year)
        self.max_gross = float(max_gross)
        if not np.isfinite(self.capital) or self.capital <= 0.0:
            raise ValueError("capital must be finite and positive")
        if not np.isfinite(self.max_gross) or self.max_gross <= 0.0:
            raise ValueError("max_gross must be finite and positive")

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
        cash_returns: Optional["pd.Series"] = None,
    ) -> BacktestResult:
        """Run the event-driven simulation.

        Parameters
        ----------
        weights:
            Close-of-bar target weights (date x symbol) at decision dates.
            Rebalances occur on the first price bar strictly after an on-grid
            decision date, or on the next available bar after an off-grid date.
        prices:
            Daily close prices (date x symbol). The fill model receives a bar
            containing ``close`` and, when ``adv`` is supplied, ``volume``.
        adv:
            Optional ADV panel (date x symbol) for the cost model's liquidity
            cap and as volume context for the execution model.
        cash_returns:
            Optional per-period simple return earned by the explicit cash
            balance. ``None`` means zero cash return.

        Returns
        -------
        BacktestResult
        """
        if prices.empty:
            raise ValueError("prices is empty.")

        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("prices must use a DatetimeIndex")
        if prices.index.hasnans:
            raise ValueError("prices index must contain valid timestamps")
        prices = prices.copy()
        if prices.index.tz is not None:
            prices.index = prices.index.tz_convert("UTC").tz_localize(None)
        prices = prices.sort_index()
        if prices.index.has_duplicates:
            raise ValueError("prices index must not contain duplicate bars")
        if prices.columns.has_duplicates or prices.shape[1] == 0:
            raise ValueError("prices must have unique, non-empty symbol columns")
        prices = prices.apply(pd.to_numeric, errors="coerce")
        if not np.all(np.isfinite(prices.to_numpy(dtype=float))):
            raise ValueError("prices must contain only finite values")
        if (prices <= 0.0).any().any():
            raise ValueError("prices must be strictly positive")
        symbols = list(prices.columns)
        n_sym = len(symbols)
        index = prices.index

        # A weight dated on a price bar is a close-of-bar decision, so execute
        # on the first bar STRICTLY after it. Off-grid dates (for example a
        # Sunday) still execute on the next available bar. Duplicate snaps keep
        # the last target for that execution bar.
        if not isinstance(weights, pd.DataFrame):
            raise TypeError("weights must be a pandas DataFrame")
        if not isinstance(weights.index, pd.DatetimeIndex):
            if weights.empty:
                weights = weights.copy()
                weights.index = index[:0]
            else:
                raise TypeError("weights must use a DatetimeIndex")
        if weights.index.hasnans:
            raise ValueError("weights index must contain valid timestamps")
        if weights.index.has_duplicates:
            raise ValueError("weights index must not contain duplicate decisions")
        if weights.columns.has_duplicates:
            raise ValueError("weights columns must be unique")
        unknown = [column for column in weights.columns if column not in symbols]
        if unknown:
            raise ValueError(f"weights contain unknown symbols: {unknown}")
        weights = weights.copy()
        if weights.index.tz is not None:
            weights.index = weights.index.tz_convert("UTC").tz_localize(None)
        supplied_symbols = list(weights.columns)
        snapped = weights.sort_index().reindex(columns=symbols)
        missing_symbols = [symbol for symbol in symbols if symbol not in supplied_symbols]
        if missing_symbols:
            snapped.loc[:, missing_symbols] = 0.0
        snapped = snapped.apply(pd.to_numeric, errors="coerce")
        if snapped[supplied_symbols].isna().any(axis=None):
            raise ValueError(
                "each target row must explicitly specify every supplied symbol"
            )
        if len(snapped.index) == 0:
            # Empty weights -> a fully flat (all-cash) run. Coerce the index to
            # the price dtype so searchsorted does not choke on an empty
            # int64 RangeIndex.
            snapped.index = index[:0]
        pos = index.searchsorted(snapped.index, side="right")
        keep = pos < len(index)
        snapped = snapped.iloc[keep]
        snapped.index = index[pos[keep]]
        snapped = snapped[~snapped.index.duplicated(keep="last")]
        rebalance_bars: set[int] = {
            int(p) for p in index.get_indexer(snapped.index)
        }

        target = snapped.reindex(index).ffill().fillna(0.0)
        target_vals = target.to_numpy(dtype=np.float64, copy=True)
        if not np.all(np.isfinite(target_vals)):
            raise ValueError("weights must contain only finite values")
        for position, row in enumerate(target_vals):
            target_vals[position] = enforce_exposure_contract(
                row,
                max_gross=self.max_gross,
                name=f"EventDrivenBacktest target row {position}",
            )
        price_vals = prices.to_numpy(dtype=np.float64)
        cash_return_series = align_cash_returns(cash_returns, index)
        cash_ret_vals = cash_return_series.to_numpy(dtype=np.float64)
        adv_aligned = None
        if adv is not None:
            if not isinstance(adv, pd.DataFrame):
                raise TypeError("adv must be a pandas DataFrame")
            if not isinstance(adv.index, pd.DatetimeIndex):
                raise TypeError("adv must use a DatetimeIndex")
            if adv.index.hasnans or adv.index.has_duplicates:
                raise ValueError("adv index must contain unique valid timestamps")
            if adv.columns.has_duplicates:
                raise ValueError("adv columns must be unique")
            adv = adv.copy()
            if adv.index.tz is not None:
                adv.index = adv.index.tz_convert("UTC").tz_localize(None)
            adv = adv.sort_index().apply(pd.to_numeric, errors="coerce")
            adv_aligned = adv.reindex(columns=symbols).reindex(index).to_numpy(
                dtype=np.float64
            )

        cash = self.capital
        shares = np.zeros(n_sym, dtype=np.float64)

        equity = np.empty(len(index), dtype=np.float64)
        gross_equity = np.empty(len(index), dtype=np.float64)
        asset_contribution = np.zeros(len(index), dtype=np.float64)
        cash_contribution = np.zeros(len(index), dtype=np.float64)
        turnover = np.zeros(len(index), dtype=np.float64)
        eff_weights = np.zeros((len(index), n_sym), dtype=np.float64)

        for i, dt in enumerate(index):
            px = price_vals[i]
            previous_equity = self.capital if i == 0 else equity[i - 1]
            previous_px = px if i == 0 else price_vals[i - 1]
            cash_before_accrual = cash
            cash *= 1.0 + cash_ret_vals[i]
            cash_contribution[i] = (
                cash_before_accrual * cash_ret_vals[i] / previous_equity
            )
            asset_contribution[i] = float(
                np.dot(shares, px - previous_px) / previous_equity
            )
            # Pre-trade NAV (mark current holdings at this bar's close).
            nav = cash + float(np.dot(shares, px))
            if not np.isfinite(nav):
                raise ValueError(f"non-finite NAV on bar {dt!r}")
            if nav <= 0.0:
                raise ValueError(f"portfolio NAV was exhausted on bar {dt!r}")
            gross_equity[i] = nav

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
                    max_gross=self.max_gross,
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
                    if not np.isfinite(fill_px) or fill_px <= 0.0:
                        raise ValueError(
                            f"execution model returned invalid fill for {symbols[j]!r}"
                        )
                    trade_cash += dq * fill_px  # buys cost cash, sells add cash

                # Explicit commission/slippage/tax charge from the cost model.
                cost_cash = cost_res.total_cost * nav

                next_cash = cash - trade_cash - cost_cash
                if (
                    self.max_gross <= 1.0 + 1e-12
                    and np.all(target_shares >= -1e-12)
                    and next_cash < -1e-10 * max(nav, 1.0)
                ):
                    raise ValueError(
                        "adverse fills and costs require unmodeled financing for "
                        "a long-only unlevered target"
                    )
                cash = next_cash
                shares = target_shares

                turnover[i] = cost_res.turnover

            post_nav = cash + float(np.dot(shares, px))
            if not np.isfinite(post_nav) or post_nav <= 0.0:
                raise ValueError(f"portfolio NAV was exhausted on bar {dt!r}")
            equity[i] = post_nav
            realized_weights = (shares * px) / post_nav
            realized_gross = float(np.abs(realized_weights).sum())
            eff_weights[i] = enforce_exposure_contract(
                realized_weights,
                max_gross=max(1.0, realized_gross) + 1e-9,
                name=f"EventDrivenBacktest realized weights on {dt}",
            )

        equity_curve = pd.Series(equity, index=index)
        net = equity_curve.pct_change(fill_method=None).fillna(0.0)
        # First bar's return relative to starting capital.
        net.iloc[0] = equity[0] / self.capital - 1.0
        previous_equity = np.concatenate(([self.capital], equity[:-1]))
        gross = pd.Series(gross_equity / previous_equity - 1.0, index=index)
        # Includes explicit fees and any execution-model price impact, all on
        # the same prior-equity denominator as the return series.
        cost_series = gross - net

        metadata = {
            "capital": self.capital,
            "periods_per_year": self.periods_per_year,
            "engine": "event_driven",
            "execution_model": repr(self.execution_model),
            "n_periods": int(len(index)),
            "n_rebalances": len(rebalance_bars),
            "execution_timing": "next_bar_close",
            "cash_return_source": cash_return_series.name,
        }

        return BacktestResult(
            returns=net,
            equity_curve=equity_curve,
            weights=pd.DataFrame(eff_weights, index=index, columns=symbols),
            gross_returns=gross,
            costs=cost_series,
            turnover=pd.Series(turnover, index=index),
            metadata=metadata,
            asset_contribution=pd.Series(asset_contribution, index=index),
            cash_contribution=pd.Series(cash_contribution, index=index),
            cash_weights=1.0 - pd.DataFrame(
                eff_weights, index=index, columns=symbols
            ).sum(axis=1),
        )
