"""Full performance tearsheet for a return stream.

Computes the standard battery of risk/return statistics (CAGR, annualized
return and volatility, Sharpe, Sortino, Calmar, drawdowns, tail metrics, hit
rate, etc.), produces a monthly-returns table, and renders a four-panel
matplotlib figure (equity curve, drawdown, rolling Sharpe, monthly heatmap).

``matplotlib.pyplot`` is imported lazily inside :meth:`Tearsheet.plot` so that
merely importing this module - or computing metrics in a headless pipeline  - 
does not pull in a GUI backend.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

__all__ = ["Tearsheet"]


class Tearsheet:
    """Performance tearsheet for a periodic return series.

    Parameters
    ----------
    returns:
        Periodic (e.g. daily) simple returns.  A :class:`~pandas.Series`;
        a :class:`~pandas.DatetimeIndex` is required for the monthly table and
        heatmap but not for the scalar metrics.
    periods_per_year:
        Number of return periods in a year used for annualization
        (default 252, i.e. trading days).
    risk_free:
        Per-period risk-free rate subtracted from returns when computing
        risk-adjusted ratios (default 0).
    """

    def __init__(
        self,
        returns: pd.Series,
        *,
        periods_per_year: float = 252,
        risk_free: float = 0.0,
    ) -> None:
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
        self.returns = returns.dropna().astype(float)
        self.periods_per_year = float(periods_per_year)
        self.risk_free = float(risk_free)

    # ------------------------------------------------------------------ #
    # Core series                                                        #
    # ------------------------------------------------------------------ #
    def equity_curve(self, starting_value: float = 1.0) -> pd.Series:
        """Cumulative compounded equity curve (growth of ``starting_value``)."""
        return starting_value * (1.0 + self.returns).cumprod()

    def drawdown_series(self) -> pd.Series:
        """Drawdown at each period: ``equity / running_peak - 1`` (<= 0)."""
        if self.returns.empty:
            return pd.Series(dtype=float)
        equity = (1.0 + self.returns).cumprod()
        peak = equity.cummax()
        return equity / peak - 1.0

    def rolling_sharpe(self, window: int = 126) -> pd.Series:
        """Annualized rolling Sharpe ratio over a trailing ``window``."""
        if self.returns.empty:
            return pd.Series(dtype=float)
        excess = self.returns - self.risk_free
        mean = excess.rolling(window).mean()
        std = excess.rolling(window).std(ddof=1)
        ann = math.sqrt(self.periods_per_year)
        return (mean / std.replace(0.0, np.nan)) * ann

    # ------------------------------------------------------------------ #
    # Scalar metric helpers                                              #
    # ------------------------------------------------------------------ #
    def _max_drawdown(self) -> float:
        dd = self.drawdown_series()
        return float(dd.min()) if not dd.empty else float("nan")

    def _max_dd_duration(self) -> int:
        """Longest run (in periods) spent below a prior equity peak."""
        dd = self.drawdown_series()
        if dd.empty:
            return 0
        under = (dd < 0).to_numpy()
        longest = current = 0
        for flag in under:
            current = current + 1 if flag else 0
            longest = max(longest, current)
        return int(longest)

    def _annualized_return(self) -> float:
        if self.returns.empty:
            return float("nan")
        return float(self.returns.mean() * self.periods_per_year)

    def _annualized_vol(self) -> float:
        if self.returns.size < 2:
            return float("nan")
        return float(self.returns.std(ddof=1) * math.sqrt(self.periods_per_year))

    def _cagr(self) -> float:
        if self.returns.empty:
            return float("nan")
        total_growth = float((1.0 + self.returns).prod())
        if total_growth <= 0.0:
            return float("nan")
        years = self.returns.size / self.periods_per_year
        if years <= 0:
            return float("nan")
        return total_growth ** (1.0 / years) - 1.0

    def _sharpe(self) -> float:
        if self.returns.size < 2:
            return float("nan")
        excess = self.returns - self.risk_free
        sd = excess.std(ddof=1)
        if sd == 0.0:
            return float("nan")
        return float(excess.mean() / sd * math.sqrt(self.periods_per_year))

    def _sortino(self) -> float:
        if self.returns.size < 2:
            return float("nan")
        excess = self.returns - self.risk_free
        downside = excess[excess < 0.0]
        if downside.empty:
            return float("inf") if excess.mean() > 0 else float("nan")
        # Downside deviation uses all observations in the denominator.
        dd = math.sqrt(float((downside ** 2).sum()) / excess.size)
        if dd == 0.0:
            return float("nan")
        return float(excess.mean() / dd * math.sqrt(self.periods_per_year))

    def _calmar(self) -> float:
        """Calmar ratio: CAGR / |max drawdown|.

        Uses the canonical convention of a geometric (CAGR) numerator rather
        than the arithmetic annualized return.
        """
        mdd = self._max_drawdown()
        if not np.isfinite(mdd) or mdd == 0.0:
            return float("nan")
        return float(self._cagr() / abs(mdd))

    def _var_cvar(self, level: float = 0.95) -> tuple[float, float]:
        """Historical VaR and CVaR (expected shortfall) at ``level``.

        Returned as losses (positive numbers).  ``var_95`` is the 5th-percentile
        loss; ``cvar_95`` the mean loss beyond it.
        """
        if self.returns.empty:
            return float("nan"), float("nan")
        q = self.returns.quantile(1.0 - level)
        var = -float(q)
        tail = self.returns[self.returns <= q]
        cvar = -float(tail.mean()) if not tail.empty else var
        return var, cvar

    def _tail_ratio(self) -> float:
        """Ratio of the 95th-percentile gain to the |5th-percentile loss|."""
        if self.returns.empty:
            return float("nan")
        right = float(self.returns.quantile(0.95))
        left = abs(float(self.returns.quantile(0.05)))
        if left == 0.0:
            return float("nan")
        return right / left

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def compute(self) -> Dict[str, float]:
        """Compute the full set of scalar performance metrics.

        Returns
        -------
        dict
            Keys: ``total_return, cagr, ann_return, ann_vol, sharpe, sortino,
            calmar, max_drawdown, max_dd_duration, var_95, cvar_95, skew,
            kurtosis, best_period, worst_period, win_rate, avg_win, avg_loss,
            profit_factor, tail_ratio``.  Undefined metrics are ``nan``.
        """
        r = self.returns
        wins = r[r > 0.0]
        losses = r[r < 0.0]
        var_95, cvar_95 = self._var_cvar(0.95)

        gross_profit = float(wins.sum())
        gross_loss = float(losses.sum())  # negative or zero
        if gross_loss < 0.0:
            profit_factor = gross_profit / abs(gross_loss)
        elif gross_profit > 0.0:
            profit_factor = float("inf")
        else:
            profit_factor = float("nan")

        return {
            "total_return": float((1.0 + r).prod() - 1.0) if not r.empty
            else float("nan"),
            "cagr": self._cagr(),
            "ann_return": self._annualized_return(),
            "ann_vol": self._annualized_vol(),
            "sharpe": self._sharpe(),
            "sortino": self._sortino(),
            "calmar": self._calmar(),
            "max_drawdown": self._max_drawdown(),
            "max_dd_duration": self._max_dd_duration(),
            "var_95": var_95,
            "cvar_95": cvar_95,
            "skew": float(r.skew()) if r.size >= 3 else float("nan"),
            "kurtosis": float(r.kurtosis()) if r.size >= 4 else float("nan"),
            "best_period": float(r.max()) if not r.empty else float("nan"),
            "worst_period": float(r.min()) if not r.empty else float("nan"),
            "win_rate": float((r > 0.0).mean()) if not r.empty else float("nan"),
            "avg_win": float(wins.mean()) if not wins.empty else float("nan"),
            "avg_loss": float(losses.mean()) if not losses.empty else float("nan"),
            "profit_factor": profit_factor,
            "tail_ratio": self._tail_ratio(),
        }

    def monthly_returns_table(self) -> pd.DataFrame:
        """Years x months pivot of compounded returns, plus a YTD column.

        Requires a :class:`~pandas.DatetimeIndex`.  Each cell is the compounded
        return for that month; the ``YTD`` column is the compounded return for
        the whole year.
        """
        if self.returns.empty:
            return pd.DataFrame()
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            raise TypeError(
                "monthly_returns_table requires a DatetimeIndex on returns"
            )

        monthly = (1.0 + self.returns).groupby(
            [self.returns.index.year, self.returns.index.month]
        ).prod() - 1.0
        monthly.index = monthly.index.set_names(["year", "month"])
        table = monthly.unstack("month")
        # Order/label the month columns present.
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        }
        table = table.reindex(columns=sorted(table.columns))
        # YTD computed before renaming columns.
        ytd = (1.0 + self.returns).groupby(self.returns.index.year).prod() - 1.0
        table = table.rename(columns=month_names)
        table["YTD"] = ytd
        table.index.name = "year"
        return table

    def summary(self) -> str:
        """Formatted plain-text summary table of the scalar metrics."""
        m = self.compute()

        def pct(x: float) -> str:
            return "n/a" if not np.isfinite(x) else f"{x * 100:.2f}%"

        def num(x: float) -> str:
            if x == float("inf"):
                return "inf"
            return "n/a" if not np.isfinite(x) else f"{x:.3f}"

        rows = [
            ("Total return", pct(m["total_return"])),
            ("CAGR", pct(m["cagr"])),
            ("Annual return", pct(m["ann_return"])),
            ("Annual volatility", pct(m["ann_vol"])),
            ("Sharpe ratio", num(m["sharpe"])),
            ("Sortino ratio", num(m["sortino"])),
            ("Calmar ratio", num(m["calmar"])),
            ("Max drawdown", pct(m["max_drawdown"])),
            ("Max DD duration (periods)", str(int(m["max_dd_duration"]))),
            ("VaR 95%", pct(m["var_95"])),
            ("CVaR 95%", pct(m["cvar_95"])),
            ("Skew", num(m["skew"])),
            ("Kurtosis", num(m["kurtosis"])),
            ("Best period", pct(m["best_period"])),
            ("Worst period", pct(m["worst_period"])),
            ("Win rate", pct(m["win_rate"])),
            ("Avg win", pct(m["avg_win"])),
            ("Avg loss", pct(m["avg_loss"])),
            ("Profit factor", num(m["profit_factor"])),
            ("Tail ratio", num(m["tail_ratio"])),
        ]
        width = max(len(label) for label, _ in rows)
        lines = ["Performance tearsheet", "=" * (width + 18)]
        for label, value in rows:
            lines.append(f"  {label:<{width}}  {value:>12}")
        lines.append("=" * (width + 18))
        return "\n".join(lines)

    def plot(self, window: int = 126):
        """Render a four-panel tearsheet figure (lazy matplotlib import).

        Panels: cumulative equity curve, drawdown, rolling Sharpe, and a
        monthly-returns heatmap (when a DatetimeIndex is available).

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt  # lazy import

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        ax_eq, ax_dd, ax_rs, ax_hm = axes.ravel()

        # Equity curve.
        equity = self.equity_curve()
        ax_eq.plot(equity.index, equity.to_numpy(), color="C0")
        ax_eq.set_title("Equity curve")
        ax_eq.set_ylabel("Growth of 1")
        ax_eq.grid(True, alpha=0.3)

        # Drawdown.
        dd = self.drawdown_series()
        ax_dd.fill_between(dd.index, dd.to_numpy(), 0.0, color="C3", alpha=0.5)
        ax_dd.set_title("Drawdown")
        ax_dd.set_ylabel("Drawdown")
        ax_dd.grid(True, alpha=0.3)

        # Rolling Sharpe.
        rs = self.rolling_sharpe(window)
        ax_rs.plot(rs.index, rs.to_numpy(), color="C2")
        ax_rs.axhline(0.0, color="k", linewidth=0.8)
        ax_rs.set_title(f"Rolling Sharpe ({window})")
        ax_rs.set_ylabel("Sharpe")
        ax_rs.grid(True, alpha=0.3)

        # Monthly heatmap.
        try:
            table = self.monthly_returns_table()
        except TypeError:
            table = pd.DataFrame()
        if not table.empty:
            data = table.drop(columns=["YTD"], errors="ignore")
            mat = data.to_numpy(dtype=float)
            im = ax_hm.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-np.nanmax(
                np.abs(mat)) if np.isfinite(mat).any() else -1.0,
                vmax=np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0)
            ax_hm.set_xticks(range(len(data.columns)))
            ax_hm.set_xticklabels(data.columns, rotation=45, ha="right")
            ax_hm.set_yticks(range(len(data.index)))
            ax_hm.set_yticklabels(data.index)
            ax_hm.set_title("Monthly returns")
            fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
        else:
            ax_hm.text(0.5, 0.5, "Monthly heatmap\n(requires DatetimeIndex)",
                       ha="center", va="center", transform=ax_hm.transAxes)
            ax_hm.set_axis_off()

        fig.tight_layout()
        return fig
