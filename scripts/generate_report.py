"""Generate charts and markdown tables from an explicit real-data source.

Runs the multi_asset_rotation strategy on real data through the mandatory-cost
engine and emits a local Markdown report plus separate diagnostic plots:

* ``report_overview.png`` - compact four-panel review image
* ``equity_vs_benchmarks.png`` - growth of $1 vs SPY and equal-weight
* ``drawdown.png`` - underwater drawdown
* ``rolling_sharpe.png`` - rolling 126-day Sharpe
* ``rolling_risk.png`` - rolling volatility and beta to SPY
* ``allocation_and_exposure.png`` - post-trade weights, gross exposure, cash
* ``turnover_and_costs.png`` - turnover and cumulative cost fractions
* ``monthly_returns.png`` - monthly net-return heatmap
* ``return_distribution.png`` - daily return distribution and normal Q-Q
* a **performance metrics** markdown table (printed to stdout)
* a **monthly returns** markdown table (printed to stdout)

Every number is computed from the supplied data. The output records the source
kind, date window, and a SHA-256 digest for local files. The repository does not
bundle market data or generated performance results.

    python scripts/generate_report.py --prices-csv local_data/rotation_prices.csv
    python scripts/generate_report.py --live-yfinance --start 2015 --end 2024
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.getLogger("hmmlearn").setLevel(logging.ERROR)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
# joblib/loky can print a physical-core detection traceback on hosts where CPU
# topology is unreadable; pin it so offline/CI output stays clean (respects an
# existing override and matches the single-threaded determinism elsewhere).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from quantcortex.backtest.costs.transaction_costs import TransactionCostModel
from quantcortex.backtest.engines.vectorized import VectorizedBacktest
from quantcortex.backtest.metrics.tearsheet import Tearsheet
from quantcortex.backtest.validation.deflated_sharpe import compute_dsr
from quantcortex.data.local_csv import load_price_matrix, sha256_file
from quantcortex.data.processors.calendar import first_session_each_week

ROTATION_UNIVERSE = ["QQQ", "VGT", "GLD", "TLT", "SPY", "VIG"]
REPORT_ARTIFACTS = (
    ("report_overview.png", "Diagnostic overview"),
    ("equity_vs_benchmarks.png", "Net growth versus gross benchmarks"),
    ("drawdown.png", "Underwater drawdown"),
    ("rolling_sharpe.png", "Rolling 126-session Sharpe"),
    ("rolling_risk.png", "Rolling volatility and beta to SPY"),
    ("allocation_and_exposure.png", "Post-trade allocation, exposure, and cash"),
    (
        "turnover_and_costs.png",
        "Executed turnover and cumulative sum of modeled cost fractions",
    ),
    ("monthly_returns.png", "Monthly net returns"),
    ("return_distribution.png", "Daily return distribution and normal Q-Q"),
)
YFINANCE_NOTICE = (
    "Live Yahoo Finance data is fetched through yfinance. Review Yahoo's terms "
    "and yfinance's legal disclaimer at https://ranaroussi.github.io/yfinance/."
)


def load_prices(
    start: str,
    end: str,
    prices_csv: Path | None = None,
    live_yfinance: bool = False,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load prices from exactly one explicit source and return source metadata."""
    if (prices_csv is not None) == live_yfinance:
        raise ValueError("choose exactly one of prices_csv or live_yfinance")

    if prices_csv is not None:
        resolved = prices_csv.expanduser().resolve()
        prices = load_price_matrix(
            resolved,
            symbols=ROTATION_UNIVERSE,
            start=start,
            end=end,
        )
        return prices, {
            "kind": "local CSV",
            "path": str(resolved),
            "sha256": sha256_file(resolved),
        }

    print(YFINANCE_NOTICE, file=sys.stderr)
    from quantcortex.data.providers.yfinance_provider import YFinanceProvider

    provider_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    prices = YFinanceProvider().get_prices(
        ROTATION_UNIVERSE, start=start, end=provider_end
    )
    if prices is None or prices.empty:
        raise RuntimeError("yfinance returned no prices")
    prices = prices.dropna(how="all").ffill(limit=5).dropna()
    if prices.empty:
        raise RuntimeError("no complete rows remain in the yfinance response")
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return prices, {
        "kind": "live yfinance",
        "provider": "Yahoo Finance via yfinance",
        "retrieved_at": fetched_at,
        "adjustment_method": "yfinance adjusted close with auto_adjust=False",
    }


def _growth(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def _ann_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")


def _rolling_volatility(returns: pd.Series, window: int = 126) -> pd.Series:
    """Annualized trailing volatility on the report's daily return clock."""
    if isinstance(window, (bool, np.bool_)) or int(window) != window or window < 2:
        raise ValueError("window must be an integer >= 2")
    return returns.rolling(int(window)).std(ddof=1) * np.sqrt(252.0)


def _rolling_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 126,
) -> pd.Series:
    """Trailing beta to a benchmark, preserving the portfolio return index."""
    if isinstance(window, (bool, np.bool_)) or int(window) != window or window < 2:
        raise ValueError("window must be an integer >= 2")
    aligned = pd.concat(
        [returns.rename("portfolio"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()
    covariance = aligned["portfolio"].rolling(int(window)).cov(aligned["benchmark"])
    variance = aligned["benchmark"].rolling(int(window)).var(ddof=1)
    beta = covariance / variance.replace(0.0, np.nan)
    return beta.reindex(returns.index)


def _allocation_frame(weights: pd.DataFrame) -> pd.DataFrame:
    """Return long-only post-trade weights with an explicit cash column."""
    if not isinstance(weights, pd.DataFrame) or weights.empty:
        raise ValueError("allocation plot requires a non-empty weight panel")
    values = weights.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("allocation weights must be finite")
    if np.any(values < -1e-10):
        raise ValueError("multi-asset rotation allocation must be long-only")
    long_weights = weights.clip(lower=0.0)
    gross = long_weights.sum(axis=1)
    if (gross > 1.0 + 1e-8).any():
        raise ValueError("allocation gross exposure must not exceed 100%")
    allocation = long_weights.copy()
    allocation["Cash"] = (1.0 - gross).clip(lower=0.0, upper=1.0)
    return allocation


def _evaluation_index(
    prices: pd.DataFrame,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
) -> pd.DatetimeIndex:
    index = prices.index
    if start is not None:
        index = index[index >= pd.Timestamp(start)]
    if end is not None:
        index = index[index <= pd.Timestamp(end)]
    if index.empty:
        raise ValueError("no price rows fall inside the evaluation window")
    return index


def _validate_warmup(
    prices: pd.DataFrame,
    evaluation_start: pd.Timestamp,
    required_sessions: int,
    *,
    enforce: bool,
) -> int:
    """Return available pre-evaluation sessions and enforce full initialization."""
    available = int((prices.index < evaluation_start).sum())
    if enforce and available < required_sessions:
        raise ValueError(
            f"data source provides {available} pre-evaluation sessions; "
            f"multi_asset_rotation requires at least {required_sessions} for "
            "full signal initialization. Supply earlier data or pass "
            "--warmup-years 0 to explicitly permit a cold-start report"
        )
    return available


def _buy_hold_returns(
    prices: pd.DataFrame, evaluation_index: pd.DatetimeIndex
) -> tuple[pd.Series, pd.Series]:
    """Return SPY and equal-weight buy-and-hold returns on one capital clock."""
    first = prices.index.get_loc(evaluation_index[0])
    last = prices.index.get_loc(evaluation_index[-1])
    base = max(0, first - 1)
    benchmark_prices = prices.iloc[base : last + 1]

    spy = benchmark_prices["SPY"].pct_change(fill_method=None).reindex(
        evaluation_index
    )
    equal_weight_curve = benchmark_prices.div(benchmark_prices.iloc[0]).mean(axis=1)
    equal_weight = equal_weight_curve.pct_change(fill_method=None).reindex(
        evaluation_index
    )
    if first == 0:
        spy.iloc[0] = 0.0
        equal_weight.iloc[0] = 0.0
    return spy, equal_weight


def compute(
    prices: pd.DataFrame,
    n_trials: int = 10,
    *,
    sr_variance: float | None = None,
    evaluation_start: str | pd.Timestamp | None = None,
    evaluation_end: str | pd.Timestamp | None = None,
    strategy=None,
) -> dict:
    from quantcortex.strategies.multi_asset_rotation import MultiAssetRotation

    strategy = strategy if strategy is not None else MultiAssetRotation()
    weekly = first_session_each_week(prices.index)
    weights = strategy.generate_weights(prices, weekly)
    if weights.empty:
        raise ValueError("strategy produced no target weights")
    cost_model = TransactionCostModel()
    result = VectorizedBacktest(cost_model, capital=1.0).run(weights, prices)
    evaluation_index = _evaluation_index(
        prices, evaluation_start, evaluation_end
    )
    rets = result.returns.reindex(evaluation_index).dropna()
    if rets.empty:
        raise ValueError("backtest produced no returns in the evaluation window")

    ts = Tearsheet(rets)
    m = ts.compute()
    m["dsr"] = compute_dsr(
        rets, n_trials=n_trials, sr_variance=sr_variance
    )
    m["dsr_n_trials"] = n_trials
    m["dsr_sr_variance"] = sr_variance
    m["annualized_turnover"] = float(
        result.turnover.reindex(rets.index).mean() * 252
    )
    m["summed_cost_fraction"] = float(result.costs.reindex(rets.index).sum())
    spy, ew = _buy_hold_returns(prices, rets.index)
    first_evaluation_position = prices.index.get_loc(rets.index[0])
    report_weights = result.weights.reindex(rets.index)
    allocation = _allocation_frame(report_weights)
    return {
        "px": prices,
        "rets": rets,
        "ts": ts,
        "m": m,
        "strat_g": _growth(rets),
        "spy_g": _growth(spy),
        "ew_g": _growth(ew),
        "spy_sharpe": _ann_sharpe(spy),
        "ew_sharpe": _ann_sharpe(ew),
        "spy_returns": spy,
        "ew_returns": ew,
        "weights": report_weights,
        "allocation": allocation,
        "gross_exposure": report_weights.abs().sum(axis=1),
        "turnover": result.turnover.reindex(rets.index).fillna(0.0),
        "costs": result.costs.reindex(rets.index).fillna(0.0),
        "monthly": ts.monthly_returns_table(),
        "cost_model": cost_model,
        "warmup_sessions": first_evaluation_position,
    }


def save_charts(d: dict, imgdir: Path) -> list[Path]:
    # Import matplotlib here (not at module load) so --help and arg validation
    # don't pay its import cost or risk a config-cache warning before argparse.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from scipy.stats import norm

    imgdir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-darkgrid")

    def save(fig, name: str) -> Path:
        path = imgdir / name
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_equity(ax) -> None:
        ax.plot(
            d["strat_g"].index,
            d["strat_g"].to_numpy(),
            label="Multi-Asset Rotation (net)",
            color="C0",
            lw=1.7,
        )
        ax.plot(
            d["spy_g"].index,
            d["spy_g"].to_numpy(),
            label="SPY buy-and-hold (gross)",
            color="C7",
            lw=1.1,
            alpha=0.85,
        )
        ax.plot(
            d["ew_g"].index,
            d["ew_g"].to_numpy(),
            label="Equal-weight 6-ETF (gross)",
            color="C2",
            lw=1.1,
            alpha=0.85,
        )
        ax.axhline(1.0, color="black", lw=0.7, alpha=0.5)
        ax.set_title("Growth of $1: net strategy versus gross benchmarks")
        ax.set_ylabel("Growth of $1")
        ax.legend(loc="best", framealpha=0.9)

    def plot_drawdown(ax) -> None:
        drawdown = d["ts"].drawdown_series()
        ax.fill_between(
            drawdown.index,
            drawdown.to_numpy(),
            0.0,
            color="C3",
            alpha=0.45,
        )
        ax.set_title("Underwater drawdown")
        ax.set_ylabel("Drawdown")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    def plot_allocation(ax, *, include_legend: bool = True) -> None:
        allocation = d["allocation"]
        colors = list(plt.get_cmap("tab20").colors[: len(allocation.columns) - 1])
        colors.append((0.72, 0.72, 0.72))
        ax.stackplot(
            allocation.index,
            *[allocation[column].to_numpy() for column in allocation.columns],
            labels=allocation.columns,
            colors=colors,
            alpha=0.88,
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Post-trade target weights and cash")
        ax.set_ylabel("NAV weight")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        if include_legend:
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=4,
                framealpha=0.9,
            )

    def plot_turnover_costs(ax) -> None:
        turnover = d["turnover"]
        nonzero = turnover[turnover > 0.0]
        ax.bar(
            nonzero.index,
            nonzero.to_numpy(),
            width=3.0,
            color="C1",
            alpha=0.65,
            label="Executed one-way turnover",
        )
        ax.set_title("Executed turnover and cumulative sum of modeled cost fractions")
        ax.set_ylabel("Turnover")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        cost_axis = ax.twinx()
        cumulative_cost = d["costs"].cumsum()
        cost_axis.plot(
            cumulative_cost.index,
            cumulative_cost.to_numpy(),
            color="C3",
            lw=1.5,
            label="Cumulative sum of modeled cost fractions",
        )
        cost_axis.set_ylabel("Cumulative sum of cost fractions")
        cost_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        handles, labels = ax.get_legend_handles_labels()
        cost_handles, cost_labels = cost_axis.get_legend_handles_labels()
        ax.legend(handles + cost_handles, labels + cost_labels, loc="best")

    def plot_monthly_heatmap(ax, *, annotate: bool = True) -> None:
        month_order = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        table = d["monthly"].drop(columns=["YTD"], errors="ignore")
        table = table.reindex(columns=month_order)
        matrix = table.to_numpy(dtype=float)
        finite = np.isfinite(matrix)
        if not finite.any():
            ax.text(0.5, 0.5, "No monthly returns", ha="center", va="center")
            ax.set_axis_off()
            return
        limit = max(float(np.nanmax(np.abs(matrix))), 0.01)
        image = ax.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            cmap="RdYlGn",
            vmin=-limit,
            vmax=limit,
        )
        ax.set_xticks(range(len(month_order)))
        ax.set_xticklabels(month_order)
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index.astype(str))
        ax.set_title("Monthly net returns")
        if annotate:
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    value = matrix[row, column]
                    if not np.isfinite(value):
                        continue
                    color = "white" if abs(value) > 0.6 * limit else "black"
                    ax.text(
                        column,
                        row,
                        f"{value * 100:.1f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=color,
                    )
        colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    paths: list[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    plot_equity(axes[0, 0])
    plot_drawdown(axes[0, 1])
    plot_allocation(axes[1, 0], include_legend=False)
    plot_turnover_costs(axes[1, 1])
    handles, labels = axes[1, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            framealpha=0.9,
        )
    window = f"{d['rets'].index[0].date()} to {d['rets'].index[-1].date()}"
    fig.suptitle(
        f"Multi-Asset Rotation diagnostic overview | {window}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.97))
    paths.append(save(fig, "report_overview.png"))

    fig, ax = plt.subplots(figsize=(11, 4.2))
    plot_equity(ax)
    fig.tight_layout()
    paths.append(save(fig, "equity_vs_benchmarks.png"))

    fig, ax = plt.subplots(figsize=(11, 3.4))
    plot_drawdown(ax)
    fig.tight_layout()
    paths.append(save(fig, "drawdown.png"))

    fig, ax = plt.subplots(figsize=(11, 3.4))
    rs = d["ts"].rolling_sharpe(126)
    ax.plot(rs.index, rs.to_numpy(), color="C4", lw=1.3)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_title("Rolling Sharpe (126-day)")
    ax.set_ylabel("Sharpe")
    fig.tight_layout()
    paths.append(save(fig, "rolling_sharpe.png"))

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    rolling_strategy_vol = _rolling_volatility(d["rets"], 126)
    rolling_spy_vol = _rolling_volatility(d["spy_returns"], 126)
    axes[0].plot(
        rolling_strategy_vol.index,
        rolling_strategy_vol.to_numpy(),
        color="C0",
        label="Strategy (net)",
    )
    axes[0].plot(
        rolling_spy_vol.index,
        rolling_spy_vol.to_numpy(),
        color="C7",
        alpha=0.85,
        label="SPY (gross)",
    )
    axes[0].set_title("Rolling annualized volatility (126 sessions)")
    axes[0].set_ylabel("Volatility")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(loc="best")
    rolling_beta = _rolling_beta(d["rets"], d["spy_returns"], 126)
    axes[1].plot(
        rolling_beta.index,
        rolling_beta.to_numpy(),
        color="C5",
        label="Beta to SPY",
    )
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].axhline(1.0, color="C7", lw=0.8, ls="--")
    axes[1].set_title("Rolling beta to SPY (126 sessions)")
    axes[1].set_ylabel("Beta")
    axes[1].legend(loc="best")
    fig.tight_layout()
    paths.append(save(fig, "rolling_risk.png"))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.0), sharex=True)
    plot_allocation(axes[0])
    axes[1].plot(
        d["gross_exposure"].index,
        d["gross_exposure"].to_numpy(),
        color="C0",
        label="Gross exposure",
    )
    cash = d["allocation"]["Cash"]
    axes[1].plot(cash.index, cash.to_numpy(), color="C7", label="Cash")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Invested exposure and cash")
    axes[1].set_ylabel("NAV fraction")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].legend(loc="best")
    fig.tight_layout()
    paths.append(save(fig, "allocation_and_exposure.png"))

    fig, ax = plt.subplots(figsize=(11, 4.0))
    plot_turnover_costs(ax)
    fig.tight_layout()
    paths.append(save(fig, "turnover_and_costs.png"))

    fig, ax = plt.subplots(figsize=(11, max(3.6, 0.5 * len(d["monthly"]) + 2.0)))
    plot_monthly_heatmap(ax)
    fig.tight_layout()
    paths.append(save(fig, "monthly_returns.png"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    returns_pct = d["rets"].to_numpy(dtype=float) * 100.0
    axes[0].hist(
        returns_pct,
        bins="fd",
        color="C0",
        alpha=0.72,
        edgecolor="white",
    )
    fifth_percentile = float(d["rets"].quantile(0.05)) * 100.0
    tail = d["rets"][d["rets"] <= d["rets"].quantile(0.05)]
    expected_shortfall = float(tail.mean()) * 100.0
    axes[0].axvline(
        fifth_percentile,
        color="C1",
        lw=1.4,
        label="5th percentile",
    )
    axes[0].axvline(
        expected_shortfall,
        color="C3",
        lw=1.4,
        ls="--",
        label="Tail mean",
    )
    axes[0].set_title("Daily net return distribution")
    axes[0].set_xlabel("Return (%)")
    axes[0].set_ylabel("Observations")
    axes[0].legend(loc="best")

    sorted_returns = np.sort(d["rets"].to_numpy(dtype=float))
    standard_deviation = float(np.std(sorted_returns, ddof=1))
    if standard_deviation > 0.0:
        standardized = (sorted_returns - float(np.mean(sorted_returns))) / standard_deviation
        probabilities = (np.arange(len(standardized)) + 0.5) / len(standardized)
        theoretical = norm.ppf(probabilities)
        axes[1].scatter(theoretical, standardized, s=8, alpha=0.55, color="C4")
        bounds = [
            min(float(theoretical.min()), float(standardized.min())),
            max(float(theoretical.max()), float(standardized.max())),
        ]
        axes[1].plot(bounds, bounds, color="black", lw=0.8, ls="--")
    else:
        axes[1].text(
            0.5,
            0.5,
            "Q-Q unavailable for zero-variance returns",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    axes[1].set_title("Normal Q-Q diagnostic")
    axes[1].set_xlabel("Theoretical normal quantile")
    axes[1].set_ylabel("Standardized observed return")
    fig.tight_layout()
    paths.append(save(fig, "return_distribution.png"))

    return paths


def markdown_metrics(d: dict) -> str:
    m = d["m"]
    rows = [
        ("CAGR", f"{m['cagr']:+.2%}"),
        ("Annualized volatility", f"{m['ann_vol']:.2%}"),
        ("Sharpe", f"{m['sharpe']:+.2f}"),
        ("Sortino", f"{m['sortino']:+.2f}"),
        ("Calmar", f"{m['calmar']:+.2f}"),
        ("Max drawdown", f"{m['max_drawdown']:+.2%}"),
        ("Annualized one-way turnover", f"{m['annualized_turnover']:.2f}x"),
        ("Sum of modeled cost fractions", f"{m['summed_cost_fraction']:.2%}"),
        ("VaR 95% (daily)", f"{m['var_95']:.2%}"),
        ("CVaR 95% (daily)", f"{m['cvar_95']:.2%}"),
        (f"Deflated Sharpe ({m['dsr_n_trials']} trials)", f"{m['dsr']:.3f}"),
        ("SPY buy & hold Sharpe (gross)", f"{d['spy_sharpe']:+.2f}"),
        ("Equal-weight 6-ETF buy & hold Sharpe (gross)", f"{d['ew_sharpe']:+.2f}"),
    ]
    out = ["| Metric | Value |", "|--------|-------|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(out)


def markdown_monthly(d: dict) -> str:
    table = d["monthly"]
    cols = list(table.columns)
    header = "| Year | " + " | ".join(cols) + " |"
    sep = "|------|" + "|".join(["-----"] * len(cols)) + "|"
    lines = [header, sep]
    for year, row in table.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append("" if pd.isna(v) else f"{v*100:+.1f}")
        lines.append(f"| {year} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown_source(source: dict[str, str], prices: pd.DataFrame) -> str:
    rows = [
        ("Source kind", source["kind"]),
        ("Provider", source.get("provider", "not supplied")),
        ("Permission basis", source.get("permission_basis", "not supplied")),
        ("Retrieved at", source.get("retrieved_at", "not supplied")),
        ("Adjustment method", source.get("adjustment_method", "not supplied")),
        ("Symbols", ", ".join(map(str, prices.columns))),
        ("Price window", f"{prices.index[0].date()} to {prices.index[-1].date()}"),
        (
            "Provenance metadata",
            source.get("provenance_metadata", "incomplete"),
        ),
    ]
    if "path" in source:
        rows.extend([("Local path", source["path"]), ("SHA-256", source["sha256"])])
    out = ["| Field | Value |", "|-------|-------|"]
    out.extend(
        f"| {field} | {str(value).replace('|', '&#124;')} |"
        for field, value in rows
    )
    return "\n".join(out)


def markdown_settings(d: dict) -> str:
    cost_model = d["cost_model"]
    rows = [
        ("Strategy", "multi_asset_rotation"),
        (
            "Evaluation window",
            f"{d['rets'].index[0].date()} to {d['rets'].index[-1].date()}",
        ),
        ("Pre-evaluation warm-up sessions", str(d["warmup_sessions"])),
        ("Full-signal warm-up requirement", str(d["required_warmup_sessions"])),
        (
            "Cold-start override",
            "enabled" if d["cold_start_allowed"] else "disabled",
        ),
        (
            "Rebalance",
            "first available session each week; close signal executes next bar close",
        ),
        ("Commission", f"{cost_model.commission * 10_000:.1f} bps per trade"),
        ("Slippage", f"{cost_model.slippage * 10_000:.1f} bps per trade"),
        ("Transfer tax", f"{cost_model.tax * 10_000:.1f} bps on sells"),
        ("ADV cap", "not applied; this report supplies no volume input"),
        ("DSR trial count", str(d["m"]["dsr_n_trials"])),
        (
            "DSR cross-trial Sharpe variance",
            "single-series estimate"
            if d["m"]["dsr_sr_variance"] is None
            else f"{d['m']['dsr_sr_variance']:.8g}",
        ),
    ]
    out = ["| Setting | Value |", "|---------|-------|"]
    out.extend(f"| {setting} | {value} |" for setting, value in rows)
    return "\n".join(out)


def write_markdown_report(
    d: dict,
    source: dict[str, str],
    prices: pd.DataFrame,
    report_path: Path,
    image_paths: list[Path],
) -> Path:
    """Write a portable local report linking every generated diagnostic."""
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    images = {path.name: path.expanduser().resolve() for path in image_paths}
    missing = [name for name, _ in REPORT_ARTIFACTS if name not in images]
    if missing:
        raise ValueError(f"report images are incomplete: {missing}")

    def image_link(name: str) -> str:
        relative = Path(os.path.relpath(images[name], report_path.parent))
        return relative.as_posix()

    lines = [
        "# quantcortex Multi-Asset Rotation Report",
        "",
        "This report was generated from an explicit data source. Strategy returns",
        "are net of the configured cost model; benchmark returns are gross.",
        "",
        "## Data Source",
        "",
        markdown_source(source, prices),
        "",
        "## Evaluation Settings",
        "",
        markdown_settings(d),
        "",
        "## Diagnostic Overview",
        "",
        f"![Diagnostic overview]({image_link('report_overview.png')})",
        "",
        "## Performance Metrics",
        "",
        markdown_metrics(d),
        "",
        "## Detailed Diagnostics",
        "",
    ]
    for name, title in REPORT_ARTIFACTS:
        if name == "report_overview.png":
            continue
        lines.extend([f"### {title}", "", f"![{title}]({image_link(name)})", ""])
    lines.extend(
        [
            "## Monthly Returns (%)",
            "",
            markdown_monthly(d),
            "",
            "## Interpretation Limits",
            "",
            "See `PERFORMANCE.md` in the repository for required disclosures and",
            "limitations. This report does not establish production readiness,",
            "capacity, or expected future performance.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def positive_int(value: str) -> int:
    """argparse type: a strictly-positive integer (the DSR needs n_trials >= 1)."""
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer (got {value!r})")
    return ivalue


def nonnegative_int(value: str) -> int:
    """argparse type: a non-negative integer."""
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(
            f"must be a non-negative integer (got {value!r})"
        )
    return ivalue


def nonnegative_float(value: str) -> float:
    """argparse type: a finite non-negative float."""
    fvalue = float(value)
    if not np.isfinite(fvalue) or fvalue < 0.0:
        raise argparse.ArgumentTypeError(
            f"must be a finite non-negative number (got {value!r})"
        )
    return fvalue


def nonempty_text(value: str) -> str:
    """argparse type: non-empty metadata text."""
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("must not be empty")
    return text


def iso_date_or_datetime(value: str) -> str:
    """argparse type: an ISO date or datetime retained as provenance text."""
    text = nonempty_text(value)
    try:
        if "T" in text or " " in text:
            datetime.fromisoformat(text.replace("Z", "+00:00"))
        else:
            date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"must be an ISO date or datetime (got {value!r})"
        ) from exc
    return text


def date_boundary(value: str, *, end_of_year: bool) -> pd.Timestamp:
    """Parse an exact ISO date or expand a four-digit year to its boundary."""
    try:
        if len(value) == 4 and value.isdigit():
            year = int(value)
            parsed = date(year, 12, 31) if end_of_year else date(year, 1, 1)
        else:
            parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"must be YYYY or YYYY-MM-DD (got {value!r})"
        ) from exc
    return pd.Timestamp(parsed)


def start_date(value: str) -> pd.Timestamp:
    """argparse type: evaluation start as a year or exact ISO date."""
    return date_boundary(value, end_of_year=False)


def end_date(value: str) -> pd.Timestamp:
    """argparse type: evaluation end as a year or exact ISO date."""
    return date_boundary(value, end_of_year=True)


def main(argv) -> int:
    ap = argparse.ArgumentParser(
        description="generate separate tearsheet charts + tables"
    )
    ap.add_argument(
        "--start",
        type=start_date,
        default=start_date("2018"),
        help="evaluation start year or YYYY-MM-DD (default 2018)",
    )
    ap.add_argument(
        "--end",
        type=end_date,
        default=end_date("2025"),
        help="evaluation end year or YYYY-MM-DD (default 2025)",
    )
    ap.add_argument(
        "--warmup-years",
        type=nonnegative_int,
        default=2,
        help="pre-evaluation history loaded for signals (default 2)",
    )
    ap.add_argument("--imgdir", type=Path, default=Path("reports/img"))
    ap.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Markdown report path (default: <imgdir>/../report.md)",
    )
    ap.add_argument(
        "--data-provider",
        type=nonempty_text,
        default=None,
        help="provider/vendor recorded in report provenance",
    )
    ap.add_argument(
        "--permission-basis",
        type=nonempty_text,
        default=None,
        help="license or permission basis, supplied by the data owner",
    )
    ap.add_argument(
        "--retrieved-at",
        type=iso_date_or_datetime,
        default=None,
        help="retrieval date or timestamp recorded in report provenance",
    )
    ap.add_argument(
        "--adjustment-method",
        type=nonempty_text,
        default=None,
        help="corporate-action adjustment method recorded in provenance",
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--prices-csv",
        type=Path,
        help="owner-supplied wide adjusted-close CSV with a date column",
    )
    source.add_argument(
        "--live-yfinance",
        action="store_true",
        help="explicitly fetch live data through yfinance",
    )
    ap.add_argument(
        "--n-trials",
        type=positive_int,
        default=10,
        help="trials assumed for the Deflated Sharpe Ratio (default 10)",
    )
    ap.add_argument(
        "--sr-variance",
        type=nonnegative_float,
        default=None,
        help="cross-trial variance of per-observation Sharpe estimates for DSR",
    )
    args = ap.parse_args(argv[1:])

    try:
        evaluation_start = args.start
        evaluation_end = args.end
        if evaluation_start > evaluation_end:
            raise ValueError("evaluation start must not be after evaluation end")
        data_start = evaluation_start - pd.DateOffset(years=args.warmup_years)
        prices, source_metadata = load_prices(
            data_start.strftime("%Y-%m-%d"),
            evaluation_end.strftime("%Y-%m-%d"),
            prices_csv=args.prices_csv,
            live_yfinance=args.live_yfinance,
        )
        supplied_metadata = {
            "provider": args.data_provider,
            "permission_basis": args.permission_basis,
            "retrieved_at": args.retrieved_at,
            "adjustment_method": args.adjustment_method,
        }
        source_metadata.update(
            {key: value for key, value in supplied_metadata.items() if value is not None}
        )
        publication_fields = (
            "provider",
            "permission_basis",
            "retrieved_at",
            "adjustment_method",
        )
        source_metadata["provenance_metadata"] = (
            "complete (owner-supplied; permission not independently verified)"
            if all(source_metadata.get(field) for field in publication_fields)
            else "incomplete - do not publish generated results"
        )
        from quantcortex.strategies.multi_asset_rotation import MultiAssetRotation

        strategy = MultiAssetRotation()
        warmup_sessions = _validate_warmup(
            prices,
            evaluation_start,
            strategy.required_history,
            enforce=args.warmup_years > 0,
        )
        d = compute(
            prices,
            n_trials=args.n_trials,
            sr_variance=args.sr_variance,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            strategy=strategy,
        )
        d["warmup_sessions"] = warmup_sessions
        d["required_warmup_sessions"] = strategy.required_history
        d["cold_start_allowed"] = args.warmup_years == 0
    except Exception as exc:
        print(f"report generation failed: {exc}", file=sys.stderr)
        return 1

    try:
        image_paths = save_charts(d, args.imgdir)
        report_path = args.report_out or args.imgdir.parent / "report.md"
        report_path = write_markdown_report(
            d,
            source_metadata,
            prices,
            report_path,
            image_paths,
        )
    except Exception as exc:
        print(f"report rendering failed: {exc}", file=sys.stderr)
        return 1
    window = f"{d['rets'].index[0].date()} to {d['rets'].index[-1].date()}"
    print(f"# Charts written to {args.imgdir}/ for window {window}\n")
    print(f"# Markdown report written to {report_path}\n")
    print("## Data source\n")
    print(markdown_source(source_metadata, prices))
    print("\n## Evaluation settings\n")
    print(markdown_settings(d))
    print("\n## Performance metrics (markdown)\n")
    print(markdown_metrics(d))
    print("\n## Monthly returns %, (markdown)\n")
    print(markdown_monthly(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
