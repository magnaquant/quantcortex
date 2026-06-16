"""Generate charts and markdown tables from an explicit real-data source.

Runs the multi_asset_rotation strategy on real data through the mandatory-cost
engine and emits, as *separate* artifacts (so they can be embedded individually
in docs):

* ``equity_vs_benchmarks.png`` - growth of $1 vs SPY and equal-weight
* ``drawdown.png``             - underwater drawdown
* ``rolling_sharpe.png``       - rolling 126-day Sharpe
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
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
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

ROTATION_UNIVERSE = ["QQQ", "VGT", "GLD", "TLT", "SPY", "VIG"]
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
    prices = prices.dropna(how="all").ffill().dropna()
    if prices.empty:
        raise RuntimeError("no complete rows remain in the yfinance response")
    return prices, {
        "kind": "live yfinance",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _growth(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def _ann_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")


def compute(prices: pd.DataFrame, n_trials: int = 10) -> dict:
    from quantcortex.strategies.multi_asset_rotation import MultiAssetRotation

    weekly = prices.index[prices.index.weekday == 0]
    weights = MultiAssetRotation().generate_weights(prices, weekly)
    cost_model = TransactionCostModel()
    result = VectorizedBacktest(cost_model, capital=1.0).run(weights, prices)
    rets = result.returns.dropna()

    ts = Tearsheet(rets)
    m = ts.compute()
    m["dsr"] = compute_dsr(rets, n_trials=n_trials)
    m["dsr_n_trials"] = n_trials
    m["annualized_turnover"] = float(result.turnover.mean() * 252)
    m["summed_cost_fraction"] = float(result.costs.sum())
    spy = prices["SPY"].pct_change().reindex(rets.index)
    equal_weight_curve = prices.div(prices.iloc[0]).mean(axis=1)
    ew = equal_weight_curve.pct_change().reindex(rets.index)
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
        "monthly": ts.monthly_returns_table(),
        "cost_model": cost_model,
    }


def save_charts(d: dict, imgdir: Path) -> None:
    # Import matplotlib here (not at module load) so --help and arg validation
    # don't pay its import cost or risk a config-cache warning before argparse.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imgdir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-darkgrid")

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(d["strat_g"].index, d["strat_g"].to_numpy(), label="Multi-Asset Rotation", color="C0", lw=1.7)
    ax.plot(d["spy_g"].index, d["spy_g"].to_numpy(), label="SPY (buy & hold)", color="C7", lw=1.1, alpha=0.85)
    ax.plot(d["ew_g"].index, d["ew_g"].to_numpy(), label="Equal-weight 6-ETF buy & hold", color="C2", lw=1.1, alpha=0.85)
    ax.set_title("Growth of $1 - strategy net of costs; benchmarks gross")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(imgdir / "equity_vs_benchmarks.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.4))
    dd = d["ts"].drawdown_series()
    ax.fill_between(dd.index, dd.to_numpy(), 0.0, color="C3", alpha=0.45)
    ax.set_title("Underwater (drawdown)")
    ax.set_ylabel("Drawdown")
    fig.tight_layout()
    fig.savefig(imgdir / "drawdown.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.4))
    rs = d["ts"].rolling_sharpe(126)
    ax.plot(rs.index, rs.to_numpy(), color="C4", lw=1.3)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_title("Rolling Sharpe (126-day)")
    ax.set_ylabel("Sharpe")
    fig.tight_layout()
    fig.savefig(imgdir / "rolling_sharpe.png", dpi=130)
    plt.close(fig)


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
        ("Price window", f"{prices.index[0].date()} to {prices.index[-1].date()}"),
    ]
    if "path" in source:
        rows.extend([("Local path", source["path"]), ("SHA-256", source["sha256"])])
    if "fetched_at_utc" in source:
        rows.append(("Fetched at", source["fetched_at_utc"]))
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
        ("Rebalance", "weekly (available Mondays)"),
        ("Commission", f"{cost_model.commission * 10_000:.1f} bps per trade"),
        ("Slippage", f"{cost_model.slippage * 10_000:.1f} bps per trade"),
        ("Transfer tax", f"{cost_model.tax * 10_000:.1f} bps on sells"),
        ("ADV cap", "not applied; this report supplies no volume input"),
        ("DSR trial count", str(d["m"]["dsr_n_trials"])),
    ]
    out = ["| Setting | Value |", "|---------|-------|"]
    out.extend(f"| {setting} | {value} |" for setting, value in rows)
    return "\n".join(out)


def positive_int(value: str) -> int:
    """argparse type: a strictly-positive integer (the DSR needs n_trials >= 1)."""
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer (got {value!r})")
    return ivalue


def main(argv) -> int:
    ap = argparse.ArgumentParser(description="generate separate tearsheet charts + tables")
    ap.add_argument("--start", default="2018")
    ap.add_argument("--end", default="2025")
    ap.add_argument("--imgdir", type=Path, default=Path("reports/img"))
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
    ap.add_argument("--n-trials", type=positive_int, default=10,
                    help="trials assumed for the Deflated Sharpe Ratio (default 10)")
    args = ap.parse_args(argv[1:])

    try:
        prices, source_metadata = load_prices(
            f"{args.start}-01-01",
            f"{args.end}-12-31",
            prices_csv=args.prices_csv,
            live_yfinance=args.live_yfinance,
        )
        d = compute(prices, n_trials=args.n_trials)
    except Exception as exc:
        print(f"report generation failed: {exc}", file=sys.stderr)
        return 1

    save_charts(d, args.imgdir)
    window = f"{d['rets'].index[0].date()} to {d['rets'].index[-1].date()}"
    print(f"# Charts written to {args.imgdir}/ for window {window}\n")
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
