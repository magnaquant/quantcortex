"""Generate separate, publication-quality charts + markdown tables for a backtest.

Runs the multi_asset_rotation strategy on real data through the mandatory-cost
engine and emits, as *separate* artifacts (so they can be embedded individually
in docs):

* ``docs/img/equity_vs_benchmarks.png`` - growth of $1 vs SPY and equal-weight
* ``docs/img/drawdown.png``             - underwater drawdown
* ``docs/img/rolling_sharpe.png``       - rolling 126-day Sharpe
* a **performance metrics** markdown table (printed to stdout)
* a **monthly returns** markdown table (printed to stdout)

Every number printed is computed from the run, so docs can quote it verbatim.
The backtest is deterministic (see timing/hmm_regime), so the figures reproduce
given the same price-data window. Honest by construction - no tuning toward the
design targets. Needs network + yfinance + matplotlib.

    python scripts/generate_report.py
    python scripts/generate_report.py --start 2015 --end 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import quantcortex
from quantcortex.backtest.costs.transaction_costs import TransactionCostModel
from quantcortex.backtest.engines.vectorized import VectorizedBacktest
from quantcortex.backtest.metrics.tearsheet import Tearsheet
from quantcortex.backtest.validation.deflated_sharpe import compute_dsr

ROTATION_UNIVERSE = ["QQQ", "VGT", "GLD", "TLT", "SPY", "VIG"]
# The snapshot is package data; resolve it via the package so the path stays
# correct regardless of where this script lives or future directory moves.
SNAPSHOT = Path(quantcortex.__file__).resolve().parent / "data" / "sample" / "rotation_prices.csv"


def load_prices(start: str, end: str, live: bool) -> pd.DataFrame:
    """Load prices for the report.

    Default: the bundled fixed snapshot (``quantcortex/data/sample/rotation_prices.csv``) so
    the committed charts/tables are exactly reproducible.  ``--live`` refetches
    from yfinance instead - note that yfinance re-adjusts historical closes over
    time, so a live fetch will differ slightly from the snapshot (the strategy
    itself is deterministic given fixed data).
    """
    if not live and SNAPSHOT.exists():
        px = pd.read_csv(SNAPSHOT, index_col="date", parse_dates=True)
        return px.loc[start:end].dropna(how="all").ffill().dropna()
    from quantcortex.data.providers.yfinance_provider import YFinanceProvider

    px = YFinanceProvider().get_prices(ROTATION_UNIVERSE, start=start, end=end)
    if px is None or px.empty:
        raise RuntimeError("could not fetch prices (need network + yfinance), and "
                           "no bundled snapshot is available")
    return px.dropna(how="all").ffill().dropna()


def _growth(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def _ann_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")


def compute(start: str, end: str, live: bool = False, n_trials: int = 10) -> dict:
    from quantcortex.strategies.multi_asset_rotation import MultiAssetRotation

    px = load_prices(start, end, live)
    weekly = px.index[px.index.weekday == 0]
    weights = MultiAssetRotation().generate_weights(px, weekly)
    rets = VectorizedBacktest(TransactionCostModel(), capital=1.0).run(weights, px).returns.dropna()

    ts = Tearsheet(rets)
    m = ts.compute()
    m["dsr"] = compute_dsr(rets, n_trials=n_trials)
    m["dsr_n_trials"] = n_trials
    spy = px["SPY"].pct_change().reindex(rets.index)
    ew = px.pct_change().mean(axis=1).reindex(rets.index)
    return {
        "px": px, "rets": rets, "ts": ts, "m": m,
        "strat_g": _growth(rets), "spy_g": _growth(spy), "ew_g": _growth(ew),
        "spy_sharpe": _ann_sharpe(spy), "ew_sharpe": _ann_sharpe(ew),
        "monthly": ts.monthly_returns_table(),
    }


def save_charts(d: dict, imgdir: Path) -> None:
    imgdir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-darkgrid")

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(d["strat_g"].index, d["strat_g"].to_numpy(), label="Multi-Asset Rotation", color="C0", lw=1.7)
    ax.plot(d["spy_g"].index, d["spy_g"].to_numpy(), label="SPY (buy & hold)", color="C7", lw=1.1, alpha=0.85)
    ax.plot(d["ew_g"].index, d["ew_g"].to_numpy(), label="Equal-weight 6-ETF", color="C2", lw=1.1, alpha=0.85)
    ax.set_title("Growth of $1 - Multi-Asset Rotation vs benchmarks (net of costs)")
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
        ("VaR 95% (daily)", f"{m['var_95']:.2%}"),
        ("CVaR 95% (daily)", f"{m['cvar_95']:.2%}"),
        (f"Deflated Sharpe ({m['dsr_n_trials']} trials)", f"{m['dsr']:.3f}"),
        ("SPY buy & hold Sharpe", f"{d['spy_sharpe']:+.2f}"),
        ("Equal-weight 6-ETF Sharpe", f"{d['ew_sharpe']:+.2f}"),
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


def main(argv) -> int:
    ap = argparse.ArgumentParser(description="generate separate tearsheet charts + tables")
    ap.add_argument("--start", default="2018")
    ap.add_argument("--end", default="2025")
    ap.add_argument("--imgdir", default="docs/img")
    ap.add_argument("--live", action="store_true",
                    help="refetch from yfinance instead of the bundled snapshot")
    ap.add_argument("--n-trials", type=int, default=10,
                    help="trials assumed for the Deflated Sharpe Ratio (default 10; the "
                         "committed README report uses this default)")
    args = ap.parse_args(argv[1:])

    try:
        d = compute(f"{args.start}-01-01", f"{args.end}-12-31", live=args.live,
                    n_trials=args.n_trials)
    except Exception as exc:
        print(f"report generation failed: {exc}")
        return 1

    save_charts(d, Path(args.imgdir))
    window = f"{d['rets'].index[0].date()} to {d['rets'].index[-1].date()}"
    print(f"# Charts written to {args.imgdir}/ for window {window}\n")
    print("## Performance metrics (markdown)\n")
    print(markdown_metrics(d))
    print("\n## Monthly returns %, (markdown)\n")
    print(markdown_monthly(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
