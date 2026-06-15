"""Measured out-of-sample performance validation against the design targets.

This harness fetches *real* split/dividend-adjusted prices (yfinance), runs the
reference strategies through the mandatory-cost backtest engine, and reports the
measured CAGR / Sharpe / Sortino / Calmar / max-drawdown plus the Deflated
Sharpe Ratio, side by side with naive buy-and-hold benchmarks.

It is deliberately *honest*: it reports whatever the data shows and makes no
attempt to tune toward the README's aspirational Sharpe targets (doing so on a
single historical window is exactly the overfitting the platform's DSR / BHY
tooling exists to catch). Run it on your own licensed point-in-time history for
a survivorship-safe read on the single-name strategies.

    python scripts/validate_performance.py            # 2018-2025, real data
    python scripts/validate_performance.py 2010 2020  # custom window

Requires network + ``yfinance``; without them it exits rather than report
meaningless synthetic numbers.
"""

from __future__ import annotations

import logging
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

from backtest.costs.transaction_costs import TransactionCostModel
from backtest.engines.vectorized import VectorizedBacktest
from backtest.metrics.tearsheet import Tearsheet
from backtest.validation.deflated_sharpe import compute_dsr

ROTATION_UNIVERSE = ["QQQ", "VGT", "GLD", "TLT", "SPY", "VIG"]
# Liquid large-caps that traded across 2018-2025. NOTE: using today's names is
# survivorship-biased (it excludes delisted/merged firms); a survivorship-safe
# run needs a point-in-time constituents source (see data/universe/).
MOMENTUM_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "V", "PG",
    "HD", "MA", "BAC", "DIS", "ADBE", "CRM", "NFLX", "XOM", "CVX", "KO",
    "PEP", "WMT", "MRK", "PFE", "CSCO", "INTC", "ORCL", "QCOM", "TXN", "COST",
]
CAPITAL = 1_000_000.0


def fetch_prices(symbols, start, end):
    """Real adjusted prices via yfinance, or None if unavailable."""
    try:
        from data.providers.yfinance_provider import YFinanceProvider
    except Exception:
        return None
    px = YFinanceProvider().get_prices(symbols, start=start, end=end)
    if px is None or px.empty:
        return None
    px = px.dropna(how="all").ffill().dropna(how="all")
    px = px.dropna(axis=1, how="any")  # keep only fully-populated names
    return px if px.shape[0] > 252 and px.shape[1] >= 2 else None


def metrics(returns: pd.Series, n_trials: int) -> dict:
    r = returns.dropna()
    ts = Tearsheet(r).compute()
    ts["dsr"] = compute_dsr(r, n_trials=n_trials)
    return ts


def ann_sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")


def backtest_weights(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    res = VectorizedBacktest(TransactionCostModel(), capital=CAPITAL).run(weights, prices)
    return res.returns.dropna()


def run_rotation(prices: pd.DataFrame) -> pd.Series:
    from strategies.multi_asset_rotation import MultiAssetRotation

    weekly = prices.index[prices.index.weekday == 0]
    weights = MultiAssetRotation().generate_weights(prices, weekly)
    return backtest_weights(weights, prices)


def run_momentum_ml(prices: pd.DataFrame) -> pd.Series:
    from strategies.momentum_ml import MomentumMLStrategy

    monthly = prices.resample("MS").first().index
    monthly = monthly[(monthly >= prices.index[0]) & (monthly <= prices.index[-1])]
    weights = MomentumMLStrategy().generate_weights(prices, monthly)
    return backtest_weights(weights, prices)


def fmt_row(name: str, m: dict, target: str = "") -> str:
    return (
        f"  {name:<26} Sharpe {m['sharpe']:+5.2f}  CAGR {m['cagr']:+7.2%}  "
        f"Sortino {m['sortino']:+5.2f}  Calmar {m['calmar']:+5.2f}  "
        f"maxDD {m['max_drawdown']:+6.1%}  DSR {m['dsr']:.3f}  {target}"
    )


def main(argv) -> int:
    start = f"{argv[1]}-01-01" if len(argv) > 1 else "2018-01-01"
    end = f"{argv[2]}-12-31" if len(argv) > 2 else "2025-12-31"
    print(f"quantcortex performance validation | window {start} -> {end}")
    print("=" * 78)

    rot_px = fetch_prices(ROTATION_UNIVERSE, start, end)
    if rot_px is None:
        print("ERROR: could not fetch real prices (need network + yfinance).")
        print("Refusing to report synthetic numbers as a performance validation.")
        return 1
    print(f"rotation data: {rot_px.shape[0]} days x {rot_px.shape[1]} symbols "
          f"({rot_px.index[0].date()} .. {rot_px.index[-1].date()})\n")

    # --- benchmarks ---
    spy = rot_px["SPY"].pct_change() if "SPY" in rot_px else rot_px.pct_change().mean(axis=1)
    ew = rot_px.pct_change().mean(axis=1)
    print("Benchmarks (buy & hold, no costs):")
    print(f"  {'SPY':<26} Sharpe {ann_sharpe(spy):+5.2f}")
    print(f"  {'Equal-weight universe':<26} Sharpe {ann_sharpe(ew):+5.2f}\n")

    print("Strategies (weekly/monthly rebalance, 3bps commission + 10bps slippage):")
    rot = metrics(run_rotation(rot_px), n_trials=10)
    print(fmt_row("multi_asset_rotation", rot, "[target Sharpe > 1.10]"))

    mom_px = fetch_prices(MOMENTUM_UNIVERSE, start, end)
    if mom_px is not None and mom_px.shape[1] >= 10:
        print(f"\nmomentum_ml data: {mom_px.shape[0]} days x {mom_px.shape[1]} names "
              "(survivorship-biased; current large-caps only)")
        mom = metrics(run_momentum_ml(mom_px), n_trials=10)
        print(fmt_row("momentum_ml", mom, "[target Sharpe > 0.9]"))

    print("\n" + "=" * 78)
    print("Note: targets are aspirational design goals, not claims about the")
    print("reference implementation. A defensive rotation underperforms buy-and-")
    print("hold in a bull-dominated window; the single-name read is survivorship-")
    print("biased. Use a licensed point-in-time feed for a clean evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
