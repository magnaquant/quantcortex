"""Measured out-of-sample performance validation against the design targets.

This harness fetches *real* split/dividend-adjusted prices (yfinance), runs the
reference strategies through the mandatory-cost backtest engine, and reports the
measured CAGR / Sharpe / Sortino / Calmar / max-drawdown plus the Deflated
Sharpe Ratio, side by side with naive buy-and-hold benchmarks.

It is deliberately *honest*: it reports whatever the data shows and makes no
attempt to tune toward the README's aspirational Sharpe targets (doing so on a
single historical window is exactly the overfitting the platform's DSR / BHY
tooling exists to catch). A survivorship-safe single-name study requires a
licensed provider with delisted-security prices; this yfinance-only harness
does not provide that dataset.

    python scripts/validate_performance.py --live-yfinance
    python scripts/validate_performance.py 2010 2020 --live-yfinance

The explicit flag acknowledges a live yfinance download. Review the provider's
terms before use. Without network + ``yfinance`` the script exits rather than
report generated numbers.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

logging.getLogger("hmmlearn").setLevel(logging.ERROR)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
# joblib/loky can print a physical-core detection traceback on hosts where CPU
# topology is unreadable; pin it so output stays clean (respects an existing
# override and matches the single-threaded determinism elsewhere).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from quantcortex.backtest.costs.transaction_costs import TransactionCostModel
from quantcortex.backtest.engines.vectorized import VectorizedBacktest
from quantcortex.backtest.metrics.tearsheet import Tearsheet
from quantcortex.backtest.validation.deflated_sharpe import compute_dsr
from quantcortex.data.processors.calendar import (
    first_session_each_week,
    last_session_each_month,
)

ROTATION_UNIVERSE = ["QQQ", "VGT", "GLD", "TLT", "SPY", "VIG"]
# Static current large-caps. Using today's names for historical research is
# survivorship-biased; --pit replaces this list with a start-date index cohort.
MOMENTUM_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "V", "PG",
    "HD", "MA", "BAC", "DIS", "ADBE", "CRM", "NFLX", "XOM", "CVX", "KO",
    "PEP", "WMT", "MRK", "PFE", "CSCO", "INTC", "ORCL", "QCOM", "TXN", "COST",
]
CAPITAL = 1_000_000.0
YFINANCE_NOTICE = (
    "Live Yahoo Finance data is fetched through yfinance. Review Yahoo's terms "
    "and yfinance's legal disclaimer at https://ranaroussi.github.io/yfinance/."
)


def fetch_prices(symbols, start, end):
    """Return real adjusted prices via yfinance, or None if insufficient."""
    try:
        from quantcortex.data.providers.yfinance_provider import YFinanceProvider

        provider_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        px = YFinanceProvider().get_prices(symbols, start=start, end=provider_end)
    except Exception as exc:
        raise RuntimeError(f"yfinance price fetch failed: {exc}") from exc
    if px is None or px.empty:
        return None
    px = px.dropna(how="all").ffill(limit=5).dropna(how="all")
    px = px.dropna(axis=1, how="any")  # keep only fully-populated names
    return px if px.shape[0] > 252 and px.shape[1] >= 2 else None


def metrics(
    returns: pd.Series, n_trials: int, sr_variance: float | None
) -> dict:
    r = returns.dropna()
    ts = Tearsheet(r).compute()
    ts["dsr"] = compute_dsr(
        r, n_trials=n_trials, sr_variance=sr_variance
    )
    return ts


def ann_sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")


def backtest_weights(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    evaluation_start: pd.Timestamp,
) -> pd.Series:
    res = VectorizedBacktest(TransactionCostModel(), capital=CAPITAL).run(weights, prices)
    return res.returns.loc[res.returns.index >= evaluation_start].dropna()


def run_rotation(
    prices: pd.DataFrame, evaluation_start: pd.Timestamp
) -> pd.Series:
    from quantcortex.strategies.multi_asset_rotation import MultiAssetRotation

    weekly = first_session_each_week(prices.index)
    weights = MultiAssetRotation().generate_weights(prices, weekly)
    return backtest_weights(weights, prices, evaluation_start)


def run_momentum_ml(
    prices: pd.DataFrame, evaluation_start: pd.Timestamp
) -> pd.Series:
    from quantcortex.strategies.momentum_ml import MomentumMLStrategy

    monthly = last_session_each_month(prices.index)
    weights = MomentumMLStrategy().generate_weights(prices, monthly)
    return backtest_weights(weights, prices, evaluation_start)


def benchmark_returns(
    prices: pd.DataFrame, evaluation_start: pd.Timestamp
) -> tuple[pd.Series, pd.Series]:
    """Return SPY and equal-initial-weight buy-and-hold basket returns."""
    first = int(prices.index.searchsorted(evaluation_start, side="left"))
    if first >= len(prices):
        raise ValueError("evaluation window begins after the available prices")
    base = max(0, first - 1)
    benchmark_prices = prices.iloc[base:]
    evaluation_index = prices.index[first:]
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


def fmt_row(name: str, m: dict, target: str = "") -> str:
    return (
        f"  {name:<26} Sharpe {m['sharpe']:+5.2f}  CAGR {m['cagr']:+7.2%}  "
        f"Sortino {m['sortino']:+5.2f}  Calmar {m['calmar']:+5.2f}  "
        f"maxDD {m['max_drawdown']:+6.1%}  DSR {m['dsr']:.3f}  {target}"
    )


def momentum_universe(start: str, pit: bool):
    """Resolve the momentum_ml universe.

    With ``--pit`` the universe is the S&P 500 *as of the backtest start*
    (reconstructed from Wikipedia), which removes current-membership look-ahead
    at the start date. It is a fixed start-date cohort, not a dynamically
    reconstituted universe. Pricing still lacks reliable delisted-name history,
    so the read is survivorship-aware, not survivorship-safe. Without ``--pit``
    a small static current-large-cap list is used.
    """
    if not pit:
        return MOMENTUM_UNIVERSE, "static large-caps (survivorship-biased)"
    from quantcortex.data.universe.sp500_universe import SP500Universe

    members = SP500Universe.from_wikipedia().constituents(start)
    if not members:
        raise RuntimeError(f"point-in-time S&P 500 universe is empty for {start}")
    return members, f"S&P 500 point-in-time members as of {start}"


def main(argv) -> int:
    import argparse

    def positive_int(value: str) -> int:
        """argparse type: a strictly-positive integer (the DSR needs n_trials >= 1)."""
        ivalue = int(value)
        if ivalue < 1:
            raise argparse.ArgumentTypeError(f"must be a positive integer (got {value!r})")
        return ivalue

    def nonnegative_int(value: str) -> int:
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError(
                f"must be a non-negative integer (got {value!r})"
            )
        return ivalue

    def nonnegative_float(value: str) -> float:
        fvalue = float(value)
        if not np.isfinite(fvalue) or fvalue < 0.0:
            raise argparse.ArgumentTypeError(
                f"must be a finite non-negative number (got {value!r})"
            )
        return fvalue

    ap = argparse.ArgumentParser(description="quantcortex performance validation")
    ap.add_argument("start_year", nargs="?", default="2018")
    ap.add_argument("end_year", nargs="?", default="2025")
    ap.add_argument(
        "--warmup-years",
        type=nonnegative_int,
        default=2,
        help="pre-evaluation history loaded for signals (default 2)",
    )
    ap.add_argument(
        "--live-yfinance",
        action="store_true",
        required=True,
        help="explicitly permit this run to fetch live data through yfinance",
    )
    ap.add_argument("--pit", action="store_true",
                    help="use the start-date S&P 500 cohort for momentum_ml")
    ap.add_argument("--n-trials", type=positive_int, default=10,
                    help="number of strategy trials assumed for the Deflated Sharpe Ratio; "
                         "set this to the true count of configurations you searched")
    ap.add_argument(
        "--sr-variance",
        type=nonnegative_float,
        default=None,
        help="cross-trial variance of per-observation Sharpe estimates for DSR",
    )
    args = ap.parse_args(argv[1:])
    print(YFINANCE_NOTICE, file=sys.stderr)
    try:
        evaluation_start = pd.Timestamp(f"{args.start_year}-01-01")
        evaluation_end = pd.Timestamp(f"{args.end_year}-12-31")
    except Exception as exc:
        ap.error(f"invalid evaluation year: {exc}")
    if evaluation_start > evaluation_end:
        ap.error("start_year must not be after end_year")
    data_start = evaluation_start - pd.DateOffset(years=args.warmup_years)
    start = evaluation_start.strftime("%Y-%m-%d")
    end = evaluation_end.strftime("%Y-%m-%d")
    fetch_start = data_start.strftime("%Y-%m-%d")
    print(f"quantcortex performance validation | window {start} -> {end}"
          + ("  [PIT universe]" if args.pit else ""))
    print(f"Deflated Sharpe assumes n_trials = {args.n_trials}")
    print(
        "Deflated Sharpe variance = "
        + (
            "single-series estimate"
            if args.sr_variance is None
            else f"{args.sr_variance:.8g} (supplied cross-trial variance)"
        )
    )
    print("=" * 78)

    try:
        rot_px = fetch_prices(ROTATION_UNIVERSE, fetch_start, end)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if rot_px is None:
        print("ERROR: could not fetch real prices (need network + yfinance).")
        print("Refusing to report synthetic numbers as a performance validation.")
        return 1
    missing_rotation = [s for s in ROTATION_UNIVERSE if s not in rot_px.columns]
    if missing_rotation:
        print(
            f"ERROR: incomplete rotation universe from yfinance: {missing_rotation}",
            file=sys.stderr,
        )
        return 1
    warmup_sessions = int((rot_px.index < evaluation_start).sum())
    print(f"rotation data: {rot_px.shape[0]} days x {rot_px.shape[1]} symbols "
          f"({rot_px.index[0].date()} .. {rot_px.index[-1].date()})\n")
    print(f"pre-evaluation warm-up: {warmup_sessions} sessions\n")

    # --- benchmarks ---
    spy, ew = benchmark_returns(rot_px, evaluation_start)
    print("Benchmarks (buy & hold, no costs):")
    print(f"  {'SPY':<26} Sharpe {ann_sharpe(spy):+5.2f}")
    print(
        f"  {'Equal-initial-weight basket':<30} Sharpe {ann_sharpe(ew):+5.2f}\n"
    )

    print("Strategies (weekly/monthly rebalance, 3bps commission + 10bps slippage):")
    rot = metrics(
        run_rotation(rot_px, evaluation_start),
        n_trials=args.n_trials,
        sr_variance=args.sr_variance,
    )
    print(fmt_row("multi_asset_rotation", rot, "[target Sharpe > 1.10]"))

    try:
        mom_syms, mom_label = momentum_universe(start, args.pit)
        mom_px = fetch_prices(mom_syms, fetch_start, end)
    except Exception as exc:
        print(f"ERROR: momentum validation failed: {exc}", file=sys.stderr)
        return 1
    if mom_px is None or mom_px.shape[1] < 10:
        print("ERROR: insufficient complete momentum universe price history.", file=sys.stderr)
        return 1

    coverage = ""
    if args.pit:
        priceable = mom_px.shape[1]
        coverage = (
            f"; {priceable}/{len(mom_syms)} PIT members have complete yfinance "
            f"history ({len(mom_syms) - priceable} missing/incomplete)"
        )
    print(f"\nmomentum_ml universe: {mom_label}{coverage}")
    print(f"momentum_ml data: {mom_px.shape[0]} days x {mom_px.shape[1]} names")
    mom = metrics(
        run_momentum_ml(mom_px, evaluation_start),
        n_trials=args.n_trials,
        sr_variance=args.sr_variance,
    )
    print(fmt_row("momentum_ml", mom, "[target Sharpe > 0.9]"))

    print("\n" + "=" * 78)
    print("Note: targets are aspirational design goals, not claims about the")
    print("reference implementation. Benchmarks are gross; strategy results use")
    print("flat costs without ADV caps. The single-name read is not fully")
    print("survivorship-safe. Use a licensed point-in-time feed for evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
