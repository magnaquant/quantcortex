"""Run the fixed, untuned experiment suite used by the research paper.

The script requires an owner-supplied adjusted-close CSV containing the six
rotation ETFs and a cash proxy. It writes aggregate tables, figures, and a
provenance manifest; it never copies the source price matrix into the repo.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.getLogger("hmmlearn").setLevel(logging.ERROR)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from quantcortex.backtest.costs.transaction_costs import TransactionCostModel
from quantcortex.backtest.engines.event_driven import EventDrivenBacktest
from quantcortex.backtest.engines.vectorized import BacktestResult, VectorizedBacktest
from quantcortex.data.local_csv import load_price_matrix, sha256_file
from quantcortex.data.processors.calendar import first_session_each_week
from quantcortex.strategies.multi_asset_rotation import MultiAssetRotation

UNIVERSE = ["QQQ", "VGT", "GLD", "TLT", "SPY", "VIG"]
VARIANTS = {
    "full": {"regime": True, "vix_scale": True},
    "no_regime": {"regime": False, "vix_scale": True},
    "no_vix": {"regime": True, "vix_scale": False},
    "signal_only": {"regime": False, "vix_scale": False},
}
COST_LEVELS_BPS = (0.0, 5.0, 13.0, 25.0, 50.0)
BASELINE_COST_BPS = 13.0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _growth(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def _cagr(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    values = returns.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return float("nan")
    growth = float(np.prod(1.0 + values))
    return growth ** (periods_per_year / values.size) - 1.0


def _sharpe(returns: pd.Series, risk_free: pd.Series | float = 0.0) -> float:
    excess = (returns - risk_free).dropna()
    standard_deviation = float(excess.std(ddof=1))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
        return float("nan")
    return float(excess.mean() / standard_deviation * np.sqrt(252.0))


def _max_drawdown(returns: pd.Series) -> float:
    growth = _growth(returns)
    return float((growth / growth.cummax().clip(lower=1.0) - 1.0).min())


def _benchmark_returns(
    prices: pd.DataFrame,
    evaluation_index: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
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
    return spy.fillna(0.0), equal_weight.fillna(0.0)


def _engine_result(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    cash_returns: pd.Series,
    *,
    cost_bps: float,
    engine: str,
) -> BacktestResult:
    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValueError("cost_bps must be finite and non-negative")
    cost_model = TransactionCostModel(
        commission=0.0,
        slippage=cost_bps / 10_000.0,
    )
    if engine == "vectorized":
        runner = VectorizedBacktest(cost_model, capital=1.0)
    elif engine == "event_driven":
        runner = EventDrivenBacktest(cost_model, capital=1.0)
    else:
        raise ValueError("engine must be 'vectorized' or 'event_driven'")
    return runner.run(weights, prices, cash_returns=cash_returns)


def _evaluation_index(
    prices: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DatetimeIndex:
    index = prices.index[
        (prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))
    ]
    if index.empty:
        raise ValueError("evaluation window contains no price rows")
    return index


def _summarize(
    result: BacktestResult,
    cash_returns: pd.Series,
    evaluation_index: pd.DatetimeIndex,
) -> tuple[dict[str, float], dict[str, pd.Series]]:
    net = result.returns.reindex(evaluation_index)
    gross = result.gross_returns.reindex(evaluation_index)
    cash = cash_returns.reindex(evaluation_index)
    if net.isna().any() or gross.isna().any() or cash.isna().any():
        raise ValueError("experiment inputs do not fully cover the evaluation window")
    active_weights = result.weights.shift(1).reindex(evaluation_index).fillna(0.0)
    exposure = active_weights.abs().sum(axis=1)
    metrics = {
        "net_cagr": _cagr(net),
        "gross_cagr": _cagr(gross),
        "net_cash_excess_sharpe": _sharpe(net, cash),
        "gross_cash_excess_sharpe": _sharpe(gross, cash),
        "annualized_volatility": float(net.std(ddof=1) * np.sqrt(252.0)),
        "max_drawdown": _max_drawdown(net),
        "annualized_one_way_turnover": float(
            result.turnover.reindex(evaluation_index).mean() * 252.0
        ),
        "sum_cost_fractions": float(result.costs.reindex(evaluation_index).sum()),
        "mean_gross_exposure": float(exposure.mean()),
        "fully_cash_fraction": float((exposure < 1e-12).mean()),
    }
    series = {
        "net": net,
        "gross": gross,
        "cash": cash,
        "exposure": exposure,
    }
    return metrics, series


def circular_block_bootstrap(
    returns: pd.Series,
    *,
    block_length: int = 21,
    replications: int = 5_000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Paired circular-block bootstrap for annualized arithmetic mean return."""
    values = returns.dropna().to_numpy(dtype=float)
    if values.size < 2:
        raise ValueError("bootstrap requires at least two observations")
    if not np.all(np.isfinite(values)):
        raise ValueError("bootstrap returns must be finite")
    if block_length <= 0 or block_length > values.size:
        raise ValueError("block_length must be in [1, number of observations]")
    if replications <= 0:
        raise ValueError("replications must be positive")

    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(values.size / block_length))
    estimates = np.empty(replications, dtype=float)
    offsets = np.arange(block_length)
    for replication in range(replications):
        starts = rng.integers(0, values.size, size=blocks)
        indices = (starts[:, None] + offsets[None, :]) % values.size
        sample = values[indices.ravel()[: values.size]]
        estimates[replication] = float(sample.mean() * 252.0)

    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "observations": int(values.size),
        "block_length": int(block_length),
        "replications": int(replications),
        "seed": int(seed),
        "annualized_mean": float(values.mean() * 252.0),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "bootstrap_probability_positive": float((estimates > 0.0).mean()),
    }


def _yearly_return_rows(series_by_name: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for name, returns in series_by_name.items():
        for year, values in returns.groupby(returns.index.year):
            rows.append(
                {
                    "series": name,
                    "year": int(year),
                    "return": float((1.0 + values).prod() - 1.0),
                }
            )
    return pd.DataFrame(rows)


def _subperiod_rows(
    series_by_name: dict[str, pd.Series],
    periods: tuple[tuple[str, str, str], ...],
    cash_returns: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for period_name, start, end in periods:
        for series_name, returns in series_by_name.items():
            subset = returns.loc[start:end]
            cash = cash_returns.reindex(subset.index)
            rows.append(
                {
                    "period": period_name,
                    "series": series_name,
                    "start": subset.index[0].date().isoformat(),
                    "end": subset.index[-1].date().isoformat(),
                    "cagr": _cagr(subset),
                    "cash_excess_sharpe": _sharpe(subset, cash),
                    "max_drawdown": _max_drawdown(subset),
                }
            )
    return pd.DataFrame(rows)


def _save_figures(
    output_dir: Path,
    baseline_series: dict[str, pd.Series],
    matched_equal_weight: pd.Series,
    yearly_returns: pd.DataFrame,
    cost_results: pd.DataFrame,
    ablation_results: pd.DataFrame,
    engine_series: dict[str, pd.Series],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    paths: list[Path] = []

    def save(fig, stem: str) -> None:
        for suffix, dpi in (("pdf", None), ("png", 180)):
            path = output_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    axes[0].plot(
        _growth(baseline_series["gross"]),
        label="Strategy before costs",
        color="C2",
    )
    axes[0].plot(
        _growth(baseline_series["net"]),
        label="Strategy after costs",
        color="C0",
    )
    axes[0].plot(
        _growth(matched_equal_weight),
        label="Exposure-matched equal weight",
        color="C1",
    )
    axes[0].plot(
        _growth(baseline_series["cash"]),
        label="SHV cash proxy",
        color="C7",
    )
    axes[0].set_title("Accounting changes the conclusion")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(fontsize=8)

    pivot = yearly_returns.pivot(index="year", columns="series", values="return")
    pivot[["strategy_net", "cash_proxy", "exposure_matched_equal_weight"]].plot.bar(
        ax=axes[1],
        color=["C0", "C7", "C1"],
        width=0.8,
    )
    axes[1].set_title("Calendar-year nominal returns")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Return")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].legend(
        ["Strategy net", "SHV", "Exposure-matched equal weight"],
        fontsize=8,
    )
    fig.tight_layout()
    save(fig, "accounting_summary")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    axes[0].plot(
        cost_results["all_in_cost_bps"],
        cost_results["net_cash_excess_sharpe"],
        marker="o",
        color="C3",
    )
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].axvline(BASELINE_COST_BPS, color="C7", lw=0.8, ls="--")
    axes[0].set_title("Cost sensitivity")
    axes[0].set_xlabel("All-in one-way cost (bps)")
    axes[0].set_ylabel("Cash-excess Sharpe")

    labels = [name.replace("_", " ") for name in ablation_results["variant"]]
    locations = np.arange(len(labels), dtype=float)
    width = 0.38
    axes[1].bar(
        locations - width / 2.0,
        ablation_results["net_cash_excess_sharpe"],
        width=width,
        color="C0",
        label="Strategy after costs",
    )
    axes[1].bar(
        locations + width / 2.0,
        ablation_results["matched_equal_weight_cash_excess_sharpe"],
        width=width,
        color="C1",
        label="Exposure-matched equal weight",
    )
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_title("Every overlay variant trails matched passive exposure")
    axes[1].set_ylabel("Cash-excess Sharpe")
    axes[1].set_xticks(locations, labels, rotation=20)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "sensitivity_and_ablation")

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.2), sharex=True)
    for name, returns in engine_series.items():
        axes[0].plot(_growth(returns), label=name.replace("_", " "))
    axes[0].set_title("Backtest-engine comparison at 13 bps")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend()
    difference = engine_series["vectorized"] - engine_series["event_driven"]
    axes[1].plot(difference.cumsum(), color="C4")
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_title("Cumulative arithmetic return difference")
    axes[1].set_ylabel("Vectorized - event driven")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.tight_layout()
    save(fig, "engine_comparison")
    return paths


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
    return path


def run_experiments(
    prices: pd.DataFrame,
    cash_returns: pd.Series,
    *,
    evaluation_start: str,
    evaluation_end: str,
    bootstrap_replications: int,
) -> dict[str, object]:
    evaluation_index = _evaluation_index(
        prices,
        evaluation_start,
        evaluation_end,
    )
    strategy_weights: dict[str, pd.DataFrame] = {}
    ablation_rows: list[dict[str, float | str]] = []
    variant_series: dict[str, dict[str, pd.Series]] = {}
    weekly = first_session_each_week(prices.index)

    for variant, configuration in VARIANTS.items():
        strategy = MultiAssetRotation(**configuration)
        weights = strategy.generate_weights(prices, weekly)
        strategy_weights[variant] = weights
        result = _engine_result(
            weights,
            prices,
            cash_returns,
            cost_bps=BASELINE_COST_BPS,
            engine="vectorized",
        )
        metrics, series = _summarize(result, cash_returns, evaluation_index)
        variant_series[variant] = series
        ablation_rows.append({"variant": variant, **metrics})

    baseline_series = variant_series["full"]
    spy, equal_weight = _benchmark_returns(prices, evaluation_index)
    ablation = pd.DataFrame(ablation_rows)
    for row_index, variant in enumerate(ablation["variant"]):
        variant_data = variant_series[variant]
        variant_exposure = variant_data["exposure"]
        variant_matched_equal_weight = (
            variant_exposure * equal_weight
            + (1.0 - variant_exposure) * variant_data["cash"]
        )
        active_returns = variant_data["net"] - variant_matched_equal_weight
        ablation.loc[row_index, "matched_equal_weight_cash_excess_sharpe"] = (
            _sharpe(variant_matched_equal_weight, variant_data["cash"])
        )
        ablation.loc[row_index, "active_information_ratio"] = _sharpe(
            active_returns
        )
        ablation.loc[row_index, "annualized_arithmetic_active_return"] = float(
            active_returns.mean() * 252.0
        )
    exposure = baseline_series["exposure"]
    cash = baseline_series["cash"]
    matched_spy = exposure * spy + (1.0 - exposure) * cash
    matched_equal_weight = exposure * equal_weight + (1.0 - exposure) * cash

    accounting_rows = []
    accounting_series = {
        "strategy_net_shv": baseline_series["net"],
        "strategy_gross_shv": baseline_series["gross"],
        "cash_proxy": cash,
        "exposure_matched_spy": matched_spy,
        "exposure_matched_equal_weight": matched_equal_weight,
    }
    zero_cash_result = _engine_result(
        strategy_weights["full"],
        prices,
        pd.Series(0.0, index=prices.index, name="zero-return cash"),
        cost_bps=BASELINE_COST_BPS,
        engine="vectorized",
    )
    zero_cash_metrics, zero_cash_series = _summarize(
        zero_cash_result,
        pd.Series(0.0, index=prices.index),
        evaluation_index,
    )
    accounting_series["strategy_net_zero_cash"] = zero_cash_series["net"]
    for name, returns in accounting_series.items():
        risk_free = 0.0 if name == "strategy_net_zero_cash" else cash
        accounting_rows.append(
            {
                "series": name,
                "nominal_cagr": _cagr(returns),
                "cash_excess_sharpe": _sharpe(returns, risk_free),
                "max_drawdown": _max_drawdown(returns),
            }
        )
    if not np.isclose(
        zero_cash_metrics["mean_gross_exposure"],
        float(baseline_series["exposure"].mean()),
    ):
        raise AssertionError("cash-return assumptions changed the strategy exposure")

    cost_rows = []
    for cost_bps in COST_LEVELS_BPS:
        result = _engine_result(
            strategy_weights["full"],
            prices,
            cash_returns,
            cost_bps=cost_bps,
            engine="vectorized",
        )
        metrics, _ = _summarize(result, cash_returns, evaluation_index)
        cost_rows.append({"all_in_cost_bps": cost_bps, **metrics})

    engine_rows = []
    engine_series = {}
    for engine in ("vectorized", "event_driven"):
        result = _engine_result(
            strategy_weights["full"],
            prices,
            cash_returns,
            cost_bps=BASELINE_COST_BPS,
            engine=engine,
        )
        metrics, series = _summarize(result, cash_returns, evaluation_index)
        engine_rows.append({"engine": engine, **metrics})
        engine_series[engine] = series["net"]

    yearly = _yearly_return_rows(
        {
            "strategy_net": baseline_series["net"],
            "cash_proxy": cash,
            "exposure_matched_equal_weight": matched_equal_weight,
        }
    )
    subperiods = _subperiod_rows(
        {
            "strategy_net": baseline_series["net"],
            "exposure_matched_equal_weight": matched_equal_weight,
        },
        (
            ("2018-2021", "2018-01-01", "2021-12-31"),
            ("2022-2025", "2022-01-01", "2025-12-31"),
        ),
        cash,
    )
    uncertainty = {
        "strategy_minus_cash": circular_block_bootstrap(
            baseline_series["net"] - cash,
            replications=bootstrap_replications,
        ),
        "strategy_minus_exposure_matched_equal_weight": circular_block_bootstrap(
            baseline_series["net"] - matched_equal_weight,
            replications=bootstrap_replications,
        ),
    }
    return {
        "evaluation_index": evaluation_index,
        "accounting": pd.DataFrame(accounting_rows),
        "ablation": ablation,
        "cost_sensitivity": pd.DataFrame(cost_rows),
        "engine_comparison": pd.DataFrame(engine_rows),
        "yearly_returns": yearly,
        "subperiods": subperiods,
        "uncertainty": uncertainty,
        "baseline_series": baseline_series,
        "matched_equal_weight": matched_equal_weight,
        "engine_series": engine_series,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices-csv", type=Path, required=True)
    parser.add_argument("--cash-proxy-symbol", default="SHV")
    parser.add_argument("--start", default="2018-01-02")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--output-dir", type=Path, default=Path("paper"))
    parser.add_argument("--data-provider", default=None)
    parser.add_argument("--permission-basis", default=None)
    parser.add_argument("--retrieved-at", default=None)
    parser.add_argument("--adjustment-method", default=None)
    parser.add_argument(
        "--bootstrap-replications",
        type=positive_int,
        default=5_000,
    )
    args = parser.parse_args(argv[1:])

    source = args.prices_csv.expanduser().resolve()
    symbols = UNIVERSE + [args.cash_proxy_symbol]
    prices_with_cash = load_price_matrix(source, symbols=symbols)
    prices = prices_with_cash.loc[:, UNIVERSE]
    cash_returns = prices_with_cash[args.cash_proxy_symbol].pct_change(
        fill_method=None
    ).fillna(0.0)
    cash_returns.name = f"{args.cash_proxy_symbol} adjusted-close proxy"
    if pd.Timestamp(args.start) <= prices.index[0]:
        raise ValueError("evaluation start must leave pre-evaluation signal history")

    experiment = run_experiments(
        prices,
        cash_returns,
        evaluation_start=args.start,
        evaluation_end=args.end,
        bootstrap_replications=args.bootstrap_replications,
    )
    output_dir = args.output_dir.expanduser().resolve()
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    artifacts = [
        _write_csv(experiment["accounting"], results_dir / "accounting.csv"),
        _write_csv(experiment["ablation"], results_dir / "ablation.csv"),
        _write_csv(
            experiment["cost_sensitivity"],
            results_dir / "cost_sensitivity.csv",
        ),
        _write_csv(
            experiment["engine_comparison"],
            results_dir / "engine_comparison.csv",
        ),
        _write_csv(
            experiment["yearly_returns"],
            results_dir / "yearly_returns.csv",
        ),
        _write_csv(experiment["subperiods"], results_dir / "subperiods.csv"),
    ]
    uncertainty_path = results_dir / "uncertainty.json"
    uncertainty_path.write_text(
        json.dumps(experiment["uncertainty"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts.append(uncertainty_path)
    artifacts.extend(
        _save_figures(
            figures_dir,
            experiment["baseline_series"],
            experiment["matched_equal_weight"],
            experiment["yearly_returns"],
            experiment["cost_sensitivity"],
            experiment["ablation"],
            experiment["engine_series"],
        )
    )

    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": {
            "path": "scripts/run_paper_experiments.py",
            "script_sha256": _sha256(script_path),
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "matplotlib",
                    "threadpoolctl",
                )
            },
        },
        "source": {
            "file_name": source.name,
            "input_sha256": sha256_file(source),
            "raw_input_committed": False,
            "symbols": symbols,
            "cash_proxy": args.cash_proxy_symbol,
            "provider": args.data_provider,
            "permission_basis": args.permission_basis,
            "retrieved_at": args.retrieved_at,
            "adjustment_method": args.adjustment_method,
            "price_window": (
                f"{prices.index[0].date()} to {prices.index[-1].date()}"
            ),
        },
        "design": {
            "evaluation_window": f"{args.start} to {args.end}",
            "strategy_variants": VARIANTS,
            "all_in_cost_levels_bps": COST_LEVELS_BPS,
            "baseline_all_in_cost_bps": BASELINE_COST_BPS,
            "engines": ["vectorized", "event_driven"],
            "bootstrap": {
                "method": "paired circular blocks",
                "block_length_sessions": 21,
                "replications": args.bootstrap_replications,
                "seed": 42,
            },
            "selection_rule": "none; all predefined variants are reported",
        },
        "artifacts": {
            str(path.relative_to(output_dir)): _sha256(path)
            for path in sorted(artifacts)
        },
    }
    manifest_path = results_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"paper experiments written to {output_dir}")
    print(f"input SHA-256: {manifest['source']['input_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
