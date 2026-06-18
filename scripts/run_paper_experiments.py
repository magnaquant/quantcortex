"""Run the fixed experiment suite used by the research paper.

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

from quantcortex.backtest.conformance import (
    target_tape_from_payload,
    target_tape_to_payload,
    target_tape_to_weights,
    weights_to_target_tape,
)
from quantcortex.backtest.costs.transaction_costs import TransactionCostModel
from quantcortex.backtest.engines.event_driven import EventDrivenBacktest
from quantcortex.backtest.engines.vectorized import BacktestResult, VectorizedBacktest
from quantcortex.backtest.metrics.plotting import (
    BRIGHT_COLORS,
    CASH,
    COUNTERFACTUAL_AMBER,
    INK,
    MUTED_INK,
    NEGATIVE_RED,
    POSITIVE_GREEN,
    REFERENCE_BLUE,
    SPINE,
    add_panel_label,
    apply_plot_style,
    style_axis,
)
from quantcortex.data.local_csv import load_price_matrix, sha256_file
from quantcortex.data.processors.calendar import first_session_each_week
from quantcortex.strategies.multi_asset_rotation import MultiAssetRotation

UNIVERSE = ["QQQ", "VGT", "GLD", "TLT", "SPY", "VIG"]
STRATEGY_PARAMETERS = {
    "top_n_groups": 2,
    "ir_lookback": 126,
    "mom_lookback": 126,
    "mom_gap": 21,
    "target_vix": 20.0,
    "max_position_weight": 0.60,
    "regime_backend": "gmm",
    "regime_n_states": 3,
    "regime_covariance_type": "full",
    "regime_n_iter": 100,
    "regime_seed": 42,
    "regime_reg_covar": 1e-5,
    "regime_feature_vol_lookback": 20,
    "vix_floor": 0.3,
    "vix_cap": 1.0,
    "vix_proxy_lookback": 21,
}
VARIANTS = {
    "full": {"regime": True, "vix_scale": True},
    "no_regime": {"regime": False, "vix_scale": True},
    "no_vol_scaler": {"regime": True, "vix_scale": False},
    "signal_only": {"regime": False, "vix_scale": False},
}
COST_LEVELS_BPS = (0.0, 5.0, 13.0, 25.0, 50.0)
BASELINE_COST_BPS = 13.0
PRIMARY_ENGINE = "event_driven"
BOOTSTRAP_BLOCK_LENGTHS = (5, 21, 63)
PRIMARY_BOOTSTRAP_BLOCK_LENGTH = 21
BOOTSTRAP_SEED = 42
PAPER_MAX_FORWARD_FILL = 0
PROVENANCE_SCHEMA_VERSION = 5
EVALUATION_CONTRACT_SCHEMA_VERSION = 1
TARGET_TAPE_SCHEMA_VERSION = 1
TARGET_EXPOSURE_COMPARATOR_SYMBOL = "equal_initial_weight_basket"
DECOMPOSITION_LABELS = {
    "active_risky_allocation": "Active risky allocation",
    "dynamic_exposure_timing": "Dynamic exposure timing",
    "passive_risky_exposure": "Passive risky exposure",
    "implementation_cost": "Modeled implementation cost",
    "net_excess_over_cash": "Net excess over cash",
}
SOURCE_TREE_FIXED_FILES = (
    "scripts/run_paper_experiments.py",
    "schemas/canonical_target_tape.schema.json",
    "schemas/evaluation_contract.schema.json",
    "pyproject.toml",
    "poetry.lock",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def nonempty_text(value: str) -> str:
    parsed = value.strip()
    if not parsed:
        raise argparse.ArgumentTypeError("value must be non-empty")
    return parsed


def iso_timestamp(value: str) -> str:
    parsed = nonempty_text(value)
    candidate = parsed[:-1] + "+00:00" if parsed.endswith("Z") else parsed
    try:
        datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be an ISO-8601 date or timestamp"
        ) from exc
    return parsed


def _git_metadata(repo_root: Path) -> dict[str, str | bool]:
    """Capture source revision and cleanliness before artifact writes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {
            "source_commit": "unavailable",
            "worktree_clean_at_start": False,
        }
    return {
        "source_commit": commit,
        "worktree_clean_at_start": not status.strip(),
    }


def _git_path_is_tracked(repo_root: Path, path: Path) -> bool:
    """Return whether ``path`` is tracked in the source repository."""
    try:
        relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return False
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError("could not determine whether paper input is tracked") from exc
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    detail = result.stderr.strip() or f"git exited with status {result.returncode}"
    raise RuntimeError(
        f"could not determine whether paper input is tracked: {detail}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _threadpool_environment() -> list[dict[str, object]]:
    """Return stable BLAS/OpenMP metadata without machine-specific paths."""
    try:
        from threadpoolctl import threadpool_info
    except ImportError:  # pragma: no cover - a locked core dependency
        return []
    keys = (
        "user_api",
        "internal_api",
        "prefix",
        "version",
        "threading_layer",
        "architecture",
        "num_threads",
    )
    return [
        {key: entry[key] for key in keys if key in entry}
        for entry in threadpool_info()
    ]


def source_tree_manifest(
    repo_root: Path,
    relative_paths: list[str] | None = None,
) -> dict[str, object]:
    """Fingerprint the complete project source tree relevant to the experiment."""
    if relative_paths is None:
        discovered = set(SOURCE_TREE_FIXED_FILES)
        package_root = repo_root / "quantcortex"
        if not package_root.is_dir():
            raise ValueError(f"paper package root is missing: {package_root}")
        discovered.update(
            path.relative_to(repo_root).as_posix()
            for path in package_root.rglob("*.py")
            if path.is_file()
        )
        relative_paths = sorted(discovered)

    paths = [repo_root / relative_path for relative_path in relative_paths]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        missing_names = ", ".join(path.as_posix() for path in missing)
        raise ValueError(f"paper source-tree fingerprint is missing: {missing_names}")
    if not paths:
        raise ValueError("paper source-tree fingerprint matched no files")

    digest = hashlib.sha256()
    files: dict[str, str] = {}
    for path in paths:
        relative = path.relative_to(repo_root).as_posix()
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[relative] = file_digest
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_digest))
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(paths),
        "files": files,
    }


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
    if excess.empty or float(np.ptp(excess.to_numpy(dtype=float))) == 0.0:
        return float("nan")
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
    """Return SPY and an equal-initial-weight buy-and-hold basket."""
    first = prices.index.get_loc(evaluation_index[0])
    last = prices.index.get_loc(evaluation_index[-1])
    base = max(0, first - 1)
    benchmark_prices = prices.iloc[base : last + 1]
    spy = benchmark_prices["SPY"].pct_change(fill_method=None).reindex(
        evaluation_index
    )
    equal_initial_weight_curve = benchmark_prices.div(
        benchmark_prices.iloc[0]
    ).mean(axis=1)
    equal_initial_weight = equal_initial_weight_curve.pct_change(
        fill_method=None
    ).reindex(
        evaluation_index
    )
    if first == 0:
        spy.iloc[0] = 0.0
        equal_initial_weight.iloc[0] = 0.0
    return spy.fillna(0.0), equal_initial_weight.fillna(0.0)


def _costed_target_exposure_comparator(
    strategy_weights: pd.DataFrame,
    prices: pd.DataFrame,
    cash_returns: pd.Series,
    evaluation_index: pd.DatetimeIndex,
    *,
    cost_bps: float,
) -> BacktestResult:
    """Run a causal, costed equal-weight basket at strategy target exposure.

    The synthetic basket is initialized at equal dollar weights at the close
    immediately before evaluation. A seed decision establishes the previously
    active target exposure on that pre-evaluation close, so its initial cost is
    outside the reported window. Later exposure targets retain the strategy's
    original decision timestamps and therefore execute on the next bar.
    """
    if not isinstance(strategy_weights, pd.DataFrame):
        raise TypeError("strategy_weights must be a pandas DataFrame")
    if not isinstance(strategy_weights.index, pd.DatetimeIndex):
        raise TypeError("strategy_weights must use a DatetimeIndex")
    if strategy_weights.index.has_duplicates:
        raise ValueError("strategy_weights must not contain duplicate decisions")
    if evaluation_index.empty:
        raise ValueError("evaluation_index must not be empty")

    first = prices.index.get_loc(evaluation_index[0])
    last = prices.index.get_loc(evaluation_index[-1])
    if first == 0:
        raise ValueError(
            "costed comparator requires one pre-evaluation price observation"
        )
    base = first - 1
    comparator_source = prices.iloc[base : last + 1]
    comparator_curve = comparator_source.div(comparator_source.iloc[0]).mean(axis=1)
    comparator_prices = comparator_curve.rename(
        TARGET_EXPOSURE_COMPARATOR_SYMBOL
    ).to_frame()

    target_exposure = strategy_weights.abs().sum(axis=1).sort_index()
    if not np.all(np.isfinite(target_exposure.to_numpy(dtype=float))):
        raise ValueError("strategy target exposure must be finite")
    if ((target_exposure < -1e-12) | (target_exposure > 1.0 + 1e-12)).any():
        raise ValueError("strategy target exposure must remain in [0, 1]")

    base_timestamp = comparator_prices.index[0]
    prior_targets = target_exposure.loc[target_exposure.index < base_timestamp]
    initial_target = float(prior_targets.iloc[-1]) if not prior_targets.empty else 0.0
    seed_timestamp = base_timestamp - pd.Timedelta(microseconds=1)
    comparator_targets = pd.concat(
        [
            pd.Series([initial_target], index=pd.DatetimeIndex([seed_timestamp])),
            target_exposure.loc[target_exposure.index >= base_timestamp],
        ]
    )
    comparator_targets = comparator_targets[
        ~comparator_targets.index.duplicated(keep="last")
    ]
    comparator_weights = comparator_targets.rename(
        TARGET_EXPOSURE_COMPARATOR_SYMBOL
    ).to_frame()
    comparator_cash = cash_returns.reindex(comparator_prices.index)
    if comparator_cash.isna().any():
        raise ValueError("cash returns do not cover the comparator window")

    return _engine_result(
        comparator_weights,
        comparator_prices,
        comparator_cash,
        cost_bps=cost_bps,
        engine=PRIMARY_ENGINE,
    )


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
        "annualized_gross_traded_notional": float(
            result.traded_notional.reindex(evaluation_index).mean() * 252.0
        ),
        "arithmetic_sum_transaction_cost_return_drag": float(
            result.costs.reindex(evaluation_index).sum()
        ),
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


def circular_block_bootstrap_frame(
    returns: pd.DataFrame,
    *,
    block_length: int = PRIMARY_BOOTSTRAP_BLOCK_LENGTH,
    replications: int = 5_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Joint circular-block bootstrap for annualized arithmetic mean returns.

    Every column uses the same resampled row indices. This preserves exact
    daily accounting identities inside every bootstrap draw.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if returns.shape[1] == 0 or returns.columns.has_duplicates:
        raise ValueError("returns must have unique, non-empty columns")
    if len(returns) < 2:
        raise ValueError("bootstrap requires at least two observations")
    if returns.isna().any(axis=None):
        raise ValueError("bootstrap returns must be complete")
    values = returns.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("bootstrap returns must be finite")
    if block_length <= 0 or block_length > len(values):
        raise ValueError("block_length must be in [1, number of observations]")
    if replications <= 0:
        raise ValueError("replications must be positive")

    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(len(values) / block_length))
    estimates = np.empty((replications, values.shape[1]), dtype=float)
    offsets = np.arange(block_length)
    for replication in range(replications):
        starts = rng.integers(0, len(values), size=blocks)
        indices = (starts[:, None] + offsets[None, :]) % len(values)
        sample = values[indices.ravel()[: len(values)]]
        estimates[replication] = sample.mean(axis=0) * 252.0

    lower, upper = np.quantile(estimates, [0.025, 0.975], axis=0)
    rows = []
    for column_index, column in enumerate(returns.columns):
        rows.append(
            {
                "series": str(column),
                "observations": int(len(values)),
                "block_length": int(block_length),
                "replications": int(replications),
                "seed": int(seed),
                "annualized_mean": float(values[:, column_index].mean() * 252.0),
                "ci_95_lower": float(lower[column_index]),
                "ci_95_upper": float(upper[column_index]),
                "positive_draw_fraction": float(
                    (estimates[:, column_index] > 0.0).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def circular_block_bootstrap(
    returns: pd.Series,
    *,
    block_length: int = PRIMARY_BOOTSTRAP_BLOCK_LENGTH,
    replications: int = 5_000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Circular-block bootstrap for one annualized arithmetic mean return."""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    name = str(returns.name) if returns.name is not None else "series"
    row = circular_block_bootstrap_frame(
        returns.rename(name).to_frame(),
        block_length=block_length,
        replications=replications,
        seed=seed,
    ).iloc[0]
    return {
        key: value.item() if hasattr(value, "item") else value
        for key, value in row.drop(labels="series").items()
    }


def circular_block_bootstrap_sharpe_frame(
    excess_returns: pd.DataFrame,
    *,
    block_length: int = PRIMARY_BOOTSTRAP_BLOCK_LENGTH,
    replications: int = 5_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Joint circular-block intervals for conventional annualized Sharpe.

    The statistic remains the conventional ``sqrt(252)`` sample Sharpe. Block
    resampling supplies dependence-sensitive uncertainty without treating the
    daily observations as independent.
    """
    if not isinstance(excess_returns, pd.DataFrame):
        raise TypeError("excess_returns must be a pandas DataFrame")
    if excess_returns.shape[1] == 0 or excess_returns.columns.has_duplicates:
        raise ValueError("excess_returns must have unique, non-empty columns")
    if len(excess_returns) < 2:
        raise ValueError("bootstrap requires at least two observations")
    if excess_returns.isna().any(axis=None):
        raise ValueError("bootstrap excess returns must be complete")
    values = excess_returns.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("bootstrap excess returns must be finite")
    if block_length <= 0 or block_length > len(values):
        raise ValueError("block_length must be in [1, number of observations]")
    if replications <= 0:
        raise ValueError("replications must be positive")

    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(len(values) / block_length))
    offsets = np.arange(block_length)
    estimates = np.full((replications, values.shape[1]), np.nan, dtype=float)
    for replication in range(replications):
        starts = rng.integers(0, len(values), size=blocks)
        indices = (starts[:, None] + offsets[None, :]) % len(values)
        sample = values[indices.ravel()[: len(values)]]
        standard_deviation = sample.std(axis=0, ddof=1)
        valid = (
            np.isfinite(standard_deviation)
            & (standard_deviation > 0.0)
            & (np.ptp(sample, axis=0) > 0.0)
        )
        estimates[replication, valid] = (
            sample[:, valid].mean(axis=0)
            / standard_deviation[valid]
            * np.sqrt(252.0)
        )

    rows = []
    for column_index, column in enumerate(excess_returns.columns):
        column_estimates = estimates[:, column_index]
        finite = column_estimates[np.isfinite(column_estimates)]
        if finite.size == 0:
            raise ValueError(f"bootstrap Sharpe is undefined for {column!r}")
        lower, upper = np.quantile(finite, [0.025, 0.975])
        rows.append(
            {
                "series": str(column),
                "observations": int(len(values)),
                "block_length": int(block_length),
                "replications": int(replications),
                "finite_replications": int(finite.size),
                "seed": int(seed),
                "sample_sharpe": _sharpe(excess_returns[column]),
                "ci_95_lower": float(lower),
                "ci_95_upper": float(upper),
                "positive_draw_fraction": float((finite > 0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def circular_block_bootstrap_sharpe(
    excess_returns: pd.Series,
    *,
    block_length: int = PRIMARY_BOOTSTRAP_BLOCK_LENGTH,
    replications: int = 5_000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Circular-block interval for one conventional annualized Sharpe."""
    if not isinstance(excess_returns, pd.Series):
        raise TypeError("excess_returns must be a pandas Series")
    name = (
        str(excess_returns.name)
        if excess_returns.name is not None
        else "series"
    )
    row = circular_block_bootstrap_sharpe_frame(
        excess_returns.rename(name).to_frame(),
        block_length=block_length,
        replications=replications,
        seed=seed,
    ).iloc[0]
    return {
        key: value.item() if hasattr(value, "item") else value
        for key, value in row.drop(labels="series").items()
    }


def return_decomposition(
    *,
    net: pd.Series,
    gross: pd.Series,
    cash: pd.Series,
    passive_basket: pd.Series,
    risky_exposure: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Decompose net cash excess into four exact arithmetic components."""
    aligned = pd.concat(
        {
            "net": net,
            "gross": gross,
            "cash": cash,
            "passive_basket": passive_basket,
            "risky_exposure": risky_exposure,
        },
        axis=1,
    )
    if aligned.isna().any(axis=None):
        raise ValueError("return decomposition inputs must share a complete index")
    if not np.all(np.isfinite(aligned.to_numpy(dtype=float))):
        raise ValueError("return decomposition inputs must be finite")
    exposure = aligned["risky_exposure"]
    if ((exposure < -1e-12) | (exposure > 1.0 + 1e-12)).any():
        raise ValueError("risky exposure must remain in [0, 1]")

    constant_exposure = float(exposure.mean())
    constant_passive = (
        constant_exposure * aligned["passive_basket"]
        + (1.0 - constant_exposure) * aligned["cash"]
    )
    matched_passive = (
        exposure * aligned["passive_basket"]
        + (1.0 - exposure) * aligned["cash"]
    )
    components = pd.DataFrame(
        {
            "active_risky_allocation": aligned["gross"] - matched_passive,
            "dynamic_exposure_timing": matched_passive - constant_passive,
            "passive_risky_exposure": constant_passive - aligned["cash"],
            "implementation_cost": aligned["net"] - aligned["gross"],
            "net_excess_over_cash": aligned["net"] - aligned["cash"],
        },
        index=aligned.index,
    )
    reconstructed = components.iloc[:, :4].sum(axis=1)
    if not np.allclose(
        reconstructed.to_numpy(),
        components["net_excess_over_cash"].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("return decomposition identity failed")
    constant_passive.name = "constant_exposure_passive_basket"
    return components, constant_passive


def invalid_same_close_diagnostic(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    cash_returns: pd.Series,
    *,
    cost_bps: float,
) -> dict[str, pd.Series]:
    """Apply close-derived targets to the return ending at that same close.

    This is deliberately invalid and exists only to measure timing-assumption
    sensitivity. It must never be used as an executable backtest path.
    """
    if not isinstance(weights.index, pd.DatetimeIndex):
        raise TypeError("weights must use a DatetimeIndex")
    if weights.index.has_duplicates:
        raise ValueError("weights index must not contain duplicate decisions")
    unknown = sorted(set(weights.columns) - set(prices.columns))
    if unknown:
        raise ValueError(f"weights contain unknown symbols: {unknown}")
    cash = cash_returns.reindex(prices.index)
    if cash.isna().any():
        raise ValueError("cash returns must cover every price bar")

    decisions = weights.sort_index().reindex(columns=prices.columns, fill_value=0.0)
    positions = prices.index.searchsorted(decisions.index, side="left")
    keep = positions < len(prices)
    decisions = decisions.iloc[keep].copy()
    source_positions = positions[keep]
    execution_positions = np.maximum(source_positions - 1, 0)
    decisions.index = pd.DatetimeIndex(
        [
            prices.index[position] - pd.Timedelta(microseconds=1)
            for position in execution_positions
        ]
    )
    decisions = decisions[~decisions.index.duplicated(keep="last")]
    result = _engine_result(
        decisions,
        prices,
        cash,
        cost_bps=cost_bps,
        engine=PRIMARY_ENGINE,
    )
    exposure = result.weights.shift(1).fillna(0.0).abs().sum(axis=1)
    return {
        "net": result.returns,
        "gross": result.gross_returns,
        "cash": cash,
        "exposure": exposure,
        "costs": result.costs,
        "turnover": result.turnover,
        "traded_notional": result.traded_notional,
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
    realized_exposure_attribution: pd.Series,
    target_exposure_costed_comparator: pd.Series,
    cost_results: pd.DataFrame,
    ablation_results: pd.DataFrame,
    ablation_uncertainty: pd.DataFrame,
    engine_series: dict[str, pd.Series],
    decomposition_results: pd.DataFrame,
    protocol_switches: pd.DataFrame,
    bootstrap_sensitivity: pd.DataFrame,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import PercentFormatter

    output_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_style("paper")
    paths: list[Path] = []

    def save(fig, stem: str) -> None:
        for suffix, dpi in (("pdf", None), ("png", 300)):
            path = output_dir / f"{stem}.{suffix}"
            metadata = None
            if suffix == "pdf":
                metadata = {
                    "CreationDate": None,
                    "ModDate": None,
                    "Creator": "quantcortex paper experiment generator",
                }
            fig.savefig(
                path,
                dpi=dpi,
                bbox_inches="tight",
                metadata=metadata,
            )
            paths.append(path)
        plt.close(fig)

    def format_year_axis(ax, interval: int = 2) -> None:
        ax.xaxis.set_major_locator(mdates.YearLocator(base=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig, ax = plt.subplots(figsize=(5.5, 1.55))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    stages = (
        ("Point-in-time\ninputs", "Known at\ndecision"),
        ("Contract-valid\ntargets", "Budgeted;\nbounded"),
        ("Next-bar\nexecution", "No same-close\nreturn"),
        ("Cash and\ncosts", "Explicit\nresidual cash"),
        ("Matched\nevaluation", "Exposure matched;\nuncertainty;\nprovenance"),
    )
    box_width = 0.174
    box_height = 0.40
    lefts = np.linspace(0.015, 0.811, len(stages))
    stage_colors = (
        REFERENCE_BLUE,
        BRIGHT_COLORS[2],
        MUTED_INK,
        COUNTERFACTUAL_AMBER,
        BRIGHT_COLORS[5],
    )
    for index, ((title, subtitle), left, stage_color) in enumerate(
        zip(stages, lefts, stage_colors, strict=True)
    ):
        box = FancyBboxPatch(
            (left, 0.37),
            box_width,
            box_height,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            linewidth=0.9,
            edgecolor=SPINE,
            facecolor="white",
        )
        ax.add_patch(box)
        ax.text(
            left + box_width / 2.0,
            0.735,
            str(index + 1),
            ha="center",
            va="center",
            fontsize=6.4,
            fontweight="bold",
            color="white",
            bbox={
                "boxstyle": "circle,pad=0.18",
                "facecolor": stage_color,
                "edgecolor": "none",
            },
        )
        ax.text(
            left + box_width / 2.0,
            0.625,
            title,
            ha="center",
            va="center",
            fontsize=6.4,
            fontweight="bold",
        )
        ax.text(
            left + box_width / 2.0,
            0.47,
            subtitle,
            ha="center",
            va="center",
            fontsize=5.2,
            color=MUTED_INK,
        )
        if index < len(stages) - 1:
            ax.annotate(
                "",
                xy=(lefts[index + 1] - 0.005, 0.57),
                xytext=(left + box_width + 0.005, 0.57),
                arrowprops={"arrowstyle": "->", "color": MUTED_INK, "lw": 0.8},
            )
    failure_box = FancyBboxPatch(
        (0.05, 0.10),
        0.90,
        0.15,
        boxstyle="round,pad=0.008,rounding_size=0.01",
        linewidth=0.7,
        edgecolor=NEGATIVE_RED,
        facecolor="#FAF0F2",
    )
    ax.add_patch(failure_box)
    ax.text(
        0.5,
        0.175,
        "Fail closed: reject missing, non-causal, non-finite, or contract-violating inputs",
        ha="center",
        va="center",
        fontsize=6.2,
        color=INK,
    )
    save(fig, "audit_protocol")

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35))
    axes[0].plot(
        _growth(baseline_series["gross"]),
        label="Strategy, gross",
        color=BRIGHT_COLORS[0],
        lw=1.0,
        ls="--",
    )
    axes[0].plot(
        _growth(baseline_series["net"]),
        label="Strategy, net",
        color=REFERENCE_BLUE,
        lw=1.35,
    )
    axes[0].plot(
        _growth(realized_exposure_attribution),
        label="Realized-exposure control, gross",
        color=COUNTERFACTUAL_AMBER,
        lw=0.9,
        ls="-.",
    )
    axes[0].plot(
        _growth(target_exposure_costed_comparator),
        label="Target-exposure comparator, net",
        color=POSITIVE_GREEN,
        lw=1.1,
        ls=(0, (3, 1.5)),
    )
    axes[0].plot(
        _growth(baseline_series["cash"]),
        label="SHV cash",
        color=MUTED_INK,
        lw=1.0,
        ls=":",
    )
    axes[0].set_title("Cumulative wealth")
    axes[0].set_ylabel("Growth of $1")
    style_axis(axes[0], grid="y")
    format_year_axis(axes[0])
    axes[0].legend(loc="upper left", ncol=1, fontsize=5.3)
    add_panel_label(axes[0], "a")

    exposure = baseline_series["exposure"]
    axes[1].fill_between(
        exposure.index,
        0.0,
        exposure,
        step="post",
        color=REFERENCE_BLUE,
        alpha=0.62,
        label="Risky exposure",
    )
    axes[1].fill_between(
        exposure.index,
        exposure,
        1.0,
        step="post",
        color=CASH,
        alpha=0.58,
        label="Residual cash",
    )
    axes[1].plot(exposure.index, exposure, color=REFERENCE_BLUE, lw=0.55)
    axes[1].axhline(
        exposure.mean(),
        color=INK,
        lw=0.8,
        ls="--",
        label=f"Mean exposure ({exposure.mean():.1%})",
    )
    axes[1].set_title("Risky exposure and residual cash")
    axes[1].set_ylabel("Capital weight")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    style_axis(axes[1], grid="y")
    format_year_axis(axes[1])
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=5.4,
    )
    add_panel_label(axes[1], "b")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    save(fig, "accounting_summary")

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.55))
    primary = decomposition_results.loc[
        decomposition_results["block_length"] == PRIMARY_BOOTSTRAP_BLOCK_LENGTH
    ].set_index("component")
    component_order = list(DECOMPOSITION_LABELS)
    component_rows = primary.loc[component_order]
    component_means = component_rows["annualized_mean"].to_numpy()
    component_errors = np.vstack(
        [
            component_means - component_rows["ci_95_lower"].to_numpy(),
            component_rows["ci_95_upper"].to_numpy() - component_means,
        ]
    )
    component_colors = [
        REFERENCE_BLUE,
        BRIGHT_COLORS[5],
        POSITIVE_GREEN,
        NEGATIVE_RED,
        INK,
    ]
    component_locations = np.arange(len(component_order), dtype=float)
    for location, mean, lower_error, upper_error, color in zip(
        component_locations,
        component_means,
        component_errors[0],
        component_errors[1],
        component_colors,
        strict=True,
    ):
        axes[0].errorbar(
            mean,
            location,
            xerr=np.array([[lower_error], [upper_error]]),
            fmt="o",
            color=color,
            markersize=4.2,
            capsize=2,
            elinewidth=0.9,
        )
    axes[0].axvline(0.0, color=INK, lw=0.8)
    axes[0].set_title("Exact return attribution")
    axes[0].set_xlabel("Annualized contribution")
    axes[0].set_yticks(
        component_locations,
        [
            "Active allocation",
            "Exposure timing",
            "Passive exposure",
            "Modeled cost",
            "Net excess",
        ],
    )
    axes[0].invert_yaxis()
    axes[0].xaxis.set_major_formatter(PercentFormatter(1.0))
    style_axis(axes[0], grid="x")
    add_panel_label(axes[0], "a")

    switch_locations = np.arange(len(protocol_switches))
    switch_colors = protocol_switches["diagnostic_class"].map(
        {
            "reference": REFERENCE_BLUE,
            "economic_counterfactual": COUNTERFACTUAL_AMBER,
            "causally_invalid": NEGATIVE_RED,
        }
    )
    if switch_colors.isna().any():
        raise ValueError("protocol switches contain an unknown diagnostic class")
    switch_bars = axes[1].barh(
        switch_locations,
        protocol_switches["shv_excess_sharpe"],
        color=switch_colors,
    )
    for bar, diagnostic_class in zip(
        switch_bars,
        protocol_switches["diagnostic_class"],
        strict=True,
    ):
        if diagnostic_class == "causally_invalid":
            bar.set_hatch("///")
    axes[1].axvline(0.0, color=INK, lw=0.8)
    axes[1].set_title("One-assumption diagnostics")
    axes[1].set_xlabel("SHV-excess Sharpe")
    axes[1].set_yticks(
        switch_locations,
        protocol_switches["protocol"].map(
            {
                "audited": "Audited",
                "zero_return_cash": "No cash return",
                "zero_modeled_costs": "No modeled cost",
                "invalid_same_close": "Invalid same-close",
            }
        ),
    )
    axes[1].invert_yaxis()
    axes[1].grid(axis="y", visible=False)
    axes[1].margins(x=0.08)
    values = protocol_switches["shv_excess_sharpe"].to_numpy(dtype=float)
    value_range = max(values.max() - values.min(), 0.1)
    for location, value in zip(switch_locations, values, strict=True):
        inside = abs(value) >= 0.08
        if inside:
            text_x = value - np.sign(value) * 0.035 * value_range
            horizontal_alignment = "right" if value >= 0.0 else "left"
            text_color = "white"
        else:
            text_x = value + np.sign(value or 1.0) * 0.025 * value_range
            horizontal_alignment = "left" if value >= 0.0 else "right"
            text_color = INK
        axes[1].text(
            text_x,
            location,
            f"{value:.2f}",
            ha=horizontal_alignment,
            va="center",
            fontsize=6.4,
            color=text_color,
        )
    style_axis(axes[1], grid="x")
    add_panel_label(axes[1], "b")
    fig.tight_layout()
    save(fig, "return_attribution_and_protocol_switches")

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35))
    axes[0].plot(
        cost_results["all_in_cost_bps"],
        cost_results["net_cash_excess_sharpe"],
        marker="o",
        color=NEGATIVE_RED,
        lw=1.25,
        markersize=3.8,
    )
    axes[0].axhline(0.0, color=INK, lw=0.8)
    axes[0].axvline(
        BASELINE_COST_BPS,
        color=MUTED_INK,
        lw=0.8,
        ls="--",
        label=f"Baseline ({BASELINE_COST_BPS:.0f} bps)",
    )
    axes[0].set_title("Cost sensitivity")
    axes[0].set_xlabel("All-in cost (bps per dollar traded)")
    axes[0].set_ylabel("Cash-excess Sharpe")
    axes[0].set_xticks(cost_results["all_in_cost_bps"])
    style_axis(axes[0], grid="y")
    axes[0].legend(loc="upper right")
    add_panel_label(axes[0], "a")

    ablation_order = list(ablation_results["variant"])
    uncertainty = ablation_uncertainty.set_index("variant").loc[ablation_order]
    active_means = uncertainty["annualized_mean"].to_numpy(dtype=float)
    active_errors = np.vstack(
        [
            active_means - uncertainty["ci_95_lower"].to_numpy(dtype=float),
            uncertainty["ci_95_upper"].to_numpy(dtype=float) - active_means,
        ]
    )
    exposure_by_variant = ablation_results.set_index("variant")[
        "mean_gross_exposure"
    ]
    variant_names = {
        "full": "Full",
        "no_regime": "No regime",
        "no_vol_scaler": "No vol scaler",
        "signal_only": "Signal only",
    }
    labels = [
        f"{variant_names[variant]} ({exposure_by_variant.loc[variant]:.0%} risky)"
        for variant in ablation_order
    ]
    locations = np.arange(len(labels), dtype=float)
    axes[1].errorbar(
        active_means,
        locations,
        xerr=active_errors,
        fmt="o",
        markersize=4.5,
        capsize=2,
        color=REFERENCE_BLUE,
        ecolor=INK,
        elinewidth=1.0,
    )
    axes[1].axvline(0.0, color=INK, lw=0.8)
    axes[1].set_title("Overlay ablations")
    axes[1].set_xlabel("Annualized gross active return")
    axes[1].set_yticks(locations, labels)
    axes[1].invert_yaxis()
    axes[1].xaxis.set_major_formatter(PercentFormatter(1.0))
    style_axis(axes[1], grid="x")
    add_panel_label(axes[1], "b")
    fig.tight_layout()
    save(fig, "sensitivity_and_ablation")

    fig, ax = plt.subplots(figsize=(4.84, 1.95))
    comparison_styles = (
        (
            "strategy_gross_minus_exposure_matched_equal_weight",
            "Gross",
            REFERENCE_BLUE,
            "o",
            -0.10,
        ),
        (
            "strategy_minus_exposure_matched_equal_weight",
            "Net",
            NEGATIVE_RED,
            "s",
            0.10,
        ),
    )
    indexed_sensitivity = bootstrap_sensitivity.set_index(
        ["comparison", "block_length"]
    )
    block_positions = np.arange(len(BOOTSTRAP_BLOCK_LENGTHS), dtype=float)
    for block_position, block_length in zip(
        block_positions,
        BOOTSTRAP_BLOCK_LENGTHS,
        strict=True,
    ):
        for comparison, label, color, marker, offset in comparison_styles:
            row = indexed_sensitivity.loc[(comparison, block_length)]
            mean = float(row["annualized_mean"])
            error = np.array(
                [
                    [mean - float(row["ci_95_lower"])],
                    [float(row["ci_95_upper"]) - mean],
                ]
            )
            ax.errorbar(
                mean,
                block_position + offset,
                xerr=error,
                fmt=marker,
                color=color,
                markersize=4.2,
                capsize=2,
                elinewidth=0.9,
                label=label if block_position == 0 else None,
            )
    ax.axvline(0.0, color=INK, lw=0.8)
    ax.set_title("Realized-exposure active-return robustness", pad=22)
    ax.set_xlabel("Annualized return vs. realized-exposure control")
    ax.set_ylabel("Bootstrap block (sessions)")
    ax.set_yticks(block_positions, [str(value) for value in BOOTSTRAP_BLOCK_LENGTHS])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    style_axis(ax, grid="x")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        columnspacing=0.9,
    )
    fig.tight_layout()
    save(fig, "bootstrap_robustness")

    fig, axes = plt.subplots(2, 1, figsize=(5.06, 3.05), sharex=True)
    engine_styles = {
        "vectorized": {
            "color": REFERENCE_BLUE,
            "linestyle": "-",
            "linewidth": 1.35,
            "label": "Vectorized",
        },
        "event_driven": {
            "color": COUNTERFACTUAL_AMBER,
            "linestyle": "--",
            "linewidth": 1.0,
            "label": "Event-driven",
        },
    }
    for name, returns in engine_series.items():
        axes[0].plot(_growth(returns), **engine_styles[name])
    axes[0].set_title("Backtest-engine wealth at 13 bps")
    axes[0].set_ylabel("Growth of $1")
    style_axis(axes[0], grid="y")
    axes[0].legend(loc="upper left", ncol=2)
    add_panel_label(axes[0], "a")
    difference = engine_series["vectorized"] - engine_series["event_driven"]
    cumulative_difference = difference.cumsum()
    axes[1].plot(cumulative_difference, color=BRIGHT_COLORS[5], lw=1.15)
    axes[1].axhline(0.0, color=INK, lw=0.8)
    axes[1].set_title("Cumulative arithmetic return difference")
    axes[1].set_ylabel("Vectorized - event-driven")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].text(
        0.01,
        0.06,
        (
            f"Final {cumulative_difference.iloc[-1]:+.3%}; "
            f"max |difference| {cumulative_difference.abs().max():.3%}"
        ),
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=MUTED_INK,
    )
    style_axis(axes[1], grid="y")
    add_panel_label(axes[1], "b")
    fig.tight_layout()
    save(fig, "engine_comparison")
    return paths


def evaluation_contract(cash_proxy_symbol: str) -> dict[str, object]:
    """Return the machine-readable contract governing the paper experiment."""
    return {
        "schema_version": EVALUATION_CONTRACT_SCHEMA_VERSION,
        "contract_id": "quantcortex.daily_target_weight_audit.v1",
        "target_tape": {
            "schema_version": TARGET_TAPE_SCHEMA_VERSION,
            "schema_path": "schemas/canonical_target_tape.schema.json",
            "columns": [
                "decision_timestamp",
                "symbol",
                "target_weight",
            ],
            "timestamp_semantics": "close-of-bar decision time",
            "constraints": {
                "long_only": True,
                "maximum_gross_exposure": 1.0,
                "complete_symbol_set_per_decision": True,
            },
        },
        "overlays": {
            "input_contract": "finite contract-valid target weights",
            "output_contract": (
                "finite long-only weights within the declared gross limit"
            ),
            "may_reduce_risky_exposure": True,
            "may_increase_declared_gross_limit": False,
        },
        "information_set": {
            "price_basis": "adjusted close",
            "point_in_time_rule": (
                "every feature value must be available by its decision timestamp"
            ),
            "missing_data_rule": "reject incomplete required observations",
        },
        "execution": {
            "timing": "first observed bar strictly after the decision timestamp",
            "accounting": "adjusted-close pseudo-shares held between rebalances",
            "target_sizing": "post-cost NAV",
        },
        "cash": {
            "residual_weight_rule": "one minus risky asset weights",
            "return_proxy": cash_proxy_symbol,
        },
        "costs": {
            "model": "proportional charge on executed buy and sell notional",
            "baseline_bps_per_dollar_traded": BASELINE_COST_BPS,
            "market_impact": "not modeled",
        },
        "order_state": {
            "intent_persistence": "persist intent before broker submission",
            "uncertain_submission": (
                "block automatic retry until broker reconciliation"
            ),
            "state_revision": "reject stale writes without the current revision",
        },
        "comparators": {
            "realized_exposure_attribution_control": {
                "purpose": "exact ex-post arithmetic attribution",
                "implementable": False,
                "costed": False,
                "exposure_basis": "strategy realized daily risky exposure",
            },
            "target_exposure_costed_comparator": {
                "purpose": "causal implementable comparison",
                "implementable": True,
                "costed": True,
                "exposure_basis": "strategy close-of-bar target gross exposure",
                "risky_asset": "equal-initial-weight buy-and-hold basket",
            },
        },
        "evidence_classes": {
            "exact_identity": "algebraic equality checked on every daily row",
            "property_test": "deterministic invariant over generated inputs",
            "economic_counterfactual": "one declared economic assumption changed",
            "fault_injection": "failure path exercised with malformed or stale state",
            "sdk_conformance": "request and response contract checked offline",
            "untested_live": "requires authenticated external infrastructure",
        },
    }


def _write_json(value: object, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
    return path


def _tex_percent(value: float, digits: int = 2) -> str:
    formatted = f"{value * 100:.{digits}f}\\%"
    return f"\\mbox{{{formatted}}}" if formatted.startswith("-") else formatted


def _tex_number(value: float, digits: int = 2) -> str:
    rounded = round(float(value), digits)
    if rounded == 0.0:
        rounded = 0.0
    formatted = f"{rounded:.{digits}f}"
    return f"\\mbox{{{formatted}}}" if formatted.startswith("-") else formatted


def _write_tex_values(
    experiment: dict[str, object],
    *,
    input_digest: str,
    source_tree_digest: str,
    path: Path,
) -> Path:
    """Write paper-facing values from the same in-memory experiment results."""
    accounting = experiment["accounting"].set_index("series")
    ablation = experiment["ablation"].set_index("variant")
    ablation_uncertainty = experiment["ablation_uncertainty"].set_index("variant")
    costs = experiment["cost_sensitivity"].set_index("all_in_cost_bps")
    comparator_diagnostics = experiment["comparator_diagnostics"].set_index(
        "comparator"
    )
    engines = experiment["engine_comparison"].set_index("engine")
    subperiods = experiment["subperiods"].set_index(["period", "series"])
    decomposition = experiment["return_decomposition"]
    primary_decomposition = decomposition.loc[
        decomposition["block_length"] == PRIMARY_BOOTSTRAP_BLOCK_LENGTH
    ].set_index("component")
    protocol_switches = experiment["protocol_switches"].set_index("protocol")
    uncertainty = experiment["uncertainty"]
    sharpe_uncertainty = experiment["sharpe_uncertainty"].set_index(
        ["block_length", "series"]
    )
    full = ablation.loc["full"]

    net_active = uncertainty["strategy_minus_exposure_matched_equal_weight"]
    gross_active = uncertainty[
        "strategy_gross_minus_exposure_matched_equal_weight"
    ]
    net_cash = uncertainty["strategy_minus_cash"]
    gross_cash = uncertainty["strategy_gross_minus_cash"]
    costed_active = uncertainty[
        "strategy_minus_target_exposure_matched_equal_weight_net"
    ]
    costed_comparator_cash = uncertainty[
        "target_exposure_matched_equal_weight_net_minus_cash"
    ]
    primary_net_sharpe = sharpe_uncertainty.loc[
        (PRIMARY_BOOTSTRAP_BLOCK_LENGTH, "strategy_net_minus_cash")
    ]
    primary_costed_active_sharpe = sharpe_uncertainty.loc[
        (
            PRIMARY_BOOTSTRAP_BLOCK_LENGTH,
            "strategy_net_minus_target_exposure_costed_comparator",
        )
    ]
    primary_costed_comparator_sharpe = sharpe_uncertainty.loc[
        (
            PRIMARY_BOOTSTRAP_BLOCK_LENGTH,
            "target_exposure_costed_comparator_minus_cash",
        )
    ]
    costed_comparator = comparator_diagnostics.loc[
        "target_exposure_matched_equal_weight_net"
    ]

    commands = {
        "PaperObservationCount": f"{len(experiment['evaluation_index']):,}",
        "PaperWarmupSessions": f"{experiment['warmup_sessions']:,}",
        "PaperRequiredWarmupSessions": (
            f"{experiment['required_warmup_sessions']:,}"
        ),
        "PaperBootstrapReplications": f"{net_active['replications']:,}",
        "PaperInputDigest": input_digest,
        "PaperSourceTreeDigest": source_tree_digest,
        "PaperNetCAGR": _tex_percent(
            accounting.loc["strategy_net_shv", "nominal_cagr"]
        ),
        "PaperGrossCAGR": _tex_percent(
            accounting.loc["strategy_gross_shv", "nominal_cagr"]
        ),
        "PaperCashCAGR": _tex_percent(accounting.loc["cash_proxy", "nominal_cagr"]),
        "PaperMatchedCAGR": _tex_percent(
            accounting.loc["exposure_matched_equal_weight", "nominal_cagr"]
        ),
        "PaperCostedComparatorCAGR": _tex_percent(
            accounting.loc[
                "target_exposure_matched_equal_weight_net", "nominal_cagr"
            ]
        ),
        "PaperZeroCashCAGR": _tex_percent(
            accounting.loc["strategy_net_zero_cash", "nominal_cagr"]
        ),
        "PaperNetSharpe": _tex_number(
            accounting.loc["strategy_net_shv", "cash_excess_sharpe"]
        ),
        "PaperGrossSharpe": _tex_number(
            accounting.loc["strategy_gross_shv", "cash_excess_sharpe"]
        ),
        "PaperMatchedSharpe": _tex_number(
            accounting.loc[
                "exposure_matched_equal_weight", "cash_excess_sharpe"
            ]
        ),
        "PaperCostedComparatorSharpe": _tex_number(
            accounting.loc[
                "target_exposure_matched_equal_weight_net",
                "cash_excess_sharpe",
            ]
        ),
        "PaperZeroCashSharpe": _tex_number(
            accounting.loc["strategy_net_zero_cash", "cash_excess_sharpe"]
        ),
        "PaperNetMaxDrawdown": _tex_percent(
            accounting.loc["strategy_net_shv", "max_drawdown"]
        ),
        "PaperGrossMaxDrawdown": _tex_percent(
            accounting.loc["strategy_gross_shv", "max_drawdown"]
        ),
        "PaperCashMaxDrawdown": _tex_percent(
            accounting.loc["cash_proxy", "max_drawdown"]
        ),
        "PaperMatchedMaxDrawdown": _tex_percent(
            accounting.loc["exposure_matched_equal_weight", "max_drawdown"]
        ),
        "PaperCostedComparatorMaxDrawdown": _tex_percent(
            accounting.loc[
                "target_exposure_matched_equal_weight_net", "max_drawdown"
            ]
        ),
        "PaperCostedComparatorTurnover": (
            f"{costed_comparator['annualized_one_way_turnover']:.2f}"
        ),
        "PaperCostedComparatorGrossTradedNotional": (
            f"{costed_comparator['annualized_gross_traded_notional']:.2f}"
        ),
        "PaperZeroCashMaxDrawdown": _tex_percent(
            accounting.loc["strategy_net_zero_cash", "max_drawdown"]
        ),
        "PaperMeanExposure": _tex_percent(full["mean_gross_exposure"]),
        "PaperFullyCash": _tex_percent(full["fully_cash_fraction"]),
        "PaperTurnover": f"{full['annualized_one_way_turnover']:.2f}",
        "PaperGrossTradedNotional": (
            f"{full['annualized_gross_traded_notional']:.2f}"
        ),
        "PaperNetCashActiveMean": _tex_percent(net_cash["annualized_mean"]),
        "PaperNetCashActiveLower": _tex_percent(net_cash["ci_95_lower"]),
        "PaperNetCashActiveUpper": _tex_percent(net_cash["ci_95_upper"]),
        "PaperGrossCashActiveMean": _tex_percent(gross_cash["annualized_mean"]),
        "PaperGrossCashActiveLower": _tex_percent(gross_cash["ci_95_lower"]),
        "PaperGrossCashActiveUpper": _tex_percent(gross_cash["ci_95_upper"]),
        "PaperNetMatchedActiveMean": _tex_percent(net_active["annualized_mean"]),
        "PaperNetMatchedActiveLower": _tex_percent(net_active["ci_95_lower"]),
        "PaperNetMatchedActiveUpper": _tex_percent(net_active["ci_95_upper"]),
        "PaperGrossMatchedActiveMean": _tex_percent(gross_active["annualized_mean"]),
        "PaperGrossMatchedActiveLower": _tex_percent(gross_active["ci_95_lower"]),
        "PaperGrossMatchedActiveUpper": _tex_percent(gross_active["ci_95_upper"]),
        "PaperGrossMatchedPositive": _tex_percent(
            gross_active["positive_draw_fraction"]
        ),
        "PaperNetMatchedPositive": _tex_percent(
            net_active["positive_draw_fraction"]
        ),
        "PaperCostedActiveMean": _tex_percent(costed_active["annualized_mean"]),
        "PaperCostedActiveLower": _tex_percent(costed_active["ci_95_lower"]),
        "PaperCostedActiveUpper": _tex_percent(costed_active["ci_95_upper"]),
        "PaperCostedActivePositive": _tex_percent(
            costed_active["positive_draw_fraction"]
        ),
        "PaperCostedComparatorCashMean": _tex_percent(
            costed_comparator_cash["annualized_mean"]
        ),
        "PaperNetSharpeLower": _tex_number(primary_net_sharpe["ci_95_lower"]),
        "PaperNetSharpeUpper": _tex_number(primary_net_sharpe["ci_95_upper"]),
        "PaperNetSharpePositive": _tex_percent(
            primary_net_sharpe["positive_draw_fraction"]
        ),
        "PaperCostedActiveSharpe": _tex_number(
            primary_costed_active_sharpe["sample_sharpe"]
        ),
        "PaperCostedActiveSharpeLower": _tex_number(
            primary_costed_active_sharpe["ci_95_lower"]
        ),
        "PaperCostedActiveSharpeUpper": _tex_number(
            primary_costed_active_sharpe["ci_95_upper"]
        ),
        "PaperCostedComparatorSharpeLower": _tex_number(
            primary_costed_comparator_sharpe["ci_95_lower"]
        ),
        "PaperCostedComparatorSharpeUpper": _tex_number(
            primary_costed_comparator_sharpe["ci_95_upper"]
        ),
        "PaperAllocationMean": _tex_percent(
            primary_decomposition.loc[
                "active_risky_allocation", "annualized_mean"
            ]
        ),
        "PaperAllocationLower": _tex_percent(
            primary_decomposition.loc[
                "active_risky_allocation", "ci_95_lower"
            ]
        ),
        "PaperAllocationUpper": _tex_percent(
            primary_decomposition.loc[
                "active_risky_allocation", "ci_95_upper"
            ]
        ),
        "PaperTimingMean": _tex_percent(
            primary_decomposition.loc[
                "dynamic_exposure_timing", "annualized_mean"
            ]
        ),
        "PaperPassiveExposureMean": _tex_percent(
            primary_decomposition.loc[
                "passive_risky_exposure", "annualized_mean"
            ]
        ),
        "PaperImplementationCostMean": _tex_percent(
            primary_decomposition.loc["implementation_cost", "annualized_mean"]
        ),
        "PaperNetExcessMean": _tex_percent(
            primary_decomposition.loc["net_excess_over_cash", "annualized_mean"]
        ),
        "PaperSameCloseCAGR": _tex_percent(
            protocol_switches.loc["invalid_same_close", "nominal_cagr"]
        ),
        "PaperSameCloseSharpe": _tex_number(
            protocol_switches.loc["invalid_same_close", "shv_excess_sharpe"]
        ),
        "PaperZeroCostCAGR": _tex_percent(
            protocol_switches.loc["zero_modeled_costs", "nominal_cagr"]
        ),
        "PaperZeroCostSharpe": _tex_number(
            protocol_switches.loc["zero_modeled_costs", "shv_excess_sharpe"]
        ),
        "PaperZeroCashSHVSharpe": _tex_number(
            protocol_switches.loc["zero_return_cash", "shv_excess_sharpe"]
        ),
        "PaperCostZeroSharpe": _tex_number(
            costs.loc[0.0, "net_cash_excess_sharpe"]
        ),
        "PaperCostFiveSharpe": _tex_number(
            costs.loc[5.0, "net_cash_excess_sharpe"]
        ),
        "PaperCostBaselineSharpe": _tex_number(
            costs.loc[13.0, "net_cash_excess_sharpe"]
        ),
        "PaperCostTwentyFiveSharpe": _tex_number(
            costs.loc[25.0, "net_cash_excess_sharpe"]
        ),
        "PaperCostFiftySharpe": _tex_number(
            costs.loc[50.0, "net_cash_excess_sharpe"]
        ),
        "PaperVectorizedCAGR": _tex_percent(
            engines.loc["vectorized", "net_cagr"], digits=3
        ),
        "PaperEventCAGR": _tex_percent(
            engines.loc["event_driven", "net_cagr"], digits=3
        ),
        "PaperVectorizedSharpe": _tex_number(
            engines.loc["vectorized", "net_cash_excess_sharpe"], digits=4
        ),
        "PaperEventSharpe": _tex_number(
            engines.loc["event_driven", "net_cash_excess_sharpe"], digits=4
        ),
    }
    variant_suffixes = {
        "full": "Full",
        "no_regime": "NoRegime",
        "no_vol_scaler": "NoVolScaler",
        "signal_only": "SignalOnly",
    }
    for variant, suffix in variant_suffixes.items():
        commands[f"PaperGrossActive{suffix}"] = _tex_percent(
            ablation.loc[variant, "annualized_arithmetic_gross_active_return"]
        )
        commands[f"PaperGrossActive{suffix}Lower"] = _tex_percent(
            ablation_uncertainty.loc[variant, "ci_95_lower"]
        )
        commands[f"PaperGrossActive{suffix}Upper"] = _tex_percent(
            ablation_uncertainty.loc[variant, "ci_95_upper"]
        )
    period_suffixes = {
        "2018-2021": "FirstHalf",
        "2022-2025": "SecondHalf",
    }
    for period, suffix in period_suffixes.items():
        strategy = subperiods.loc[(period, "strategy_net")]
        matched = subperiods.loc[(period, "exposure_matched_equal_weight")]
        costed = subperiods.loc[
            (period, "target_exposure_matched_equal_weight_net")
        ]
        commands[f"PaperStrategyCAGR{suffix}"] = _tex_percent(strategy["cagr"])
        commands[f"PaperStrategySharpe{suffix}"] = _tex_number(
            strategy["cash_excess_sharpe"]
        )
        commands[f"PaperMatchedCAGR{suffix}"] = _tex_percent(matched["cagr"])
        commands[f"PaperMatchedSharpe{suffix}"] = _tex_number(
            matched["cash_excess_sharpe"]
        )
        commands[f"PaperCostedComparatorCAGR{suffix}"] = _tex_percent(
            costed["cagr"]
        )
        commands[f"PaperCostedComparatorSharpe{suffix}"] = _tex_number(
            costed["cash_excess_sharpe"]
        )

    lines = [
        "% Generated by scripts/run_paper_experiments.py; do not edit.",
        *[
            f"\\newcommand{{\\{name}}}{{{value}}}"
            for name, value in commands.items()
        ],
    ]
    lines.append("\\newcommand{\\PaperDecompositionRows}{%")
    for component in DECOMPOSITION_LABELS:
        row = primary_decomposition.loc[component]
        positive_draws = int(
            round(row["positive_draw_fraction"] * row["replications"])
        )
        lines.append(
            "  "
            + DECOMPOSITION_LABELS[component]
            + " & "
            + _tex_percent(row["annualized_mean"])
            + " & "
            + (
                f"[{_tex_percent(row['ci_95_lower'])}, "
                f"{_tex_percent(row['ci_95_upper'])}]"
            )
            + f" & {positive_draws}/{int(row['replications'])}"
            + r" \\"
        )
    lines.append("}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
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
    target_tape_metadata: dict[str, dict[str, object]] = {}
    ablation_rows: list[dict[str, float | str]] = []
    variant_series: dict[str, dict[str, pd.Series]] = {}
    gross_active_by_variant: dict[str, pd.Series] = {}
    weekly = first_session_each_week(prices.index)
    strategies = {
        variant: MultiAssetRotation(**STRATEGY_PARAMETERS, **configuration)
        for variant, configuration in VARIANTS.items()
    }
    available_warmup_sessions = int(prices.index.get_loc(evaluation_index[0]))
    required_warmup_sessions = max(
        strategy.required_history for strategy in strategies.values()
    )
    if available_warmup_sessions < required_warmup_sessions:
        raise ValueError(
            f"paper experiment provides {available_warmup_sessions} warm-up "
            f"sessions; the fixed strategy variants require at least "
            f"{required_warmup_sessions}"
        )

    for variant, strategy in strategies.items():
        raw_weights = strategy.generate_weights(prices, weekly)
        target_tape = weights_to_target_tape(raw_weights, max_gross=1.0)
        target_payload = target_tape_to_payload(
            target_tape,
            max_gross=1.0,
            expected_symbols=list(prices.columns),
        )
        validated_tape = target_tape_from_payload(target_payload)
        weights = target_tape_to_weights(
            validated_tape,
            max_gross=1.0,
            expected_symbols=list(prices.columns),
        )
        strategy_weights[variant] = weights
        target_tape_metadata[variant] = {
            "schema_version": TARGET_TAPE_SCHEMA_VERSION,
            "decision_count": int(weights.shape[0]),
            "record_count": int(len(target_tape)),
            "symbols": list(target_payload["symbols"]),
            "max_gross": 1.0,
            "canonical_payload_sha256": _json_sha256(target_payload),
        }
        result = _engine_result(
            weights,
            prices,
            cash_returns,
            cost_bps=BASELINE_COST_BPS,
            engine=PRIMARY_ENGINE,
        )
        metrics, series = _summarize(result, cash_returns, evaluation_index)
        variant_series[variant] = series
        ablation_rows.append({"variant": variant, **metrics})

    baseline_series = variant_series["full"]
    spy, equal_initial_weight = _benchmark_returns(prices, evaluation_index)
    ablation = pd.DataFrame(ablation_rows)
    for row_index, variant in enumerate(ablation["variant"]):
        variant_data = variant_series[variant]
        variant_exposure = variant_data["exposure"]
        variant_matched_equal_weight = (
            variant_exposure * equal_initial_weight
            + (1.0 - variant_exposure) * variant_data["cash"]
        )
        net_active_returns = variant_data["net"] - variant_matched_equal_weight
        gross_active_returns = variant_data["gross"] - variant_matched_equal_weight
        gross_active_by_variant[variant] = gross_active_returns.rename(variant)
        ablation.loc[row_index, "matched_equal_weight_cash_excess_sharpe"] = (
            _sharpe(variant_matched_equal_weight, variant_data["cash"])
        )
        ablation.loc[row_index, "net_active_information_ratio"] = _sharpe(
            net_active_returns
        )
        ablation.loc[row_index, "gross_active_information_ratio"] = _sharpe(
            gross_active_returns
        )
        ablation.loc[
            row_index, "annualized_arithmetic_net_active_return"
        ] = float(net_active_returns.mean() * 252.0)
        ablation.loc[
            row_index, "annualized_arithmetic_gross_active_return"
        ] = float(
            gross_active_returns.mean() * 252.0
        )
    ablation_uncertainty = circular_block_bootstrap_frame(
        pd.DataFrame(gross_active_by_variant),
        block_length=PRIMARY_BOOTSTRAP_BLOCK_LENGTH,
        replications=bootstrap_replications,
        seed=BOOTSTRAP_SEED,
    ).rename(columns={"series": "variant"})
    exposure = baseline_series["exposure"]
    cash = baseline_series["cash"]
    matched_spy = exposure * spy + (1.0 - exposure) * cash
    matched_equal_weight = (
        exposure * equal_initial_weight + (1.0 - exposure) * cash
    )
    costed_comparator_result = _costed_target_exposure_comparator(
        strategy_weights["full"],
        prices,
        cash_returns,
        evaluation_index,
        cost_bps=BASELINE_COST_BPS,
    )
    costed_comparator_metrics, costed_comparator_series = _summarize(
        costed_comparator_result,
        cash_returns,
        evaluation_index,
    )
    decomposition_series, constant_exposure_passive = return_decomposition(
        net=baseline_series["net"],
        gross=baseline_series["gross"],
        cash=cash,
        passive_basket=equal_initial_weight,
        risky_exposure=exposure,
    )
    decomposition_rows = []
    for block_length in BOOTSTRAP_BLOCK_LENGTHS:
        statistics = circular_block_bootstrap_frame(
            decomposition_series,
            block_length=block_length,
            replications=bootstrap_replications,
            seed=BOOTSTRAP_SEED,
        ).rename(columns={"series": "component"})
        statistics.insert(
            1,
            "label",
            statistics["component"].map(DECOMPOSITION_LABELS),
        )
        decomposition_rows.append(statistics)
    decomposition_results = pd.concat(decomposition_rows, ignore_index=True)

    accounting_rows = []
    accounting_series = {
        "strategy_net_shv": baseline_series["net"],
        "strategy_gross_shv": baseline_series["gross"],
        "cash_proxy": cash,
        "exposure_matched_spy": matched_spy,
        "exposure_matched_equal_weight": matched_equal_weight,
        "target_exposure_matched_equal_weight_net": costed_comparator_series["net"],
        "constant_exposure_passive_diagnostic": constant_exposure_passive,
    }
    zero_cash_result = _engine_result(
        strategy_weights["full"],
        prices,
        pd.Series(0.0, index=prices.index, name="zero-return cash"),
        cost_bps=BASELINE_COST_BPS,
        engine=PRIMARY_ENGINE,
    )
    _, zero_cash_series = _summarize(
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
    cost_rows = []
    zero_cost_series: dict[str, pd.Series] | None = None
    for cost_bps in COST_LEVELS_BPS:
        result = _engine_result(
            strategy_weights["full"],
            prices,
            cash_returns,
            cost_bps=cost_bps,
            engine=PRIMARY_ENGINE,
        )
        metrics, series = _summarize(result, cash_returns, evaluation_index)
        cost_rows.append({"all_in_cost_bps": cost_bps, **metrics})
        if cost_bps == 0.0:
            zero_cost_series = series
    if zero_cost_series is None:
        raise AssertionError("cost grid must include the zero-cost diagnostic")

    same_close = invalid_same_close_diagnostic(
        strategy_weights["full"],
        prices,
        cash_returns,
        cost_bps=BASELINE_COST_BPS,
    )
    same_close = {
        name: values.reindex(evaluation_index) for name, values in same_close.items()
    }
    if any(values.isna().any() for values in same_close.values()):
        raise AssertionError("same-close diagnostic does not cover the evaluation window")

    def protocol_row(
        *,
        protocol: str,
        display_name: str,
        changed_assumption: str,
        returns: pd.Series,
        diagnostic_class: str,
        causally_valid: bool,
    ) -> dict[str, float | bool | str]:
        excess = returns - cash
        return {
            "protocol": protocol,
            "display_name": display_name,
            "changed_assumption": changed_assumption,
            "diagnostic_class": diagnostic_class,
            "causally_valid": causally_valid,
            "nominal_cagr": _cagr(returns),
            "shv_excess_sharpe": _sharpe(returns, cash),
            "max_drawdown": _max_drawdown(returns),
            "annualized_arithmetic_shv_excess": float(excess.mean() * 252.0),
        }

    protocol_switches = pd.DataFrame(
        [
            protocol_row(
                protocol="audited",
                display_name="Audited",
                changed_assumption="none",
                returns=baseline_series["net"],
                diagnostic_class="reference",
                causally_valid=True,
            ),
            protocol_row(
                protocol="zero_return_cash",
                display_name="Cash return omitted",
                changed_assumption="residual cash earns zero instead of SHV",
                returns=zero_cash_series["net"],
                diagnostic_class="economic_counterfactual",
                causally_valid=True,
            ),
            protocol_row(
                protocol="zero_modeled_costs",
                display_name="Modeled costs omitted",
                changed_assumption="all proportional trading costs set to zero",
                returns=zero_cost_series["net"],
                diagnostic_class="economic_counterfactual",
                causally_valid=True,
            ),
            protocol_row(
                protocol="invalid_same_close",
                display_name="Invalid same-close return capture",
                changed_assumption=(
                    "close-derived target earns the return ending at that close"
                ),
                returns=same_close["net"],
                diagnostic_class="causally_invalid",
                causally_valid=False,
            ),
        ]
    )

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
            "target_exposure_matched_equal_weight_net": costed_comparator_series[
                "net"
            ],
        }
    )
    subperiods = _subperiod_rows(
        {
            "strategy_net": baseline_series["net"],
            "exposure_matched_equal_weight": matched_equal_weight,
            "target_exposure_matched_equal_weight_net": costed_comparator_series[
                "net"
            ],
        },
        (
            ("2018-2021", "2018-01-01", "2021-12-31"),
            ("2022-2025", "2022-01-01", "2025-12-31"),
        ),
        cash,
    )
    bootstrap_series = {
        "strategy_minus_cash": baseline_series["net"] - cash,
        "strategy_gross_minus_cash": baseline_series["gross"] - cash,
        "strategy_minus_exposure_matched_equal_weight": (
            baseline_series["net"] - matched_equal_weight
        ),
        "strategy_gross_minus_exposure_matched_equal_weight": (
            baseline_series["gross"] - matched_equal_weight
        ),
        "strategy_minus_target_exposure_matched_equal_weight_net": (
            baseline_series["net"] - costed_comparator_series["net"]
        ),
        "target_exposure_matched_equal_weight_net_minus_cash": (
            costed_comparator_series["net"] - cash
        ),
    }
    uncertainty: dict[str, dict[str, float | int]] = {}
    bootstrap_rows: list[dict[str, float | int | str]] = []
    for comparison, active_returns in bootstrap_series.items():
        for block_length in BOOTSTRAP_BLOCK_LENGTHS:
            statistics = circular_block_bootstrap(
                active_returns,
                block_length=block_length,
                replications=bootstrap_replications,
                seed=BOOTSTRAP_SEED,
            )
            bootstrap_rows.append({"comparison": comparison, **statistics})
            if block_length == PRIMARY_BOOTSTRAP_BLOCK_LENGTH:
                uncertainty[comparison] = statistics
    sharpe_returns = pd.DataFrame(
        {
            "strategy_net_minus_cash": baseline_series["net"] - cash,
            "strategy_gross_minus_cash": baseline_series["gross"] - cash,
            "strategy_net_minus_realized_exposure_attribution": (
                baseline_series["net"] - matched_equal_weight
            ),
            "strategy_gross_minus_realized_exposure_attribution": (
                baseline_series["gross"] - matched_equal_weight
            ),
            "strategy_net_minus_target_exposure_costed_comparator": (
                baseline_series["net"] - costed_comparator_series["net"]
            ),
            "target_exposure_costed_comparator_minus_cash": (
                costed_comparator_series["net"] - cash
            ),
        }
    )
    sharpe_rows = []
    for block_length in BOOTSTRAP_BLOCK_LENGTHS:
        sharpe_rows.append(
            circular_block_bootstrap_sharpe_frame(
                sharpe_returns,
                block_length=block_length,
                replications=bootstrap_replications,
                seed=BOOTSTRAP_SEED,
            )
        )
    return {
        "evaluation_index": evaluation_index,
        "warmup_sessions": available_warmup_sessions,
        "required_warmup_sessions": required_warmup_sessions,
        "target_tape_metadata": target_tape_metadata,
        "accounting": pd.DataFrame(accounting_rows),
        "ablation": ablation,
        "ablation_uncertainty": ablation_uncertainty,
        "cost_sensitivity": pd.DataFrame(cost_rows),
        "comparator_diagnostics": pd.DataFrame(
            [
                {
                    "comparator": "target_exposure_matched_equal_weight_net",
                    **costed_comparator_metrics,
                }
            ]
        ),
        "engine_comparison": pd.DataFrame(engine_rows),
        "yearly_returns": yearly,
        "subperiods": subperiods,
        "uncertainty": uncertainty,
        "bootstrap_sensitivity": pd.DataFrame(bootstrap_rows),
        "sharpe_uncertainty": pd.concat(sharpe_rows, ignore_index=True),
        "return_decomposition": decomposition_results,
        "protocol_switches": protocol_switches,
        "baseline_series": baseline_series,
        "matched_equal_weight": matched_equal_weight,
        "costed_comparator_metrics": costed_comparator_metrics,
        "costed_comparator_series": costed_comparator_series,
        "engine_series": engine_series,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices-csv", type=Path, required=True)
    parser.add_argument("--cash-proxy-symbol", default="SHV")
    parser.add_argument("--start", default="2018-01-02")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--output-dir", type=Path, default=Path("paper"))
    parser.add_argument("--data-provider", type=nonempty_text, required=True)
    parser.add_argument("--permission-basis", type=nonempty_text, required=True)
    parser.add_argument("--retrieved-at", type=iso_timestamp, required=True)
    parser.add_argument("--adjustment-method", type=nonempty_text, required=True)
    parser.add_argument(
        "--generated-at",
        type=iso_timestamp,
        help=(
            "explicit artifact timestamp; supply this for a byte-reproducible "
            "release manifest"
        ),
    )
    parser.add_argument(
        "--require-clean-source",
        action="store_true",
        help="fail unless the source Git worktree is clean before generation",
    )
    parser.add_argument(
        "--bootstrap-replications",
        type=positive_int,
        default=5_000,
    )
    args = parser.parse_args(argv[1:])

    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    git_metadata = _git_metadata(repo_root)
    if args.require_clean_source and not git_metadata["worktree_clean_at_start"]:
        raise RuntimeError(
            "paper artifact generation requires a clean source worktree"
        )
    source_tree = source_tree_manifest(repo_root)
    dependency_lock_path = repo_root / "poetry.lock"
    if not dependency_lock_path.is_file():
        raise ValueError(f"dependency lock is missing: {dependency_lock_path}")

    source = args.prices_csv.expanduser().resolve()
    raw_input_committed = _git_path_is_tracked(repo_root, source)
    if raw_input_committed:
        raise RuntimeError("paper price input must not be tracked by Git")
    input_digest = sha256_file(source)
    symbols = UNIVERSE + [args.cash_proxy_symbol]
    prices_with_cash = load_price_matrix(
        source,
        symbols=symbols,
        max_ffill=PAPER_MAX_FORWARD_FILL,
        require_complete=True,
    )
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
    contract = evaluation_contract(args.cash_proxy_symbol)
    artifacts = [
        _write_csv(experiment["accounting"], results_dir / "accounting.csv"),
        _write_csv(experiment["ablation"], results_dir / "ablation.csv"),
        _write_csv(
            experiment["ablation_uncertainty"],
            results_dir / "ablation_uncertainty.csv",
        ),
        _write_csv(
            experiment["cost_sensitivity"],
            results_dir / "cost_sensitivity.csv",
        ),
        _write_csv(
            experiment["comparator_diagnostics"],
            results_dir / "comparator_diagnostics.csv",
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
        _write_csv(
            experiment["bootstrap_sensitivity"],
            results_dir / "bootstrap_sensitivity.csv",
        ),
        _write_csv(
            experiment["sharpe_uncertainty"],
            results_dir / "sharpe_uncertainty.csv",
        ),
        _write_csv(
            experiment["return_decomposition"],
            results_dir / "return_decomposition.csv",
        ),
        _write_csv(
            experiment["protocol_switches"],
            results_dir / "protocol_switches.csv",
        ),
        _write_tex_values(
            experiment,
            input_digest=input_digest,
            source_tree_digest=source_tree["sha256"],
            path=results_dir / "generated_values.tex",
        ),
        _write_json(contract, results_dir / "evaluation_contract.json"),
        _write_json(
            experiment["target_tape_metadata"],
            results_dir / "target_tape_hashes.json",
        ),
    ]
    uncertainty_path = results_dir / "uncertainty.json"
    artifacts.append(_write_json(experiment["uncertainty"], uncertainty_path))
    artifacts.extend(
        _save_figures(
            figures_dir,
            experiment["baseline_series"],
            experiment["matched_equal_weight"],
            experiment["costed_comparator_series"]["net"],
            experiment["cost_sensitivity"],
            experiment["ablation"],
            experiment["ablation_uncertainty"],
            experiment["engine_series"],
            experiment["return_decomposition"],
            experiment["protocol_switches"],
            experiment["bootstrap_sensitivity"],
        )
    )

    manifest = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "generated_at": (
            args.generated_at
            if args.generated_at is not None
            else datetime.now(timezone.utc).isoformat(timespec="seconds")
        ),
        "generator": {
            "path": "scripts/run_paper_experiments.py",
            "script_sha256": _sha256(script_path),
            "source_tree": source_tree,
            "git": git_metadata,
            "dependency_lock": {
                "path": "poetry.lock",
                "sha256": _sha256(dependency_lock_path),
            },
            "python": platform.python_version(),
            "platform": platform.platform(),
            "threadpools": _threadpool_environment(),
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
            "input_sha256": input_digest,
            "raw_input_committed": raw_input_committed,
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
            "input_validation": {
                "complete_rows_required": True,
                "max_forward_fill_sessions": PAPER_MAX_FORWARD_FILL,
            },
            "warmup": {
                "available_sessions": experiment["warmup_sessions"],
                "required_sessions": experiment["required_warmup_sessions"],
            },
            "universe": {
                "symbols": UNIVERSE,
                "groups": MultiAssetRotation.GROUPS,
                "group_display_names": MultiAssetRotation.GROUP_DISPLAY_NAMES,
                "benchmark": MultiAssetRotation.BENCHMARK,
            },
            "strategy_parameters": STRATEGY_PARAMETERS,
            "strategy_variants": VARIANTS,
            "signal_fallback": (
                "hold cash when mature selected-group residual scores are all "
                "non-positive; plain momentum is used only when the benchmark, "
                "group signal, or regression estimate is unavailable"
            ),
            "rebalance_rule": "first observed session of each calendar week",
            "signal_timing": (
                "close-derived targets execute on the first strictly later bar"
            ),
            "cash_account": (
                f"residual weight earns {args.cash_proxy_symbol} adjusted-close return"
            ),
            "comparators": {
                "realized_exposure_attribution_control": (
                    "equal-initial-weight buy-and-hold basket scaled by the "
                    "strategy's realized daily risky exposure; residual weight "
                    "earns the cash proxy; ex-post and gross of comparator costs"
                ),
                "target_exposure_costed_comparator": (
                    "event-driven equal-initial-weight buy-and-hold basket using "
                    "the strategy's close-of-bar target gross exposure, next-bar "
                    "execution, residual cash, and the baseline cost model"
                ),
            },
            "return_decomposition": {
                "method": "exact daily arithmetic identity",
                "components": {
                    "active_risky_allocation": (
                        "gross strategy minus daily exposure-matched passive basket"
                    ),
                    "dynamic_exposure_timing": (
                        "daily exposure-matched passive basket minus passive basket "
                        "held at the full-sample mean risky exposure"
                    ),
                    "passive_risky_exposure": (
                        "passive basket held at mean risky exposure minus cash"
                    ),
                    "implementation_cost": "net strategy minus gross strategy",
                },
                "total": "net strategy minus cash",
                "annualization": "252 times the arithmetic daily mean",
            },
            "protocol_switches": {
                "purpose": (
                    "diagnose one-assumption sensitivity; counterfactuals are not "
                    "candidate strategies, and same-close capture is causally invalid"
                ),
                "variants": [
                    "zero_return_cash",
                    "zero_modeled_costs",
                    "invalid_same_close",
                ],
            },
            "cost_model": {
                "all_in_proportional_cost_bps_per_dollar_traded": (
                    BASELINE_COST_BPS
                ),
                "traded_notional_basis": "pre-trade NAV",
                "transaction_cost_charge_basis": (
                    "all-in rate times executed buy and sell notional"
                ),
                "reported_cost_series_basis": (
                    "daily return drag against prior-close NAV; gross minus net return"
                ),
                "market_impact": "not modeled",
            },
            "all_in_cost_levels_bps": COST_LEVELS_BPS,
            "baseline_all_in_cost_bps": BASELINE_COST_BPS,
            "engines": ["vectorized", "event_driven"],
            "primary_engine": PRIMARY_ENGINE,
            "primary_engine_semantics": (
                "explicit adjusted-close pseudo-shares drift between rebalances; "
                "targets are sized against post-cost NAV"
            ),
            "bootstrap": {
                "method": "joint paired circular blocks",
                "interval": "two-sided unstudentized percentile interval",
                "block_length_sessions": PRIMARY_BOOTSTRAP_BLOCK_LENGTH,
                "sensitivity_block_lengths_sessions": BOOTSTRAP_BLOCK_LENGTHS,
                "ablation_intervals": (
                    "joint across all named variants at the primary block length"
                ),
                "sharpe_intervals": (
                    "joint circular-block intervals for the conventional sqrt(252) "
                    "sample Sharpe statistic"
                ),
                "replications": args.bootstrap_replications,
                "seed": BOOTSTRAP_SEED,
            },
            "selection_rule": (
                "all four named audit variants are reported; the audit run does not "
                "select a winner; earlier research trial history is unknown"
            ),
        },
        "evaluation_contract": {
            "path": "results/evaluation_contract.json",
            "schema_version": EVALUATION_CONTRACT_SCHEMA_VERSION,
            "canonical_sha256": _json_sha256(contract),
        },
        "decision_streams": {
            "path": "results/target_tape_hashes.json",
            "variants": experiment["target_tape_metadata"],
        },
        "artifacts": {
            str(path.relative_to(output_dir)): _sha256(path)
            for path in sorted(artifacts)
        },
    }
    manifest["configuration_sha256"] = _json_sha256(manifest["design"])
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
