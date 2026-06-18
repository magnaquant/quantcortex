#!/usr/bin/env python3
"""Run the frozen multi-panel evaluation-contract expansion."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from quantcortex.backtest.conformance import (
    target_tape_to_payload,
    weights_to_target_tape,
)
from quantcortex.backtest.metrics.plotting import (
    BRIGHT_COLORS,
    INK,
    MUTED_INK,
    NEGATIVE_RED,
    REFERENCE_BLUE,
    SPINE,
    add_panel_label,
    apply_plot_style,
    style_axis,
)
from quantcortex.data.local_csv import load_price_matrix, sha256_file
from quantcortex.research.expansion import (
    FROZEN_PROTOCOL_COMMIT,
    FROZEN_PROTOCOL_SHA256,
    bootstrap_metric_difference,
    cross_sectional_momentum_targets,
    exposure_matched_comparator_targets,
    invalid_same_close_result,
    learned_gbrt_targets,
    monthly_decision_dates,
    performance_metrics,
    run_engine,
    short_term_reversal_targets,
    time_series_momentum_targets,
    validate_price_panel,
)

VARIANTS = (
    "baseline",
    "same_close",
    "zero_cash",
    "zero_cost",
    "costed_comparator",
    "vectorized",
)
SWITCHES = {
    "same_close_minus_baseline": ("same_close", "baseline"),
    "zero_cash_minus_baseline": ("zero_cash", "baseline"),
    "zero_cost_minus_baseline": ("zero_cost", "baseline"),
    "strategy_minus_costed_comparator": ("baseline", "costed_comparator"),
    "vectorized_minus_event": ("vectorized", "baseline"),
}
STRATEGY_LABELS = {
    "ts_momentum": "TS momentum",
    "cross_sectional_momentum": "Cross-sectional momentum",
    "short_term_reversal": "Short-term reversal",
    "learned_gbrt": "Learned GBRT",
}
SWITCH_LABELS = {
    "same_close_minus_baseline": "Same-close - next-bar",
    "zero_cash_minus_baseline": "Zero cash - SHV cash",
    "zero_cost_minus_baseline": "Zero cost - 13 bp cost",
    "strategy_minus_costed_comparator": "Strategy - costed comparator",
    "vectorized_minus_event": "Vectorized - event-driven",
}
PANEL_LABELS = {
    "us_sector_etfs": "U.S. sector ETFs",
    "country_equity_etfs": "Country equity ETFs",
}
PROVIDER_NAME = "Yahoo Finance via yfinance"
TERMS_URLS = (
    "https://ranaroussi.github.io/yfinance/",
    "https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html",
)
SOURCE_FILES = (
    "scripts/fetch_expansion_data.py",
    "scripts/release_expansion_artifacts.sh",
    "scripts/run_expansion_experiments.py",
    "paper/preregistration.md",
    "paper/expansion/protocol.json",
    "schemas/canonical_target_tape.schema.json",
    "pyproject.toml",
    "poetry.lock",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_aware_timestamp(value: str, *, name: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a UTC offset")
    return parsed


def _threadpool_environment() -> list[dict[str, object]]:
    """Return stable BLAS/OpenMP metadata without machine-specific paths."""
    from threadpoolctl import threadpool_info

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


def _git_metadata(repo_root: Path) -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "source_commit": commit,
        "tracked_worktree_clean_at_start": not tracked_status.strip(),
    }


def _source_tree_manifest(repo_root: Path) -> dict[str, object]:
    relative_paths = set(SOURCE_FILES)
    relative_paths.update(
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "quantcortex").rglob("*.py")
        if path.is_file()
    )
    files = {}
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        path = repo_root / relative
        if not path.is_file():
            raise ValueError(f"source fingerprint is missing {relative}")
        content_digest = _sha256(path)
        files[relative] = content_digest
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_digest.encode("ascii"))
        digest.update(b"\n")
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def _load_protocol(path: Path) -> dict[str, object]:
    if _sha256(path) != FROZEN_PROTOCOL_SHA256:
        raise ValueError("expansion protocol differs from the pre-data freeze")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported expansion protocol version")
    if protocol.get("status") != (
        "repository_frozen_prospective_not_externally_registered"
    ):
        raise ValueError("expansion protocol is not frozen")
    if protocol.get("historical_case_confirmatory") is not False:
        raise ValueError("historical case must remain non-confirmatory")
    return protocol


def _load_panel(
    repo_root: Path,
    panel_dir: Path,
    panel_name: str,
    risky_symbols: list[str],
    protocol: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    data = protocol["data"]
    cash_symbol = data["cash_proxy"]
    csv_path = panel_dir / f"{panel_name}.csv"
    metadata_path = panel_dir / f"{panel_name}.metadata.json"
    if not csv_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"missing local panel or metadata for {panel_name}")
    try:
        relative_csv = csv_path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        relative_csv = None
    if relative_csv is not None:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative_csv],
            cwd=repo_root,
            check=False,
            capture_output=True,
        )
        if tracked.returncode == 0:
            raise ValueError(f"raw panel must not be tracked: {csv_path}")
        if tracked.returncode != 1:
            raise RuntimeError(f"could not determine tracking status for {csv_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_protocol_digest = _sha256(repo_root / "paper/expansion/protocol.json")
    if metadata.get("schema_version") != 1:
        raise ValueError(f"unsupported metadata version for {panel_name}")
    if metadata.get("panel") != panel_name:
        raise ValueError(f"metadata panel mismatch for {panel_name}")
    if metadata.get("protocol_sha256") != expected_protocol_digest:
        raise ValueError(f"metadata protocol digest mismatch for {panel_name}")
    if metadata.get("input_sha256") != sha256_file(csv_path):
        raise ValueError(f"input digest mismatch for {panel_name}")
    if metadata.get("raw_data_committed") is not False:
        raise ValueError(f"metadata must mark raw {panel_name} data uncommitted")
    symbols = [*risky_symbols, cash_symbol]
    if metadata.get("symbols") != symbols:
        raise ValueError(f"metadata symbol mismatch for {panel_name}")
    if metadata.get("request") != data["provider_request"]:
        raise ValueError(f"metadata request mismatch for {panel_name}")
    if metadata.get("protocol_path") != "paper/expansion/protocol.json":
        raise ValueError(f"metadata protocol path mismatch for {panel_name}")
    if metadata.get("provider") != PROVIDER_NAME:
        raise ValueError(f"metadata provider mismatch for {panel_name}")
    if metadata.get("terms_urls") != list(TERMS_URLS):
        raise ValueError(f"metadata terms URLs mismatch for {panel_name}")
    if not isinstance(metadata.get("yfinance_version"), str) or not metadata[
        "yfinance_version"
    ].strip():
        raise ValueError(f"metadata provider version is missing for {panel_name}")
    if metadata.get("provider_terms_independently_verified") is not False:
        raise ValueError(f"metadata terms status mismatch for {panel_name}")
    _parse_aware_timestamp(metadata.get("retrieved_at"), name="retrieved_at")
    prices = load_price_matrix(
        csv_path,
        symbols=symbols,
        start=data["start"],
        end=data["evaluation_end"],
        max_ffill=None,
        require_complete=True,
    )
    validated = validate_price_panel(
        prices,
        risky_symbols=risky_symbols,
        cash_symbol=cash_symbol,
    )
    evaluation = validated.loc[data["evaluation_start"] : data["evaluation_end"]]
    pre_evaluation_sessions = int(
        (validated.index < pd.Timestamp(data["evaluation_start"])).sum()
    )
    pre_evaluation_months = int(
        validated.index[
            validated.index < pd.Timestamp(data["evaluation_start"])
        ].to_period("M").nunique()
    )
    if pre_evaluation_sessions < data["minimum_pre_evaluation_sessions"]:
        raise ValueError(f"panel {panel_name} has insufficient pre-evaluation history")
    expected_months = pd.period_range(
        pd.Timestamp(data["evaluation_start"]).to_period("M"),
        pd.Timestamp(data["evaluation_end"]).to_period("M"),
        freq="M",
    )
    missing_months = expected_months.difference(evaluation.index.to_period("M").unique())
    if len(missing_months):
        raise ValueError(
            f"panel {panel_name} is missing evaluation months: "
            + ", ".join(str(month) for month in missing_months)
        )
    expected_metadata = {
        "complete_rows": len(validated),
        "evaluation_sessions": len(evaluation),
        "first_date": validated.index[0].date().isoformat(),
        "last_date": validated.index[-1].date().isoformat(),
        "pre_evaluation_months": pre_evaluation_months,
        "pre_evaluation_sessions": pre_evaluation_sessions,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"metadata {key} mismatch for {panel_name}")
    provider_rows = metadata.get("provider_rows")
    dropped_rows = metadata.get("dropped_incomplete_rows")
    missing_by_symbol = metadata.get("missing_by_symbol")
    if (
        isinstance(provider_rows, bool)
        or not isinstance(provider_rows, int)
        or isinstance(dropped_rows, bool)
        or not isinstance(dropped_rows, int)
        or provider_rows != len(validated) + dropped_rows
        or dropped_rows < 0
    ):
        raise ValueError(f"metadata provider-row accounting mismatch for {panel_name}")
    if not isinstance(missing_by_symbol, dict) or set(missing_by_symbol) != set(symbols):
        raise ValueError(f"metadata missingness symbols mismatch for {panel_name}")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in missing_by_symbol.values()
    ):
        raise ValueError(f"metadata missingness counts are invalid for {panel_name}")
    if sum(missing_by_symbol.values()) < dropped_rows:
        raise ValueError(f"metadata missingness accounting mismatch for {panel_name}")
    return validated, metadata


def _strategy_targets(
    prices: pd.DataFrame,
    risky_symbols: list[str],
    protocol: dict[str, object],
) -> tuple[dict[str, dict[str, pd.DataFrame]], list[dict[str, object]]]:
    data = protocol["data"]
    strategies = protocol["strategies"]
    all_decisions = monthly_decision_dates(prices.index, end=data["evaluation_end"])
    evaluation_decisions = monthly_decision_dates(
        prices.index,
        start=data["evaluation_start"],
        end=data["evaluation_end"],
    )
    targets: dict[str, dict[str, pd.DataFrame]] = {
        "ts_momentum": {
            "deterministic": time_series_momentum_targets(
                prices,
                symbols=risky_symbols,
                decisions=evaluation_decisions,
                lookback=strategies["ts_momentum"]["lookback_sessions"],
            )
        },
        "cross_sectional_momentum": {
            "deterministic": cross_sectional_momentum_targets(
                prices,
                symbols=risky_symbols,
                decisions=evaluation_decisions,
                lookback_start=strategies["cross_sectional_momentum"][
                    "lookback_start_sessions"
                ],
                skip_recent=strategies["cross_sectional_momentum"][
                    "skip_recent_sessions"
                ],
                selection_count=strategies["cross_sectional_momentum"][
                    "selection_count"
                ],
            )
        },
        "short_term_reversal": {
            "deterministic": short_term_reversal_targets(
                prices,
                symbols=risky_symbols,
                decisions=evaluation_decisions,
                lookback=strategies["short_term_reversal"]["lookback_sessions"],
                selection_count=strategies["short_term_reversal"][
                    "selection_count"
                ],
            )
        },
        "learned_gbrt": {},
    }
    learned_diagnostics = []
    learned_config = strategies["learned_gbrt"]
    for seed in learned_config["seeds"]:
        result = learned_gbrt_targets(
            prices,
            symbols=risky_symbols,
            all_decisions=all_decisions,
            evaluation_decisions=evaluation_decisions,
            config=learned_config,
            seed=seed,
        )
        targets["learned_gbrt"][str(seed)] = result.weights
        learned_diagnostics.append(
            {
                "seed": int(seed),
                "first_training_rows": int(result.training_rows.iloc[0]),
                "minimum_training_rows": int(result.training_rows.min()),
                "maximum_training_rows": int(result.training_rows.max()),
                "minimum_training_months": int(result.training_months.min()),
                "maximum_training_months": int(result.training_months.max()),
            }
        )
    return targets, learned_diagnostics


def _evaluation_index(prices: pd.DataFrame, protocol: dict[str, object]) -> pd.DatetimeIndex:
    data = protocol["data"]
    index = prices.index[
        (prices.index >= pd.Timestamp(data["evaluation_start"]))
        & (prices.index <= pd.Timestamp(data["evaluation_end"]))
    ]
    if index.empty:
        raise ValueError("evaluation window contains no observations")
    return pd.DatetimeIndex(index)


def _target_hash(weights: pd.DataFrame) -> tuple[str, int]:
    tape = weights_to_target_tape(weights, max_gross=1.0)
    payload = target_tape_to_payload(
        tape,
        max_gross=1.0,
        expected_symbols=list(weights.columns),
    )
    return _canonical_json_digest(payload), int(len(payload["records"]))


def _run_target_variant(
    weights: pd.DataFrame,
    risky_prices: pd.DataFrame,
    cash_returns: pd.Series,
    *,
    cost_rate: float,
) -> dict[str, object]:
    zero_cash = pd.Series(0.0, index=cash_returns.index, name="zero_cash")
    comparator_targets = exposure_matched_comparator_targets(weights)
    return {
        "baseline": run_engine(
            weights,
            risky_prices,
            cash_returns,
            cost_rate=cost_rate,
            engine="event_driven",
        ),
        "same_close": invalid_same_close_result(
            weights,
            risky_prices,
            cash_returns,
            cost_rate=cost_rate,
        ),
        "zero_cash": run_engine(
            weights,
            risky_prices,
            zero_cash,
            cost_rate=cost_rate,
            engine="event_driven",
        ),
        "zero_cost": run_engine(
            weights,
            risky_prices,
            cash_returns,
            cost_rate=0.0,
            engine="event_driven",
        ),
        "costed_comparator": run_engine(
            comparator_targets,
            risky_prices,
            cash_returns,
            cost_rate=cost_rate,
            engine="event_driven",
        ),
        "vectorized": run_engine(
            weights,
            risky_prices,
            cash_returns,
            cost_rate=cost_rate,
            engine="vectorized",
        ),
    }


def _panel_experiment(
    panel_name: str,
    prices: pd.DataFrame,
    risky_symbols: list[str],
    protocol: dict[str, object],
) -> dict[str, object]:
    cash_symbol = protocol["data"]["cash_proxy"]
    cost_rate = protocol["execution"]["cost_per_one_way_gross_notional"]
    risky_prices = prices.loc[:, risky_symbols]
    cash_returns = prices[cash_symbol].pct_change(fill_method=None).fillna(0.0)
    cash_returns.name = cash_symbol
    evaluation_index = _evaluation_index(prices, protocol)
    targets, learned_diagnostics = _strategy_targets(
        prices,
        risky_symbols,
        protocol,
    )
    metric_rows = []
    engine_rows = []
    target_rows = []
    variant_series: dict[str, dict[str, dict[str, pd.Series]]] = {}
    for strategy, seed_targets in targets.items():
        variant_series[strategy] = {}
        for seed, weights in seed_targets.items():
            target_digest, record_count = _target_hash(weights)
            target_rows.append(
                {
                    "panel": panel_name,
                    "strategy": strategy,
                    "seed": seed,
                    "sha256": target_digest,
                    "decision_count": int(len(weights)),
                    "record_count": record_count,
                    "symbols": list(weights.columns),
                }
            )
            results = _run_target_variant(
                weights,
                risky_prices,
                cash_returns,
                cost_rate=cost_rate,
            )
            variant_series[strategy][seed] = {}
            for variant in VARIANTS:
                result = results[variant]
                metrics = performance_metrics(result, cash_returns, evaluation_index)
                metric_rows.append(
                    {
                        "panel": panel_name,
                        "strategy": strategy,
                        "seed": seed,
                        "variant": variant,
                        **metrics,
                    }
                )
                variant_series[strategy][seed][variant] = result.returns.reindex(
                    evaluation_index
                )
            event_returns = variant_series[strategy][seed]["baseline"]
            vectorized_returns = variant_series[strategy][seed]["vectorized"]
            engine_rows.append(
                {
                    "panel": panel_name,
                    "strategy": strategy,
                    "seed": seed,
                    "observations": int(len(evaluation_index)),
                    "max_absolute_daily_return_difference": float(
                        (event_returns - vectorized_returns).abs().max()
                    ),
                    "final_wealth_difference": float(
                        (1.0 + vectorized_returns).prod()
                        - (1.0 + event_returns).prod()
                    ),
                    "event_return_sha256": _series_digest(event_returns),
                    "vectorized_return_sha256": _series_digest(vectorized_returns),
                }
            )
    return {
        "metrics": pd.DataFrame(metric_rows),
        "engine": pd.DataFrame(engine_rows),
        "targets": target_rows,
        "series": variant_series,
        "cash": cash_returns.reindex(evaluation_index),
        "learned_diagnostics": learned_diagnostics,
    }


def _series_digest(series: pd.Series) -> str:
    payload = pd.DataFrame(
        {
            "date": series.index.strftime("%Y-%m-%d"),
            "return": series.to_numpy(dtype=float),
        }
    ).to_csv(index=False, float_format="%.17g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _family_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in metrics.select_dtypes(include=[np.number]).columns
        if column not in {"observations", "family_size"}
    ]
    grouped = metrics.groupby(["panel", "strategy", "variant"], sort=True)
    summary = grouped[numeric].mean().reset_index()
    summary["observations"] = grouped["observations"].min().to_numpy()
    summary["family_size"] = grouped.size().to_numpy()
    return summary.loc[
        :,
        [
            "panel",
            "strategy",
            "variant",
            "family_size",
            "observations",
            *numeric,
        ],
    ]


def _contract_effects(
    panel_results: dict[str, dict[str, object]],
    protocol: dict[str, object],
) -> pd.DataFrame:
    uncertainty = protocol["uncertainty"]
    block_lengths = [
        uncertainty["primary_block_sessions"],
        *uncertainty["sensitivity_block_sessions"],
    ]
    rows = []
    for panel_name, panel in panel_results.items():
        for strategy, seeds in panel["series"].items():
            ordered_seeds = sorted(
                seeds,
                key=lambda value: int(value) if value.isdigit() else -1,
            )
            for switch, (lhs_variant, rhs_variant) in SWITCHES.items():
                lhs = [seeds[seed][lhs_variant] for seed in ordered_seeds]
                rhs = [seeds[seed][rhs_variant] for seed in ordered_seeds]
                for block_length in block_lengths:
                    estimate = bootstrap_metric_difference(
                        lhs,
                        rhs,
                        panel["cash"],
                        block_length=block_length,
                        replications=uncertainty["replications"],
                        seed=uncertainty["seed"],
                    )
                    rows.append(
                        {
                            "panel": panel_name,
                            "strategy": strategy,
                            "switch": switch,
                            **estimate,
                        }
                    )
    return pd.DataFrame(rows)


def _rank_reversals(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for panel, panel_frame in summary.groupby("panel"):
        baseline = panel_frame.loc[panel_frame["variant"] == "baseline"]
        baseline_order = _metric_order(baseline)
        for variant in VARIANTS:
            variant_frame = panel_frame.loc[panel_frame["variant"] == variant]
            order = _metric_order(variant_frame)
            for strategy in sorted(baseline_order):
                rows.append(
                    {
                        "panel": panel,
                        "variant": variant,
                        "strategy": strategy,
                        "baseline_rank": baseline_order[strategy],
                        "variant_rank": order[strategy],
                        "rank_change": baseline_order[strategy] - order[strategy],
                    }
                )
    return pd.DataFrame(rows)


def _metric_order(frame: pd.DataFrame) -> dict[str, int]:
    values = {
        row.strategy: float(row.cash_excess_sharpe)
        for row in frame.itertuples(index=False)
    }
    ordered = sorted(values, key=lambda strategy: (-values[strategy], strategy))
    return {strategy: rank for rank, strategy in enumerate(ordered, start=1)}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def _write_generated_values(
    *,
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    engine: pd.DataFrame,
    metrics: pd.DataFrame,
    ranks: pd.DataFrame,
    path: Path,
) -> None:
    """Write paper-facing LaTeX values from the aggregate result tables."""

    def macro(name: str, value: str | int) -> str:
        return f"\\newcommand{{\\{name}}}{{{value}}}"

    def percent(value: float) -> str:
        return f"{value * 100.0:.2f}\\%"

    baseline = summary.loc[summary["variant"] == "baseline"].copy()
    baseline_rows = []
    for panel in PANEL_LABELS:
        frame = baseline.loc[baseline["panel"] == panel].set_index("strategy")
        for strategy in STRATEGY_LABELS:
            row = frame.loc[strategy]
            baseline_rows.append(
                " & ".join(
                    [
                        PANEL_LABELS[panel],
                        STRATEGY_LABELS[strategy],
                        percent(float(row["annualized_arithmetic_return"])),
                        f"{float(row['cash_excess_sharpe']):.2f}",
                        percent(float(row["cagr"])),
                        percent(float(row["max_drawdown"])),
                    ]
                )
                + r" \\"
            )

    primary = effects.loc[effects["block_length"] == 21].copy()
    effect_rows = []
    for switch in SWITCHES:
        frame = primary.loc[primary["switch"] == switch]
        return_below = int((frame["annualized_mean_ci_95_upper"] < 0.0).sum())
        return_above = int((frame["annualized_mean_ci_95_lower"] > 0.0).sum())
        sharpe_below = int((frame["sharpe_ci_95_upper"] < 0.0).sum())
        sharpe_above = int((frame["sharpe_ci_95_lower"] > 0.0).sum())
        count = int(len(frame))
        effect_rows.append(
            " & ".join(
                [
                    SWITCH_LABELS[switch],
                    f"{return_below}/{count - return_below - return_above}/{return_above}",
                    f"{sharpe_below}/{count - sharpe_below - sharpe_above}/{sharpe_above}",
                ]
            )
            + r" \\"
        )

    learned = metrics.loc[
        (metrics["strategy"] == "learned_gbrt")
        & (metrics["variant"] == "baseline")
    ]
    seed_ranges = {}
    for panel in PANEL_LABELS:
        values = learned.loc[
            learned["panel"] == panel,
            "cash_excess_sharpe",
        ]
        seed_ranges[panel] = f"{values.min():.2f}--{values.max():.2f}"

    cost_effect = primary.loc[primary["switch"] == "zero_cost_minus_baseline"]
    same_close = primary.loc[
        primary["switch"] == "same_close_minus_baseline"
    ]
    zero_cash = primary.loc[primary["switch"] == "zero_cash_minus_baseline"]
    rank_changes = {
        variant: int(
            (
                ranks.loc[ranks["variant"] == variant, "rank_change"]
                != 0
            ).sum()
        )
        for variant in ("same_close", "zero_cost", "costed_comparator")
    }
    lines = [
        "% Generated by scripts/run_expansion_experiments.py; do not edit.",
        macro("ExpansionPanelCount", len(PANEL_LABELS)),
        macro("ExpansionFamilyCount", len(STRATEGY_LABELS)),
        macro("ExpansionFamilyPanelCount", len(baseline)),
        macro(
            "ExpansionPositiveBaselineSharpeCount",
            int((baseline["cash_excess_sharpe"] > 0.0).sum()),
        ),
        macro("ExpansionEvaluationSessions", f"{int(baseline['observations'].min()):,}"),
        macro("ExpansionBootstrapReplications", f"{int(primary['replications'].min()):,}"),
        macro("ExpansionBaselineRows", "\n".join(baseline_rows)),
        macro("ExpansionEffectRows", "\n".join(effect_rows)),
        macro(
            "ExpansionCostEffectRange",
            (
                f"{cost_effect['annualized_mean_difference'].min() * 100.0:.2f}--"
                f"{cost_effect['annualized_mean_difference'].max() * 100.0:.2f}"
                " percentage points"
            ),
        ),
        macro(
            "ExpansionSameCloseBelowCount",
            int((same_close["annualized_mean_ci_95_upper"] < 0.0).sum()),
        ),
        macro(
            "ExpansionSameCloseOverlapCount",
            int(
                (
                    (same_close["annualized_mean_ci_95_lower"] <= 0.0)
                    & (same_close["annualized_mean_ci_95_upper"] >= 0.0)
                ).sum()
            ),
        ),
        macro(
            "ExpansionZeroCashBelowCount",
            int((zero_cash["annualized_mean_ci_95_upper"] < 0.0).sum()),
        ),
        macro(
            "ExpansionZeroCashOverlapCount",
            int(
                (
                    (zero_cash["annualized_mean_ci_95_lower"] <= 0.0)
                    & (zero_cash["annualized_mean_ci_95_upper"] >= 0.0)
                ).sum()
            ),
        ),
        macro(
            "ExpansionEngineMaxDailyBp",
            f"{engine['max_absolute_daily_return_difference'].max() * 10_000.0:.2f}",
        ),
        macro(
            "ExpansionEngineMaxWealthGap",
            percent(float(engine["final_wealth_difference"].abs().max())),
        ),
        macro("ExpansionSectorSeedRange", seed_ranges["us_sector_etfs"]),
        macro("ExpansionCountrySeedRange", seed_ranges["country_equity_etfs"]),
        macro("ExpansionSameCloseRankChanges", rank_changes["same_close"]),
        macro("ExpansionZeroCostRankChanges", rank_changes["zero_cost"]),
        macro(
            "ExpansionComparatorRankChanges",
            rank_changes["costed_comparator"],
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _plot_baseline(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_plot_style()
    baseline = summary.loc[summary["variant"] == "baseline"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharex=True, sharey=True)
    colors = dict(
        zip(
            STRATEGY_LABELS,
            BRIGHT_COLORS[: len(STRATEGY_LABELS)],
            strict=True,
        )
    )
    annotation_style = {
        ("country_equity_etfs", "learned_gbrt"): {
            "xytext": (-6, -7),
            "ha": "right",
        },
        ("country_equity_etfs", "ts_momentum"): {
            "xytext": (6, -1),
            "ha": "left",
        },
    }
    for panel_index, panel in enumerate(PANEL_LABELS):
        axis = axes[panel_index]
        frame = baseline.loc[baseline["panel"] == panel]
        for row in frame.itertuples(index=False):
            axis.scatter(
                row.annualized_arithmetic_return * 100.0,
                row.cash_excess_sharpe,
                color=colors[row.strategy],
                s=45,
                zorder=3,
            )
            annotation = annotation_style.get(
                (panel, row.strategy),
                {"xytext": (5, 5), "ha": "left"},
            )
            axis.annotate(
                STRATEGY_LABELS[row.strategy],
                (row.annualized_arithmetic_return * 100.0, row.cash_excess_sharpe),
                xytext=annotation["xytext"],
                textcoords="offset points",
                ha=annotation["ha"],
                fontsize=7.5,
            )
        axis.axhline(0.0, color=SPINE, linewidth=0.8)
        axis.axvline(0.0, color=SPINE, linewidth=0.8)
        axis.set_title(PANEL_LABELS[panel])
        axis.set_xlabel("Annualized arithmetic return (%)")
        style_axis(axis)
        add_panel_label(axis, chr(ord("a") + panel_index))
    axes[0].set_ylabel("Cash-excess Sharpe")
    fig.suptitle("Frozen-strategy performance after costs", color=INK, y=1.01)
    fig.tight_layout()
    _save_figure(fig, output / "baseline_performance")
    plt.close(fig)


def _plot_effects(effects: pd.DataFrame, output: Path, *, metric: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_plot_style()
    primary = effects.loc[effects["block_length"] == 21].copy()
    if metric == "return":
        estimate = "annualized_mean_difference"
        lower = "annualized_mean_ci_95_lower"
        upper = "annualized_mean_ci_95_upper"
        scale = 100.0
        label = "Annualized arithmetic return difference (pp)"
        stem = "contract_effects_return"
    elif metric == "sharpe":
        estimate = "sharpe_difference"
        lower = "sharpe_ci_95_lower"
        upper = "sharpe_ci_95_upper"
        scale = 1.0
        label = "Cash-excess Sharpe difference"
        stem = "contract_effects_sharpe"
    else:
        raise ValueError("metric must be 'return' or 'sharpe'")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 8.5), sharex=True, sharey=True)
    ordered_pairs: list[tuple[str, str]] = []
    effect_positions: list[int] = []
    tick_positions: list[int] = []
    tick_labels: list[str] = []
    heading_indexes: list[int] = []
    cursor = 0
    for strategy in STRATEGY_LABELS:
        heading_indexes.append(len(tick_positions))
        tick_positions.append(cursor)
        tick_labels.append(STRATEGY_LABELS[strategy])
        cursor += 1
        for switch in SWITCHES:
            ordered_pairs.append((strategy, switch))
            effect_positions.append(cursor)
            tick_positions.append(cursor)
            tick_labels.append(SWITCH_LABELS[switch])
            cursor += 1
        cursor += 1
    for panel_index, panel in enumerate(PANEL_LABELS):
        axis = axes[panel_index]
        frame = primary.loc[primary["panel"] == panel].set_index(
            ["strategy", "switch"]
        )
        y = np.asarray(effect_positions)
        values = np.array([frame.loc[pair, estimate] for pair in ordered_pairs]) * scale
        lows = np.array([frame.loc[pair, lower] for pair in ordered_pairs]) * scale
        highs = np.array([frame.loc[pair, upper] for pair in ordered_pairs]) * scale
        categories = np.where(
            lows > 0.0,
            "above",
            np.where(highs < 0.0, "below", "overlap"),
        )
        axis.hlines(y, lows, highs, color=MUTED_INK, linewidth=1.0, zorder=2)
        for category, color, marker in (
            ("above", REFERENCE_BLUE, "^"),
            ("overlap", MUTED_INK, "o"),
            ("below", NEGATIVE_RED, "v"),
        ):
            mask = categories == category
            axis.scatter(
                values[mask],
                y[mask],
                color=color,
                marker=marker,
                s=27,
                zorder=3,
            )
        axis.axvline(0.0, color=SPINE, linewidth=0.9)
        axis.set_title(PANEL_LABELS[panel])
        axis.set_yticks(tick_positions)
        if panel_index == 0:
            axis.set_yticklabels(tick_labels)
            for index in heading_indexes:
                axis.get_yticklabels()[index].set_fontweight("bold")
        else:
            axis.tick_params(labelleft=False)
        for position in effect_positions:
            axis.axhline(position, color="#D9DEE5", linewidth=0.65, zorder=0)
        style_axis(axis, grid=None)
        add_panel_label(axis, chr(ord("a") + panel_index))
    axes[0].invert_yaxis()
    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle="none",
            markersize=5,
            label=legend_label,
        )
        for color, marker, legend_label in (
            (REFERENCE_BLUE, "^", "95% interval above zero"),
            (MUTED_INK, "o", "95% interval overlaps zero"),
            (NEGATIVE_RED, "v", "95% interval below zero"),
        )
    ]
    fig.suptitle(
        "One-switch effects with 21-session block-bootstrap intervals",
        color=INK,
        y=0.995,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
    )
    fig.supxlabel(label, y=0.012)
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.93))
    _save_figure(fig, output / stem)
    plt.close(fig)


def _plot_engine_conformance(engine: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_plot_style()
    frame = (
        engine.groupby(["panel", "strategy"])[
            "max_absolute_daily_return_difference"
        ]
        .max()
        .unstack("strategy")
        .reindex(index=list(PANEL_LABELS), columns=list(STRATEGY_LABELS))
        * 10_000.0
    )
    fig, axis = plt.subplots(figsize=(7.4, 3.4))
    values = frame.to_numpy()
    image = axis.imshow(values, cmap="Blues", aspect="auto")
    contrast_threshold = float(np.nanmin(values) + 0.58 * np.ptp(values))
    for row in range(frame.shape[0]):
        for column in range(frame.shape[1]):
            value = frame.iloc[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value >= contrast_threshold else INK,
            )
    axis.set_xticks(
        np.arange(frame.shape[1]),
        [STRATEGY_LABELS[name] for name in frame.columns],
        rotation=20,
        ha="right",
    )
    axis.set_yticks(
        np.arange(frame.shape[0]),
        [PANEL_LABELS[name] for name in frame.index],
    )
    axis.set_title("Maximum absolute daily return difference between engines (bp)")
    fig.colorbar(image, ax=axis, label="Basis points")
    fig.tight_layout()
    _save_figure(fig, output / "engine_conformance")
    plt.close(fig)


def _plot_learned_seeds(metrics: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_plot_style()
    frame = metrics.loc[
        (metrics["strategy"] == "learned_gbrt")
        & (metrics["variant"] == "baseline")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
    for panel_index, panel in enumerate(PANEL_LABELS):
        axis = axes[panel_index]
        selected = frame.loc[frame["panel"] == panel].sort_values("seed")
        axis.scatter(
            selected["seed"].astype(int),
            selected["cash_excess_sharpe"],
            color=REFERENCE_BLUE,
            s=30,
            zorder=3,
        )
        family_mean = float(selected["cash_excess_sharpe"].mean())
        axis.axhline(
            family_mean,
            color=MUTED_INK,
            linewidth=0.9,
            linestyle="--",
            label="Seed mean",
        )
        axis.axhline(0.0, color=SPINE, linewidth=0.8)
        axis.set_title(PANEL_LABELS[panel])
        axis.set_xlabel("Frozen random seed")
        axis.set_xticks(selected["seed"].astype(int))
        style_axis(axis)
        add_panel_label(axis, chr(ord("a") + panel_index))
    axes[1].legend(loc="lower right")
    axes[0].set_ylabel("Cash-excess Sharpe")
    fig.suptitle("Learned-model seed sensitivity", color=INK)
    fig.tight_layout()
    _save_figure(fig, output / "learned_seed_sensitivity")
    plt.close(fig)


def _save_figure(fig, path_without_suffix: Path) -> None:
    fig.savefig(
        path_without_suffix.with_suffix(".png"),
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        path_without_suffix.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )


def _artifact_manifest(root: Path) -> dict[str, str]:
    artifacts = {}
    for directory in (root / "results", root / "figures"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                artifacts[path.relative_to(root).as_posix()] = _sha256(path)
    return artifacts


def run_expansion(
    *,
    repo_root: Path,
    protocol_path: Path,
    panel_dir: Path,
    output_dir: Path,
    generated_at: str,
) -> dict[str, object]:
    """Run every frozen panel and write aggregate artifacts."""
    generated_timestamp = _parse_aware_timestamp(generated_at, name="generated_at")
    git = _git_metadata(repo_root)
    if git["source_commit"] == "unavailable" or not git[
        "tracked_worktree_clean_at_start"
    ]:
        raise ValueError("commit tracked source changes before running expansion")
    protocol = _load_protocol(protocol_path)
    panel_results: dict[str, dict[str, object]] = {}
    data_records = []
    for panel_name, risky_symbols in protocol["panels"].items():
        prices, metadata = _load_panel(
            repo_root,
            panel_dir,
            panel_name,
            list(risky_symbols),
            protocol,
        )
        retrieved_timestamp = _parse_aware_timestamp(
            metadata["retrieved_at"],
            name="retrieved_at",
        )
        if retrieved_timestamp > generated_timestamp:
            raise ValueError(f"panel {panel_name} was retrieved after generation")
        panel_results[panel_name] = _panel_experiment(
            panel_name,
            prices,
            list(risky_symbols),
            protocol,
        )
        data_records.append(
            {
                key: value
                for key, value in metadata.items()
                if key != "local_file"
            }
        )

    metrics = pd.concat(
        [result["metrics"] for result in panel_results.values()],
        ignore_index=True,
    )
    summary = _family_summary(metrics)
    effects = _contract_effects(panel_results, protocol)
    engine = pd.concat(
        [result["engine"] for result in panel_results.values()],
        ignore_index=True,
    )
    ranks = _rank_reversals(summary)
    targets = [
        row
        for result in panel_results.values()
        for row in result["targets"]
    ]
    learned_diagnostics = [
        {"panel": panel, **row}
        for panel, result in panel_results.items()
        for row in result["learned_diagnostics"]
    ]

    with tempfile.TemporaryDirectory(prefix="quantcortex-expansion-") as temp:
        generated = Path(temp)
        result_dir = generated / "results"
        figure_dir = generated / "figures"
        result_dir.mkdir(parents=True)
        figure_dir.mkdir(parents=True)
        _write_csv(metrics, result_dir / "seed_variant_metrics.csv")
        _write_csv(summary, result_dir / "family_summary.csv")
        _write_csv(effects, result_dir / "contract_effects.csv")
        _write_csv(engine, result_dir / "engine_conformance.csv")
        _write_csv(ranks, result_dir / "rank_reversals.csv")
        _write_csv(
            pd.DataFrame(learned_diagnostics),
            result_dir / "learned_fit_diagnostics.csv",
        )
        _write_generated_values(
            summary=summary,
            effects=effects,
            engine=engine,
            metrics=metrics,
            ranks=ranks,
            path=result_dir / "generated_values.tex",
        )
        (result_dir / "target_tape_hashes.json").write_text(
            json.dumps(targets, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (result_dir / "data_provenance.json").write_text(
            json.dumps(data_records, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _plot_baseline(summary, figure_dir)
        _plot_effects(effects, figure_dir, metric="return")
        _plot_effects(effects, figure_dir, metric="sharpe")
        _plot_engine_conformance(engine, figure_dir)
        _plot_learned_seeds(metrics, figure_dir)
        manifest = {
            "schema_version": 1,
            "generated_at": generated_at,
            "protocol": {
                "path": protocol_path.relative_to(repo_root).as_posix(),
                "sha256": _sha256(protocol_path),
                "freeze_commit": FROZEN_PROTOCOL_COMMIT,
                "status": protocol["status"],
            },
            "git": git,
            "source_tree": _source_tree_manifest(repo_root),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": importlib.metadata.version("numpy"),
                "pandas": importlib.metadata.version("pandas"),
                "scikit_learn": importlib.metadata.version("scikit-learn"),
                "matplotlib": importlib.metadata.version("matplotlib"),
                "threadpoolctl": importlib.metadata.version("threadpoolctl"),
                "threadpools": _threadpool_environment(),
            },
            "data": data_records,
            "counts": {
                "panels": len(panel_results),
                "strategy_families": len(STRATEGY_LABELS),
                "seed_variant_rows": int(len(metrics)),
                "contract_effect_rows": int(len(effects)),
                "target_tapes": int(len(targets)),
            },
            "artifacts": _artifact_manifest(generated),
        }
        (result_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for name in ("results", "figures"):
            destination = output_dir / name
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(generated / name, destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("paper/expansion/protocol.json"),
    )
    parser.add_argument(
        "--panel-dir",
        type=Path,
        default=Path("local_data/expansion"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/expansion"),
    )
    parser.add_argument("--generated-at", required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    manifest = run_expansion(
        repo_root=repo_root,
        protocol_path=args.protocol.resolve(),
        panel_dir=args.panel_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        generated_at=args.generated_at,
    )
    print(
        f"wrote {len(manifest['artifacts'])} aggregate expansion artifacts "
        f"from source commit {manifest['git']['source_commit']}"
    )


if __name__ == "__main__":
    main()
