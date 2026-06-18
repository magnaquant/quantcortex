from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
from scripts.fetch_expansion_data import _adjusted_close, _validate_panel
from scripts.run_expansion_experiments import (
    PANEL_LABELS,
    STRATEGY_LABELS,
    SWITCHES,
    _family_summary,
    _load_panel,
    _load_protocol,
    _plot_baseline,
    _plot_effects,
    _plot_engine_conformance,
    _plot_learned_seeds,
    _rank_reversals,
    _source_tree_manifest,
    _write_generated_values,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _trend_panel(periods: int = 340) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-02", periods=periods)
    time = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "A": np.exp(0.001 * time),
            "B": np.exp(-0.001 * time),
            "C": np.ones(periods),
            "D": np.exp(0.001 * time),
            "CASH": np.exp(0.0001 * time),
        },
        index=index,
    )


def _learned_config() -> dict[str, object]:
    return {
        "label_horizon_sessions": 21,
        "feature_return_windows": [5, 21, 63, 126, 252],
        "feature_volatility_windows": [21, 63],
        "training_decision_months": 60,
        "minimum_training_decision_months": 24,
        "n_estimators": 10,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_samples_leaf": 5,
        "subsample": 0.8,
        "selection_count": 3,
    }


def test_panel_loader_validates_published_metadata(tmp_path):
    protocol_path = REPO_ROOT / "paper" / "expansion" / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    dates = pd.bdate_range("2024-01-02", periods=30)
    protocol["data"] = {
        **protocol["data"],
        "start": dates[0].date().isoformat(),
        "evaluation_start": dates[10].date().isoformat(),
        "evaluation_end": dates[-1].date().isoformat(),
        "minimum_pre_evaluation_sessions": 5,
    }
    panel = pd.DataFrame(
        {
            "A": np.linspace(100.0, 110.0, len(dates)),
            "B": np.linspace(90.0, 105.0, len(dates)),
            "SHV": np.linspace(100.0, 100.5, len(dates)),
        },
        index=dates,
    )
    panel.index.name = "date"
    csv_path = tmp_path / "fixture_panel.csv"
    metadata_path = tmp_path / "fixture_panel.metadata.json"
    panel.to_csv(csv_path, float_format="%.17g")
    pre_evaluation = dates < dates[10]
    metadata = {
        "schema_version": 1,
        "panel": "fixture_panel",
        "symbols": ["A", "B", "SHV"],
        "provider": "Yahoo Finance via yfinance",
        "provider_terms_independently_verified": False,
        "retrieved_at": "2026-06-18T22:37:51Z",
        "request": protocol["data"]["provider_request"],
        "protocol_path": "paper/expansion/protocol.json",
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "yfinance_version": "1.4.1",
        "terms_urls": [
            "https://ranaroussi.github.io/yfinance/",
            "https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html",
        ],
        "raw_data_committed": False,
        "input_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "provider_rows": len(panel),
        "complete_rows": len(panel),
        "dropped_incomplete_rows": 0,
        "missing_by_symbol": {"A": 0, "B": 0, "SHV": 0},
        "pre_evaluation_sessions": int(pre_evaluation.sum()),
        "pre_evaluation_months": int(dates[pre_evaluation].to_period("M").nunique()),
        "evaluation_sessions": int((~pre_evaluation).sum()),
        "first_date": dates[0].date().isoformat(),
        "last_date": dates[-1].date().isoformat(),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    loaded, published = _load_panel(
        REPO_ROOT,
        tmp_path,
        "fixture_panel",
        ["A", "B"],
        protocol,
    )
    pd.testing.assert_frame_equal(loaded, panel, check_freq=False)
    assert published == metadata

    tampered = copy.deepcopy(metadata)
    tampered["provider_rows"] += 1
    metadata_path.write_text(
        json.dumps(tampered, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="provider-row accounting"):
        _load_panel(REPO_ROOT, tmp_path, "fixture_panel", ["A", "B"], protocol)


def test_monthly_schedule_and_price_validation_are_strict():
    prices = _trend_panel()
    validated = validate_price_panel(
        prices,
        risky_symbols=["A", "B", "C", "D"],
        cash_symbol="CASH",
    )
    decisions = monthly_decision_dates(validated.index, start="2020-03-10")

    assert decisions[0] == pd.Timestamp("2020-03-10")
    assert decisions.to_period("M").is_unique
    assert list(validated.columns) == ["A", "B", "C", "D", "CASH"]


def test_rule_targets_follow_frozen_selection_and_tie_breaks():
    prices = _trend_panel()
    decision = pd.DatetimeIndex([prices.index[300]])
    symbols = ["A", "B", "C", "D"]

    time_series = time_series_momentum_targets(
        prices,
        symbols=symbols,
        decisions=decision,
    )
    cross_sectional = cross_sectional_momentum_targets(
        prices,
        symbols=symbols,
        decisions=decision,
        selection_count=1,
    )

    assert time_series.iloc[0].to_dict() == {
        "A": 0.5,
        "B": 0.0,
        "C": 0.0,
        "D": 0.5,
    }
    assert cross_sectional.iloc[0].to_dict() == {
        "A": 1.0,
        "B": 0.0,
        "C": 0.0,
        "D": 0.0,
    }


def test_reversal_does_not_renormalize_unused_exposure():
    index = pd.bdate_range("2024-01-02", periods=8)
    prices = pd.DataFrame(
        {
            "A": [100, 101, 102, 103, 104, 105, 106, 107],
            "B": [100, 100, 100, 100, 100, 90, 90, 90],
            "C": [100] * 8,
            "D": [100, 101, 101, 101, 101, 101, 101, 101],
        },
        index=index,
    )
    targets = short_term_reversal_targets(
        prices,
        symbols=["A", "B", "C", "D"],
        decisions=pd.DatetimeIndex([index[-1]]),
        selection_count=3,
    )

    assert targets.iloc[0].to_dict() == {
        "A": 0.0,
        "B": 1.0 / 3.0,
        "C": 0.0,
        "D": 0.0,
    }
    assert np.isclose(targets.iloc[0].sum(), 1.0 / 3.0)


def test_learned_targets_are_deterministic_and_ignore_future_prices():
    index = pd.bdate_range("2014-01-02", periods=1_500)
    rng = np.random.default_rng(7)
    innovations = rng.normal(
        loc=np.array([0.0003, 0.0001, -0.0001, 0.0002]),
        scale=np.array([0.008, 0.009, 0.007, 0.01]),
        size=(len(index), 4),
    )
    prices = pd.DataFrame(
        np.exp(np.cumsum(innovations, axis=0)) * 100.0,
        index=index,
        columns=["A", "B", "C", "D"],
    )
    all_decisions = monthly_decision_dates(index)
    evaluation = all_decisions[-12:]
    cutoff = evaluation[3]

    first = learned_gbrt_targets(
        prices,
        symbols=list(prices.columns),
        all_decisions=all_decisions,
        evaluation_decisions=evaluation,
        config=_learned_config(),
        seed=11,
    )
    repeated = learned_gbrt_targets(
        prices,
        symbols=list(prices.columns),
        all_decisions=all_decisions,
        evaluation_decisions=evaluation,
        config=_learned_config(),
        seed=11,
    )
    perturbed = prices.copy()
    perturbed.loc[perturbed.index > cutoff, "A"] *= 4.0
    future_changed = learned_gbrt_targets(
        perturbed,
        symbols=list(prices.columns),
        all_decisions=all_decisions,
        evaluation_decisions=evaluation[:4],
        config=_learned_config(),
        seed=11,
    )

    pd.testing.assert_frame_equal(first.weights, repeated.weights)
    pd.testing.assert_series_equal(first.weights.loc[cutoff], future_changed.weights.loc[cutoff])
    assert (first.training_months >= 24).all()
    assert (first.weights.sum(axis=1) <= 1.0 + 1e-12).all()
    assert (first.weights >= 0.0).all(axis=None)


def test_same_close_diagnostic_moves_the_first_earned_return_backward():
    index = pd.bdate_range("2024-01-02", periods=5)
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1, 146.41]}, index=index)
    cash = pd.Series(0.0, index=index)
    weights = pd.DataFrame({"A": [1.0]}, index=pd.DatetimeIndex([index[1]]))

    causal = run_engine(
        weights,
        prices,
        cash,
        cost_rate=0.0,
        engine="event_driven",
    )
    invalid = invalid_same_close_result(
        weights,
        prices,
        cash,
        cost_rate=0.0,
    )

    assert np.isclose(causal.returns.loc[index[1]], 0.0)
    assert np.isclose(causal.returns.loc[index[2]], 0.0)
    assert np.isclose(causal.returns.loc[index[3]], 0.10)
    assert np.isclose(invalid.returns.loc[index[1]], 0.10)


def test_exposure_matched_comparator_preserves_residual_cash():
    index = pd.DatetimeIndex(["2024-01-02", "2024-02-01"])
    weights = pd.DataFrame(
        [[0.6, 0.0, 0.0], [0.0, 0.0, 0.0]],
        index=index,
        columns=["A", "B", "C"],
    )

    comparator = exposure_matched_comparator_targets(weights)

    np.testing.assert_allclose(comparator.iloc[0], [0.2, 0.2, 0.2])
    np.testing.assert_allclose(comparator.iloc[1], [0.0, 0.0, 0.0])


def test_exposure_matched_comparator_rejects_short_targets():
    weights = pd.DataFrame(
        [[0.6, -0.1, 0.0]],
        index=pd.DatetimeIndex(["2024-01-02"]),
        columns=["A", "B", "C"],
    )

    with pytest.raises(ValueError, match="long-only"):
        exposure_matched_comparator_targets(weights)


def test_bootstrap_family_effect_is_reproducible_and_matches_point_estimates():
    index = pd.bdate_range("2024-01-02", periods=12)
    cash = pd.Series(np.linspace(0.0001, 0.0002, len(index)), index=index)
    lhs = [
        pd.Series(np.linspace(-0.01, 0.02, len(index)), index=index),
        pd.Series(np.linspace(-0.008, 0.018, len(index)), index=index),
    ]
    rhs = [
        pd.Series(np.linspace(-0.006, 0.011, len(index)), index=index),
        pd.Series(np.linspace(-0.004, 0.010, len(index)), index=index),
    ]

    first = bootstrap_metric_difference(
        lhs,
        rhs,
        cash,
        block_length=3,
        replications=250,
        seed=42,
    )
    second = bootstrap_metric_difference(
        lhs,
        rhs,
        cash,
        block_length=3,
        replications=250,
        seed=42,
    )

    assert first == second
    expected = np.mean(
        [
            (left - right).mean() * 252.0
            for left, right in zip(lhs, rhs, strict=True)
        ]
    )
    assert np.isclose(first["annualized_mean_difference"], expected)
    assert first["family_size"] == 2


def test_bootstrap_point_estimate_uses_the_joint_family_intersection():
    dates = pd.bdate_range("2024-01-02", periods=4)
    lhs = [
        pd.Series([1.0, 0.01, 0.03], index=dates[:3]),
        pd.Series([0.02, 0.04, 2.0], index=dates[1:]),
    ]
    rhs = [
        pd.Series([0.0, 0.0, 0.0], index=dates[:3]),
        pd.Series([0.0, 0.0, 0.0], index=dates[1:]),
    ]
    cash = pd.Series([0.001, 0.002, 0.0015, 0.0025], index=dates)

    estimate = bootstrap_metric_difference(
        lhs,
        rhs,
        cash,
        block_length=2,
        replications=100,
        seed=42,
    )

    expected = np.mean([0.02, 0.03]) * 252.0
    assert estimate["observations"] == 2
    assert np.isclose(estimate["annualized_mean_difference"], expected)


def test_performance_metrics_use_shv_excess_returns():
    prices = _trend_panel(periods=30)
    risky = prices[["A"]]
    cash = prices["CASH"].pct_change(fill_method=None).fillna(0.0)
    weights = pd.DataFrame({"A": [1.0]}, index=pd.DatetimeIndex([prices.index[0]]))
    result = run_engine(
        weights,
        risky,
        cash,
        cost_rate=0.0,
        engine="event_driven",
    )
    metrics = performance_metrics(result, cash, prices.index)

    assert metrics["observations"] == len(prices)
    assert metrics["annualized_arithmetic_return"] > 0.0
    assert metrics["cash_excess_sharpe"] > 0.0


def test_machine_protocol_matches_frozen_model_and_panel_choices():
    protocol_path = REPO_ROOT / "paper" / "expansion" / "protocol.json"
    protocol = _load_protocol(protocol_path)

    assert protocol["status"] == (
        "repository_frozen_prospective_not_externally_registered"
    )
    assert protocol["historical_case_confirmatory"] is False
    assert list(protocol["panels"]) == [
        "us_sector_etfs",
        "country_equity_etfs",
    ]
    learned = protocol["strategies"]["learned_gbrt"]
    assert learned["estimator"] == (
        "sklearn.ensemble.GradientBoostingRegressor"
    )
    assert learned["seeds"] == [11, 29, 47, 71, 97]
    assert protocol["execution"]["cost_per_one_way_gross_notional"] == 0.0013
    assert protocol["uncertainty"] == {
        "method": "joint circular block bootstrap",
        "replications": 5000,
        "primary_block_sessions": 21,
        "sensitivity_block_sessions": [5, 63],
        "seed": 20260618,
        "interval": "two-sided 95 percent percentile",
        "annualized_arithmetic_return": "252 times daily arithmetic mean",
        "sharpe": (
            "sqrt(252) times mean daily strategy-minus-SHV return divided by "
            "sample standard deviation"
        ),
    }
    assert FROZEN_PROTOCOL_COMMIT == (
        "4018f4063f46889f41d6981db5a71079e1dbd713"
    )
    assert FROZEN_PROTOCOL_SHA256 == (
        "e49e41a12a19fa5404a573ba5e21eb8a2888e616985f8c610d9652866923315c"
    )


def test_provider_adjusted_close_selection_and_complete_intersection():
    index = pd.bdate_range("2017-01-02", "2018-03-30")
    columns = pd.MultiIndex.from_product(
        [["Adj Close", "Close"], ["A", "B", "SHV"]]
    )
    values = np.tile(np.arange(1, len(columns) + 1, dtype=float), (len(index), 1))
    download = pd.DataFrame(values, index=index, columns=columns)
    download.loc[index[5], ("Adj Close", "B")] = np.nan

    adjusted = _adjusted_close(download, ["A", "B", "SHV"])
    complete, diagnostics = _validate_panel(
        adjusted,
        evaluation_start="2018-01-02",
        evaluation_end="2018-03-30",
        minimum_pre_evaluation_sessions=200,
        minimum_mature_training_months=12,
    )

    assert list(complete.columns) == ["A", "B", "SHV"]
    assert index[5] not in complete.index
    assert diagnostics["dropped_incomplete_rows"] == 1
    assert diagnostics["missing_by_symbol"]["B"] == 1


def test_provider_panel_validation_rejects_a_missing_evaluation_month():
    index = pd.bdate_range("2016-01-04", "2018-03-30")
    frame = pd.DataFrame(100.0, index=index, columns=["A", "B", "SHV"])
    frame = frame.loc[frame.index.to_period("M") != pd.Period("2018-02", freq="M")]

    with pytest.raises(ValueError, match="missing evaluation months: 2018-02"):
        _validate_panel(
            frame,
            evaluation_start="2018-01-02",
            evaluation_end="2018-03-30",
            minimum_pre_evaluation_sessions=252,
            minimum_mature_training_months=24,
        )


def _plot_fixture_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    metric_rows = []
    engine_rows = []
    for panel_index, panel in enumerate(PANEL_LABELS):
        for strategy_index, strategy in enumerate(STRATEGY_LABELS):
            for variant_index, variant in enumerate(
                [
                    "baseline",
                    "same_close",
                    "zero_cash",
                    "zero_cost",
                    "costed_comparator",
                    "vectorized",
                ]
            ):
                summary_rows.append(
                    {
                        "panel": panel,
                        "strategy": strategy,
                        "variant": variant,
                        "family_size": 1,
                        "observations": 100,
                        "annualized_arithmetic_return": (
                            0.01 * (strategy_index + 1) - 0.002 * variant_index
                        ),
                        "cash_excess_sharpe": (
                            0.2 * strategy_index - 0.03 * variant_index
                        ),
                        "cagr": 0.008 * (strategy_index + 1) - 0.001 * variant_index,
                        "max_drawdown": -0.05 * (strategy_index + 1),
                    }
                )
            for seed in [11, 29, 47, 71, 97]:
                metric_rows.append(
                    {
                        "panel": panel,
                        "strategy": "learned_gbrt",
                        "seed": str(seed),
                        "variant": "baseline",
                        "cash_excess_sharpe": 0.01 * seed - panel_index,
                    }
                )
            engine_rows.append(
                {
                    "panel": panel,
                    "strategy": strategy,
                    "max_absolute_daily_return_difference": (
                        0.00001 * (strategy_index + 1)
                    ),
                    "final_wealth_difference": 0.001 * (strategy_index + 1),
                }
            )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(engine_rows),
    )


def test_family_summary_and_rank_reversals_are_deterministic():
    summary, _, _ = _plot_fixture_frames()
    seed_metrics = pd.concat([summary.assign(seed="11"), summary.assign(seed="29")])

    aggregated = _family_summary(seed_metrics)
    ranks = _rank_reversals(aggregated)

    assert (aggregated["family_size"] == 2).all()
    assert len(ranks) == len(PANEL_LABELS) * 6 * len(STRATEGY_LABELS)
    baseline = ranks.loc[ranks["variant"] == "baseline"]
    assert (baseline["rank_change"] == 0).all()


def test_all_expansion_figures_render_from_complete_tables(tmp_path):
    summary, learned_metrics, engine = _plot_fixture_frames()
    effect_rows = []
    for panel in PANEL_LABELS:
        for strategy in STRATEGY_LABELS:
            for switch_index, switch in enumerate(SWITCHES):
                value = (switch_index - 2) * 0.002
                effect_rows.append(
                    {
                        "panel": panel,
                        "strategy": strategy,
                        "switch": switch,
                        "block_length": 21,
                        "annualized_mean_difference": value,
                        "annualized_mean_ci_95_lower": value - 0.01,
                        "annualized_mean_ci_95_upper": value + 0.01,
                        "sharpe_difference": value * 10.0,
                        "sharpe_ci_95_lower": value * 10.0 - 0.1,
                        "sharpe_ci_95_upper": value * 10.0 + 0.1,
                    }
                )
    effects = pd.DataFrame(effect_rows)

    _plot_baseline(summary, tmp_path)
    _plot_effects(effects, tmp_path, metric="return")
    _plot_effects(effects, tmp_path, metric="sharpe")
    _plot_engine_conformance(engine, tmp_path)
    _plot_learned_seeds(learned_metrics, tmp_path)

    assert len(list(tmp_path.glob("*.png"))) == 5
    assert len(list(tmp_path.glob("*.pdf"))) == 5
    assert all(path.stat().st_size > 1_000 for path in tmp_path.iterdir())


def test_expansion_latex_values_are_derived_from_aggregate_tables(tmp_path):
    summary, learned_metrics, engine = _plot_fixture_frames()
    effects = []
    for panel in PANEL_LABELS:
        for strategy in STRATEGY_LABELS:
            for switch_index, switch in enumerate(SWITCHES):
                value = (switch_index - 2) * 0.01
                effects.append(
                    {
                        "panel": panel,
                        "strategy": strategy,
                        "switch": switch,
                        "block_length": 21,
                        "replications": 5000,
                        "annualized_mean_difference": value,
                        "annualized_mean_ci_95_lower": value - 0.001,
                        "annualized_mean_ci_95_upper": value + 0.001,
                        "sharpe_ci_95_lower": value - 0.001,
                        "sharpe_ci_95_upper": value + 0.001,
                    }
                )
    ranks = _rank_reversals(summary)
    path = tmp_path / "generated_values.tex"

    _write_generated_values(
        summary=summary,
        effects=pd.DataFrame(effects),
        engine=engine,
        metrics=learned_metrics,
        ranks=ranks,
        data_records=[
            {"panel": "us_sector_etfs", "input_sha256": "a" * 64},
            {"panel": "country_equity_etfs", "input_sha256": "b" * 64},
        ],
        path=path,
    )

    generated = path.read_text(encoding="ascii")
    assert "\\newcommand{\\ExpansionPanelCount}{2}" in generated
    assert "\\newcommand{\\ExpansionFamilyCount}{4}" in generated
    assert "\\newcommand{\\ExpansionBootstrapReplications}{5,000}" in generated
    assert f"\\newcommand{{\\ExpansionProtocolDigest}}{{{FROZEN_PROTOCOL_SHA256}}}" in generated
    assert f"\\newcommand{{\\ExpansionSectorInputDigest}}{{{'a' * 64}}}" in generated
    assert f"\\newcommand{{\\ExpansionCountryInputDigest}}{{{'b' * 64}}}" in generated
    assert "U.S. sector ETFs & TS momentum" in generated
    assert "95%" not in generated


def test_expansion_source_fingerprint_includes_contract_and_operational_sources():
    manifest = _source_tree_manifest(REPO_ROOT)

    assert "scripts/fetch_expansion_data.py" in manifest["files"]
    assert "scripts/release_expansion_artifacts.sh" in manifest["files"]
    assert "scripts/run_expansion_experiments.py" in manifest["files"]
    assert "paper/preregistration.md" in manifest["files"]
    assert "paper/expansion/protocol.json" in manifest["files"]
    assert "schemas/canonical_target_tape.schema.json" in manifest["files"]
    assert len(manifest["sha256"]) == 64
