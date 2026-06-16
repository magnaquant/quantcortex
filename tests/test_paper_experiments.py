from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantcortex.strategies.multi_asset_rotation import MultiAssetRotation
from scripts.run_paper_experiments import (
    BOOTSTRAP_BLOCK_LENGTHS,
    PAPER_MAX_FORWARD_FILL,
    PRIMARY_BOOTSTRAP_BLOCK_LENGTH,
    SOURCE_TREE_FIXED_FILES,
    STRATEGY_PARAMETERS,
    _benchmark_returns,
    _cagr,
    _max_drawdown,
    _save_figures,
    _tex_number,
    circular_block_bootstrap,
    circular_block_bootstrap_frame,
    invalid_same_close_diagnostic,
    iso_timestamp,
    nonempty_text,
    return_decomposition,
    run_experiments,
    source_tree_manifest,
)


def test_circular_block_bootstrap_is_deterministic_and_reports_sign():
    returns = pd.Series([0.01, 0.02, -0.005, 0.015] * 20)
    first = circular_block_bootstrap(
        returns,
        block_length=5,
        replications=200,
        seed=7,
    )
    second = circular_block_bootstrap(
        returns,
        block_length=5,
        replications=200,
        seed=7,
    )

    assert first == second
    assert first["annualized_mean"] == pytest.approx(returns.mean() * 252.0)
    assert first["positive_draw_fraction"] == 1.0
    assert first["ci_95_lower"] < first["ci_95_upper"]


@pytest.mark.parametrize("block_length", [0, 5])
def test_circular_block_bootstrap_rejects_invalid_blocks(block_length):
    with pytest.raises(ValueError, match="block_length"):
        circular_block_bootstrap(
            pd.Series([0.01, 0.02, 0.03, 0.04]),
            block_length=block_length,
            replications=10,
        )


def test_benchmark_returns_share_one_capital_clock():
    index = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 110.0, 99.0, 108.9],
            "QQQ": [100.0, 100.0, 100.0, 100.0],
        },
        index=index,
    )

    spy, equal_weight = _benchmark_returns(prices, index[1:])

    assert spy.iloc[0] == pytest.approx(0.10)
    assert _cagr(spy, periods_per_year=3.0) == pytest.approx(0.089)
    expected_curve = prices.div(prices.iloc[0]).mean(axis=1)
    expected = expected_curve.pct_change(fill_method=None).iloc[1:]
    pd.testing.assert_series_equal(equal_weight, expected, check_names=False)


def test_joint_bootstrap_uses_complete_finite_rows():
    frame = pd.DataFrame(
        {
            "first": [0.01, 0.02, -0.01, 0.03],
            "second": [0.02, -0.01, 0.01, 0.00],
        }
    )
    frame["total"] = frame["first"] + frame["second"]

    result = circular_block_bootstrap_frame(
        frame,
        block_length=2,
        replications=100,
        seed=11,
    ).set_index("series")

    assert result.loc["total", "annualized_mean"] == pytest.approx(
        result.loc["first", "annualized_mean"]
        + result.loc["second", "annualized_mean"]
    )
    incomplete = frame.copy()
    incomplete.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="complete"):
        circular_block_bootstrap_frame(
            incomplete,
            block_length=2,
            replications=10,
        )


def test_return_decomposition_is_an_exact_daily_identity():
    index = pd.bdate_range("2024-01-01", periods=4)
    cash = pd.Series([0.001, 0.001, 0.001, 0.001], index=index)
    passive = pd.Series([0.01, -0.01, 0.02, 0.00], index=index)
    exposure = pd.Series([0.0, 0.5, 1.0, 0.5], index=index)
    gross = pd.Series([0.001, 0.004, 0.015, -0.002], index=index)
    net = gross - pd.Series([0.0, 0.001, 0.002, 0.001], index=index)

    components, constant_passive = return_decomposition(
        net=net,
        gross=gross,
        cash=cash,
        passive_basket=passive,
        risky_exposure=exposure,
    )

    pd.testing.assert_series_equal(
        components.iloc[:, :4].sum(axis=1),
        components["net_excess_over_cash"],
        check_names=False,
    )
    expected_constant = 0.5 * passive + 0.5 * cash
    pd.testing.assert_series_equal(
        constant_passive,
        expected_constant.rename("constant_exposure_passive_basket"),
    )


def test_same_close_diagnostic_is_explicitly_lookahead():
    index = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=index)
    weights = pd.DataFrame({"A": [1.0]}, index=index[1:2])
    cash = pd.Series(0.0, index=index)

    diagnostic = invalid_same_close_diagnostic(
        weights,
        prices,
        cash,
        cost_bps=0.0,
    )

    assert diagnostic["net"].iloc[0] == 0.0
    assert diagnostic["net"].iloc[1] == pytest.approx(0.10)
    assert diagnostic["exposure"].iloc[1] == pytest.approx(1.0)


def test_drawdown_uses_running_peak():
    returns = pd.Series([0.10, -0.20, 0.10])
    assert _max_drawdown(returns) == pytest.approx(-0.20)
    assert np.isfinite(_cagr(returns))


def test_paper_number_formatting_normalizes_rounded_zero():
    assert _tex_number(-0.000021, digits=2) == "0.00"
    assert _tex_number(-0.1547, digits=4) == "-0.1547"


def test_paper_provenance_text_rejects_empty_values():
    assert nonempty_text("  Yahoo Finance  ") == "Yahoo Finance"
    with pytest.raises(argparse.ArgumentTypeError, match="non-empty"):
        nonempty_text("   ")


@pytest.mark.parametrize("value", ["2026-06-16", "2026-06-16T12:30:00Z"])
def test_paper_retrieval_timestamp_accepts_iso_8601(value):
    assert iso_timestamp(value) == value


def test_paper_retrieval_timestamp_rejects_invalid_values():
    with pytest.raises(argparse.ArgumentTypeError, match="ISO-8601"):
        iso_timestamp("June 16, 2026")


def test_paper_strategy_parameters_are_explicit_and_constructible():
    strategy = MultiAssetRotation(**STRATEGY_PARAMETERS)

    assert strategy.top_n_groups == 2
    assert strategy.ir_lookback == 126
    assert strategy.mom_lookback == 126
    assert strategy.mom_gap == 21
    assert strategy.max_position_weight == pytest.approx(0.60)
    assert strategy._hmm.n_states == 3
    assert strategy._hmm.covariance_type == "full"
    assert strategy._hmm.n_iter == 100
    assert strategy._hmm.seed == 42
    assert strategy._hmm.reg_covar == pytest.approx(1e-5)
    assert strategy.regime_feature_vol_lookback == 20
    assert strategy._vix_scaler.floor == pytest.approx(0.3)
    assert strategy._vix_scaler.cap == pytest.approx(1.0)
    assert strategy.vix_proxy_lookback == 21
    assert strategy.required_history == 274
    assert PAPER_MAX_FORWARD_FILL == 0
    assert BOOTSTRAP_BLOCK_LENGTHS == (5, 21, 63)
    assert PRIMARY_BOOTSTRAP_BLOCK_LENGTH == 21


def test_paper_experiment_rejects_insufficient_signal_warmup():
    index = pd.bdate_range("2024-01-01", periods=40)
    prices = pd.DataFrame(
        {
            symbol: 100.0 + np.arange(len(index), dtype=float)
            for symbol in ("QQQ", "VGT", "GLD", "TLT", "SPY", "VIG")
        },
        index=index,
    )
    cash = pd.Series(0.0, index=index)

    with pytest.raises(ValueError, match="require at least 274"):
        run_experiments(
            prices,
            cash,
            evaluation_start=index[-5].date().isoformat(),
            evaluation_end=index[-1].date().isoformat(),
            bootstrap_replications=10,
        )


def test_source_tree_fingerprint_is_deterministic():
    repo_root = Path(__file__).resolve().parent.parent
    paths = [
        "scripts/run_paper_experiments.py",
        "quantcortex/strategies/multi_asset_rotation.py",
    ]
    first = source_tree_manifest(repo_root, paths)
    second = source_tree_manifest(repo_root, paths)

    assert first == second
    assert first["file_count"] == 2
    assert len(first["sha256"]) == 64

    complete = source_tree_manifest(repo_root)
    expected_paths = set(SOURCE_TREE_FIXED_FILES)
    expected_paths.update(
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "quantcortex").rglob("*.py")
        if path.is_file()
    )
    assert set(complete["files"]) == expected_paths
    assert complete["file_count"] == len(expected_paths)


def test_paper_figure_files_are_byte_deterministic(tmp_path):
    index = pd.bdate_range("2024-01-01", periods=6)
    baseline = {
        "gross": pd.Series([0.0, 0.01, -0.005, 0.004, 0.002, 0.003], index=index),
        "net": pd.Series([0.0, 0.009, -0.006, 0.003, 0.001, 0.002], index=index),
        "cash": pd.Series([0.0, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001], index=index),
        "exposure": pd.Series([0.0, 0.5, 1.0, 0.3, 0.0, 0.8], index=index),
    }
    matched = pd.Series([0.0, 0.006, -0.002, 0.003, 0.002, 0.004], index=index)
    costs = pd.DataFrame(
        {
            "all_in_cost_bps": [0.0, 13.0],
            "net_cash_excess_sharpe": [0.2, -0.1],
        }
    )
    ablation = pd.DataFrame(
        {
            "variant": ["full", "signal_only"],
            "mean_gross_exposure": [0.3, 0.8],
        }
    )
    ablation_uncertainty = pd.DataFrame(
        {
            "variant": ["full", "signal_only"],
            "annualized_mean": [-0.03, -0.05],
            "ci_95_lower": [-0.05, -0.08],
            "ci_95_upper": [-0.01, -0.02],
        }
    )
    engines = {"vectorized": baseline["net"], "event_driven": baseline["net"]}
    decomposition = pd.DataFrame(
        {
            "component": [
                "active_risky_allocation",
                "dynamic_exposure_timing",
                "passive_risky_exposure",
                "implementation_cost",
                "net_excess_over_cash",
            ],
            "block_length": [21] * 5,
            "annualized_mean": [-0.03, -0.005, 0.04, -0.02, -0.015],
            "ci_95_lower": [-0.05, -0.02, 0.02, -0.03, -0.04],
            "ci_95_upper": [-0.01, 0.01, 0.06, -0.01, 0.01],
        }
    )
    switches = pd.DataFrame(
        {
            "protocol": ["audited", "invalid_same_close"],
            "display_name": ["Audited", "Invalid same-close"],
            "diagnostic_class": ["reference", "causally_invalid"],
            "causally_valid": [True, False],
            "shv_excess_sharpe": [-0.1, 0.2],
        }
    )
    bootstrap_sensitivity = pd.DataFrame(
        [
            {
                "comparison": comparison,
                "block_length": block_length,
                "annualized_mean": mean,
                "ci_95_lower": lower,
                "ci_95_upper": upper,
            }
            for comparison, mean, lower, upper in (
                (
                    "strategy_gross_minus_exposure_matched_equal_weight",
                    -0.03,
                    -0.05,
                    -0.01,
                ),
                (
                    "strategy_minus_exposure_matched_equal_weight",
                    -0.05,
                    -0.08,
                    -0.02,
                ),
            )
            for block_length in BOOTSTRAP_BLOCK_LENGTHS
        ]
    )

    first = tmp_path / "first"
    second = tmp_path / "second"
    _save_figures(
        first,
        baseline,
        matched,
        costs,
        ablation,
        ablation_uncertainty,
        engines,
        decomposition,
        switches,
        bootstrap_sensitivity,
    )
    _save_figures(
        second,
        baseline,
        matched,
        costs,
        ablation,
        ablation_uncertainty,
        engines,
        decomposition,
        switches,
        bootstrap_sensitivity,
    )

    for first_path in sorted(first.iterdir()):
        second_path = second / first_path.name
        assert first_path.read_bytes() == second_path.read_bytes(), first_path.name
