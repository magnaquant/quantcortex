from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

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
    _costed_target_exposure_comparator,
    _git_metadata,
    _git_path_is_tracked,
    _max_drawdown,
    _save_figures,
    _tex_number,
    circular_block_bootstrap,
    circular_block_bootstrap_frame,
    circular_block_bootstrap_sharpe,
    circular_block_bootstrap_sharpe_frame,
    evaluation_contract,
    invalid_same_close_diagnostic,
    iso_timestamp,
    nonempty_text,
    return_decomposition,
    run_experiments,
    source_tree_manifest,
)
from scripts.run_paper_experiments import (
    main as paper_main,
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


def test_sharpe_block_bootstrap_is_deterministic_and_reports_sample_statistic():
    returns = pd.Series(
        np.random.default_rng(3).normal(0.001, 0.01, size=150),
        name="excess",
    )
    first = circular_block_bootstrap_sharpe(
        returns,
        block_length=5,
        replications=250,
        seed=17,
    )
    second = circular_block_bootstrap_sharpe(
        returns,
        block_length=5,
        replications=250,
        seed=17,
    )

    assert first == second
    assert first["sample_sharpe"] == pytest.approx(
        returns.mean() / returns.std(ddof=1) * np.sqrt(252.0)
    )
    assert first["ci_95_lower"] < first["ci_95_upper"]

    with pytest.raises(ValueError, match="undefined"):
        circular_block_bootstrap_sharpe(
            pd.Series([0.01] * 10),
            block_length=2,
            replications=20,
        )


def test_joint_sharpe_bootstrap_rejects_incomplete_rows():
    frame = pd.DataFrame(
        {
            "first": [0.01, 0.02, -0.01],
            "second": [0.00, np.nan, 0.01],
        }
    )
    with pytest.raises(ValueError, match="complete"):
        circular_block_bootstrap_sharpe_frame(
            frame,
            block_length=2,
            replications=20,
        )


def test_costed_target_exposure_comparator_is_causal_and_charges_rebalances():
    index = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame(
        {
            "A": [100.0, 100.0, 110.0, 110.0, 121.0, 121.0],
            "B": [100.0, 100.0, 110.0, 110.0, 121.0, 121.0],
        },
        index=index,
    )
    weights = pd.DataFrame(
        {"A": [0.5, 1.0], "B": [0.0, 0.0]},
        index=pd.DatetimeIndex([index[0], index[2]]),
    )
    cash = pd.Series(0.0, index=index)
    evaluation_index = index[2:]

    no_cost = _costed_target_exposure_comparator(
        weights,
        prices,
        cash,
        evaluation_index,
        cost_bps=0.0,
    )
    costed = _costed_target_exposure_comparator(
        weights,
        prices,
        cash,
        evaluation_index,
        cost_bps=13.0,
    )

    expected = pd.Series([0.05, 0.0, 0.10, 0.0], index=evaluation_index)
    pd.testing.assert_series_equal(
        no_cost.returns.reindex(evaluation_index),
        expected,
    )
    assert costed.costs.loc[index[2]] == 0.0
    assert costed.costs.loc[index[3]] > 0.0
    assert _growth_for_test(costed.returns.reindex(evaluation_index)) < (
        _growth_for_test(no_cost.returns.reindex(evaluation_index))
    )


def _growth_for_test(returns: pd.Series) -> float:
    return float((1.0 + returns).prod())


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
    assert _tex_number(-0.1547, digits=4) == r"\mbox{-0.1547}"


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


def test_git_metadata_captures_cleanliness_before_writes(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "tracked.txt").write_text("source\n", encoding="ascii")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "test: source"],
        cwd=tmp_path,
        check=True,
    )

    clean = _git_metadata(tmp_path)
    assert len(clean["source_commit"]) == 40
    assert clean["worktree_clean_at_start"] is True

    (tmp_path / "generated.txt").write_text("artifact\n", encoding="ascii")
    dirty = _git_metadata(tmp_path)
    assert dirty["source_commit"] == clean["source_commit"]
    assert dirty["worktree_clean_at_start"] is False


def test_git_path_tracking_is_measured_from_the_source_repository(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.csv"
    untracked = tmp_path / "untracked.csv"
    outside = tmp_path.parent / f"{tmp_path.name}-outside.csv"
    tracked.write_text("date,A\n2024-01-01,1\n", encoding="ascii")
    untracked.write_text("date,A\n2024-01-01,1\n", encoding="ascii")
    outside.write_text("date,A\n2024-01-01,1\n", encoding="ascii")
    subprocess.run(["git", "add", "tracked.csv"], cwd=tmp_path, check=True)

    assert _git_path_is_tracked(tmp_path, tracked) is True
    assert _git_path_is_tracked(tmp_path, untracked) is False
    assert _git_path_is_tracked(tmp_path, outside) is False


def test_git_path_tracking_fails_closed_on_git_errors(monkeypatch, tmp_path):
    candidate = tmp_path / "prices.csv"
    candidate.write_text("date,A\n2024-01-01,1\n", encoding="ascii")
    monkeypatch.setattr(
        "scripts.run_paper_experiments.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=128, stderr="fatal"),
    )

    with pytest.raises(RuntimeError, match="could not determine"):
        _git_path_is_tracked(tmp_path, candidate)


def test_paper_cli_requires_clean_source_before_reading_input(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scripts.run_paper_experiments._git_metadata",
        lambda _repo_root: {
            "source_commit": "0" * 40,
            "worktree_clean_at_start": False,
        },
    )
    with pytest.raises(RuntimeError, match="clean source worktree"):
        paper_main(
            [
                "run_paper_experiments.py",
                "--prices-csv",
                str(tmp_path / "missing.csv"),
                "--data-provider",
                "test provider",
                "--permission-basis",
                "test permission",
                "--retrieved-at",
                "2026-06-16",
                "--adjustment-method",
                "test adjustment",
                "--require-clean-source",
            ]
        )


def test_paper_cli_rejects_a_git_tracked_price_input(monkeypatch):
    monkeypatch.setattr(
        "scripts.run_paper_experiments._git_metadata",
        lambda _repo_root: {
            "source_commit": "0" * 40,
            "worktree_clean_at_start": True,
        },
    )
    tracked_fixture = (
        Path(__file__).resolve().parent / "fixtures" / "conformance" / "prices.csv"
    )

    with pytest.raises(RuntimeError, match="must not be tracked by Git"):
        paper_main(
            [
                "run_paper_experiments.py",
                "--prices-csv",
                str(tracked_fixture),
                "--data-provider",
                "test provider",
                "--permission-basis",
                "test permission",
                "--retrieved-at",
                "2026-06-16",
                "--adjustment-method",
                "test adjustment",
            ]
        )


def test_paper_cli_writes_a_complete_test_only_artifact_set(monkeypatch, tmp_path):
    warmup = pd.bdate_range(end="2017-12-29", periods=300)
    first_period = pd.bdate_range("2018-01-02", periods=40)
    second_period = pd.bdate_range("2022-01-03", periods=40)
    index = warmup.append(first_period).append(second_period)
    rng = np.random.default_rng(7)
    common = rng.normal(0.0002, 0.006, size=len(index))
    prices = {}
    for offset, symbol in enumerate(["QQQ", "VGT", "GLD", "TLT", "SPY", "VIG"]):
        idiosyncratic = rng.normal(0.0, 0.002 + offset * 0.0001, size=len(index))
        prices[symbol] = 100.0 * np.exp(np.cumsum(common + idiosyncratic))
    prices["SHV"] = 100.0 * np.exp(np.cumsum(np.full(len(index), 0.00005)))
    source = tmp_path / "test_only_prices.csv"
    pd.DataFrame(prices, index=index).rename_axis("date").reset_index().to_csv(
        source,
        index=False,
    )
    output_dir = tmp_path / "paper"
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setattr(
        "scripts.run_paper_experiments._git_metadata",
        lambda _repo_root: {
            "source_commit": "1" * 40,
            "worktree_clean_at_start": True,
        },
    )
    monkeypatch.setattr(
        "scripts.run_paper_experiments._threadpool_environment",
        lambda: [{"user_api": "test", "num_threads": 1}],
    )
    from scripts import run_paper_experiments as paper_experiments

    parsed_payloads = []
    original_parser = paper_experiments.target_tape_from_payload

    def recording_parser(payload):
        parsed_payloads.append(payload)
        return original_parser(payload)

    monkeypatch.setattr(paper_experiments, "target_tape_from_payload", recording_parser)

    assert (
        paper_main(
            [
                "run_paper_experiments.py",
                "--prices-csv",
                str(source),
                "--start",
                "2018-01-02",
                "--end",
                "2022-02-25",
                "--output-dir",
                str(output_dir),
                "--data-provider",
                "deterministic synthetic test fixture",
                "--permission-basis",
                "test-only generated data",
                "--retrieved-at",
                "2026-06-16",
                "--adjustment-method",
                "test-only generated positive price paths",
                "--generated-at",
                "2026-06-16T00:00:00Z",
                "--bootstrap-replications",
                "10",
                "--require-clean-source",
            ]
        )
        == 0
    )

    manifest = json.loads(
        (output_dir / "results" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 5
    assert manifest["generated_at"] == "2026-06-16T00:00:00Z"
    assert manifest["generator"]["git"] == {
        "source_commit": "1" * 40,
        "worktree_clean_at_start": True,
    }
    assert manifest["source"]["raw_input_committed"] is False
    assert manifest["source"]["provider"] == "deterministic synthetic test fixture"
    assert "results/evaluation_contract.json" in manifest["artifacts"]
    assert "results/target_tape_hashes.json" in manifest["artifacts"]
    assert "results/sharpe_uncertainty.csv" in manifest["artifacts"]
    assert "figures/accounting_summary.pdf" in manifest["artifacts"]
    for relative_path in manifest["artifacts"]:
        assert (output_dir / relative_path).is_file(), relative_path

    target_tapes = manifest["decision_streams"]["variants"]
    assert set(target_tapes) == {"full", "no_regime", "no_vol_scaler", "signal_only"}
    assert all(metadata["schema_version"] == 1 for metadata in target_tapes.values())
    assert all(metadata["decision_count"] > 0 for metadata in target_tapes.values())
    assert all(metadata["symbols"] == sorted(metadata["symbols"]) for metadata in target_tapes.values())
    assert all(
        metadata["record_count"]
        == metadata["decision_count"] * len(metadata["symbols"])
        for metadata in target_tapes.values()
    )
    assert len(parsed_payloads) == 4


def test_evaluation_contract_separates_attribution_and_tradable_comparator():
    contract = evaluation_contract("SHV")

    assert contract["schema_version"] == 1
    assert contract["cash"]["return_proxy"] == "SHV"
    assert contract["overlays"]["may_reduce_risky_exposure"] is True
    assert contract["overlays"]["may_increase_declared_gross_limit"] is False
    assert "block automatic retry" in contract["order_state"][
        "uncertain_submission"
    ]
    comparators = contract["comparators"]
    assert comparators["realized_exposure_attribution_control"] == {
        "purpose": "exact ex-post arithmetic attribution",
        "implementable": False,
        "costed": False,
        "exposure_basis": "strategy realized daily risky exposure",
    }
    assert comparators["target_exposure_costed_comparator"]["implementable"] is True
    assert comparators["target_exposure_costed_comparator"]["costed"] is True


def test_paper_figure_files_are_byte_deterministic(tmp_path):
    index = pd.bdate_range("2024-01-01", periods=6)
    baseline = {
        "gross": pd.Series([0.0, 0.01, -0.005, 0.004, 0.002, 0.003], index=index),
        "net": pd.Series([0.0, 0.009, -0.006, 0.003, 0.001, 0.002], index=index),
        "cash": pd.Series([0.0, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001], index=index),
        "exposure": pd.Series([0.0, 0.5, 1.0, 0.3, 0.0, 0.8], index=index),
    }
    matched = pd.Series([0.0, 0.006, -0.002, 0.003, 0.002, 0.004], index=index)
    costed_comparator = pd.Series(
        [0.0, 0.005, -0.0025, 0.0025, 0.0015, 0.0035],
        index=index,
    )
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
        costed_comparator,
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
        costed_comparator,
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
