from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts import generate_report, validate_performance


def run_script(
    script: str,
    *args: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "PYTHONPATH": "."}
    environment.pop("ALPACA_API_KEY", None)
    environment.pop("ALPACA_SECRET_KEY", None)
    environment.update(extra_env or {})
    return subprocess.run(
        [sys.executable, script, *args],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )


@pytest.mark.parametrize(
    ("script", "message"),
    [
        (
            "scripts/generate_report.py",
            "one of the arguments --prices-csv --live-yfinance is required",
        ),
        (
            "scripts/validate_performance.py",
            "the following arguments are required: --live-yfinance",
        ),
        (
            "scripts/paper_trade_cycle.py",
            "one of the arguments --offline --live-yfinance is required",
        ),
        (
            "scripts/survivorship_demo.py",
            "the following arguments are required: --live-yfinance",
        ),
    ],
)
def test_data_scripts_require_an_explicit_source(script, message):
    result = run_script(script)

    assert result.returncode == 2
    assert message in result.stderr


def test_paper_cycle_cannot_submit_synthetic_orders():
    result = run_script("scripts/paper_trade_cycle.py", "--offline", "--submit")

    assert result.returncode == 2
    assert "--submit requires --live-yfinance" in result.stderr


def test_paper_cycle_submit_requires_credentials_before_fetching_data():
    result = run_script(
        "scripts/paper_trade_cycle.py", "--live-yfinance", "--submit"
    )

    assert result.returncode == 2
    assert "paper submission requires ALPACA_API_KEY" in result.stderr


def test_report_rejects_invalid_warmup_and_dsr_variance_before_loading_data():
    warmup = run_script(
        "scripts/generate_report.py",
        "--prices-csv",
        "missing.csv",
        "--warmup-years",
        "-1",
    )
    variance = run_script(
        "scripts/generate_report.py",
        "--prices-csv",
        "missing.csv",
        "--sr-variance",
        "nan",
    )

    assert warmup.returncode == 2
    assert "must be a non-negative integer" in warmup.stderr
    assert variance.returncode == 2
    assert "must be a finite non-negative number" in variance.stderr


def test_report_date_boundaries_accept_years_and_exact_iso_dates():
    assert generate_report.start_date("2018") == pd.Timestamp("2018-01-01")
    assert generate_report.end_date("2025") == pd.Timestamp("2025-12-31")
    assert generate_report.start_date("2018-03-14") == pd.Timestamp("2018-03-14")
    assert generate_report.end_date("2025-09-30") == pd.Timestamp("2025-09-30")


def test_report_rejects_malformed_date_boundaries_before_loading_data():
    result = run_script(
        "scripts/generate_report.py",
        "--prices-csv",
        "missing.csv",
        "--end",
        "2025-12-31-12-31",
    )

    assert result.returncode == 2
    assert "must be YYYY or YYYY-MM-DD" in result.stderr

    provenance = run_script(
        "scripts/generate_report.py",
        "--prices-csv",
        "missing.csv",
        "--retrieved-at",
        "yesterday",
    )
    assert provenance.returncode == 2
    assert "must be an ISO date or datetime" in provenance.stderr


def test_report_rejects_cash_proxy_inside_strategy_universe():
    result = run_script(
        "scripts/generate_report.py",
        "--prices-csv",
        "missing.csv",
        "--cash-proxy-symbol",
        "spy",
    )

    assert result.returncode == 2
    assert "cash proxy must be distinct" in result.stderr


def test_report_and_validation_benchmarks_share_the_evaluation_clock():
    dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 110.0, 121.0],
            "BOND": [100.0, 100.0, 100.0],
        },
        index=dates,
    )
    evaluation_index = dates[1:]

    report_spy, report_equal = generate_report._buy_hold_returns(
        prices, evaluation_index
    )
    validation_spy, validation_equal = validate_performance.benchmark_returns(
        prices, evaluation_index[0]
    )

    pd.testing.assert_series_equal(report_spy, validation_spy)
    pd.testing.assert_series_equal(report_equal, validation_equal)
    assert report_spy.tolist() == pytest.approx([0.1, 0.1])
    assert report_equal.iloc[0] == pytest.approx(0.05)


def test_report_warmup_guard_requires_explicit_cold_start_override():
    evaluation_start = pd.Timestamp("2024-01-02")
    prices = pd.DataFrame(
        {"SPY": [100.0] * 11},
        index=pd.bdate_range(end=evaluation_start, periods=11),
    )

    with pytest.raises(ValueError, match="requires at least 274"):
        generate_report._validate_warmup(
            prices,
            evaluation_start,
            required_sessions=274,
            enforce=True,
        )

    assert generate_report._validate_warmup(
        prices,
        evaluation_start,
        required_sessions=274,
        enforce=False,
    ) == 10


def test_report_rolling_beta_and_allocation_cash_are_hand_calculable():
    dates = pd.bdate_range("2024-01-02", periods=6)
    benchmark = pd.Series([-0.02, -0.01, 0.0, 0.01, 0.02, 0.03], index=dates)
    portfolio = 2.0 * benchmark

    beta = generate_report._rolling_beta(portfolio, benchmark, window=3)
    assert beta.dropna().tolist() == pytest.approx([2.0, 2.0, 2.0, 2.0])

    weights = pd.DataFrame(
        {"AAA": [0.5, 0.6], "BBB": [0.3, 0.4]},
        index=dates[:2],
    )
    allocation = generate_report._allocation_frame(weights)
    assert allocation["Cash"].tolist() == pytest.approx([0.2, 0.0])
    assert allocation.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_report_cash_attribution_and_exposure_matching_are_causal():
    dates = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.DataFrame(
        100.0,
        index=dates,
        columns=generate_report.ROTATION_UNIVERSE,
    )
    cash_returns = pd.Series(0.01, index=dates, name="cash proxy")

    class HalfInvestedStrategy:
        def generate_weights(self, _prices, _rebalance_dates):
            weights = pd.DataFrame(
                0.0,
                index=pd.DatetimeIndex([dates[0]]),
                columns=generate_report.ROTATION_UNIVERSE,
            )
            weights.loc[dates[0], "SPY"] = 0.5
            return weights

    report = generate_report.compute(
        prices,
        n_trials=1,
        cash_returns=cash_returns,
        strategy=HalfInvestedStrategy(),
    )

    assert report["active_risky_exposure"].tolist() == pytest.approx(
        [0.0, 0.0, 0.5, 0.5]
    )
    assert report["gross_returns"].tolist() == pytest.approx(
        [0.01, 0.01, 0.005, 0.005]
    )
    assert report["exposure_matched_spy_returns"].tolist() == pytest.approx(
        [0.01, 0.01, 0.005, 0.005]
    )
    assert report["m"]["cash_contribution_sum"] == pytest.approx(0.03)


def test_report_generates_complete_local_diagnostic_gallery(tmp_path):
    rng = np.random.default_rng(20240615)
    dates = pd.bdate_range("2017-01-03", periods=850, name="date")
    market = rng.normal(0.00025, 0.008, size=(len(dates), 1))
    loadings = np.linspace(0.8, 1.2, len(generate_report.ROTATION_UNIVERSE))
    noise = rng.normal(
        0.0,
        0.006,
        size=(len(dates), len(generate_report.ROTATION_UNIVERSE)),
    )
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(market * loadings + noise, axis=0)),
        index=dates,
        columns=generate_report.ROTATION_UNIVERSE,
    )
    prices_path = tmp_path / "prices.csv"
    prices.to_csv(prices_path, index_label="date")

    image_dir = tmp_path / "reports" / "img"
    report_path = tmp_path / "reports" / "report.md"
    manifest_path = tmp_path / "reports" / "performance_manifest.json"
    result = run_script(
        "scripts/generate_report.py",
        "--prices-csv",
        str(prices_path),
        "--start",
        "2019-01-02",
        "--end",
        "2020-04-06",
        "--n-trials",
        "3",
        "--data-provider",
        "test-only fixture generator",
        "--permission-basis",
        "repository test fixture; not for publication",
        "--retrieved-at",
        "2024-06-15",
        "--adjustment-method",
        "synthetic geometric return construction",
        "--imgdir",
        str(image_dir),
        "--report-out",
        str(report_path),
        "--manifest-out",
        str(manifest_path),
        extra_env={"MPLCONFIGDIR": str(tmp_path / "mpl")},
    )

    assert result.returncode == 0, result.stderr
    for filename, _ in generate_report.REPORT_ARTIFACTS:
        artifact = image_dir / filename
        assert artifact.exists(), filename
        assert artifact.stat().st_size > 1_000, filename

    report = report_path.read_text(encoding="utf-8")
    assert "Strategy returns" in report
    assert "SHA-256" in report
    assert "Deflated cash-excess Sharpe (3 trials)" in report
    assert "complete (owner-supplied; permission not independently verified)" in report
    for filename, _ in generate_report.REPORT_ARTIFACTS:
        assert f"img/{filename}" in report

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["source"]["input_sha256"]
    assert manifest["evaluation"]["dsr_trials_assumed"] == 3
    assert manifest["evaluation"]["sharpe_basis"].startswith("per-period return")
    assert set(manifest["artifacts"]) == {
        filename for filename, _ in generate_report.REPORT_ARTIFACTS
    }
