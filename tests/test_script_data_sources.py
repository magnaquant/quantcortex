from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd
import pytest

from scripts import generate_report, validate_performance


def run_script(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "PYTHONPATH": "."}
    environment.pop("ALPACA_API_KEY", None)
    environment.pop("ALPACA_SECRET_KEY", None)
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
