from __future__ import annotations

import os
import subprocess
import sys

import pytest


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
