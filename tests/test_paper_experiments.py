from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_paper_experiments import (
    _benchmark_returns,
    _cagr,
    _max_drawdown,
    _save_figures,
    circular_block_bootstrap,
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
    assert first["bootstrap_probability_positive"] == 1.0
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


def test_drawdown_uses_running_peak():
    returns = pd.Series([0.10, -0.20, 0.10])
    assert _max_drawdown(returns) == pytest.approx(-0.20)
    assert np.isfinite(_cagr(returns))


def test_paper_figure_files_are_byte_deterministic(tmp_path):
    index = pd.bdate_range("2024-01-01", periods=6)
    baseline = {
        "gross": pd.Series([0.0, 0.01, -0.005, 0.004, 0.002, 0.003], index=index),
        "net": pd.Series([0.0, 0.009, -0.006, 0.003, 0.001, 0.002], index=index),
        "cash": pd.Series([0.0, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001], index=index),
    }
    matched = pd.Series([0.0, 0.006, -0.002, 0.003, 0.002, 0.004], index=index)
    yearly = pd.DataFrame(
        {
            "series": [
                "strategy_net",
                "cash_proxy",
                "exposure_matched_equal_weight",
            ],
            "year": [2024, 2024, 2024],
            "return": [0.01, 0.001, 0.02],
        }
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
            "gross_cash_excess_sharpe": [0.1, 0.3],
            "matched_equal_weight_cash_excess_sharpe": [0.5, 0.6],
        }
    )
    engines = {"vectorized": baseline["net"], "event_driven": baseline["net"]}

    first = tmp_path / "first"
    second = tmp_path / "second"
    _save_figures(first, baseline, matched, yearly, costs, ablation, engines)
    _save_figures(second, baseline, matched, yearly, costs, ablation, engines)

    for first_path in sorted(first.iterdir()):
        second_path = second / first_path.name
        assert first_path.read_bytes() == second_path.read_bytes(), first_path.name
