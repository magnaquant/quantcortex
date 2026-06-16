from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_paper_experiments import (
    _benchmark_returns,
    _cagr,
    _max_drawdown,
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
