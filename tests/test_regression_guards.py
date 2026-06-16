"""Regression guards for bugs fixed during the platform's audit passes.

Each test encodes a hand-derived or canonical expected value (not a snapshot of
current behaviour) so it would fail if the corresponding fix regressed. All use
only the scientific core (numpy/pandas/scipy/sklearn), so they run in CI without
the optional extras.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcortex.backtest.costs.transaction_costs import TransactionCostModel
from quantcortex.backtest.engines.event_driven import EventDrivenBacktest
from quantcortex.backtest.engines.vectorized import VectorizedBacktest
from quantcortex.backtest.engines.walk_forward import WalkForwardOptimizer
from quantcortex.backtest.validation.deflated_sharpe import (
    compute_dsr,
    probabilistic_sharpe_ratio,
)
from quantcortex.execution.pre_trade_risk import PreTradeRiskCheck
from quantcortex.portfolio.base import PortfolioMode
from quantcortex.portfolio.hrp import HierarchicalRiskParity


# --------------------------------------------------------------------------- #
# Backtest engine: causal accounting + mandatory costs
# --------------------------------------------------------------------------- #
def test_vectorized_engine_accounting_hand_example():
    """Weights set at close t earn t->t+1; the rebalance cost is charged once.

    Asset A rises +10%/day. A pre-sample decision executes on day 0 and buys
    100% A at a 10 bps one-way cost. Expected net returns are
    [-0.0010, +0.10, +0.10, +0.10] and final equity
    (1-0.001)*1.1**3.
    """
    dates = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1], "B": [100.0] * 4}, index=dates)
    weights = pd.DataFrame(
        {"A": [1.0], "B": [0.0]}, index=[dates[0] - pd.Timedelta(days=1)]
    )
    cm = TransactionCostModel(commission=0.001, slippage=0.0, tax=0.0)

    res = VectorizedBacktest(cm, capital=1.0).run(weights, prices)
    got = res.returns.reindex(dates).fillna(0.0).to_numpy()
    assert np.allclose(got, [-0.001, 0.10, 0.10, 0.10], atol=1e-12)
    assert res.equity_curve.iloc[-1] == pytest.approx((1 - 0.001) * 1.1 ** 3, abs=1e-12)


def test_vectorized_requires_cost_model():
    with pytest.raises(ValueError):
        VectorizedBacktest(None)


def test_vectorized_empty_weights_is_all_cash():
    # Regression: the date-snapping rewrite once raised TypeError on empty weights.
    dates = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame({"A": 100.0, "B": 100.0}, index=dates)
    res = VectorizedBacktest(TransactionCostModel(), capital=1000.0).run(
        pd.DataFrame(columns=["A", "B"]), prices
    )
    assert res.equity_curve.iloc[-1] == pytest.approx(1000.0)


@pytest.mark.parametrize("engine", [VectorizedBacktest, EventDrivenBacktest])
def test_backtest_cash_returns_compound_for_an_all_cash_book(engine):
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=dates)
    cash_returns = pd.Series([0.01, 0.02, 0.03], index=dates, name="cash proxy")

    result = engine(TransactionCostModel(), capital=1_000.0).run(
        pd.DataFrame(columns=["A"]),
        prices,
        cash_returns=cash_returns,
    )

    expected = 1_000.0 * 1.01 * 1.02 * 1.03
    assert result.equity_curve.iloc[-1] == pytest.approx(expected)
    assert result.gross_returns.to_list() == pytest.approx([0.01, 0.02, 0.03])
    assert result.asset_contribution.to_list() == pytest.approx([0.0, 0.0, 0.0])
    assert result.cash_contribution.to_list() == pytest.approx([0.01, 0.02, 0.03])
    assert result.cash_weights.to_list() == pytest.approx([1.0, 1.0, 1.0])


@pytest.mark.parametrize("engine", [VectorizedBacktest, EventDrivenBacktest])
def test_backtest_cash_and_asset_contributions_reconcile(engine):
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=dates)
    weights = pd.DataFrame(
        {"A": [0.5, 0.5, 0.5]},
        index=[dates[0] - pd.Timedelta(days=1), dates[0], dates[1]],
    )
    cash_returns = pd.Series(0.01, index=dates, name="cash proxy")

    result = engine(TransactionCostModel(), capital=1.0).run(
        weights,
        prices,
        cash_returns=cash_returns,
    )

    assert result.gross_returns.to_list() == pytest.approx([0.01, 0.055, 0.055])
    assert result.asset_contribution.to_list() == pytest.approx([0.0, 0.05, 0.05])
    assert result.cash_contribution.to_list() == pytest.approx([0.01, 0.005, 0.005])
    pd.testing.assert_series_equal(
        result.gross_returns,
        result.asset_contribution + result.cash_contribution,
        check_names=False,
    )
    assert result.cash_weights.to_list() == pytest.approx([0.5, 0.5, 0.5])


@pytest.mark.parametrize("engine", [VectorizedBacktest, EventDrivenBacktest])
def test_backtest_cash_returns_fail_closed_on_missing_bars(engine):
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0]}, index=dates)
    cash_returns = pd.Series([0.0, 0.0], index=dates[:2])

    with pytest.raises(ValueError, match="cover every price bar"):
        engine(TransactionCostModel()).run(
            pd.DataFrame(columns=["A"]),
            prices,
            cash_returns=cash_returns,
        )


def test_vectorized_cost_uses_current_pretrade_nav_denominator():
    dates = pd.bdate_range("2024-01-01", periods=2)
    prices = pd.DataFrame({"A": [100.0, 110.0]}, index=dates)
    weights = pd.DataFrame(
        {"A": [1.0, 0.0]},
        index=[dates[0] - pd.Timedelta(days=1), dates[0]],
    )
    model = TransactionCostModel(commission=0.01, slippage=0.0, tax=0.0)

    result = VectorizedBacktest(model, capital=1.0).run(weights, prices)

    assert result.gross_returns.iloc[1] == pytest.approx(0.10)
    assert result.costs.iloc[1] == pytest.approx(0.011)
    assert result.returns.iloc[1] == pytest.approx(0.089)
    assert result.equity_curve.iloc[-1] == pytest.approx(0.99 * 1.10 * 0.99)


@pytest.mark.parametrize("engine", [VectorizedBacktest, EventDrivenBacktest])
def test_backtest_engines_reject_excess_gross_targets(engine):
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame(
        {"A": [100.0, 101.0, 102.0], "B": [100.0, 99.0, 98.0]},
        index=dates,
    )
    weights = pd.DataFrame({"A": [1.0], "B": [0.5]}, index=[dates[0]])

    with pytest.raises(ValueError, match="gross exposure"):
        engine(TransactionCostModel()).run(weights, prices)


# --------------------------------------------------------------------------- #
# Walk-forward: purge + embargo
# --------------------------------------------------------------------------- #
def test_walk_forward_purge_and_embargo():
    wfo = WalkForwardOptimizer(train_size=100, test_size=30, embargo_gap=10, label_horizon=5)
    folds = list(wfo.split(300))
    assert len(folds) > 0
    assert wfo.n_splits(300) == len(folds)  # n_splits must match what split() yields
    for train_idx, test_idx in folds:
        assert max(train_idx) < min(test_idx)  # no overlap
        # exactly embargo_gap + label_horizon bars sit strictly between them
        assert int(min(test_idx) - max(train_idx) - 1) == 10 + 5


# --------------------------------------------------------------------------- #
# Deflated / Probabilistic Sharpe
# --------------------------------------------------------------------------- #
def test_dsr_monotonic_and_psr_equivalence():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0006, 0.01, 750))
    d1 = compute_dsr(r, n_trials=1)
    d10 = compute_dsr(r, n_trials=10)
    d100 = compute_dsr(r, n_trials=100)
    assert d1 == pytest.approx(probabilistic_sharpe_ratio(r), abs=1e-9)  # n=1 == PSR
    assert d1 > d10 > d100  # deflation strengthens with more trials
    assert 0.0 <= d100 <= 1.0


def test_psr_of_zero_mean_is_half():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 500))
    r = r - r.mean()  # exactly zero-mean -> observed Sharpe 0 -> PSR 0.5
    assert probabilistic_sharpe_ratio(r) == pytest.approx(0.5, abs=1e-6)


# --------------------------------------------------------------------------- #
# HRP weight contract
# --------------------------------------------------------------------------- #
def test_hrp_long_only_contract_on_block_correlated_universe():
    rng = np.random.default_rng(1)
    f1, f2 = rng.normal(0, 0.01, 500), rng.normal(0, 0.01, 500)
    R = pd.DataFrame({
        "A": f1 + rng.normal(0, 0.002, 500), "B": f1 + rng.normal(0, 0.002, 500),
        "C": f2 + rng.normal(0, 0.002, 500), "D": f2 + rng.normal(0, 0.002, 500),
    })
    w = HierarchicalRiskParity().optimize(R)
    assert w.shape == (4,)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-12).all() and (w <= 1.0 + 1e-12).all()


# --------------------------------------------------------------------------- #
# Pre-trade gate: validates the EXPOSURE contract (post-overlay), not sum==1
# --------------------------------------------------------------------------- #
def test_pretrade_accepts_derisked_book():
    # Regression: the gate once required sum==1.0 and rejected every gated book.
    ok, violations = PreTradeRiskCheck(max_position_weight=0.6).check_weights(
        np.array([0.489, 0.011, 0.0, 0.0, 0.0, 0.0])
    )
    assert ok, violations


def test_pretrade_rejects_real_violations():
    chk = PreTradeRiskCheck(max_position_weight=0.2)
    assert not chk.check_weights(np.array([0.7, 0.3, 0.0, 0.0]))[0]  # concentration
    assert not chk.check_weights(np.array([0.6, -0.1, 0.0, 0.0]))[0]  # short in long-only
    # market-neutral that is not dollar-neutral
    assert not PreTradeRiskCheck(max_position_weight=0.9).check_weights(
        np.array([0.5, 0.3]), mode=PortfolioMode.MARKET_NEUTRAL
    )[0]


def test_pretrade_accepts_balanced_market_neutral():
    ok, _ = PreTradeRiskCheck(max_position_weight=0.6).check_weights(
        np.array([0.5, -0.5]), mode=PortfolioMode.MARKET_NEUTRAL
    )
    assert ok
