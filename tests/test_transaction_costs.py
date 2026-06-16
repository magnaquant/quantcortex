"""Tests for the mandatory transaction cost model."""

from __future__ import annotations

import numpy as np
import pytest

from quantcortex.backtest.costs.transaction_costs import (
    TransactionCostModel,
    apply_costs,
)


def test_buy_only_applies_commission_plus_slippage():
    model = TransactionCostModel(commission=0.0003, slippage=0.0010, tax=0.0)
    w_prev = np.array([0.0, 0.0, 0.0])
    w_new = np.array([0.5, 0.3, 0.2])  # all buys, +1.0 total turnover
    res = model.apply_costs(w_prev, w_new)

    expected = w_new.sum() * (0.0003 + 0.0010)
    assert res.total_cost == pytest.approx(expected)
    # no sell-side cost on a buy-only rebalance
    assert res.sell_cost.sum() == pytest.approx(0.0)
    assert res.buy_cost.sum() == pytest.approx(expected)
    assert res.turnover == pytest.approx(1.0)
    assert res.traded_notional == pytest.approx(1.0)


def test_sell_only_applies_commission_plus_slippage_plus_tax():
    model = TransactionCostModel(
        commission=0.0003, slippage=0.0010, tax=0.0005
    )
    w_prev = np.array([0.5, 0.3, 0.2])
    w_new = np.array([0.0, 0.0, 0.0])  # liquidate everything
    res = model.apply_costs(w_prev, w_new)

    expected = w_prev.sum() * (0.0003 + 0.0010 + 0.0005)
    assert res.total_cost == pytest.approx(expected)
    assert res.buy_cost.sum() == pytest.approx(0.0)
    assert res.sell_cost.sum() == pytest.approx(expected)


def test_mixed_trade_splits_buy_and_sell():
    model = TransactionCostModel(commission=0.001, slippage=0.0, tax=0.0)
    w_prev = np.array([0.5, 0.5])
    w_new = np.array([0.8, 0.2])  # buy 0.3 of A, sell 0.3 of B
    res = model.apply_costs(w_prev, w_new)
    assert res.buy_cost.sum() == pytest.approx(0.3 * 0.001)
    assert res.sell_cost.sum() == pytest.approx(0.3 * 0.001)
    assert res.total_cost == pytest.approx(0.6 * 0.001)
    assert res.turnover == pytest.approx(0.3)
    assert res.traded_notional == pytest.approx(0.6)


def test_adv_caps_cannot_create_an_overgross_intermediate_book():
    model = TransactionCostModel(volume_cap=0.10)
    result = model.apply_costs(
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        adv=np.array([0.0, 1_000_000.0]),
        capital=1.0,
        max_gross=1.0,
    )

    assert np.abs(result.executed_weights).sum() <= 1.0 + 1e-12
    assert result.executed_change[1] == pytest.approx(0.0, abs=1e-12)


def test_gross_cap_preserves_feasible_risk_reduction_before_replacement_buy():
    model = TransactionCostModel(volume_cap=0.10)
    result = model.apply_costs(
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        adv=np.array([2.0, 1_000_000.0]),
        capital=1.0,
        max_gross=1.0,
    )

    assert result.executed_change == pytest.approx([-0.2, 0.2])
    assert result.executed_weights == pytest.approx([0.8, 0.2])
    assert np.abs(result.executed_weights).sum() == pytest.approx(1.0)


def test_overgross_book_executes_reductions_and_opens_no_new_exposure():
    model = TransactionCostModel(volume_cap=0.10)
    result = model.apply_costs(
        np.array([1.2, 0.0]),
        np.array([0.0, 1.0]),
        adv=np.array([1.0, 1_000_000.0]),
        capital=1.0,
        max_gross=1.0,
    )

    assert result.executed_change == pytest.approx([-0.1, 0.0])
    assert result.executed_weights == pytest.approx([1.1, 0.0])


def test_adv_cap_truncates_oversized_orders():
    # volume_cap=10%, capital=$1,000,000.  Symbol A's ADV is tiny so a 50%
    # weight buy ($500k) far exceeds 10% of ADV and must be truncated.
    model = TransactionCostModel(volume_cap=0.10)
    w_prev = np.array([0.0, 0.0])
    w_new = np.array([0.5, 0.5])
    capital = 1_000_000.0
    # dollar ADV: A is illiquid ($100k/day), B is liquid ($100M/day)
    adv = np.array([100_000.0, 100_000_000.0])

    res = model.apply_costs(w_prev, w_new, adv=adv, capital=capital)

    # A: desired $500k, cap = 10% * 100k = $10k -> executed weight 0.01
    assert res.executed_change[0] == pytest.approx(10_000.0 / capital)
    assert res.capped[0]
    # B: desired $500k, cap = 10% * 100M = $10M (not binding) -> full 0.5
    assert res.executed_change[1] == pytest.approx(0.5)
    assert not res.capped[1]


def test_adv_cap_with_share_volume_and_prices():
    model = TransactionCostModel(volume_cap=0.10)
    w_prev = np.array([0.0])
    w_new = np.array([1.0])
    capital = 1_000.0
    prices = np.array([10.0])
    adv_shares = np.array([50.0])  # dollar ADV = 500; cap = 50; executed = 0.05
    res = model.apply_costs(
        w_prev, w_new, prices=prices, adv=adv_shares, capital=capital
    )
    assert res.executed_change[0] == pytest.approx(50.0 / capital)
    assert res.capped[0]


def test_module_level_apply_costs_wrapper():
    res = apply_costs(np.array([0.0]), np.array([1.0]))
    assert res.total_cost == pytest.approx(1.0 * (0.0003 + 0.0010))


def test_net_return_subtracts_costs():
    model = TransactionCostModel(commission=0.001, slippage=0.0, tax=0.0)
    res = model.apply_costs(
        np.array([0.0, 0.0]),
        np.array([0.5, 0.5]),
        gross_returns=np.array([0.02, 0.02]),
    )
    gross = 0.5 * 0.02 + 0.5 * 0.02
    assert res.net_return == pytest.approx(gross - res.total_cost)
