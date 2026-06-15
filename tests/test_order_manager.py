"""Tests for the order lifecycle state machine."""

from __future__ import annotations

import pytest

from quantcortex.execution.order_manager import (
    DuplicateOrderError,
    InvalidOrderTransitionError,
    OrderManager,
    OrderSide,
    OrderStatus,
    OrderType,
    UnknownOrderError,
)


def test_full_lifecycle_new_submitted_filled():
    om = OrderManager()
    order = om.create_order("o1", "AAPL", OrderSide.BUY, 100)
    assert order.status is OrderStatus.NEW

    om.submit("o1")
    assert om.get("o1").status is OrderStatus.SUBMITTED

    om.fill("o1", 100, fill_price=150.0)
    filled = om.get("o1")
    assert filled.status is OrderStatus.FILLED
    assert filled.filled_quantity == pytest.approx(100)
    assert filled.avg_fill_price == pytest.approx(150.0)
    assert filled.history == [
        OrderStatus.NEW,
        OrderStatus.SUBMITTED,
        OrderStatus.FILLED,
    ]


def test_partial_fills_then_complete():
    om = OrderManager()
    om.create_order("o2", "MSFT", OrderSide.BUY, 100)
    om.submit("o2")
    om.fill("o2", 40, fill_price=100.0)
    assert om.get("o2").status is OrderStatus.PARTIALLY_FILLED
    om.fill("o2", 60, fill_price=110.0)
    o = om.get("o2")
    assert o.status is OrderStatus.FILLED
    # volume-weighted average price
    assert o.avg_fill_price == pytest.approx((40 * 100 + 60 * 110) / 100)


def test_duplicate_order_id_raises():
    om = OrderManager()
    om.create_order("dup", "AAPL", OrderSide.BUY, 1)
    with pytest.raises(DuplicateOrderError):
        om.create_order("dup", "AAPL", OrderSide.SELL, 1)


def test_illegal_transition_raises():
    om = OrderManager()
    om.create_order("o3", "AAPL", OrderSide.BUY, 10)
    om.submit("o3")
    om.fill("o3")  # fully filled -> terminal
    with pytest.raises(InvalidOrderTransitionError):
        om.cancel("o3")  # cannot cancel a filled order


def test_cannot_fill_before_submit():
    om = OrderManager()
    om.create_order("o4", "AAPL", OrderSide.BUY, 10)
    with pytest.raises(InvalidOrderTransitionError):
        om.fill("o4")  # NEW -> FILLED is illegal (must be submitted first)


def test_unknown_order_raises():
    om = OrderManager()
    with pytest.raises(UnknownOrderError):
        om.get("nope")


def test_reject_records_reason():
    om = OrderManager()
    om.create_order("o5", "AAPL", OrderSide.BUY, 10)
    om.reject("o5", reason="insufficient buying power")
    o = om.get("o5")
    assert o.status is OrderStatus.REJECTED
    assert o.reject_reason == "insufficient buying power"


def test_overfill_rejected():
    om = OrderManager()
    om.create_order("o6", "AAPL", OrderSide.BUY, 10)
    om.submit("o6")
    with pytest.raises(Exception):
        om.fill("o6", 20)  # more than the order quantity


def test_limit_order_requires_price():
    om = OrderManager()
    with pytest.raises(Exception):
        om.create_order("o7", "AAPL", OrderSide.BUY, 10, order_type=OrderType.LIMIT)
