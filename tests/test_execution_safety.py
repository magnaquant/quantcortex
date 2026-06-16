"""Regression tests for order, position, broker, and paper-cycle safety."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantcortex.backtest.execution_models.ideal_fill import IdealFill
from quantcortex.backtest.execution_models.market_impact import AlmgrenChriss
from quantcortex.backtest.execution_models.vwap_fill import VWAPFill
from quantcortex.execution.brokers import alpaca_broker as alpaca_broker_module
from quantcortex.execution.brokers.alpaca_broker import (
    AlpacaBroker,
    is_alpaca_live_endpoint,
    is_alpaca_paper_endpoint,
)
from quantcortex.execution.brokers.base import AccountInfo, BrokerError, Position
from quantcortex.execution.brokers.ccxt_broker import CCXTBroker
from quantcortex.execution.brokers.ib_broker import IBBroker
from quantcortex.execution.order_manager import (
    Order,
    OrderError,
    OrderManager,
    OrderSide,
    OrderStatus,
)
from quantcortex.execution.position_manager import PositionManager
from quantcortex.execution.pre_trade_risk import PreTradeRiskCheck, PreTradeRiskError
from quantcortex.execution.state_persistence import (
    StatePersistence,
    StatePersistenceError,
)
from quantcortex.portfolio.base import PortfolioMode
from scripts import paper_trade_cycle


class _MemoryState:
    def __init__(self):
        self.data = {}

    def load_state(self, key, default=None):
        return self.data.get(key, default)

    def save_state(self, key, value):
        self.data[key] = value


def test_execution_models_reject_missing_liquidity_and_oversized_orders():
    bar = pd.Series({"close": 100.0, "vwap": 100.0})
    with pytest.raises(ValueError, match="volume"):
        VWAPFill().fill("AAPL", 10.0, bar)
    with pytest.raises(ValueError, match="exceeds cap"):
        VWAPFill(participation=0.1).fill(
            "AAPL", 11.0, pd.Series({"close": 100.0, "vwap": 100.0, "volume": 100.0})
        )
    with pytest.raises(ValueError, match="ADV"):
        AlmgrenChriss().fill("AAPL", 10.0, pd.Series({"close": 100.0}))


def test_execution_models_validate_prices_and_stable_trajectory():
    with pytest.raises(ValueError, match="positive"):
        IdealFill().fill("AAPL", 1.0, pd.Series({"close": 0.0}))

    trajectory = AlmgrenChriss().optimal_execution_trajectory(
        1_000.0, 1.0, n=10, risk_aversion=1e6
    )
    assert np.isfinite(trajectory.to_numpy(dtype=float)).all()
    assert trajectory.iloc[0]["holdings"] == 1_000.0
    assert trajectory.iloc[-1]["holdings"] == 0.0


@pytest.mark.parametrize("quantity", [0.0, -1.0, np.nan, np.inf])
def test_order_rejects_nonpositive_or_nonfinite_quantity(quantity):
    with pytest.raises(OrderError, match="finite and positive"):
        OrderManager().create_order("o1", "AAPL", OrderSide.BUY, quantity)


def test_fill_rejects_invalid_price_without_mutating_order():
    manager = OrderManager()
    manager.create_order("o1", "AAPL", OrderSide.BUY, 10)
    manager.submit("o1")
    with pytest.raises(OrderError, match="Fill price"):
        manager.fill("o1", 5, fill_price=np.nan)
    assert manager.get("o1").filled_quantity == 0.0


def test_order_registry_uses_canonical_ids_for_duplicates_and_lookup():
    manager = OrderManager()
    order = manager.create_order("  o1  ", "AAPL", OrderSide.BUY, 1.0)

    assert order.order_id == "o1"
    assert manager.get(" o1 ") is order
    with pytest.raises(OrderError, match="already exists"):
        manager.create_order("o1", "AAPL", OrderSide.BUY, 1.0)


def test_position_manager_applies_only_incremental_cumulative_fills():
    orders = OrderManager()
    positions = PositionManager()
    orders.create_order("o1", "AAPL", OrderSide.BUY, 10)
    orders.submit("o1")

    partial = orders.fill("o1", 4, fill_price=100.0)
    positions.update_fill(partial)
    assert positions.get_position("AAPL").quantity == pytest.approx(4.0)

    complete = orders.fill("o1", 6, fill_price=110.0)
    positions.update_fill(complete)
    position = positions.get_position("AAPL")
    assert position.quantity == pytest.approx(10.0)
    assert position.avg_price == pytest.approx(106.0)

    positions.update_fill(complete)
    assert positions.get_position("AAPL").quantity == pytest.approx(10.0)


def test_pretrade_rejects_malformed_orders_and_prices():
    check = PreTradeRiskCheck()
    orders = [{"symbol": "AAPL", "side": "HOLD", "quantity": np.nan}]
    ok, violations = check.check_orders(orders, {"AAPL": np.nan})
    assert not ok
    assert any("quantity" in violation for violation in violations)

    ok, violations = check.check_orders([], 123.0)
    assert not ok
    assert any("mapping" in violation for violation in violations)

    duplicate_prices = pd.Series(
        [100.0, 101.0], index=pd.Index(["AAPL", "AAPL"])
    )
    ok, violations = check.check_orders([], duplicate_prices)
    assert not ok
    assert any("duplicate" in violation for violation in violations)


def test_pretrade_reconstructs_and_rejects_unsafe_post_trade_book():
    risk = PreTradeRiskCheck(max_position_weight=0.6, max_gross=1.0)
    orders = [
        {"symbol": "AAA", "side": OrderSide.SELL, "quantity": 20.0},
    ]

    ok, violations = risk.check_post_trade_positions(
        orders,
        {"AAA": 100.0},
        capital=1_000.0,
        current_positions={"AAA": 10.0},
        mode=PortfolioMode.LONG_ONLY,
    )

    assert not ok
    assert any("short legs" in message for message in violations)


def test_pretrade_assertion_requires_current_positions_for_orders():
    risk = PreTradeRiskCheck(max_position_weight=1.0)
    orders = [{"symbol": "AAA", "side": OrderSide.BUY, "quantity": 1.0}]

    with pytest.raises(PreTradeRiskError, match="current_positions"):
        risk.assert_safe(
            orders=orders,
            prices={"AAA": 100.0},
            capital=1_000.0,
        )


def test_weight_to_order_translation_rejects_missing_quote_for_held_name():
    manager = PositionManager()
    with pytest.raises(ValueError, match="price"):
        manager.target_weights_to_orders(
            {}, {"AAPL": 100.0}, capital=10_000.0, current_positions={"MSFT": 5}
        )


def test_whole_share_sizing_cannot_exceed_target_gross_exposure():
    manager = PositionManager()
    weights = {"AAA": 0.5, "BBB": 0.5}
    prices = {"AAA": 30.0, "BBB": 30.0}

    orders = manager.target_weights_to_orders(
        weights, prices, capital=100.0, current_positions={}
    )

    assert [order["quantity"] for order in orders] == [1.0, 1.0]
    risk = PreTradeRiskCheck(max_position_weight=1.0, max_gross=1.0)
    risk.assert_safe(
        weights=np.array([0.5, 0.5]),
        mode=PortfolioMode.LONG_ONLY,
        orders=orders,
        prices=prices,
        capital=100.0,
        current_positions={},
    )


def test_whole_share_sizing_rejects_fractional_current_positions():
    with pytest.raises(ValueError, match="allow_fractional=True"):
        PositionManager().target_weights_to_orders(
            {"AAA": 0.0},
            {"AAA": 30.0},
            capital=100.0,
            current_positions={"AAA": 0.5},
        )


def test_alpaca_request_is_validated_before_sdk_call():
    class FakeApi:
        called = False

        def submit_order(self, **kwargs):
            self.called = True
            return SimpleNamespace(id="unexpected", status="new")

    broker = AlpacaBroker()
    broker._api = FakeApi()
    with pytest.raises(OrderError):
        broker.submit_order("AAPL", OrderSide.BUY, np.nan)
    assert not broker._api.called


def test_alpaca_client_order_id_is_validated_and_forwarded():
    class FakeApi:
        called = False

        def submit_order(self, **kwargs):
            self.called = True
            self.kwargs = kwargs
            return SimpleNamespace(
                id="a1",
                client_order_id=kwargs["client_order_id"],
                status="new",
                filled_qty="0",
            )

    broker = AlpacaBroker()
    broker._api = FakeApi()
    with pytest.raises(BrokerError, match="client_order_id"):
        broker.submit_order(
            "AAPL",
            OrderSide.BUY,
            1.0,
            client_order_id="   ",
        )
    assert not broker._api.called

    broker.submit_order(
        "AAPL",
        OrderSide.BUY,
        1.0,
        client_order_id=" qc-order-1 ",
    )
    assert broker._api.kwargs["client_order_id"] == "qc-order-1"


def test_alpaca_client_order_lookup_only_treats_definite_404_as_missing():
    raw = SimpleNamespace(
        id="a1",
        client_order_id="qc-order-1",
        symbol="AAPL",
        side="buy",
        qty="1",
        type="market",
        limit_price=None,
        status="filled",
        filled_qty="1",
        filled_avg_price="100",
    )

    class MissingError(Exception):
        status_code = 404

    class FakeApi:
        error = None

        def get_order_by_client_order_id(self, client_order_id):
            if self.error is not None:
                raise self.error
            return raw

    broker = AlpacaBroker()
    broker._api = FakeApi()
    found = broker.find_order_by_client_order_id("qc-order-1")
    assert found is not None
    assert found.status is OrderStatus.FILLED

    broker._api.error = MissingError("missing")
    assert broker.find_order_by_client_order_id("qc-order-1") is None
    broker._api.error = TimeoutError("timed out")
    with pytest.raises(BrokerError, match="lookup failed"):
        broker.find_order_by_client_order_id("qc-order-1")


def test_alpaca_open_order_reconciliation_fails_on_truncated_results():
    raw = SimpleNamespace(
        id="a1",
        client_order_id="qc-order-1",
        symbol="AAPL",
        side="buy",
        qty="1",
        type="market",
        limit_price=None,
        status="new",
        filled_qty="0",
        filled_avg_price=None,
    )
    broker = AlpacaBroker()
    broker._api = SimpleNamespace(
        list_orders=lambda **kwargs: [raw],
    )
    orders = broker.get_open_orders()
    assert len(orders) == 1
    assert orders[0].status is OrderStatus.SUBMITTED

    broker._api = SimpleNamespace(
        list_orders=lambda **kwargs: [raw] * 500,
    )
    with pytest.raises(BrokerError, match="truncated"):
        broker.get_open_orders()


def test_ccxt_paper_mode_refuses_exchange_without_sandbox(monkeypatch):
    class NoSandboxExchange:
        def __init__(self, config):
            self.has = {"sandbox": False}

    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace(mock=NoSandboxExchange))
    with pytest.raises(BrokerError, match="no sandbox"):
        CCXTBroker(exchange="mock", paper=True).connect()


def test_broker_payloads_reject_nonfinite_values():
    with pytest.raises(BrokerError, match="finite"):
        Position("AAPL", np.nan)
    with pytest.raises(BrokerError, match="finite"):
        AccountInfo(equity=np.inf)
    with pytest.raises(BrokerError, match="boolean"):
        Position("AAPL", True)
    with pytest.raises(BrokerError, match="boolean"):
        AccountInfo(cash=True)
    with pytest.raises(BrokerError, match="dictionary"):
        AccountInfo(extra="not-a-dict")


def test_execution_numeric_contracts_reject_booleans():
    with pytest.raises(OrderError, match="boolean"):
        OrderManager().create_order("o1", "AAPL", OrderSide.BUY, True)
    with pytest.raises(ValueError, match="capital"):
        PositionManager().target_weights_to_orders(
            {"AAPL": 1.0}, {"AAPL": 100.0}, capital=True
        )
    with pytest.raises(TypeError, match="allow_fractional"):
        PositionManager().target_weights_to_orders(
            {"AAPL": 1.0},
            {"AAPL": 100.0},
            capital=1_000.0,
            allow_fractional="yes",
        )
    with pytest.raises(TypeError, match="boolean"):
        PreTradeRiskCheck(max_gross=True)


def test_position_manager_rejects_duplicate_series_symbols():
    duplicate_weights = pd.Series([0.5, 0.5], index=["AAPL", "AAPL"])
    with pytest.raises(ValueError, match="duplicate"):
        PositionManager().target_weights_to_orders(
            duplicate_weights, {"AAPL": 100.0}, capital=1_000.0
        )


def test_alpaca_unknown_order_status_fails_closed():
    broker = AlpacaBroker()
    raw = SimpleNamespace(id="a1", status="mystery", filled_qty="0")
    with pytest.raises(BrokerError, match="Unknown Alpaca"):
        broker._to_order(raw, "AAPL", OrderSide.BUY, 1.0, "MARKET", None)


def test_ccxt_unknown_order_status_fails_closed():
    broker = CCXTBroker()
    with pytest.raises(BrokerError, match="Unknown ccxt"):
        broker._to_order(
            {"id": "c1", "status": "mystery"},
            "BTC/USDT",
            OrderSide.BUY,
            1.0,
            "MARKET",
            None,
        )


def test_ccxt_account_fails_closed_on_unvalued_assets():
    broker = CCXTBroker(account_currency="USDT")
    broker._exchange = SimpleNamespace(
        fetch_balance=lambda: {
            "free": {"USDT": 1_000.0, "BTC": 0.5},
            "total": {"USDT": 1_000.0, "BTC": 0.5},
        }
    )

    with pytest.raises(BrokerError, match="unvalued assets"):
        broker.get_account()

    broker._exchange = SimpleNamespace(
        fetch_balance=lambda: {
            "free": {"USDT": 900.0},
            "total": {"USDT": 1_000.0},
        }
    )
    account = broker.get_account()
    assert account.cash == 900.0
    assert account.equity == 1_000.0
    assert account.extra["valuation_scope"] == "single_currency_cash_only"


def test_ccxt_account_currency_can_be_selected_from_environment(monkeypatch):
    monkeypatch.setenv("CCXT_ACCOUNT_CURRENCY", " usdt ")
    broker = CCXTBroker()

    assert broker.account_currency == "USDT"


def test_ib_unknown_order_status_fails_closed():
    broker = IBBroker()
    trade = SimpleNamespace(
        order=SimpleNamespace(orderId=1),
        orderStatus=SimpleNamespace(status="Mystery", filled=0.0),
    )
    with pytest.raises(BrokerError, match="Unknown Interactive Brokers"):
        broker._to_order(trade, "AAPL", OrderSide.BUY, 1.0, "MARKET", None)


def test_ib_account_summary_requires_an_account_selector_for_multiple_accounts():
    broker = IBBroker()
    broker._ib = SimpleNamespace(
        accountSummary=lambda: [
            SimpleNamespace(
                account="A1", tag="NetLiquidation", value="100", currency="USD"
            ),
            SimpleNamespace(
                account="A2", tag="NetLiquidation", value="200", currency="USD"
            ),
        ]
    )

    with pytest.raises(BrokerError, match="multiple accounts"):
        broker.get_account()


def test_state_persistence_rejects_corrupt_file(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text("{not json", encoding="utf-8")
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("QC_STATE_PATH", str(state_path))
    store = StatePersistence(url=None, namespace="test")

    with pytest.raises(StatePersistenceError, match="could not read"):
        store.load_state("positions")


def test_state_persistence_default_is_not_temporary(monkeypatch, tmp_path):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("QC_STATE_PATH", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state-home"))
    store = StatePersistence(url=None, namespace="test")

    assert Path(store._file_path).is_relative_to(tmp_path / "state-home")


def test_state_persistence_explicit_file_path_is_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("REDIS_URL", "redis://must-not-be-used:6379/0")
    monkeypatch.setenv("QC_STATE_PATH", str(tmp_path / "default.json"))
    explicit = tmp_path / "isolated" / "state.json"
    store = StatePersistence(
        url=None,
        namespace="notebook",
        file_path=explicit,
    )

    store.save_state("probe", {"value": 1})

    assert Path(store._file_path) == explicit.resolve()
    assert store.backend == "file"
    assert explicit.exists()
    assert not (tmp_path / "default.json").exists()


def test_file_state_keys_are_canonical_across_save_load_and_delete(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("QC_STATE_PATH", str(tmp_path / "state.json"))
    store = StatePersistence(url=None, namespace="test")

    store.save_state(" key ", {"value": 1})
    assert store.load_state("key") == {"value": 1}
    store.delete_state(" key")
    assert store.load_state("key") is None


def test_state_persistence_rejects_illegal_order_history(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        '{"orders": [{"order_id": "o1", "symbol": "AAPL", '
        '"side": "BUY", "quantity": 1, "status": "FILLED", '
        '"filled_quantity": 1, "history": ["NEW", "FILLED"]}]}',
        encoding="utf-8",
    )
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("QC_STATE_PATH", str(state_path))
    store = StatePersistence(url=None, namespace="test")

    with pytest.raises(StatePersistenceError, match="persisted orders"):
        store.load_orders()


def test_state_persistence_validates_before_writing(monkeypatch, tmp_path):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("QC_STATE_PATH", str(tmp_path / "state.json"))
    store = StatePersistence(url=None, namespace="test")

    with pytest.raises(StatePersistenceError, match="does not match"):
        store.save_positions({"AAPL": Position("MSFT", 1.0)})
    with pytest.raises(StatePersistenceError, match="invalid"):
        store.save_orders(
            [
                {
                    "order_id": "o1",
                    "symbol": "AAPL",
                    "side": "BUY",
                    "quantity": True,
                }
            ]
        )


def test_latest_flat_target_generates_liquidation_orders(monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"QQQ": [100.0, 101.0, 102.0]}, index=dates)

    class FlatStrategy:
        def generate_weights(self, prices, rebalance_dates):
            return pd.DataFrame(
                {"QQQ": [1.0, 0.0]}, index=[dates[0], dates[-1]]
            )

    monkeypatch.setattr(paper_trade_cycle, "MultiAssetRotation", FlatStrategy)
    target = paper_trade_cycle.latest_target(prices)
    assert target.empty
    orders = paper_trade_cycle.build_orders(
        target, prices, capital=10_000.0, current={"QQQ": 5.0}
    )
    assert orders == [
        {"symbol": "QQQ", "side": OrderSide.SELL, "quantity": 5.0}
    ]


def test_paper_order_builder_places_sells_before_buys():
    prices = pd.DataFrame(
        {"QQQ": [100.0], "VGT": [100.0]},
        index=pd.DatetimeIndex(["2024-01-02"]),
    )
    orders = paper_trade_cycle.build_orders(
        pd.Series({"VGT": 0.5}),
        prices,
        capital=1_000.0,
        current={"QQQ": 5.0},
    )

    assert [order["side"] for order in orders] == [
        OrderSide.SELL,
        OrderSide.BUY,
    ]


def test_paper_client_order_ids_are_stable_and_intent_specific():
    intent = {"symbol": "QQQ", "side": OrderSide.BUY, "quantity": 5.0}
    first = paper_trade_cycle.make_client_order_id("2024-01-02", intent)
    second = paper_trade_cycle.make_client_order_id(
        pd.Timestamp("2024-01-02 16:00"), intent
    )
    changed = paper_trade_cycle.make_client_order_id(
        "2024-01-02",
        {**intent, "quantity": 6.0},
    )

    assert first == second
    assert first != changed
    assert len(first) < 48


def test_paper_submission_timeout_is_persisted_and_not_retried_implicitly():
    class FakeBroker:
        fail = True
        submit_calls = 0

        def find_order_by_client_order_id(self, client_order_id):
            return None

        def submit_order(
            self,
            symbol,
            side,
            quantity,
            *,
            client_order_id,
        ):
            self.submit_calls += 1
            if self.fail:
                raise BrokerError("submission timed out")
            return Order(
                order_id="a1",
                symbol=symbol,
                side=side,
                quantity=quantity,
                status=OrderStatus.SUBMITTED,
            )

    broker = FakeBroker()
    store = _MemoryState()
    ledger = {}
    intent = {"symbol": "QQQ", "side": OrderSide.BUY, "quantity": 5.0}
    with pytest.raises(BrokerError, match="timed out"):
        paper_trade_cycle._reconcile_or_submit(
            broker,
            store,
            ledger,
            "2024-01-02",
            intent,
        )
    client_id = paper_trade_cycle.make_client_order_id("2024-01-02", intent)
    assert ledger[client_id]["state"] == "UNKNOWN"

    with pytest.raises(BrokerError, match="Prior submission attempt"):
        paper_trade_cycle._reconcile_or_submit(
            broker,
            store,
            ledger,
            "2024-01-02",
            intent,
        )
    assert broker.submit_calls == 1

    broker.fail = False
    placed, submitted, returned_id = paper_trade_cycle._reconcile_or_submit(
        broker,
        store,
        ledger,
        "2024-01-02",
        intent,
        retry_confirmed_missing=True,
    )
    assert submitted
    assert returned_id == client_id
    assert placed.order_id == "a1"
    assert broker.submit_calls == 2


def test_paper_cycle_refuses_to_trade_with_open_orders(monkeypatch):
    class FakeBroker:
        account_requested = False

        def __init__(self, paper):
            assert paper

        def connect(self):
            return None

        def disconnect(self):
            return None

        def get_open_orders(self):
            return [
                Order(
                    order_id="a1",
                    symbol="QQQ",
                    side=OrderSide.BUY,
                    quantity=1.0,
                    status=OrderStatus.SUBMITTED,
                )
            ]

        def get_account(self):
            self.account_requested = True
            raise AssertionError("account should not be read with open orders")

    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
    monkeypatch.setattr(alpaca_broker_module, "AlpacaBroker", FakeBroker)
    prices = pd.DataFrame(
        {"QQQ": [100.0]},
        index=pd.DatetimeIndex(["2024-01-02"]),
    )

    result = paper_trade_cycle.run_paper(
        prices,
        pd.Series({"QQQ": 0.5}),
        submit=False,
    )
    assert result == 1


def test_paper_cycle_defers_buys_until_sell_phase_settles(monkeypatch):
    class FakeBroker:
        instance = None

        def __init__(self, paper):
            assert paper
            self.submitted = []
            type(self).instance = self

        def connect(self):
            return None

        def disconnect(self):
            return None

        def get_open_orders(self):
            return []

        def get_account(self):
            return AccountInfo(cash=500.0, equity=1_000.0, buying_power=1_000.0)

        def get_positions(self):
            return [Position("QQQ", 5.0, market_price=100.0)]

        def find_order_by_client_order_id(self, client_order_id):
            return None

        def submit_order(
            self,
            symbol,
            side,
            quantity,
            *,
            client_order_id,
        ):
            self.submitted.append((symbol, side, quantity, client_order_id))
            return Order(
                order_id=f"a{len(self.submitted)}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                status=OrderStatus.SUBMITTED,
            )

    store = _MemoryState()
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
    monkeypatch.setattr(alpaca_broker_module, "AlpacaBroker", FakeBroker)
    monkeypatch.setattr(
        paper_trade_cycle,
        "StatePersistence",
        lambda namespace: store,
    )
    prices = pd.DataFrame(
        {"QQQ": [100.0], "VGT": [100.0]},
        index=pd.DatetimeIndex(["2024-01-02"]),
    )

    result = paper_trade_cycle.run_paper(
        prices,
        pd.Series({"VGT": 0.5}),
        submit=True,
    )

    assert result == 1
    assert FakeBroker.instance is not None
    assert [item[1] for item in FakeBroker.instance.submitted] == [OrderSide.SELL]


def test_execution_identifiers_require_string_symbols_and_strip_whitelists():
    manager = PositionManager()
    with pytest.raises(ValueError, match="non-empty strings"):
        manager.target_weights_to_orders(
            {1: 1.0}, {1: 100.0}, capital=1_000.0
        )

    risk = PreTradeRiskCheck(allowed_symbols=[" AAPL "])
    ok, violations = risk.check_orders(
        [{"symbol": "AAPL", "side": OrderSide.BUY, "quantity": 1.0}],
        {"AAPL": 100.0},
    )
    assert ok, violations


@pytest.mark.parametrize(
    "url",
    [
        "https://api.alpaca.markets",
        "http://paper-api.alpaca.markets",
        "https://paper-api.alpaca.markets.evil.example",
        "https://user@paper-api.alpaca.markets",
        "https://paper-api.alpaca.markets:8443",
        "https://paper-api.alpaca.markets/v2",
    ],
)
def test_paper_endpoint_validation_rejects_lookalikes(url):
    assert not is_alpaca_paper_endpoint(url)


def test_paper_endpoint_validation_accepts_documented_host():
    assert is_alpaca_paper_endpoint("https://paper-api.alpaca.markets")


def test_alpaca_paper_broker_rejects_a_live_endpoint():
    with pytest.raises(BrokerError, match="paper endpoint"):
        AlpacaBroker(base_url="https://api.alpaca.markets", paper=True)


def test_alpaca_live_broker_rejects_non_live_endpoint():
    assert is_alpaca_live_endpoint("https://api.alpaca.markets")
    with pytest.raises(BrokerError, match="live endpoint"):
        AlpacaBroker(base_url="https://paper-api.alpaca.markets", paper=False)


def test_ib_broker_requires_explicit_live_mode_for_known_live_ports():
    with pytest.raises(BrokerError, match="live port"):
        IBBroker(port=7496)
    live = IBBroker(port=7496, paper=False)
    assert not live.paper
