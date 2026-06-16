"""Conformance checks against installed optional broker SDKs, without network I/O."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quantcortex.data.providers.alpaca_provider import AlpacaProvider
from quantcortex.execution.brokers.alpaca_broker import AlpacaBroker
from quantcortex.execution.brokers.ib_broker import IBBroker
from quantcortex.execution.order_manager import OrderSide, OrderStatus, OrderType


def test_broker_adapters_do_not_reference_retired_sdk_imports():
    broker_dir = (
        Path(__file__).resolve().parent.parent
        / "quantcortex"
        / "execution"
        / "brokers"
    )
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(broker_dir.glob("*.py"))
    )

    assert "ib_insync" not in source
    assert "alpaca_trade_api" not in source


def test_alpaca_py_trading_request_conformance():
    pytest.importorskip("alpaca")
    from alpaca.common.enums import Sort
    from alpaca.trading.enums import (
        OrderSide as AlpacaOrderSide,
    )
    from alpaca.trading.enums import (
        OrderType as AlpacaOrderType,
    )
    from alpaca.trading.enums import (
        QueryOrderStatus,
        TimeInForce,
    )
    from alpaca.trading.requests import (
        GetOrdersRequest,
        LimitOrderRequest,
        MarketOrderRequest,
    )

    class Client:
        def submit_order(self, *, order_data):
            self.order_data = order_data
            return SimpleNamespace(
                id="alpaca-1",
                client_order_id=order_data.client_order_id,
                status="new",
                filled_qty="0",
                filled_avg_price=None,
            )

    broker = AlpacaBroker()
    broker._api = Client()
    broker._sdk = {
        "MarketOrderRequest": MarketOrderRequest,
        "LimitOrderRequest": LimitOrderRequest,
        "GetOrdersRequest": GetOrdersRequest,
        "OrderSide": AlpacaOrderSide,
        "OrderType": AlpacaOrderType,
        "TimeInForce": TimeInForce,
        "QueryOrderStatus": QueryOrderStatus,
        "Sort": Sort,
    }

    order = broker.submit_order(
        "AAPL",
        OrderSide.BUY,
        1.5,
        OrderType.MARKET,
        client_order_id="qc-sdk-check",
    )

    assert isinstance(broker._api.order_data, MarketOrderRequest)
    assert broker._api.order_data.side is AlpacaOrderSide.BUY
    assert broker._api.order_data.time_in_force is TimeInForce.DAY
    assert order.status is OrderStatus.SUBMITTED


def test_alpaca_py_market_data_request_conformance(monkeypatch):
    pytest.importorskip("alpaca")
    sdk = AlpacaProvider._load_sdk()

    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get_stock_bars(self, request):
            self.request = request
            index = pd.MultiIndex.from_tuples(
                [("AAPL", pd.Timestamp("2024-01-02", tz="UTC"))],
                names=["symbol", "timestamp"],
            )
            frame = pd.DataFrame(
                {
                    "open": [100.0],
                    "high": [102.0],
                    "low": [99.0],
                    "close": [101.0],
                    "volume": [1_000.0],
                },
                index=index,
            )
            return SimpleNamespace(df=frame)

    sdk["StockHistoricalDataClient"] = Client
    provider = AlpacaProvider("key", "secret")
    monkeypatch.setattr(provider, "_load_sdk", lambda: sdk)

    result = provider.fetch_ohlcv("AAPL", start="2024-01-01", end="2024-01-03")

    assert result["AAPL"].loc[pd.Timestamp("2024-01-02"), "close"] == 101.0


def test_ib_async_order_constructor_conformance():
    pytest.importorskip("ib_async")
    from ib_async import LimitOrder, MarketOrder, Stock

    class Client:
        def qualifyContracts(self, contract):
            return [contract]

        def placeOrder(self, contract, order):
            self.contract = contract
            self.order = order
            order.orderId = 7
            return SimpleNamespace(
                order=order,
                orderStatus=SimpleNamespace(
                    status="Submitted",
                    filled=0.0,
                    avgFillPrice=0.0,
                ),
            )

    broker = IBBroker()
    broker._ib = Client()
    broker._sdk = {
        "Stock": Stock,
        "MarketOrder": MarketOrder,
        "LimitOrder": LimitOrder,
    }

    order = broker.submit_order("AAPL", OrderSide.BUY, 2.0)

    assert broker._ib.contract.symbol == "AAPL"
    assert broker._ib.order.action == "BUY"
    assert broker._ib.order.totalQuantity == 2.0
    assert order.order_id == "7"
    assert order.status is OrderStatus.SUBMITTED
