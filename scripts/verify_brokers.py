"""Offline behavioral verification of the broker adapters via faithful mocks.

The companion to the live-SDK *API-conformance* check (which confirmed the real
SDKs expose the methods the adapters call): this exercises the adapters'
*behavior* end to end by injecting SDK-shaped fake clients and asserting that
each adapter constructs the right request and parses the response into the
correct ``Order`` / ``Position`` / ``AccountInfo``. It needs no network and no
broker account. It does not prove authenticated connectivity, account
permissions, SDK-version compatibility, or venue-side fill/rejection behavior.

    python scripts/verify_brokers.py
"""

from __future__ import annotations

from quantcortex.execution.brokers.alpaca_broker import AlpacaBroker
from quantcortex.execution.brokers.base import BrokerError
from quantcortex.execution.brokers.ccxt_broker import CCXTBroker
from quantcortex.execution.brokers.ib_broker import IBBroker
from quantcortex.execution.order_manager import OrderSide, OrderStatus, OrderType

_results = []


def _check(name, cond, detail=""):
    _results.append((name, bool(cond), detail))


# --------------------------------------------------------------------------- #
# Alpaca (alpaca-py TradingClient shapes)
# --------------------------------------------------------------------------- #
class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Request(_Obj):
    pass


class _OrderSide:
    BUY = "buy"
    SELL = "sell"


class _OrderType:
    MARKET = "market"
    LIMIT = "limit"


class _TimeInForce:
    DAY = "day"


class _QueryOrderStatus:
    OPEN = "open"


class _Sort:
    ASC = "asc"


_ALPACA_SDK = {
    "MarketOrderRequest": _Request,
    "LimitOrderRequest": _Request,
    "GetOrdersRequest": _Request,
    "OrderSide": _OrderSide,
    "OrderType": _OrderType,
    "TimeInForce": _TimeInForce,
    "QueryOrderStatus": _QueryOrderStatus,
    "Sort": _Sort,
}


class _FakeAlpacaREST:
    def __init__(self):
        self.orders_by_client_id = {}
        self.open_orders = [
            _Obj(
                id="alp-open",
                client_order_id="open-1",
                symbol="MSFT",
                side="sell",
                qty="2",
                type="limit",
                limit_price="310",
                status="new",
                filled_qty="0",
                filled_avg_price=None,
            )
        ]

    def submit_order(self, *, order_data):
        symbol = order_data.symbol
        qty = order_data.qty
        side = order_data.side
        order_type = order_data.type
        limit_price = getattr(order_data, "limit_price", None)
        client_order_id = order_data.client_order_id
        self.last = dict(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=order_data.time_in_force,
            limit_price=limit_price,
            client_order_id=client_order_id,
        )
        raw = _Obj(
            id="alp-123",
            status="filled",
            filled_qty=str(qty),
            filled_avg_price="150.25",
            client_order_id=client_order_id or "auto-1",
            symbol=symbol,
            side=side,
            qty=str(qty),
            type=order_type,
            limit_price=limit_price,
        )
        if client_order_id is not None:
            self.orders_by_client_id[client_order_id] = raw
        return raw

    def get_order_by_client_id(self, client_order_id):
        return self.orders_by_client_id[client_order_id]

    def get_orders(self, *, filter):
        self.list_args = (filter.status, filter.limit, filter.direction)
        return self.open_orders

    def get_account(self):
        return _Obj(cash="50000", equity="100000", buying_power="200000",
                    currency="USD", portfolio_value="100000")

    def get_all_positions(self):
        return [_Obj(symbol="AAPL", qty="10", avg_entry_price="140.0",
                     current_price="150.0")]

    def cancel_order_by_id(self, broker_order_id):
        self.cancelled = broker_order_id


def verify_alpaca():
    b = AlpacaBroker()
    b._api = _FakeAlpacaREST()  # inject; skip the network connect()
    b._sdk = _ALPACA_SDK
    o = b.submit_order(
        "AAPL",
        OrderSide.BUY,
        10,
        OrderType.MARKET,
        client_order_id="qc-check-1",
    )
    _check("alpaca.submit request kwargs", b._api.last["side"] == "buy"
           and b._api.last["type"] == "market" and b._api.last["qty"] == 10
           and b._api.last["client_order_id"] == "qc-check-1")
    _check("alpaca.submit parses Order",
           o.order_id == "alp-123" and o.status is OrderStatus.FILLED
           and abs(o.filled_quantity - 10) < 1e-9 and abs((o.avg_fill_price or 0) - 150.25) < 1e-9,
           f"{o.order_id}/{o.status}/{o.filled_quantity}/{o.avg_fill_price}")
    found = b.find_order_by_client_order_id("qc-check-1")
    _check(
        "alpaca.client-order lookup parses",
        found is not None
        and found.order_id == "alp-123"
        and found.status is OrderStatus.FILLED,
    )
    open_orders = b.get_open_orders()
    _check(
        "alpaca.open-order reconciliation",
        len(open_orders) == 1
        and open_orders[0].order_id == "alp-open"
        and open_orders[0].status is OrderStatus.SUBMITTED
        and b._api.list_args == ("open", 500, "asc"),
    )
    acct = b.get_account()
    _check("alpaca.get_account parses", abs(acct.equity - 100000) < 1e-9
           and abs(acct.buying_power - 200000) < 1e-9 and acct.currency == "USD")
    pos = b.get_positions()
    _check("alpaca.get_positions parses", len(pos) == 1 and pos[0].symbol == "AAPL"
           and abs(pos[0].quantity - 10) < 1e-9 and abs(pos[0].market_price - 150.0) < 1e-9)
    b.cancel_order("alp-123")
    _check("alpaca.cancel_order", b._api.cancelled == "alp-123")
    # limit order forwards the price
    b.submit_order("AAPL", OrderSide.SELL, 5, OrderType.LIMIT, limit_price=160.0)
    _check("alpaca.limit forwards price", b._api.last["limit_price"] == 160.0
           and b._api.last["type"] == "limit" and b._api.last["side"] == "sell")


# --------------------------------------------------------------------------- #
# CCXT (ccxt unified shapes: dict responses)
# --------------------------------------------------------------------------- #
class _FakeCCXT:
    def create_order(self, symbol, type, side, amount, price=None, params=None):
        self.last = dict(symbol=symbol, type=type, side=side, amount=amount, price=price)
        return {"id": "ccxt-9", "status": "closed", "filled": amount, "average": 50000.0}

    def fetch_balance(self):
        return {"total": {"BTC": 0.5, "USDT": 1000.0, "ETH": 0.0},
                "free": {"BTC": 0.5, "USDT": 1000.0}}

    def cancel_order(self, oid, symbol=None):
        self.cancelled = (oid, symbol)


def verify_ccxt():
    b = CCXTBroker(exchange="binance")
    b._exchange = _FakeCCXT()  # inject
    o = b.submit_order("BTC/USDT", OrderSide.BUY, 0.5, OrderType.MARKET)
    _check("ccxt.submit request", b._exchange.last["side"] == "buy"
           and b._exchange.last["type"] == "market" and abs(b._exchange.last["amount"] - 0.5) < 1e-12)
    _check("ccxt.submit parses Order (closed->FILLED)",
           o.order_id == "ccxt-9" and o.status is OrderStatus.FILLED
           and abs(o.filled_quantity - 0.5) < 1e-12 and abs((o.avg_fill_price or 0) - 50000) < 1e-6,
           f"{o.order_id}/{o.status}/{o.filled_quantity}/{o.avg_fill_price}")
    pos = b.get_positions()
    syms = {p.symbol: p.quantity for p in pos}
    _check("ccxt.get_positions (nonzero balances only)",
           syms.get("BTC") == 0.5 and syms.get("USDT") == 1000.0 and "ETH" not in syms, str(syms))
    try:
        b.get_account()
    except BrokerError as exc:
        _check(
            "ccxt.get_account refuses incomplete cross-asset valuation",
            "unvalued assets" in str(exc),
            str(exc),
        )
    else:
        _check(
            "ccxt.get_account refuses incomplete cross-asset valuation",
            False,
            "mixed-asset balance was accepted",
        )
    b.cancel_order("ccxt-9", "BTC/USDT")
    _check("ccxt.cancel_order", b._exchange.cancelled == ("ccxt-9", "BTC/USDT"))


# --------------------------------------------------------------------------- #
# Interactive Brokers (ib_async shapes)
# --------------------------------------------------------------------------- #
class _FakeContract:
    def __init__(self, symbol, exchange="SMART", currency="USD"):
        self.symbol = symbol


class _FakeIBOrder:
    def __init__(self, action, quantity, limit_price=None):
        self.action = action
        self.totalQuantity = quantity
        self.lmtPrice = limit_price
        self.orderId = 42
        self.permId = 4242


class _FakeTrade:
    def __init__(self, order):
        self.order = order
        self.orderStatus = _Obj(status="Filled", filled=order.totalQuantity, avgFillPrice=150.0)


class _FakeIB:
    def qualifyContracts(self, contract):
        self.qualified = contract

    def placeOrder(self, contract, order):
        self.last = (contract, order)
        return _FakeTrade(order)

    def positions(self):
        return [_Obj(contract=_FakeContract("MSFT"), position=20.0, avgCost=300.0)]

    def accountSummary(self):
        return [_Obj(tag="NetLiquidation", value="250000", currency="USD"),
                _Obj(tag="TotalCashValue", value="50000", currency="USD"),
                _Obj(tag="BuyingPower", value="500000", currency="USD")]

    def cancelOrder(self, *a, **k):
        self.cancelled = True

    def disconnect(self):
        self.disconnected = True


def verify_ib():
    b = IBBroker()
    b._ib = _FakeIB()  # inject
    b._sdk = {"Stock": _FakeContract, "MarketOrder": _FakeIBOrder, "LimitOrder": _FakeIBOrder}
    o = b.submit_order("AAPL", OrderSide.BUY, 10, OrderType.MARKET)
    _check("ib.submit places order", b._ib.last[1].action == "BUY"
           and abs(b._ib.last[1].totalQuantity - 10) < 1e-9)
    _check("ib.submit parses Order", o.order_id == "42"
           and abs(o.filled_quantity - 10) < 1e-9 and abs((o.avg_fill_price or 0) - 150.0) < 1e-9,
           f"{o.order_id}/{o.filled_quantity}/{o.avg_fill_price}")
    pos = b.get_positions()
    _check("ib.get_positions parses", len(pos) == 1 and pos[0].symbol == "MSFT"
           and abs(pos[0].quantity - 20) < 1e-9 and abs(pos[0].avg_price - 300.0) < 1e-9)
    acct = b.get_account()
    _check("ib.get_account parses NetLiquidation->equity",
           abs(acct.equity - 250000) < 1e-9, f"equity={acct.equity}")


def main() -> int:
    for fn in (verify_alpaca, verify_ccxt, verify_ib):
        try:
            fn()
        except Exception as exc:  # an adapter raising is itself a failure
            _check(f"{fn.__name__} ran without error", False, repr(exc))
    npass = sum(1 for _, ok, _ in _results if ok)
    print("Broker adapter behavioral verification (faithful SDK mocks)")
    print("=" * 66)
    for name, ok, detail in _results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if not ok else ""))
    print("=" * 66)
    print(f"{npass}/{len(_results)} checks passed")
    print(
        "Note: mocks cover request construction and response parsing, not "
        "authenticated connectivity, account permissions, or venue behavior."
    )
    return 0 if npass == len(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
