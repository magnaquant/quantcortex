"""Run one full rebalance cycle through the execution layer.

This is the operational counterpart to ``research/05_live_trading_bridge.ipynb``:
it wires data -> strategy -> pre-trade risk -> order translation -> broker, and
supports a deterministic local dry run and paper-only submission.

Choose the price source explicitly:

* **offline dry-run** (``--offline``): generates deterministic test prices,
  computes the target book, runs the pre-trade risk gate, translates to orders
  against a notional account, and walks them through the local order-lifecycle
  state machine (NEW -> SUBMITTED -> FILLED). Nothing leaves the process.
* **paper preview** (``--live-yfinance`` and ``ALPACA_*`` set): connects to the
  Alpaca *paper* account, reads equity/positions, computes the orders that would
  be sent, and prints them without submitting.
* **paper submit** (add ``--submit``): places the orders on the paper account.
  Submission is idempotent across restarts, refuses unresolved open orders,
  and runs sell and buy legs in separate account snapshots.

    python scripts/paper_trade_cycle.py --offline
    python scripts/paper_trade_cycle.py --live-yfinance
    python scripts/paper_trade_cycle.py --live-yfinance --submit

It never touches a live (real-money) endpoint: it forces a paper broker and
refuses to submit unless the configured base URL is a paper endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logging.getLogger("hmmlearn").setLevel(logging.ERROR)
# The CLI reports live-fetch failures itself; suppress duplicate provider logs.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
# joblib/loky can print a physical-core detection traceback on hosts where CPU
# topology is unreadable; pin it so the --offline dry-run stays clean (respects
# an existing override and matches the single-threaded determinism elsewhere).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from quantcortex.data.processors.calendar import first_session_each_week
from quantcortex.execution.brokers.alpaca_broker import is_alpaca_paper_endpoint
from quantcortex.execution.brokers.base import BrokerError
from quantcortex.execution.order_manager import (
    Order,
    OrderError,
    OrderManager,
    OrderSide,
    OrderStatus,
    OrderType,
    validate_order_request,
)
from quantcortex.execution.position_manager import PositionManager
from quantcortex.execution.pre_trade_risk import PreTradeRiskCheck, PreTradeRiskError
from quantcortex.execution.state_persistence import (
    StatePersistence,
    StatePersistenceError,
)
from quantcortex.portfolio.base import PortfolioMode
from quantcortex.strategies.multi_asset_rotation import MultiAssetRotation

UNIVERSE = ["QQQ", "VGT", "GLD", "TLT", "SPY", "VIG"]
DEFAULT_CAPITAL = 100_000.0
MAX_POSITION_WEIGHT = MultiAssetRotation.DEFAULT_MAX_POSITION_WEIGHT
SUBMISSION_LEDGER_KEY = "paper_submission_attempts"
YFINANCE_NOTICE = (
    "Live Yahoo Finance data is fetched through yfinance. Review Yahoo's terms "
    "and yfinance's legal disclaimer at https://ranaroussi.github.io/yfinance/."
)


def load_prices(offline: bool = False) -> pd.DataFrame:
    """Load synthetic dry-run prices or explicitly requested live prices.

    With ``offline=True`` the network is never touched: the synthetic series is
    used directly, so the dry-run is deterministic and emits no provider noise.
    """
    if offline:
        rng = np.random.default_rng(0)
        dates = pd.bdate_range("2022-01-01", periods=600)
        values = 100 * np.exp(
            np.cumsum(rng.normal(0.0003, 0.011, (600, len(UNIVERSE))), axis=0)
        )
        return pd.DataFrame(values, index=dates, columns=UNIVERSE)

    print(YFINANCE_NOTICE, file=sys.stderr)
    from quantcortex.data.providers.yfinance_provider import YFinanceProvider

    prices = YFinanceProvider().get_prices(UNIVERSE, start="2022-01-01")
    if prices is None or prices.empty or prices.shape[0] <= 200:
        raise RuntimeError("yfinance returned insufficient price history")
    return prices.dropna(how="all").ffill(limit=5).dropna()


def latest_target(prices: pd.DataFrame) -> pd.Series:
    """Return the most recent weekly target, including an explicit flat book."""
    weekly = first_session_each_week(prices.index)
    weights = MultiAssetRotation().generate_weights(prices, weekly)
    if weights.empty:
        raise RuntimeError("strategy produced no weekly targets")
    target = weights.iloc[-1]
    return target[target.abs() > 1e-9]


def build_orders(target: pd.Series, prices: pd.DataFrame, capital: float, current: dict):
    last_px = prices.iloc[-1]
    pm = PositionManager()
    orders = pm.target_weights_to_orders(
        target,
        last_px,
        capital=capital,
        current_positions=current,
    )
    return sorted(
        orders,
        key=lambda order: (
            0 if OrderSide(order["side"]) is OrderSide.SELL else 1,
            order["symbol"],
        ),
    )


def _intent_request(intent: dict) -> Order:
    if not isinstance(intent, dict):
        raise OrderError("paper order intent must be a dictionary")
    try:
        return validate_order_request(
            intent["symbol"],
            intent["side"],
            intent["quantity"],
            OrderType.MARKET,
        )
    except KeyError as exc:
        raise OrderError(f"paper order intent is missing {exc.args[0]!r}") from exc


def make_client_order_id(as_of, intent: dict) -> str:
    """Build a deterministic id for one dated rebalance intent."""
    timestamp = pd.Timestamp(as_of)
    if pd.isna(timestamp):
        raise ValueError("paper order as-of timestamp must be valid")
    request = _intent_request(intent)
    date_token = timestamp.date().strftime("%Y%m%d")
    payload = "|".join(
        [
            "quantcortex-multi-asset-rotation",
            date_token,
            request.symbol,
            request.side.value,
            format(request.quantity, ".12g"),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"qc-mar-{date_token}-{digest}"


def _load_submission_ledger(store: StatePersistence) -> dict:
    ledger = store.load_state(SUBMISSION_LEDGER_KEY, default={}) or {}
    if not isinstance(ledger, dict):
        raise StatePersistenceError("paper submission ledger must be a JSON object")
    for client_id, record in ledger.items():
        if not isinstance(client_id, str) or not client_id.strip():
            raise StatePersistenceError("paper submission ledger has an invalid id")
        if not isinstance(record, dict):
            raise StatePersistenceError(
                f"paper submission ledger entry {client_id!r} is invalid"
            )
    return ledger


def _ledger_record(as_of, request: Order, state: str) -> dict:
    return {
        "as_of": pd.Timestamp(as_of).date().isoformat(),
        "symbol": request.symbol,
        "side": request.side.value,
        "quantity": request.quantity,
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _assert_order_matches_intent(order: Order, request: Order) -> None:
    if (
        order.symbol != request.symbol
        or order.side is not request.side
        or order.order_type is not OrderType.MARKET
        or not math.isclose(
            order.quantity,
            request.quantity,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise BrokerError(
            "Alpaca order returned for the client id does not match the "
            "requested symbol, side, quantity, and market-order type."
        )


def _reconcile_or_submit(
    broker,
    store: StatePersistence,
    ledger: dict,
    as_of,
    intent: dict,
    *,
    retry_confirmed_missing: bool = False,
) -> tuple[Order, bool, str]:
    """Reconcile a deterministic client id, then submit at most once."""
    request = _intent_request(intent)
    client_id = make_client_order_id(as_of, intent)
    existing = broker.find_order_by_client_order_id(client_id)
    if existing is not None:
        _assert_order_matches_intent(existing, request)
        record = _ledger_record(as_of, request, existing.status.value)
        record["broker_order_id"] = existing.order_id
        ledger[client_id] = record
        store.save_state(SUBMISSION_LEDGER_KEY, ledger)
        return existing, False, client_id

    prior = ledger.get(client_id)
    if prior is not None and not retry_confirmed_missing:
        raise BrokerError(
            f"Prior submission attempt {client_id!r} is absent from the broker "
            "lookup, so its outcome is uncertain. Do not retry until the Alpaca "
            "dashboard or support confirms it was not routed; then use "
            "--retry-confirmed-missing."
        )

    record = _ledger_record(as_of, request, "ATTEMPTING")
    if prior is not None:
        record["operator_confirmed_missing_at"] = datetime.now(
            timezone.utc
        ).isoformat()
    ledger[client_id] = record
    # Persist before the network call. A crash or timeout must leave a durable
    # marker that prevents an automatic duplicate submission on the next run.
    store.save_state(SUBMISSION_LEDGER_KEY, ledger)
    try:
        placed = broker.submit_order(
            request.symbol,
            request.side,
            request.quantity,
            client_order_id=client_id,
        )
        _assert_order_matches_intent(placed, request)
    except (BrokerError, OrderError):
        ledger[client_id] = _ledger_record(as_of, request, "UNKNOWN")
        try:
            store.save_state(SUBMISSION_LEDGER_KEY, ledger)
        except StatePersistenceError:
            pass
        raise

    record = _ledger_record(as_of, request, placed.status.value)
    record["broker_order_id"] = placed.order_id
    ledger[client_id] = record
    store.save_state(SUBMISSION_LEDGER_KEY, ledger)
    return placed, True, client_id


def show_orders(orders, last_px) -> None:
    if not orders:
        print("  (no orders: target matches current positions)")
        return
    for o in orders:
        notional = o["quantity"] * float(last_px.get(o["symbol"], float("nan")))
        print(f"  {o['side'].value:>4} {o['quantity']:>10.2f}  {o['symbol']:<6} (~${notional:,.0f})")


def run_offline(prices, target, forced: bool = False) -> int:
    reason = "--offline forced; no broker calls" if forced else "no ALPACA_* credentials found"
    print(f"MODE: offline dry-run ({reason})\n")
    capital = DEFAULT_CAPITAL
    w_vec = target.reindex(UNIVERSE).fillna(0.0).to_numpy()
    risk = PreTradeRiskCheck(max_position_weight=MAX_POSITION_WEIGHT)
    ok, violations = risk.check_weights(
        w_vec, mode=PortfolioMode.LONG_ONLY
    )
    print(f"pre-trade risk: ok={ok} violations={violations}")
    orders = build_orders(target, prices, capital, current={})
    risk.assert_safe(
        weights=w_vec,
        mode=PortfolioMode.LONG_ONLY,
        orders=orders,
        prices=prices.iloc[-1],
        capital=capital,
        current_positions={},
    )
    print(f"\norders that would be sent (notional capital ${capital:,.0f}):")
    show_orders(orders, prices.iloc[-1])

    # Walk them through the local lifecycle to demonstrate the state machine.
    om = OrderManager()
    last_px = prices.iloc[-1]
    for i, o in enumerate(orders):
        oid = f"sim-{i:03d}"
        om.create_order(oid, o["symbol"], OrderSide(o["side"]), float(o["quantity"]))
        om.submit(oid)
        om.fill(oid, fill_price=float(last_px[o["symbol"]]))
    states = {o.order_id: o.status.value for o in om.orders}
    if states:
        assert all(s == "FILLED" for s in states.values())
        print(f"\nsimulated lifecycle: {len(states)} orders NEW -> SUBMITTED -> FILLED")
    print(
        "\nSet ALPACA_API_KEY / ALPACA_SECRET_KEY (paper), rerun with "
        "--live-yfinance, and add --submit to place paper orders."
    )
    return 0


def run_paper(
    prices,
    target,
    submit: bool,
    *,
    retry_confirmed_missing: bool = False,
) -> int:
    from quantcortex.execution.brokers.alpaca_broker import AlpacaBroker

    base = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    if not is_alpaca_paper_endpoint(base):
        print(f"REFUSING: ALPACA_BASE_URL={base!r} is not a paper endpoint. "
              "This script only trades paper.")
        return 2

    broker = AlpacaBroker(paper=True)
    try:
        broker.connect()
        open_orders = broker.get_open_orders()
        if open_orders:
            print(
                "REFUSING: the paper account has unresolved open orders. "
                "Reconcile or cancel them before computing a new target."
            )
            for order in open_orders:
                print(
                    f"  {order.order_id} {order.side.value} {order.quantity:g} "
                    f"{order.symbol} -> {order.status.value}"
                )
            return 1

        acct = broker.get_account()
        positions = {p.symbol: p.quantity for p in broker.get_positions()}
        capital = float(acct.equity)
        if not np.isfinite(capital) or capital <= 0.0:
            raise BrokerError("Alpaca paper account returned non-positive equity")
        print(
            f"MODE: paper {'SUBMIT' if submit else 'preview'} | "
            f"equity ${capital:,.2f} | {len(positions)} open positions\n"
        )

        w_vec = target.reindex(UNIVERSE).fillna(0.0).to_numpy()
        risk = PreTradeRiskCheck(max_position_weight=MAX_POSITION_WEIGHT)
        orders = build_orders(target, prices, capital, current=positions)
        risk.assert_safe(
            weights=w_vec,
            mode=PortfolioMode.LONG_ONLY,
            orders=orders,
            prices=prices.iloc[-1],
            capital=capital,
            current_positions=positions,
        )

        print("orders:")
        show_orders(orders, prices.iloc[-1])
        if not submit:
            print("\npreview only; re-run with --submit to place paper orders.")
            return 0

        store = StatePersistence(namespace="qc-paper-cycle")
        ledger = _load_submission_ledger(store)
        om = OrderManager()
        failures = 0
        reconciled = False
        sell_orders = [
            order for order in orders if OrderSide(order["side"]) is OrderSide.SELL
        ]
        buy_orders = [
            order for order in orders if OrderSide(order["side"]) is OrderSide.BUY
        ]
        phase = sell_orders if sell_orders else buy_orders
        for order in phase:
            try:
                placed, submitted_now, client_id = _reconcile_or_submit(
                    broker,
                    store,
                    ledger,
                    prices.index[-1],
                    order,
                    retry_confirmed_missing=retry_confirmed_missing,
                )
                if not submitted_now:
                    reconciled = True
                    print(
                        f"  reconciled {client_id}: {placed.order_id} "
                        f"-> {placed.status.value}; no order resent"
                    )
                    break
                om.register(placed)
                print(
                    f"  submitted {order['side'].value} {order['quantity']:.2f} "
                    f"{order['symbol']} [{client_id}] -> {placed.status.value}"
                )
                if placed.status in {OrderStatus.CANCELLED, OrderStatus.REJECTED}:
                    failures += 1
                    break
            except (BrokerError, OrderError) as exc:
                failures += 1
                print(f"  FAILED {order['symbol']}: {exc}")
                break

        print(f"\nsubmitted {len(om.orders)} paper orders in this invocation.")
        if sell_orders and buy_orders and not failures and not reconciled:
            print(
                "buy orders were deliberately deferred. Re-run after all sell "
                "orders fill so positions and buying power are refreshed."
            )
            return 1
        if reconciled:
            print(
                "A prior order was reconciled. Re-run from a fresh account "
                "snapshot before submitting any remaining intent."
            )
            return 1
        if om.orders:
            print(
                "Reconcile fills before the next cycle; open orders block a new "
                "rebalance."
            )
        return 1 if failures else 0
    except PreTradeRiskError as exc:
        print(f"pre-trade risk REJECTED the book: {exc}")
        return 1
    except (BrokerError, StatePersistenceError, ValueError) as exc:
        print(f"paper cycle failed: {exc}")
        return 1
    finally:
        broker.disconnect()


def main(argv) -> int:
    ap = argparse.ArgumentParser(description="quantcortex paper rebalance cycle")
    ap.add_argument("--submit", action="store_true", help="actually place paper orders")
    ap.add_argument(
        "--retry-confirmed-missing",
        action="store_true",
        help=(
            "retry a locally recorded submission only after Alpaca confirms the "
            "client order id was not routed"
        ),
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--offline",
        action="store_true",
        help="use the labeled synthetic dry-run; never touch network or broker",
    )
    source.add_argument(
        "--live-yfinance",
        action="store_true",
        help="explicitly fetch live prices through yfinance",
    )
    args = ap.parse_args(argv[1:])
    if args.offline and args.submit:
        ap.error("--submit requires --live-yfinance")
    if args.retry_confirmed_missing and not args.submit:
        ap.error("--retry-confirmed-missing requires --submit")
    has_creds = bool(
        os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY")
    )
    if args.submit and not has_creds:
        print(
            "paper submission requires ALPACA_API_KEY and ALPACA_SECRET_KEY",
            file=sys.stderr,
        )
        return 2

    try:
        prices = load_prices(offline=args.offline)
        target = latest_target(prices)
    except Exception as exc:
        print(f"cycle setup failed: {exc}", file=sys.stderr)
        return 1
    print(f"universe: {UNIVERSE}")
    print(f"target weights:\n{target.round(3).to_string() if not target.empty else '  (flat / no signal)'}\n")
    if args.offline:
        return run_offline(prices, target, forced=True)
    return (
        run_paper(
            prices,
            target,
            args.submit,
            retry_confirmed_missing=args.retry_confirmed_missing,
        )
        if has_creds
        else run_offline(prices, target)
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
