"""Run one full rebalance cycle through the execution layer.

This is the operational counterpart to ``research/05_live_trading_bridge.ipynb``:
it wires data -> strategy -> pre-trade risk -> order translation -> broker, and
is safe to run repeatedly.

Modes (chosen automatically by what is available):

* **offline dry-run** (no ``ALPACA_*`` env vars): synthesises/fetches prices,
  computes the target book, runs the pre-trade risk gate, translates to orders
  against a notional account, and walks them through the local order-lifecycle
  state machine (NEW -> SUBMITTED -> FILLED). Nothing leaves the process.
* **paper preview** (``ALPACA_*`` set, no ``--submit``): connects to the Alpaca
  *paper* account, reads real equity/positions, computes the orders that *would*
  be sent, and prints them without submitting.
* **paper submit** (``ALPACA_*`` set, ``--submit``): actually places the orders
  on the **paper** account.

    python scripts/paper_trade_cycle.py            # offline or paper preview
    python scripts/paper_trade_cycle.py --submit   # place paper orders (paper only)

It never touches a live (real-money) endpoint: it forces a paper broker and
refuses to submit unless the configured base URL is a paper endpoint.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)
# yfinance logs network failures (DNS/HTTP) at error level; the offline fallback
# is expected, so quiet its logger to keep the dry-run output clean.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
# joblib/loky can print a physical-core detection traceback on hosts where CPU
# topology is unreadable; pin it so the --offline dry-run stays clean (respects
# an existing override and matches the single-threaded determinism elsewhere).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from quantcortex.execution.order_manager import OrderManager, OrderSide
from quantcortex.execution.position_manager import PositionManager
from quantcortex.execution.pre_trade_risk import PreTradeRiskCheck, PreTradeRiskError
from quantcortex.portfolio.base import PortfolioMode
from quantcortex.strategies.multi_asset_rotation import MultiAssetRotation

UNIVERSE = ["QQQ", "VGT", "GLD", "TLT", "SPY", "VIG"]
DEFAULT_CAPITAL = 100_000.0


def load_prices(offline: bool = False) -> pd.DataFrame:
    """Recent daily prices for the universe; synthetic fallback if offline.

    With ``offline=True`` the network is never touched: the synthetic series is
    used directly, so the dry-run is deterministic and emits no provider noise.
    """
    if not offline:
        try:
            from quantcortex.data.providers.yfinance_provider import YFinanceProvider

            px = YFinanceProvider().get_prices(UNIVERSE, start="2022-01-01")
            if px is not None and not px.empty and px.shape[0] > 200:
                return px.dropna(how="all").ffill().dropna()
        except Exception as exc:
            print(f"[offline] yfinance unavailable ({type(exc).__name__}); synthetic prices.")
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-01-01", periods=600)
    px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, (600, len(UNIVERSE))), axis=0))
    return pd.DataFrame(px, index=dates, columns=UNIVERSE)


def latest_target(prices: pd.DataFrame) -> pd.Series:
    """Most recent INVESTED weekly target (the regime gate can flatten weeks)."""
    weekly = prices.index[prices.index.weekday == 0]
    W = MultiAssetRotation().generate_weights(prices, weekly)
    invested = W[W.abs().sum(axis=1) > 1e-9]
    if invested.empty:
        return pd.Series(dtype=float)
    target = invested.iloc[-1]
    return target[target.abs() > 1e-9]


def build_orders(target: pd.Series, prices: pd.DataFrame, capital: float, current: dict):
    last_px = prices.iloc[-1]
    pm = PositionManager()
    return pm.target_weights_to_orders(target, last_px, capital=capital, current_positions=current)


def show_orders(orders, last_px) -> None:
    if not orders:
        print("  (no orders: target matches current positions)")
        return
    for o in orders:
        notional = o["quantity"] * float(last_px.get(o["symbol"], float("nan")))
        print(f"  {o['side'].value:>4} {o['quantity']:>10.2f}  {o['symbol']:<6} (~${notional:,.0f})")


def run_offline(prices, target) -> int:
    print("MODE: offline dry-run (no ALPACA_* credentials found)\n")
    capital = DEFAULT_CAPITAL
    w_vec = target.reindex(UNIVERSE).fillna(0.0).to_numpy()
    ok, violations = PreTradeRiskCheck(max_position_weight=0.6).check_weights(
        w_vec, mode=PortfolioMode.LONG_ONLY
    )
    print(f"pre-trade risk: ok={ok} violations={violations}")
    orders = build_orders(target, prices, capital, current={})
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
    print("\nSet ALPACA_API_KEY / ALPACA_SECRET_KEY (paper) in .env to run against "
          "a real paper account; add --submit to actually place orders.")
    return 0


def run_paper(prices, target, submit: bool) -> int:
    from quantcortex.execution.brokers.alpaca_broker import AlpacaBroker, BrokerError

    base = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    if "paper" not in base:
        print(f"REFUSING: ALPACA_BASE_URL={base!r} is not a paper endpoint. "
              "This script only trades paper.")
        return 2

    broker = AlpacaBroker(paper=True)
    try:
        broker.connect()
        acct = broker.get_account()
        positions = {p.symbol: p.quantity for p in broker.get_positions()}
    except BrokerError as exc:
        print(f"Alpaca connection failed: {exc}")
        return 1

    capital = acct.equity or acct.buying_power or DEFAULT_CAPITAL
    print(f"MODE: paper {'SUBMIT' if submit else 'preview'} | equity ${capital:,.2f} | "
          f"{len(positions)} open positions\n")

    w_vec = target.reindex(UNIVERSE).fillna(0.0).to_numpy()
    try:
        PreTradeRiskCheck(max_position_weight=0.6).assert_safe(
            weights=w_vec, mode=PortfolioMode.LONG_ONLY
        )
    except PreTradeRiskError as exc:
        print(f"pre-trade risk REJECTED the book: {exc}")
        return 1

    orders = build_orders(target, prices, capital, current=positions)
    print("orders:")
    show_orders(orders, prices.iloc[-1])
    if not submit:
        print("\npreview only; re-run with --submit to place these paper orders.")
        return 0

    om = OrderManager()
    for i, o in enumerate(orders):
        try:
            placed = broker.submit_order(o["symbol"], OrderSide(o["side"]), float(o["quantity"]))
            om._orders[placed.order_id] = placed  # track the broker-returned order
            print(f"  submitted {o['side'].value} {o['quantity']:.2f} {o['symbol']} -> {placed.status.value}")
        except BrokerError as exc:
            print(f"  FAILED {o['symbol']}: {exc}")
    print(f"\nsubmitted {len(om.orders)} paper orders.")
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description="quantcortex paper rebalance cycle")
    ap.add_argument("--submit", action="store_true", help="actually place paper orders")
    ap.add_argument("--offline", action="store_true",
                    help="force the synthetic-price dry-run; never touch the network "
                         "(deterministic, no provider noise) even if ALPACA_* is set")
    args = ap.parse_args(argv[1:])

    prices = load_prices(offline=args.offline)
    target = latest_target(prices)
    print(f"universe: {UNIVERSE}")
    print(f"target weights:\n{target.round(3).to_string() if not target.empty else '  (flat / no signal)'}\n")
    if target.empty:
        print("strategy is flat this cycle; nothing to trade.")
        return 0

    if args.offline:
        return run_offline(prices, target)
    has_creds = bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"))
    return run_paper(prices, target, args.submit) if has_creds else run_offline(prices, target)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
