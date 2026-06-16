# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. It is the authoritative agent guide; `AGENTS.md` is a short orientation for other agents and must stay consistent with this file.

## Commands

A `.venv` holds the scientific core plus the optional stack; the core alone
(numpy, pandas, scipy, scikit-learn, matplotlib, pyarrow) is enough for tests
and lint. Run everything from the repo root.

- Tests: `.venv/bin/python -m pytest tests/ -q` (pytest config sets
  `pythonpath = ["."]`, so the `quantcortex` package resolves from the root).
- Single test: `.venv/bin/python -m pytest tests/test_weight_interface.py::test_equal_weight_sums_to_one -v`
- Lint: `.venv/bin/ruff check .` (CI enforces this; must stay clean). Auto-fix: `ruff check . --fix`.
- Operational scripts import `quantcortex.*`, so run them with the root on the
  path: `PYTHONPATH=. .venv/bin/python scripts/<name>.py`
  (validate_performance, generate_report, survivorship_demo, verify_brokers,
  paper_trade_cycle). Performance commands require an explicit source:
  `generate_report.py --prices-csv local_data/rotation_prices.csv` or
  `generate_report.py --live-yfinance`; `validate_performance.py` requires
  `--live-yfinance`; `survivorship_demo.py` requires `--live-yfinance`;
  `paper_trade_cycle.py` requires either `--offline` or `--live-yfinance`.
- CI (`.github/workflows/ci.yml`) runs ruff + pytest with a 60% coverage floor
  on Python 3.11-3.14 using core deps only; optional extras are deliberately not
  installed there. Separate jobs execute all notebooks on deterministic
  test-only fixtures and build/smoke-install the wheel.

## Working norms

Treat every change as unaudited until the relevant tests and invariants have
been checked directly.
- Keep the gates green: `ruff check .` and `pytest tests/ -q` must pass before a
  change is done (CI enforces both). When fixing a bug, add a regression test
  under `tests/` (see `tests/test_regression_guards.py` for the style: assert a
  hand-derived/canonical value, not a snapshot of current behaviour).
- Make surgical changes: touch only what the task needs; do not reformat,
  re-sort imports, or "improve" unrelated code. The regressions found here
  (regime non-determinism, the pre-trade contract, factor-cap ordering, the
  engine's executed-vs-target accounting) came from subtle interactions, so
  incidental edits to the money path are high-risk.
- Resist over-engineering: this codebase already leans thorough. Do not add
  config, abstractions, or flexibility beyond what was asked.
- Confirm high-blast-radius changes before doing them: anything that rewrites
  every import, moves the package layout, or touches the money path belongs in
  its own reviewed change, not bundled into unrelated work.

## The weight contract (keystone: quantcortex/portfolio/base.py)

Every component that emits weights validates them through one of two functions;
which one depends on where it sits in the pipeline:

- `enforce_weight_contract` (STRICT): float64 `(n_assets,)`; long-only weights
  are in `[0, 1]` and sum to `1.0`, while market-neutral weights are in
  `[-1, 1]` and sum to `0.0`. This is the **allocation**
  layer contract; `PortfolioOptimizer.optimize` applies it automatically, so
  subclasses only implement `_compute_weights` returning a raw vector that
  already satisfies the contract for the configured mode.
- `enforce_exposure_contract` (RELAXED): same box, but gross (`sum |w|`) `<= cap`
  and the sum may be below 1 (the remainder is cash). This is the contract for
  **timing/risk overlays and the post-overlay strategy output**, because
  overlays legitimately scale exposure down (a flat book sums to 0, a
  half-scaled long-only book to 0.5).

Getting this distinction wrong is the most common mistake here: a regime-gated
or vol-scaled book is NOT required to sum to 1. Violations raise
`WeightContractViolationError` immediately.

## The pipeline (quantcortex/strategies/base_strategy.py)

`w_t = R_t( T_t( A_t( S_t( X<=t ) ) ) )`: Select -> Allocate -> Timing -> Risk.
- Subclass `Strategy` and implement `select(ctx) -> pd.Series` (alpha scores
  indexed by the chosen symbols). Override `allocate` for score-driven weights;
  the default delegates to the injected `PortfolioOptimizer`.
- Timing and risk overlays are `callable(weights, ctx) -> weights`; concrete
  strategies wire components (HMMRegime, VIXScaler, CircuitBreaker, ...) with
  small lambdas that pull the right data out of the `StrategyContext`.
- `generate_weights(prices, rebalance_dates)` produces the date-by-symbol target
  panel that the backtest engines consume.

## Conventions that are load-bearing (violating them breaks the build)

- **Lazy imports.** Every heavy/optional dependency (torch, lightgbm/xgboost/
  catboost, hmmlearn, transformers, stable-baselines3, gymnasium, alpaca-trade-
  api, ib_insync, ccxt, yfinance, fredapi, polygon, redis, sqlalchemy, lxml) is
  imported *inside* the method that uses it, with an offline fallback or a clear
  ImportError. Modules must import with only the scientific core. Do not add a
  heavy import at module top level.
- **Mandatory costs.** Backtest engines (`quantcortex/backtest/engines/`) raise
  if constructed without a `TransactionCostModel`. Slippage in that model is a
  flat per-trade rate; size/impact-aware costs live in
  `quantcortex/backtest/execution_models/market_impact.py`.
- **Causality / PIT controls.** Research code must use only information
  available at the decision time. `pit_enforcer.py` validates fundamentals
  against `announcement_date`; date-only records use strict-before matching,
  while exact matches require observed intraday timestamps and an explicit
  opt-in. The look-ahead tools are diagnostics, not proof that an arbitrary
  pipeline is leakage-free; walk-forward purge/embargo controls only apply when
  the caller actually uses that engine.
- **Determinism.** `quantcortex/timing/hmm_regime.py` pins BLAS to one thread
  (threadpoolctl) around the HMM fit so backtests are bit-for-bit reproducible; a
  non-converged EM near a regime boundary otherwise flips under multithreaded
  float ordering. Keep model fits seeded and deterministic.
- **Explicit research data.** Do not commit market-data snapshots, generated
  performance charts, or executed notebook outputs. Reports and notebooks must
  use either an owner-supplied permitted CSV or an explicit live-provider opt-in;
  they must never silently substitute generated prices after a fetch failure.
  Synthetic fixtures remain appropriate for tests and the clearly labeled
  `paper_trade_cycle.py --offline` dry run.

## Layer ABCs to subclass

`DataProvider` (quantcortex/data/providers/base.py): `fetch_ohlcv/
fetch_fundamentals/fetch_macro`, canonical UTC-naive OHLCV schema. `Universe`
(quantcortex/data/universe/base.py): point-in-time membership;
`SP500Universe.from_wikipedia()` reconstructs historical constituents from the
current Wikipedia change table, rejects dates before source coverage, and is
still only an approximation. Named index classes do not silently select their
survivorship-biased demo subsets. `Broker` (quantcortex/execution/brokers/
base.py): `submit_order/get_positions/get_account`, adapters lazy-load their SDK.
`OrderManager`: a NEW -> SUBMITTED -> FILLED state machine that validates the
transition before mutating the order.

The current Alpaca and IB adapters target deprecated/archived SDKs
(`alpaca-trade-api` and `ib_insync`). Treat migration to `alpaca-py` and a
maintained IB client, followed by authenticated conformance tests, as a release
requirement rather than a cosmetic dependency update.

## Packaging

All importable code lives under one top-level package, `quantcortex` (e.g.
`from quantcortex.portfolio.base import enforce_weight_contract`). `tests/`,
`scripts/`, `research/`, and `docs/` sit at the repo root, outside the package.
Keep new modules inside `quantcortex/` and import them absolutely as
`quantcortex.<subpkg>...`; there are no relative imports and no top-level
package squatting. Docker commands follow the same namespace rule. Keep `.env`,
`local_data/`, and `reports/` excluded from the Docker build context via
`.dockerignore`.

## Honesty norms

The strategies' Sharpe targets (1.10 / 0.9 in the README) are aspirational
design goals, not measured claims. Do not tune toward a single backtest; record
the true trial count for DSR/BHY analysis and report unfavorable results as-is.
Every published run needs source, permission, date-window, adjustment, and input
digest metadata. See PERFORMANCE.md. Use ASCII punctuation in source and docs;
the README directory-tree block and established proper-name diacritics are the
only intentional exceptions.
