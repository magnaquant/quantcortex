# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

A `.venv` holds the scientific core plus the optional stack; the core alone
(numpy, pandas, scipy, scikit-learn, matplotlib, pyarrow) is enough for tests
and lint. Run everything from the repo root.

- Tests: `.venv/bin/python -m pytest tests/ -q` (pytest config sets
  `pythonpath = ["."]`, so the top-level packages resolve from the root).
- Single test: `.venv/bin/python -m pytest tests/test_weight_interface.py::test_equal_weight_sums_to_one -v`
- Lint: `.venv/bin/ruff check .` (CI enforces this; must stay clean). Auto-fix: `ruff check . --fix`.
- Operational scripts import the top-level packages directly, so run them with
  the root on the path: `PYTHONPATH=. .venv/bin/python scripts/<name>.py`
  (validate_performance, generate_report, survivorship_demo, verify_brokers,
  paper_trade_cycle). `generate_report.py` and `validate_performance.py --pit`
  print results; `--live` refetches data instead of using the bundled snapshot.
- CI (`.github/workflows/ci.yml`) runs ruff + pytest on Python 3.11/3.12 with
  core deps only; optional extras are deliberately not installed there.

## The weight contract (keystone: portfolio/base.py)

Every component that emits weights validates them through one of two functions;
which one depends on where it sits in the pipeline:

- `enforce_weight_contract` (STRICT): float64 `(n_assets,)`, each in `[-1, 1]`,
  sum `== 1.0` long-only or `== 0.0` market-neutral. This is the **allocation**
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

## The pipeline (strategies/base_strategy.py)

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
- **Mandatory costs.** Backtest engines (`backtest/engines/`) raise if
  constructed without a `TransactionCostModel`. Slippage in that model is a flat
  per-trade rate; size/impact-aware costs live in `execution_models/market_impact.py`.
- **Strict causality / PIT.** Factors and features use only past data;
  `pit_enforcer.py` keys fundamentals off `announcement_date`,
  `lookahead_detector.py` scans for leakage, and `walk_forward.py` applies a
  purge + embargo gap.
- **Determinism.** `timing/hmm_regime.py` pins BLAS to one thread (threadpoolctl)
  around the HMM fit so backtests are bit-for-bit reproducible; a non-converged
  EM near a regime boundary otherwise flips under multithreaded float ordering.
  Keep model fits seeded and deterministic.
- **Reproducible results.** `generate_report.py` and the README "Results" read
  the fixed snapshot `data/sample/rotation_prices.csv`, because live yfinance
  re-adjusts historical closes on every fetch. Quote numbers from the generator
  verbatim rather than hand-typing them.

## Layer ABCs to subclass

`DataProvider` (data/providers/base.py): `fetch_ohlcv/fetch_fundamentals/
fetch_macro`, canonical UTC-naive OHLCV schema. `Universe` (data/universe/
base.py): point-in-time membership; `SP500Universe.from_wikipedia()`
reconstructs real historical constituents (survivorship-safe membership).
`Broker` (execution/brokers/base.py): `submit_order/get_positions/get_account`,
adapters lazy-load their SDK. `OrderManager`: a NEW -> SUBMITTED -> FILLED state
machine that validates the transition before mutating the order.

## Known structural issue

The eight packages are TOP-LEVEL (`from portfolio.base import ...`,
`import data`), not namespaced under a `quantcortex/` package. This works via
`pythonpath = ["."]` but squats generic names and is hostile to `pip install`.
Do not paper over it by reordering imports; the proper fix is a deliberate,
repo-wide move into a `quantcortex/` package (its own reviewed change).

## Honesty norms

The strategies' Sharpe targets (1.10 / 0.9 in the README) are aspirational
design goals; the measured baselines (rotation ~0.17, momentum_ml ~0.63) are
reported honestly and are NOT to be tuned toward a single backtest (that is the
overfitting the Deflated Sharpe Ratio and BHY tooling exist to catch). See
PERFORMANCE.md. Source and docs are ASCII-only (no em-dashes, en-dashes, or
arrows); the README directory-tree block is the one exception (box-drawing).
