# Architecture

quantcortex is a weight-centric research and guarded paper-execution platform.
All importable code lives under `quantcortex`; scripts, notebooks, tests, paper
artifacts, and operational documentation remain at the repository root.

## Pipeline Boundaries

```text
data -> alpha -> strategy selection -> portfolio allocation
                                      -> timing overlays
                                      -> risk overlays
                                      -> target weights
                                         |-> backtest
                                         `-> pre-trade risk -> broker adapter
```

| Package | Primary modules |
|---|---|
| `quantcortex.data` | Provider adapters, `local_csv.py`, PIT processors, universes, storage |
| `quantcortex.alpha` | Classical factors, Alpha158-style features, GBDT/NLP baselines, validation |
| `quantcortex.portfolio` | Weight contracts, equal weight, minimum variance, HRP, risk parity, RL |
| `quantcortex.timing` | GMM/HMM regimes, time-series momentum, KAMA, VIX scaling |
| `quantcortex.risk` | Circuit breakers, VaR/CVaR, factor exposure, volatility targeting, Kelly sizing |
| `quantcortex.backtest` | Vectorized and event-driven engines, fills, costs, metrics, validation |
| `quantcortex.execution` | Pre-trade risk, order/position management, brokers, state persistence |
| `quantcortex.strategies` | End-to-end select, allocate, overlay, and rebalance workflows |

## Weight Contracts

Portfolio allocators return finite one-dimensional `float64` arrays. Long-only
allocators must be bounded in `[0, 1]` and sum to one; market-neutral allocators
must sum to zero and satisfy their configured per-asset bounds. Violations raise
`WeightContractViolationError`.

Timing and risk overlays may intentionally reduce exposure. Their output uses
the relaxed exposure contract: finite bounded weights with gross exposure no
greater than the configured cap. Residual capital is cash; it is not an
accounting omission.

Use the helpers in `quantcortex.portfolio.base` rather than duplicating weight
validation in a strategy.

## Timing and Accounting

A target computed from close data at time `t` executes on the first strictly
later available bar and starts earning returns after that execution. Both
backtest engines enforce this decision/holding distinction.

Every backtest requires a `TransactionCostModel`. Risky returns, residual-cash
returns, turnover, and costs use one capital clock. A nonzero cash return must
be supplied explicitly; missing cash-proxy observations fail the run.

The event-driven engine is the reference accounting path: it holds drifting
adjusted-close pseudo-shares, sizes targets against post-cost NAV, and applies
fill semantics. Those pseudo-shares are total-return accounting units, not
nominal broker shares. The vectorized engine re-pegs target weights between explicit
rebalances and is intended for approximations and sweeps. Agreement in one
experiment is a diagnostic, not proof that the engines are interchangeable.

`quantcortex.backtest.conformance` defines the canonical long-form target tape
used to compare engines without coupling them to a strategy implementation.
The tape validates complete timestamp-symbol decisions, finite long-only
weights, and gross exposure. See `docs/evaluation-contracts.md`.

The paper keeps two comparison roles separate. Its realized-exposure control is
an exact ex-post arithmetic attribution and is gross of comparator costs. Its
target-exposure comparator is causal, follows the original decision timestamps,
and pays the same modeled cost rate as the strategy.

## Point-in-Time Data

Fundamental records become available according to announcement timestamps, not
period ends. Date-only inputs use strict-before matching unless a source with
verified intraday release times explicitly opts into same-timestamp use.

Historical index membership and historical prices are separate requirements.
The Wikipedia S&P 500 reconstruction provides approximate membership within its
coverage window; it does not provide delisted-security prices.

## Execution State

Paper execution persists positions, orders, submission intents, and metadata as
one versioned snapshot. Writes are atomic and use optimistic concurrency. An
intent is stored as `ATTEMPTING` before broker submission. If the outcome is
uncertain, automatic retry is blocked until reconciliation with the broker.
Intent state is distinct from the order lifecycle, which separately tracks
`NEW`, `SUBMITTED`, partial or complete fills, cancellation, and rejection.

Broker SDK imports remain lazy. Offline conformance and SDK model-construction
tests validate request and response mapping, not authenticated transport or
venue behavior.

## Extension Rules

1. Add importable modules inside the appropriate `quantcortex` package.
2. Preserve absolute imports and existing layer ownership.
3. Keep provider, broker, and heavyweight ML dependencies lazy.
4. Make timestamp, adjustment, universe, and cash assumptions explicit.
5. Add focused regression tests for accounting, causality, weight, or execution-state changes.
6. Treat generated reports and paper artifacts as outputs of reviewed scripts, never hand-edited evidence.
