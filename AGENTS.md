# Repository Guidelines

## Project Structure & Module Organization

All importable code lives under the single top-level `quantcortex` package and
is organized by pipeline layer (import absolutely, e.g.
`from quantcortex.portfolio.base import enforce_weight_contract`):

- `quantcortex/data/`: providers, storage, processors, point-in-time universe utilities, and the bundled `sample/` price snapshot.
- `quantcortex/alpha/`: factor libraries, feature engineering, and validation.
- `quantcortex/portfolio/`: optimizers plus canonical weight and exposure contracts.
- `quantcortex/timing/` and `quantcortex/risk/`: regime, volatility, drawdown, VaR/CVaR, Kelly, and exposure overlays.
- `quantcortex/backtest/`: engines, execution models, costs, metrics, and validation.
- `quantcortex/execution/`: broker adapters, order/position management, risk checks, and persistence.
- `quantcortex/strategies/`: complete strategy pipelines.
- Repo root, outside the package: `tests/` (pytest suite), `research/` (notebooks), `scripts/` (utilities), and `docs/img/` (charts).

## Build, Test, and Development Commands

Run from the repo root:

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check .
PYTHONPATH=. .venv/bin/python scripts/verify_brokers.py
PYTHONPATH=. .venv/bin/python scripts/generate_report.py
```

`pytest` runs the offline core suite. `ruff check .` matches CI lint. Scripts need `PYTHONPATH=.` unless installed editable. If matplotlib cannot write its config cache (sandboxed or CI environments), point it at a writable dir, e.g. `MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl"`.

## Coding Style & Naming Conventions

Target Python 3.11+. Use 4-space indentation, `snake_case` modules/functions, `PascalCase` classes, and source ASCII unless an existing file uses otherwise. Keep comments concise and only for non-obvious logic. Prefer existing layer boundaries and helpers, especially `quantcortex.portfolio.base` contracts.

## Testing Guidelines

Tests use pytest and deterministic synthetic data. Add focused regression tests for contract, accounting, causality, and execution-state changes. Keep optional integrations lazy and mock external brokers/providers; CI installs only the scientific core. Use `test_*.py` and `test_*`.

## Load-Bearing Constraints

Do not weaken the strict allocation contract or relaxed post-overlay exposure contract. Backtests must use a `TransactionCostModel`. Factors and strategy features must stay point-in-time and causal. Broker SDKs, ML libraries, Redis, and data providers should stay lazy imports with clear fallbacks or errors.

## Commit & Pull Request Guidelines

Recent history uses conventional prefixes such as `docs:`, `ci:`, `chore:`, and `feat:`; keep messages short and imperative. PRs should explain behavior changes, list verification commands, call out data or API assumptions, and link issues. Include screenshots only for generated charts or notebook/report changes.

## Agent-Specific Instructions

`CLAUDE.md` is the authoritative agent guide (architecture, the weight contract, load-bearing conventions); this file is the short orientation and must stay consistent with it. Read `CLAUDE.md` for depth. Work from the requested change and verify it directly. State assumptions when ambiguity affects correctness. Keep edits surgical, avoid speculative abstractions, and do not refactor unrelated code. For money-path code, add or update regression tests.

## Security & Configuration Tips

Never commit `.env`, credentials, broker account data, local state, caches, or large generated artifacts. Use `.env.example` as the template. Preserve pre-trade risk checks, paper-mode defaults, and point-in-time data discipline.
