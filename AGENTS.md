# Repository Guidelines

## Project Structure & Module Organization

All importable code lives under `quantcortex`; use absolute imports such as
`from quantcortex.portfolio.base import enforce_weight_contract`. Pipeline
layers are `data`, `alpha`, `portfolio`, `timing`, `risk`, `backtest`,
`execution`, and `strategies`. Root directories include `tests/`, `research/`,
`scripts/`, ignored `local_data/`, and ignored `reports/`. See `CLAUDE.md` for
layer details.

## Build, Test, and Development Commands

Run from the repo root:

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check .
PYTHONPATH=. .venv/bin/python scripts/verify_brokers.py
PYTHONPATH=. .venv/bin/python scripts/generate_report.py \
  --prices-csv local_data/rotation_prices.csv
```

`pytest` runs the offline core suite; `ruff check .` matches CI lint. Scripts need `PYTHONPATH=.` unless installed editable. If matplotlib cannot write its cache (sandbox/CI), set `MPLCONFIGDIR` to a writable dir.

## Coding Style & Naming Conventions

Target Python 3.11+. Use 4-space indentation, `snake_case` modules/functions, `PascalCase` classes, and source ASCII unless an existing file uses otherwise. Keep comments concise and only for non-obvious logic. Prefer existing layer boundaries and helpers, especially `quantcortex.portfolio.base` contracts.

## Testing Guidelines

Tests use pytest and deterministic synthetic data. Add focused regression tests for contract, accounting, causality, and execution-state changes. Keep optional integrations lazy and mock external brokers/providers; CI installs only the scientific core. Use `test_*.py` and `test_*`.

## Load-Bearing Constraints

Do not weaken the strict allocation contract or relaxed post-overlay exposure contract. Backtests must use a `TransactionCostModel`. Factors and strategy features must stay point-in-time and causal. Broker SDKs, ML libraries, Redis, and data providers should stay lazy imports with clear fallbacks or errors.

## Commit & Pull Request Guidelines

Use conventional commit prefixes (`feat:`, `fix:`, `docs:`, `ci:`); keep messages short and imperative. PRs should explain behavior changes, list verification commands, call out data or API assumptions, and link issues. Include screenshots only for generated charts or notebook/report changes.

## Agent-Specific Instructions

`CLAUDE.md` is the authoritative agent guide; this file is the short orientation and must stay consistent with it. Read `CLAUDE.md` for depth. Work from the requested change and verify it directly. State assumptions when ambiguity affects correctness. Keep edits surgical, avoid speculative abstractions, and do not refactor unrelated code. For money-path code, add or update regression tests.

## Security & Configuration Tips

Never commit `.env`, credentials, broker account data, local state, market-data
snapshots, or executed notebook outputs. Generated performance charts require
explicit owner approval, adjacent provenance, an input digest, and artifact
hashes; ordinary reports remain ignored. Use `.env.example` as the template.
Synthetic data is limited to tests and clearly labeled dry runs. Preserve
pre-trade risk checks, paper-mode defaults, and point-in-time data discipline.
