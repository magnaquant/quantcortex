# Contributing to quantcortex

## Development Setup

Use Python 3.11 through 3.14 from the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements/dev.lock
.venv/bin/python -m pip install --no-deps -e .
```

Run the required local checks before opening a pull request:

```bash
.venv/bin/ruff check .
.venv/bin/python -m pytest tests/ -q --cov=quantcortex --cov-fail-under=60
PYTHONPATH=. .venv/bin/python scripts/verify_brokers.py
```

## Change Discipline

Keep changes within the existing pipeline boundaries. Preserve point-in-time
data handling, allocation and exposure contracts, explicit transaction costs,
paper-mode defaults, and fail-closed execution behavior. Changes to accounting,
risk checks, order state, broker adapters, or causality require focused
regression tests.

Do not commit credentials, account data, raw or cached market data, executed
notebook outputs, local state, or generated reports. Published derived figures
must include provenance and be reproducible from an owner-supplied input.

## Dependencies

Edit `pyproject.toml`, then regenerate every lock artifact with:

```bash
scripts/update_dependency_locks.sh
```

Commit `poetry.lock` and all changed files under `requirements/`. CI rejects
lock drift.

## Pull Requests

Use short conventional commits such as `fix: reconcile cash returns`. A pull
request should describe behavior, data and API assumptions, verification, and
remaining limitations. Link relevant issues and include updated figures only
when report output changes.
