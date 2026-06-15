"""Import-surface smoke test: every module imports with only the scientific core.

This guards the load-bearing invariants established by the single-package layout:

* **Lazy optional deps.** Importing any module must succeed with only the
  scientific core installed (numpy/pandas/scipy/sklearn/matplotlib/pyarrow).
  A heavy/optional dependency (torch, lightgbm, hmmlearn, ccxt, ...) imported at
  module top level instead of lazily inside the using function would break this.
* **Single namespace, no squatting.** All code lives under ``quantcortex``; the
  legacy top-level package directories (``portfolio/``, ``data/``, ...) must not
  reappear at the repo root.
* **Surface completeness.** A curated set of money-path modules across every
  layer must be present and importable (a meaningful floor, not just a count).
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import quantcortex

REPO_ROOT = Path(__file__).resolve().parent.parent

# The eight packages that used to sit at the repo root before namespacing.
LEGACY_TOPLEVEL = [
    "data", "alpha", "portfolio", "timing",
    "risk", "backtest", "execution", "strategies",
]

# Representative money-path modules spanning all eight layers. If the package
# is restructured or a layer goes missing, importing these fails loudly.
REQUIRED_MODULES = [
    "quantcortex.portfolio.base",                     # weight-contract keystone
    "quantcortex.strategies.base_strategy",           # pipeline
    "quantcortex.backtest.engines.vectorized",        # backtest engine
    "quantcortex.backtest.costs.transaction_costs",   # mandatory costs
    "quantcortex.data.providers.base",                # data provider ABC
    "quantcortex.data.processors.pit_enforcer",       # PIT / causality
    "quantcortex.execution.order_manager",            # execution state machine
    "quantcortex.timing.hmm_regime",                  # timing / determinism
    "quantcortex.risk.circuit_breaker",               # risk overlay
    "quantcortex.alpha.feature_engineering.alpha158",  # alpha
]


def _all_submodules():
    return [m.name for m in pkgutil.walk_packages(quantcortex.__path__, "quantcortex.")]


def test_every_module_imports_with_core_only():
    failures = {}
    for name in _all_submodules():
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - we want to report any failure
            failures[name] = f"{type(exc).__name__}: {exc}"
    assert not failures, "modules failed to import with the scientific core only:\n" + "\n".join(
        f"  {n}: {e}" for n, e in failures.items()
    )


def test_no_legacy_toplevel_package_dirs():
    """The pre-namespacing package dirs must not exist at the repo root."""
    squatters = [d for d in LEGACY_TOPLEVEL if (REPO_ROOT / d / "__init__.py").is_file()]
    assert not squatters, (
        f"legacy top-level package dirs reappeared at the repo root: {squatters} "
        "(all code must live under quantcortex/)"
    )


def test_required_subpackages_present_and_importable():
    surface = set(_all_submodules())
    missing = [m for m in REQUIRED_MODULES if m not in surface]
    assert not missing, f"required modules absent from the package surface: {missing}"
    for name in REQUIRED_MODULES:
        importlib.import_module(name)  # must import with core deps only
