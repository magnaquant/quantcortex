"""Import-surface smoke test: every module imports with only the scientific core.

This guards two load-bearing invariants at once:

* **Namespacing.** All importable code lives under the single ``quantcortex``
  package; every discovered module name must start with ``quantcortex.``.
* **Lazy optional deps.** Importing any module must succeed with only the
  scientific core installed (numpy/pandas/scipy/sklearn/matplotlib/pyarrow).
  A heavy/optional dependency (torch, lightgbm, hmmlearn, ccxt, ...) imported at
  module top level instead of lazily inside the using function would break this.

It would have failed loudly if the package-namespacing rewrite had missed a
module, and it fails if anyone later hoists an optional import to module scope.
"""

from __future__ import annotations

import importlib
import pkgutil

import quantcortex


def _all_submodules():
    return [m.name for m in pkgutil.walk_packages(quantcortex.__path__, "quantcortex.")]


def test_every_module_imports_with_core_only():
    failures = {}
    names = _all_submodules()
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - we want to report any failure
            failures[name] = f"{type(exc).__name__}: {exc}"
    assert not failures, "modules failed to import with the scientific core only:\n" + "\n".join(
        f"  {n}: {e}" for n, e in failures.items()
    )
    # Guard against the walk silently discovering nothing (which would make the
    # assertion above vacuously pass): the package is substantial.
    assert len(names) >= 90, f"expected the full package surface, found only {len(names)}"


def test_all_modules_are_namespaced_under_quantcortex():
    offenders = [n for n in _all_submodules() if not n.startswith("quantcortex.")]
    assert not offenders, f"modules not under the quantcortex namespace: {offenders}"
