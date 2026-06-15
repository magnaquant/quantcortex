"""Portfolio optimization base layer and the canonical *weight contract*.

This module is the keystone of quantcortex.  Every portfolio, timing and risk
component ultimately emits a weight vector, and **all** of them route that
vector through :func:`enforce_weight_contract` before it is allowed to leave the
component.  Centralising the contract here means a single, authoritative
definition of "a valid set of portfolio weights".

The contract
------------
A valid weight vector ``w`` satisfies:

* ``w`` is a :class:`numpy.ndarray` with ``dtype == float64`` and shape
  ``(n_assets,)`` (1-D).
* every element is finite and lies in ``[-1.0, 1.0]``.
* ``w.sum()`` equals ``1.0`` for a long-only book or ``0.0`` for a
  market-neutral book (within a small numerical tolerance).

Any violation raises :class:`WeightContractViolationError` *immediately* - we
fail loud and early rather than letting a malformed allocation propagate into a
backtest or, worse, a live order.
"""

from __future__ import annotations

import abc
from enum import Enum
from typing import Optional, Sequence, Union

import numpy as np

__all__ = [
    "WeightContractViolationError",
    "PortfolioMode",
    "enforce_weight_contract",
    "enforce_exposure_contract",
    "normalize_long_only",
    "normalize_market_neutral",
    "PortfolioOptimizer",
]

# Default absolute tolerance for the sum / bound checks.  Floating point
# arithmetic on hundreds of assets accumulates error well above machine
# epsilon, so 1e-6 is the practical "equal to 1.0" threshold.
DEFAULT_TOLERANCE: float = 1e-6


class WeightContractViolationError(ValueError):
    """Raised when a weight vector violates the canonical weight contract."""


class PortfolioMode(str, Enum):
    """Allocation regimes recognised by the weight contract."""

    LONG_ONLY = "long_only"
    MARKET_NEUTRAL = "market_neutral"

    @classmethod
    def coerce(cls, value: Union["PortfolioMode", str]) -> "PortfolioMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as exc:  # pragma: no cover - defensive
            raise WeightContractViolationError(
                f"Unknown portfolio mode {value!r}; expected one of "
                f"{[m.value for m in cls]}"
            ) from exc

    @property
    def target_sum(self) -> float:
        return 1.0 if self is PortfolioMode.LONG_ONLY else 0.0


def enforce_weight_contract(
    weights: Union[np.ndarray, Sequence[float]],
    *,
    mode: Union[PortfolioMode, str] = PortfolioMode.LONG_ONLY,
    tolerance: float = DEFAULT_TOLERANCE,
    lower: float = -1.0,
    upper: float = 1.0,
    name: Optional[str] = None,
) -> np.ndarray:
    """Validate ``weights`` against the canonical contract.

    Parameters
    ----------
    weights:
        Candidate weight vector (anything coercible to a 1-D float64 array).
    mode:
        ``"long_only"`` (weights sum to 1.0) or ``"market_neutral"`` (sum 0.0).
    tolerance:
        Absolute tolerance for the bound and sum checks.
    lower, upper:
        Per-asset weight bounds.  Defaults implement the ``[-1, 1]`` contract.
    name:
        Optional label included in error messages (e.g. the producing class).

    Returns
    -------
    numpy.ndarray
        The validated weights as a contiguous ``float64`` array.

    Raises
    ------
    WeightContractViolationError
        If any clause of the contract is violated.
    """
    label = f" [{name}]" if name else ""
    mode = PortfolioMode.coerce(mode)

    try:
        arr = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise WeightContractViolationError(
            f"Weights{label} are not coercible to a float64 array: {exc}"
        ) from exc

    if arr.ndim != 1:
        raise WeightContractViolationError(
            f"Weights{label} must be 1-D with shape (n_assets,), got shape {arr.shape}"
        )
    if arr.size == 0:
        raise WeightContractViolationError(f"Weights{label} are empty")

    if not np.all(np.isfinite(arr)):
        bad = np.where(~np.isfinite(arr))[0].tolist()
        raise WeightContractViolationError(
            f"Weights{label} contain non-finite values at indices {bad}"
        )

    below = np.where(arr < lower - tolerance)[0]
    above = np.where(arr > upper + tolerance)[0]
    if below.size or above.size:
        offenders = {int(i): float(arr[i]) for i in np.concatenate([below, above])}
        raise WeightContractViolationError(
            f"Weights{label} outside bounds [{lower}, {upper}]: {offenders}"
        )

    total = float(arr.sum())
    target = mode.target_sum
    if abs(total - target) > tolerance:
        raise WeightContractViolationError(
            f"Weights{label} sum to {total:.10f}; {mode.value} requires "
            f"{target:.1f} (tolerance {tolerance:g})"
        )

    return np.ascontiguousarray(arr, dtype=np.float64)


def enforce_exposure_contract(
    weights: Union[np.ndarray, Sequence[float]],
    *,
    max_gross: float = 1.0,
    tolerance: float = DEFAULT_TOLERANCE,
    lower: float = -1.0,
    upper: float = 1.0,
    name: Optional[str] = None,
) -> np.ndarray:
    """Relaxed contract for *exposure-scaling* layers (timing & risk overlays).

    Unlike :func:`enforce_weight_contract`, this does **not** require the weights
    to sum to exactly 1.0 (or 0.0): timing and risk overlays legitimately scale
    gross exposure down - a fully de-risked book is flat (sum 0) and a
    half-scaled long-only book sums to 0.5, with the remainder held in cash.

    It still guarantees the structural invariants every downstream consumer
    relies on: a finite, 1-D ``float64`` array, every weight in ``[-1, 1]`` and
    gross exposure (``sum |w|``) no greater than ``max_gross``.
    """
    label = f" [{name}]" if name else ""
    try:
        arr = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise WeightContractViolationError(
            f"Weights{label} are not coercible to a float64 array: {exc}"
        ) from exc

    if arr.ndim != 1:
        raise WeightContractViolationError(
            f"Weights{label} must be 1-D, got shape {arr.shape}"
        )
    if arr.size == 0:
        raise WeightContractViolationError(f"Weights{label} are empty")
    if not np.all(np.isfinite(arr)):
        bad = np.where(~np.isfinite(arr))[0].tolist()
        raise WeightContractViolationError(
            f"Weights{label} contain non-finite values at indices {bad}"
        )

    below = np.where(arr < lower - tolerance)[0]
    above = np.where(arr > upper + tolerance)[0]
    if below.size or above.size:
        offenders = {int(i): float(arr[i]) for i in np.concatenate([below, above])}
        raise WeightContractViolationError(
            f"Weights{label} outside bounds [{lower}, {upper}]: {offenders}"
        )

    gross = float(np.abs(arr).sum())
    if gross > max_gross + tolerance:
        raise WeightContractViolationError(
            f"Weights{label} gross exposure {gross:.6f} exceeds cap {max_gross}"
        )
    return np.ascontiguousarray(arr, dtype=np.float64)


def normalize_long_only(raw: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """Project an arbitrary score vector onto the long-only simplex.

    Negative scores are floored at zero and the result is renormalised to sum
    to 1.0.  If every score is non-positive we fall back to equal weight.
    """
    arr = np.asarray(raw, dtype=np.float64)
    clipped = np.clip(arr, 0.0, None)
    s = clipped.sum()
    if s <= 0.0 or not np.isfinite(s):
        return np.full(arr.shape, 1.0 / arr.size, dtype=np.float64)
    return clipped / s


def normalize_market_neutral(raw: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """Demean and L1-normalise scores into a dollar-neutral weight vector.

    The result sums to 0.0 and has gross exposure (``sum |w|``) of 1.0, then is
    clipped to the ``[-1, 1]`` contract.
    """
    arr = np.asarray(raw, dtype=np.float64)
    demeaned = arr - arr.mean()
    gross = np.abs(demeaned).sum()
    if gross <= 0.0 or not np.isfinite(gross):
        return np.zeros(arr.shape, dtype=np.float64)
    w = demeaned / gross
    w = np.clip(w, -1.0, 1.0)
    # Re-centre after clipping so the sum is exactly 0.0.
    w = w - w.mean()
    return w


class PortfolioOptimizer(abc.ABC):
    """Abstract base class for every portfolio optimizer in quantcortex.

    Subclasses implement :meth:`_compute_weights`, returning a raw weight
    vector.  The public :meth:`optimize` wraps that call and *guarantees* the
    output satisfies the weight contract - subclasses never have to remember to
    validate.
    """

    def __init__(
        self,
        mode: Union[PortfolioMode, str] = PortfolioMode.LONG_ONLY,
        *,
        tolerance: float = DEFAULT_TOLERANCE,
        weight_bounds: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        self.mode = PortfolioMode.coerce(mode)
        self.tolerance = float(tolerance)
        self.weight_bounds = (float(weight_bounds[0]), float(weight_bounds[1]))

    @property
    def name(self) -> str:
        return type(self).__name__

    @abc.abstractmethod
    def _compute_weights(self, returns, **kwargs) -> np.ndarray:
        """Return a raw (pre-validation) weight vector.

        ``returns`` is conventionally a ``pandas.DataFrame`` of asset returns
        (columns = assets), but optimizers are free to accept additional
        keyword arguments (expected returns, views, covariance, etc.).
        """
        raise NotImplementedError

    def optimize(self, returns, **kwargs) -> np.ndarray:
        """Compute weights and enforce the contract before returning them."""
        raw = self._compute_weights(returns, **kwargs)
        return enforce_weight_contract(
            raw,
            mode=self.mode,
            tolerance=self.tolerance,
            lower=self.weight_bounds[0],
            upper=self.weight_bounds[1],
            name=self.name,
        )

    # Convenience so optimizers can `return self.fit(returns)` style calls.
    __call__ = optimize
