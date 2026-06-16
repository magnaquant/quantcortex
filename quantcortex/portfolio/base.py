"""Portfolio optimization base layer and the canonical *weight contract*.

This module is the keystone of quantcortex.  Every portfolio, timing and risk
component ultimately emits a weight vector, and **all** of them route that
vector through either the strict allocation contract
(:func:`enforce_weight_contract`) or the relaxed post-overlay exposure contract
(:func:`enforce_exposure_contract`) before it leaves the component.
Centralising both contracts here gives downstream code one authoritative
definition of valid portfolio exposure.

The contract
------------
A valid weight vector ``w`` satisfies:

* ``w`` is a :class:`numpy.ndarray` with ``dtype == float64`` and shape
  ``(n_assets,)`` (1-D).
* every element is finite and lies in ``[-1.0, 1.0]``; long-only weights are
  additionally non-negative.
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
import pandas as pd

__all__ = [
    "WeightContractViolationError",
    "PortfolioMode",
    "enforce_weight_contract",
    "enforce_exposure_contract",
    "normalize_long_only",
    "normalize_market_neutral",
    "project_bounded_sum",
    "validate_return_panel",
    "prepare_return_panel",
    "PortfolioOptimizer",
]

# Default absolute tolerance for the sum / bound checks.  Floating point
# arithmetic on hundreds of assets accumulates error well above machine
# epsilon, so 1e-6 is the practical "equal to 1.0" threshold.
DEFAULT_TOLERANCE: float = 1e-6


def _contains_boolean(values) -> bool:
    """Return whether an array-like input contains boolean scalars."""
    try:
        flat = np.asarray(values, dtype=object).ravel()
    except Exception:
        return False
    return any(isinstance(value, (bool, np.bool_)) for value in flat)


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
        Per-asset weight bounds. Defaults implement the ``[-1, 1]`` box;
        ``long_only`` additionally enforces a zero lower bound.
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

    if _contains_boolean((tolerance, lower, upper)):
        raise WeightContractViolationError(
            f"Tolerance and bounds{label} must be numeric, not boolean"
        )
    try:
        tolerance = float(tolerance)
        lower = float(lower)
        upper = float(upper)
    except (TypeError, ValueError) as exc:
        raise WeightContractViolationError(
            f"Tolerance and bounds{label} must be numeric"
        ) from exc
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise WeightContractViolationError(
            f"Tolerance{label} must be finite and non-negative"
        )
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise WeightContractViolationError(
            f"Weight bounds{label} must be finite and ordered, got [{lower}, {upper}]"
        )

    if _contains_boolean(weights):
        raise WeightContractViolationError(
            f"Weights{label} must be numeric, not boolean"
        )
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

    if mode is PortfolioMode.LONG_ONLY:
        short = np.where(arr < -tolerance)[0]
        if short.size:
            offenders = {int(i): float(arr[i]) for i in short}
            raise WeightContractViolationError(
                f"Weights{label} contain short positions in long_only mode: {offenders}"
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
    if _contains_boolean((tolerance, max_gross, lower, upper)):
        raise WeightContractViolationError(
            f"Exposure limits{label} must be numeric, not boolean"
        )
    try:
        tolerance = float(tolerance)
        max_gross = float(max_gross)
        lower = float(lower)
        upper = float(upper)
    except (TypeError, ValueError) as exc:
        raise WeightContractViolationError(
            f"Exposure limits{label} must be numeric"
        ) from exc
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise WeightContractViolationError(
            f"Tolerance{label} must be finite and non-negative"
        )
    if not np.isfinite(max_gross) or max_gross < 0.0:
        raise WeightContractViolationError(
            f"max_gross{label} must be finite and non-negative"
        )
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise WeightContractViolationError(
            f"Weight bounds{label} must be finite and ordered, got [{lower}, {upper}]"
        )
    if _contains_boolean(weights):
        raise WeightContractViolationError(
            f"Weights{label} must be numeric, not boolean"
        )
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
    to 1.0. A vector with no positive score is undefined and raises; callers
    must explicitly choose a flat target or another allocation rule.
    """
    if _contains_boolean(raw):
        raise WeightContractViolationError("raw scores must be numeric, not boolean")
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise WeightContractViolationError("raw scores must be a non-empty 1-D vector")
    if not np.all(np.isfinite(arr)):
        raise WeightContractViolationError("raw scores must all be finite")
    clipped = np.clip(arr, 0.0, None)
    s = clipped.sum()
    if s <= 0.0 or not np.isfinite(s):
        raise WeightContractViolationError(
            "long-only score normalization requires at least one positive score"
        )
    return clipped / s


def normalize_market_neutral(raw: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """Demean and L1-normalise scores into a dollar-neutral weight vector.

    The result sums to 0.0 and has gross exposure (``sum |w|``) of 1.0, then is
    clipped to the ``[-1, 1]`` contract.
    """
    if _contains_boolean(raw):
        raise WeightContractViolationError("raw scores must be numeric, not boolean")
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise WeightContractViolationError("raw scores must be a non-empty 1-D vector")
    if not np.all(np.isfinite(arr)):
        raise WeightContractViolationError("raw scores must all be finite")
    demeaned = arr - arr.mean()
    gross = np.abs(demeaned).sum()
    if gross <= 0.0 or not np.isfinite(gross):
        raise WeightContractViolationError(
            "market-neutral score normalization requires cross-sectional dispersion"
        )
    w = demeaned / gross
    w = np.clip(w, -1.0, 1.0)
    # Re-centre after clipping so the sum is exactly 0.0.
    w = w - w.mean()
    return w


def project_bounded_sum(
    raw: Union[np.ndarray, Sequence[float]],
    *,
    target_sum: float,
    lower: float,
    upper: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """Euclidean projection onto a box-constrained constant-sum hyperplane.

    The solution has the form ``clip(raw - theta, lower, upper)``. A monotone
    bisection finds ``theta``; infeasible bounds raise instead of returning a
    portfolio that violates the requested contract.
    """
    if _contains_boolean((target_sum, lower, upper, tolerance)) or _contains_boolean(raw):
        raise WeightContractViolationError(
            "projection inputs must be numeric, not boolean"
        )
    try:
        arr = np.asarray(raw, dtype=np.float64)
        target_sum = float(target_sum)
        lower = float(lower)
        upper = float(upper)
        tolerance = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise WeightContractViolationError(
            "projection inputs must be numeric"
        ) from exc
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise WeightContractViolationError(
            "projection requires a non-empty finite 1-D vector"
        )
    if (
        not np.isfinite(target_sum)
        or not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower > upper
        or not np.isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise WeightContractViolationError(
            "projection target, bounds, and tolerance must be finite and valid"
        )

    n = arr.size
    minimum = n * lower
    maximum = n * upper
    if target_sum < minimum - tolerance or target_sum > maximum + tolerance:
        raise WeightContractViolationError(
            f"weight bounds [{lower}, {upper}] are infeasible for {n} assets "
            f"with target sum {target_sum}"
        )
    if abs(target_sum - minimum) <= tolerance:
        return np.full(n, lower, dtype=np.float64)
    if abs(target_sum - maximum) <= tolerance:
        return np.full(n, upper, dtype=np.float64)

    theta_low = float(np.min(arr - upper))
    theta_high = float(np.max(arr - lower))
    for _ in range(128):
        theta = 0.5 * (theta_low + theta_high)
        candidate = np.clip(arr - theta, lower, upper)
        if float(candidate.sum()) > target_sum:
            theta_low = theta
        else:
            theta_high = theta

    projected = np.clip(arr - 0.5 * (theta_low + theta_high), lower, upper)
    residual = target_sum - float(projected.sum())
    if residual > 0.0:
        for position in np.where(projected < upper)[0]:
            increment = min(residual, upper - projected[position])
            projected[position] += increment
            residual -= increment
            if residual <= tolerance:
                break
    elif residual < 0.0:
        for position in np.where(projected > lower)[0]:
            decrement = min(-residual, projected[position] - lower)
            projected[position] -= decrement
            residual += decrement
            if residual >= -tolerance:
                break
    if abs(target_sum - float(projected.sum())) > max(tolerance, 1e-12):
        raise WeightContractViolationError("bounded weight projection did not converge")
    return np.ascontiguousarray(projected, dtype=np.float64)


def validate_return_panel(returns, *, name: str = "returns") -> pd.DataFrame:
    """Coerce a return panel while preserving missing observations.

    Nonnumeric and infinite observations are rejected rather than converted to
    missing values, because silently dropping the affected asset changes the
    investable universe.
    """
    original = pd.DataFrame(returns).copy()
    if original.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one asset")
    if original.columns.has_duplicates:
        raise ValueError(f"{name} columns must be unique")
    if original.index.has_duplicates:
        raise ValueError(f"{name} index must be unique")
    if _contains_boolean(original.to_numpy(dtype=object)):
        raise ValueError(f"{name} contains boolean observations")
    numeric = original.apply(pd.to_numeric, errors="coerce")
    if (numeric.isna() & original.notna()).any().any():
        raise ValueError(f"{name} contains non-numeric observations")
    if np.isinf(numeric.to_numpy(dtype=np.float64)).any():
        raise ValueError(f"{name} contains infinite observations")
    return numeric.astype(np.float64)


def prepare_return_panel(
    returns, *, min_observations: int = 2, name: str = "returns"
) -> pd.DataFrame:
    """Return finite complete cases without forward/backward filling returns."""
    if (
        isinstance(min_observations, (bool, np.bool_))
        or not isinstance(min_observations, (int, np.integer))
        or min_observations < 1
    ):
        raise ValueError("min_observations must be a positive integer")
    numeric = validate_return_panel(returns, name=name)
    complete = numeric.dropna(axis=0, how="any")
    if len(complete) < min_observations:
        raise ValueError(
            f"{name} needs at least {min_observations} complete observations; "
            f"got {len(complete)}"
        )
    return complete


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
        if isinstance(tolerance, (bool, np.bool_)):
            raise TypeError("tolerance must be numeric, not boolean")
        self.tolerance = float(tolerance)
        if not np.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative")
        try:
            bounds_len = len(weight_bounds)
        except TypeError as exc:
            raise ValueError("weight_bounds must contain exactly (lower, upper)") from exc
        if bounds_len != 2:
            raise ValueError("weight_bounds must contain exactly (lower, upper)")
        if _contains_boolean(weight_bounds):
            raise TypeError("weight_bounds must be numeric, not boolean")
        lower, upper = float(weight_bounds[0]), float(weight_bounds[1])
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ValueError("weight_bounds must be finite and ordered")
        if self.mode is PortfolioMode.LONG_ONLY and upper <= 0.0:
            raise ValueError("long_only weight_bounds require a positive upper bound")
        self.weight_bounds = (lower, upper)

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

    def _project_configured_bounds(self, weights) -> np.ndarray:
        """Project deliberate optimizer output onto this instance's feasible set."""
        lower, upper = self.weight_bounds
        if self.mode is PortfolioMode.LONG_ONLY:
            lower = max(0.0, lower)
        return project_bounded_sum(
            weights,
            target_sum=self.mode.target_sum,
            lower=lower,
            upper=upper,
            tolerance=self.tolerance,
        )

    # Convenience so optimizers can `return self.fit(returns)` style calls.
    __call__ = optimize
