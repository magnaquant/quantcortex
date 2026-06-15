"""Value-at-Risk and Conditional Value-at-Risk (Expected Shortfall) estimators.

This module bundles the classic tail-risk metrics the platform reports for
every book.  Three estimation families are provided:

* **Historical / empirical** - read VaR straight off the empirical loss
  distribution; no distributional assumption.
* **Parametric (Gaussian)** - assume returns are normal and use closed-form
  quantiles / expected-shortfall formulae.
* **Cornish-Fisher** - a parametric estimate that corrects the Gaussian
  quantile for sample skewness and excess kurtosis (fatter, asymmetric tails).

Sign convention
----------------
Returns are signed (positive = gain, negative = loss).  **All VaR/CVaR figures
are reported as POSITIVE numbers representing a loss.**  A VaR of ``0.031`` at
``alpha=0.95`` means "we expect to lose no more than 3.1% on 95% of days".  A
*negative* reported number therefore means the relevant quantile is actually a
gain (possible for very strongly positively-drifting series).

Everything here is strictly causal: estimators consume a realised return
history and never peek ahead.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["VaRCVaR"]

ArrayLike = Union[pd.Series, np.ndarray, "list[float]"]
TRADING_DAYS = 252


def _as_1d(returns: ArrayLike) -> np.ndarray:
    """Coerce ``returns`` to a finite 1-D ``float64`` array."""
    arr = np.asarray(returns, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("returns is empty.")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("returns contains no finite observations.")
    return arr


class VaRCVaR:
    """Value-at-Risk / Conditional VaR estimator at confidence level ``alpha``.

    Parameters
    ----------
    alpha:
        Confidence level in ``(0, 1)``, e.g. ``0.95`` for 95% VaR.  The tail
        probability is ``1 - alpha``.

    Notes
    -----
    Losses are reported as **positive** numbers (see module docstring).
    """

    def __init__(self, alpha: float = 0.95) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly in (0, 1).")
        self.alpha = float(alpha)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    @property
    def tail(self) -> float:
        """Tail probability ``1 - alpha`` (e.g. 0.05 for 95% VaR)."""
        return 1.0 - self.alpha

    # ------------------------------------------------------------------ #
    # Historical / empirical estimators                                  #
    # ------------------------------------------------------------------ #
    def historical_var(self, returns: ArrayLike) -> float:
        """Empirical VaR: the ``(1 - alpha)`` quantile loss of ``returns``.

        We take the empirical ``tail`` quantile of the return distribution
        (a typically-negative return) and flip its sign so the loss is
        reported positive.
        """
        r = _as_1d(returns)
        # The loss threshold is the alpha-th quantile of the *loss* = -return,
        # equivalently the (1-alpha) quantile of the return.
        q = float(np.quantile(r, self.tail, method="linear"))
        return -q

    def historical_cvar(self, returns: ArrayLike) -> float:
        """Empirical CVaR / Expected Shortfall.

        Mean loss conditional on the loss exceeding (being worse than) the
        historical VaR threshold.  Reported positive.
        """
        r = _as_1d(returns)
        threshold = np.quantile(r, self.tail, method="linear")
        tail_losses = r[r <= threshold]
        if tail_losses.size == 0:
            # Degenerate (e.g. tiny sample); fall back to the VaR point itself.
            return -float(threshold)
        return -float(tail_losses.mean())

    # ------------------------------------------------------------------ #
    # Parametric (Gaussian) estimators                                   #
    # ------------------------------------------------------------------ #
    def parametric_var(self, returns: ArrayLike) -> float:
        """Gaussian VaR: ``-(mu + z_{1-alpha} * sigma)``.

        ``z_{1-alpha} = norm.ppf(1 - alpha)`` is negative for typical
        ``alpha``, so the quantile return is below the mean and the reported
        loss is positive.
        """
        r = _as_1d(returns)
        mu = float(r.mean())
        sigma = float(r.std(ddof=1)) if r.size > 1 else 0.0
        z = float(stats.norm.ppf(self.tail))
        return -(mu + z * sigma)

    def parametric_cvar(self, returns: ArrayLike) -> float:
        """Gaussian Expected Shortfall (closed form).

        ``ES = sigma * pdf(z) / (1 - alpha) - mu`` where ``z = norm.ppf(tail)``
        and ``pdf`` is the standard-normal density.  Reported positive.
        """
        r = _as_1d(returns)
        mu = float(r.mean())
        sigma = float(r.std(ddof=1)) if r.size > 1 else 0.0
        z = float(stats.norm.ppf(self.tail))
        pdf = float(stats.norm.pdf(z))
        return sigma * pdf / self.tail - mu

    def cornish_fisher_var(self, returns: ArrayLike) -> float:
        """Cornish-Fisher (modified) VaR adjusting the Gaussian quantile.

        The standard-normal quantile ``z`` is expanded with sample skewness
        ``S`` and excess kurtosis ``K`` so the estimate accounts for asymmetric,
        fat-tailed return distributions::

            z_cf = z + (z^2 - 1)/6 * S
                     + (z^3 - 3z)/24 * K
                     - (2 z^3 - 5z)/36 * S^2

        VaR is then ``-(mu + z_cf * sigma)``, reported positive.
        """
        r = _as_1d(returns)
        mu = float(r.mean())
        sigma = float(r.std(ddof=1)) if r.size > 1 else 0.0
        # Fisher=False -> excess kurtosis is reported as (kurt - 3); request
        # bias-corrected sample moments for a causal point estimate.
        s = float(stats.skew(r, bias=False)) if r.size > 2 else 0.0
        k = float(stats.kurtosis(r, fisher=True, bias=False)) if r.size > 3 else 0.0
        z = float(stats.norm.ppf(self.tail))
        z_cf = (
            z
            + (z**2 - 1.0) / 6.0 * s
            + (z**3 - 3.0 * z) / 24.0 * k
            - (2.0 * z**3 - 5.0 * z) / 36.0 * s**2
        )
        return -(mu + z_cf * sigma)

    # ------------------------------------------------------------------ #
    # Portfolio-level VaR                                                #
    # ------------------------------------------------------------------ #
    def portfolio_var(
        self,
        weights: ArrayLike,
        returns_matrix: Union[pd.DataFrame, np.ndarray],
        method: str = "historical",
    ) -> float:
        """VaR of the ``weights`` portfolio over a per-asset return matrix.

        Builds the portfolio return series ``returns_matrix @ weights`` then
        dispatches to the chosen single-series estimator.

        Parameters
        ----------
        weights:
            1-D weight vector, ``(n_assets,)``.
        returns_matrix:
            ``(T, n_assets)`` matrix of per-asset returns (DataFrame or array).
        method:
            ``"historical"``, ``"parametric"`` or ``"cornish_fisher"``.
        """
        w = np.asarray(weights, dtype=np.float64).ravel()
        rm = np.asarray(returns_matrix, dtype=np.float64)
        if rm.ndim != 2:
            raise ValueError("returns_matrix must be 2-D (T, n_assets).")
        if rm.shape[1] != w.size:
            raise ValueError(
                f"weights length {w.size} does not match returns_matrix "
                f"columns {rm.shape[1]}."
            )
        port = rm @ w
        dispatch = {
            "historical": self.historical_var,
            "parametric": self.parametric_var,
            "cornish_fisher": self.cornish_fisher_var,
        }
        if method not in dispatch:
            raise ValueError(
                f"Unknown method {method!r}; expected one of {sorted(dispatch)}."
            )
        return dispatch[method](port)

    # ------------------------------------------------------------------ #
    # Annualisation helper                                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def annualize(
        var_value: float,
        *,
        periods_per_year: int = TRADING_DAYS,
    ) -> float:
        """Scale a per-period VaR/CVaR to an annual horizon (sqrt-of-time).

        Assumes i.i.d. returns so risk scales with ``sqrt(periods_per_year)``.
        """
        return float(var_value) * float(np.sqrt(periods_per_year))
