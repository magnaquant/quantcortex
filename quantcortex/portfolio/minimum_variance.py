"""Global minimum-variance portfolio optimization.

The global minimum-variance (GMV) portfolio is the allocation that minimises
portfolio variance subject to the budget constraint, ignoring expected returns
entirely:

.. math::

    \\min_{w}\\; w^{\\top}\\Sigma w \\quad\\text{s.t.}\\quad \\mathbf{1}^{\\top}w = 1.

Because it depends only on the covariance matrix - the most *estimable* moment
of a return series - the GMV portfolio is a workhorse benchmark and a robust
default allocation.  This implementation supports Ledoit-Wolf shrinkage, ridge
regularisation of :math:`\\Sigma`, a constrained long-only solver and the
closed-form long/short solution, with an equal-weight fallback.  All paths
return weights that satisfy the canonical *weight contract*.
"""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from quantcortex.portfolio.base import (
    PortfolioMode,
    PortfolioOptimizer,
    normalize_market_neutral,
)

__all__ = ["MinimumVariance"]

_RIDGE: float = 1e-8


class MinimumVariance(PortfolioOptimizer):
    """Global minimum-variance optimizer.

    Parameters
    ----------
    mode:
        :class:`~portfolio.base.PortfolioMode`.  ``LONG_ONLY`` solves the
        bounded simplex QP; ``MARKET_NEUTRAL`` uses the closed-form long/short
        GMV direction projected onto a dollar-neutral book.
    shrinkage:
        When ``True`` (default) the covariance is estimated with Ledoit-Wolf
        shrinkage; otherwise the plain sample covariance is used.
    **kw:
        Forwarded to :class:`~portfolio.base.PortfolioOptimizer`
        (``tolerance``, ``weight_bounds``).
    """

    def __init__(
        self,
        mode: Union[PortfolioMode, str] = PortfolioMode.LONG_ONLY,
        *,
        shrinkage: bool = True,
        **kw,
    ) -> None:
        super().__init__(mode, **kw)
        self.shrinkage = bool(shrinkage)

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------
    @staticmethod
    def _clean(returns: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(returns).apply(pd.to_numeric, errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how="all")
        df = df.ffill().bfill().fillna(0.0)
        return df

    def _covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate a symmetric, ridge-regularised covariance matrix."""
        X = self._clean(returns)
        n_assets = X.shape[1]

        if n_assets == 1 or X.shape[0] < 2:
            var = X.var(axis=0, ddof=0).to_numpy(dtype=np.float64)
            sigma = np.diag(np.where(np.isfinite(var) & (var > 0), var, 1.0))
        elif self.shrinkage:
            sigma = LedoitWolf().fit(X.to_numpy(dtype=np.float64)).covariance_
        else:
            sigma = np.cov(X.to_numpy(dtype=np.float64), rowvar=False, ddof=0)

        sigma = np.atleast_2d(np.asarray(sigma, dtype=np.float64))
        sigma = 0.5 * (sigma + sigma.T)
        sigma += _RIDGE * np.eye(n_assets)
        return sigma

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------
    def _solve_long_only(self, sigma: np.ndarray) -> np.ndarray:
        """Bounded simplex QP minimising ``w' Sigma w``."""
        n = sigma.shape[0]
        lower = max(0.0, self.weight_bounds[0])
        upper = self.weight_bounds[1]
        bounds = [(lower, upper)] * n
        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

        def variance(w: np.ndarray) -> float:
            return float(w @ sigma @ w)

        def grad(w: np.ndarray) -> np.ndarray:
            return 2.0 * (sigma @ w)

        x0 = np.full(n, 1.0 / n, dtype=np.float64)
        res = minimize(
            variance,
            x0,
            jac=grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not res.success or not np.all(np.isfinite(res.x)):
            return x0
        w = np.clip(res.x, lower, upper)
        total = w.sum()
        if total <= 0.0 or not np.isfinite(total):
            return x0
        return w / total

    def _solve_market_neutral(self, sigma: np.ndarray) -> np.ndarray:
        """Closed-form GMV direction ``Sigma^-1 1`` projected to dollar-neutral.

        The unconstrained GMV weights ``Sigma^-1 1 / (1' Sigma^-1 1)`` are the
        natural minimum-variance direction; demeaning and L1-normalising them
        yields a valid sum-zero, unit-gross market-neutral book.
        """
        n = sigma.shape[0]
        ones = np.ones(n, dtype=np.float64)
        try:
            inv_ones = np.linalg.solve(sigma, ones)
        except np.linalg.LinAlgError:
            inv_ones = np.linalg.lstsq(sigma, ones, rcond=None)[0]
        if not np.all(np.isfinite(inv_ones)):
            return np.zeros(n, dtype=np.float64)
        return normalize_market_neutral(inv_ones)

    # ------------------------------------------------------------------
    # PortfolioOptimizer API
    # ------------------------------------------------------------------
    def _compute_weights(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        """Compute raw minimum-variance weights satisfying the contract.

        Columns with fewer than 2 finite observations (dead assets) carry no
        usable risk information; forward/backward/zero-filling them would
        create zero-variance pseudo-assets that absorb most of the book.  They
        are excluded from the optimization and re-inserted with weight 0.0 at
        their original positions.  If *all* columns are dead the optimizer
        falls back to the mode's neutral allocation with a warning.
        """
        df = pd.DataFrame(returns).apply(pd.to_numeric, errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)
        n_total = df.shape[1]
        alive = (df.notna().sum(axis=0) >= 2).to_numpy()

        if not alive.any():
            warnings.warn(
                "MinimumVariance: every column has fewer than 2 finite "
                "observations; falling back to the mode's neutral allocation.",
                RuntimeWarning,
                stacklevel=2,
            )
            if self.mode is PortfolioMode.MARKET_NEUTRAL:
                return np.zeros(n_total, dtype=np.float64)
            return np.full(n_total, 1.0 / n_total, dtype=np.float64)

        sub = df.loc[:, df.columns[alive]]
        w_sub = self._optimize_subset(sub)

        # Renormalise the optimized sub-vector to the mode's target sum, then
        # re-insert weight 0.0 at the dead-column positions.
        if self.mode is PortfolioMode.LONG_ONLY:
            s = float(w_sub.sum())
            if s > 0.0 and np.isfinite(s):
                w_sub = w_sub / s
        out = np.zeros(n_total, dtype=np.float64)
        out[alive] = w_sub
        return out

    def _optimize_subset(self, returns: pd.DataFrame) -> np.ndarray:
        """Run the minimum-variance machinery on the live-asset subset."""
        sigma = self._covariance(returns)
        n = sigma.shape[0]

        if n == 1:
            return np.array(
                [1.0 if self.mode is PortfolioMode.LONG_ONLY else 0.0],
                dtype=np.float64,
            )

        try:
            if self.mode is PortfolioMode.MARKET_NEUTRAL:
                return self._solve_market_neutral(sigma)
            return self._solve_long_only(sigma)
        except Exception:
            if self.mode is PortfolioMode.MARKET_NEUTRAL:
                return np.zeros(n, dtype=np.float64)
            return np.full(n, 1.0 / n, dtype=np.float64)
