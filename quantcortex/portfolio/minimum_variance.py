"""Global minimum-variance portfolio optimization.

The global minimum-variance (GMV) portfolio is the allocation that minimises
portfolio variance subject to the budget constraint, ignoring expected returns
entirely:

.. math::

    \\min_{w}\\; w^{\\top}\\Sigma w \\quad\\text{s.t.}\\quad \\mathbf{1}^{\\top}w = 1.

Because it depends only on the covariance matrix - the most *estimable* moment
of a return series - the GMV portfolio is a workhorse benchmark and a robust
default allocation. This implementation supports Ledoit-Wolf shrinkage, ridge
regularisation of :math:`\\Sigma`, and a constrained long-only solver. Failed
or undefined problems raise rather than silently substituting a portfolio.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from quantcortex.portfolio.base import (
    PortfolioMode,
    PortfolioOptimizer,
    prepare_return_panel,
    validate_return_panel,
)

__all__ = ["MinimumVariance"]

_RIDGE: float = 1e-8


class MinimumVariance(PortfolioOptimizer):
    """Global minimum-variance optimizer.

    Parameters
    ----------
    mode:
        :class:`~quantcortex.portfolio.base.PortfolioMode`.  ``LONG_ONLY`` solves the
        bounded simplex QP. ``MARKET_NEUTRAL`` is rejected because minimum
        variance with only a zero-net constraint has the trivial zero solution.
    shrinkage:
        When ``True`` (default) the covariance is estimated with Ledoit-Wolf
        shrinkage; otherwise the plain sample covariance is used.
    **kw:
        Forwarded to :class:`~quantcortex.portfolio.base.PortfolioOptimizer`
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
        if not isinstance(shrinkage, (bool, np.bool_)):
            raise TypeError("shrinkage must be a boolean")
        if self.mode is not PortfolioMode.LONG_ONLY:
            raise ValueError(
                "MinimumVariance requires LONG_ONLY; a non-zero market-neutral "
                "minimum needs an additional exposure or return constraint"
            )
        self.shrinkage = bool(shrinkage)

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------
    @staticmethod
    def _clean(returns: pd.DataFrame) -> pd.DataFrame:
        return prepare_return_panel(returns, name="MinimumVariance returns")

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

        x0 = self._project_configured_bounds(
            np.full(n, 1.0 / n, dtype=np.float64)
        )
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
            raise RuntimeError(f"MinimumVariance solver failed: {res.message}")
        return self._project_configured_bounds(res.x)

    # ------------------------------------------------------------------
    # PortfolioOptimizer API
    # ------------------------------------------------------------------
    def _compute_weights(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        """Compute raw minimum-variance weights satisfying the contract.

        Columns with fewer than 2 finite observations (dead assets) carry no
        usable risk information; forward/backward/zero-filling them would
        create zero-variance pseudo-assets that absorb most of the book.  They
        are excluded from the optimization and re-inserted with weight 0.0 at
        their original positions. If all columns are dead, optimization fails.
        """
        df = validate_return_panel(returns, name="MinimumVariance returns")
        n_total = df.shape[1]
        alive = (df.notna().sum(axis=0) >= 2).to_numpy()

        if not alive.any():
            raise ValueError(
                "MinimumVariance requires at least one asset with two observations"
            )
        if (~alive).any() and max(0.0, self.weight_bounds[0]) > self.tolerance:
            raise ValueError(
                "MinimumVariance cannot assign the configured positive minimum "
                "weight to assets without enough observations"
            )

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
            return self._project_configured_bounds(
                np.array([1.0], dtype=np.float64)
            )

        return self._solve_long_only(sigma)
