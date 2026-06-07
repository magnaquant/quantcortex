"""Risk-parity (equal risk contribution) portfolio optimization.

A risk-parity portfolio allocates capital so that every asset contributes the
same amount of risk to the total portfolio variance, rather than equalising the
dollar weights.  Formally, the *risk contribution* of asset :math:`i` is

.. math::

    \\mathrm{RC}_i = w_i\\,(\\Sigma w)_i,

and these sum to the portfolio variance, :math:`\\sum_i \\mathrm{RC}_i =
w^{\\top}\\Sigma w`.  The equal-risk-contribution (ERC) portfolio is the long-only
vector with :math:`\\mathrm{RC}_i` equal across all assets.

The ERC portfolio is found by minimising the dispersion of risk contributions
subject to a full-investment, non-negativity constraint.  It is long-only by
construction, so this optimizer rejects the market-neutral mode.  As always,
the returned weights satisfy the canonical *weight contract*.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from portfolio.base import PortfolioMode, PortfolioOptimizer

__all__ = ["RiskParity"]

_RIDGE: float = 1e-8


class RiskParity(PortfolioOptimizer):
    """Equal risk contribution (ERC) optimizer.

    Parameters
    ----------
    mode:
        Must be :attr:`~portfolio.base.PortfolioMode.LONG_ONLY`; risk parity is
        a long-only construction and the market-neutral mode is rejected.
    shrinkage:
        When ``True`` the covariance is estimated with Ledoit-Wolf shrinkage;
        defaults to ``False`` (plain sample covariance) for risk parity.
    max_iter:
        Maximum iterations granted to the underlying SLSQP solver.
    tol:
        Solver convergence tolerance on the ERC dispersion objective.
    **kw:
        Forwarded to :class:`~portfolio.base.PortfolioOptimizer`
        (``tolerance``, ``weight_bounds``).
    """

    def __init__(
        self,
        mode: Union[PortfolioMode, str] = PortfolioMode.LONG_ONLY,
        *,
        shrinkage: bool = False,
        max_iter: int = 1000,
        tol: float = 1e-10,
        **kw,
    ) -> None:
        super().__init__(mode, **kw)
        if self.mode is not PortfolioMode.LONG_ONLY:
            raise ValueError(
                "RiskParity is a long-only construction; "
                f"mode={self.mode.value!r} is not supported."
            )
        self.shrinkage = bool(shrinkage)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def risk_contributions(
        weights: Union[np.ndarray, pd.Series],
        cov: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """Return the per-asset risk contributions ``w_i (Sigma w)_i``.

        The contributions sum to the portfolio variance ``w' Sigma w``.

        Parameters
        ----------
        weights:
            Weight vector of shape ``(n,)``.
        cov:
            Covariance matrix of shape ``(n, n)``.
        """
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        sigma = np.atleast_2d(np.asarray(cov, dtype=np.float64))
        marginal = sigma @ w
        return w * marginal

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
    # Solver
    # ------------------------------------------------------------------
    def _solve(self, sigma: np.ndarray) -> np.ndarray:
        """Solve for the equal-risk-contribution weights.

        Uses the standard convex reformulation of the ERC problem

        .. math::

            \\min_{x>0}\\; \\tfrac12 x^{\\top}\\Sigma x
                - \\tfrac1n \\sum_i \\log x_i,

        whose unique positive minimiser has equal risk contributions; the final
        portfolio is recovered by renormalising ``x`` to sum to one.  The
        log-barrier keeps every weight strictly positive, sidesteps the badly
        scaled dispersion objective and converges reliably.
        """
        n = sigma.shape[0]
        inv_n = 1.0 / n

        def objective(x: np.ndarray) -> float:
            return 0.5 * float(x @ sigma @ x) - inv_n * float(np.sum(np.log(x)))

        def grad(x: np.ndarray) -> np.ndarray:
            return sigma @ x - inv_n / x

        # Inverse-volatility seed: a strong, cheap warm start for ERC.
        vol = np.sqrt(np.clip(np.diag(sigma), 1e-18, None))
        x0 = 1.0 / vol

        # Strictly-positive lower bound so log(x) and the gradient stay finite.
        bounds = [(1e-12, None)] * n
        res = minimize(
            objective,
            x0,
            jac=grad,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol},
        )
        candidate = res.x if (res.success and np.all(np.isfinite(res.x))) else x0

        total = candidate.sum()
        if total <= 0.0 or not np.isfinite(total):
            return np.full(n, 1.0 / n, dtype=np.float64)
        w = candidate / total

        # Respect the configured upper bound; only relevant for tight books.
        upper = self.weight_bounds[1]
        if np.any(w > upper):
            w = np.clip(w, 0.0, upper)
            s = w.sum()
            if s <= 0.0 or not np.isfinite(s):
                return np.full(n, 1.0 / n, dtype=np.float64)
            w = w / s
        return w

    # ------------------------------------------------------------------
    # PortfolioOptimizer API
    # ------------------------------------------------------------------
    def _compute_weights(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        """Compute raw ERC weights satisfying the long-only contract."""
        sigma = self._covariance(returns)
        n = sigma.shape[0]

        if n == 1:
            return np.array([1.0], dtype=np.float64)

        try:
            return self._solve(sigma)
        except Exception:
            return np.full(n, 1.0 / n, dtype=np.float64)
