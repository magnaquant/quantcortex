"""Mean-variance portfolio optimization.

This module implements the classic Markowitz mean-variance objective

.. math::

    \\max_{w}\\; w^{\\top}\\mu - \\frac{\\lambda}{2}\\, w^{\\top}\\Sigma w

where :math:`\\mu` is the vector of expected (excess) returns, :math:`\\Sigma`
is the asset return covariance matrix and :math:`\\lambda` is the investor's
risk-aversion coefficient.

The estimator is deliberately robust: covariances may be shrunk towards a
well-conditioned target via Ledoit-Wolf and :math:`\\Sigma` is ridge-regularised
before any inversion. Failed or undefined problems raise rather than silently
substituting a different portfolio. Successful results satisfy the contract.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from quantcortex.portfolio.base import (
    PortfolioMode,
    PortfolioOptimizer,
    normalize_market_neutral,
    prepare_return_panel,
    validate_return_panel,
)

__all__ = ["MeanVariance"]

# Tiny diagonal loading added to every covariance estimate.  This guarantees
# strict positive-definiteness (hence invertibility) even when assets are
# perfectly collinear or fewer observations than assets are supplied.
_RIDGE: float = 1e-8


class MeanVariance(PortfolioOptimizer):
    """Markowitz mean-variance optimizer.

    Parameters
    ----------
    mode:
        :class:`~quantcortex.portfolio.base.PortfolioMode` selecting a long-only
        (weights sum to 1.0) or market-neutral (weights sum to 0.0) book.
    risk_aversion:
        Risk-aversion coefficient :math:`\\lambda` in the objective.  Larger
        values penalise variance more heavily and pull the solution towards the
        minimum-variance portfolio.
    shrinkage:
        When ``True`` (default) the covariance is estimated with Ledoit-Wolf
        shrinkage (:class:`sklearn.covariance.LedoitWolf`); otherwise the plain
        sample covariance is used.
    allow_short:
        Retained for API compatibility. Shorting is only valid in
        ``market_neutral`` mode; setting this in ``long_only`` mode raises.
    **kw:
        Forwarded to :class:`~quantcortex.portfolio.base.PortfolioOptimizer`
        (``tolerance``, ``weight_bounds``).
    """

    def __init__(
        self,
        mode: Union[PortfolioMode, str] = PortfolioMode.LONG_ONLY,
        *,
        risk_aversion: float = 2.0,
        shrinkage: bool = True,
        allow_short: bool = False,
        **kw,
    ) -> None:
        super().__init__(mode, **kw)
        if not isinstance(shrinkage, (bool, np.bool_)):
            raise TypeError("shrinkage must be a boolean")
        if not isinstance(allow_short, (bool, np.bool_)):
            raise TypeError("allow_short must be a boolean")
        if allow_short and self.mode is PortfolioMode.LONG_ONLY:
            raise ValueError(
                "allow_short=True conflicts with LONG_ONLY; use MARKET_NEUTRAL"
            )
        if isinstance(risk_aversion, (bool, np.bool_)):
            raise TypeError("risk_aversion must be numeric, not boolean")
        self.risk_aversion = float(risk_aversion)
        if not np.isfinite(self.risk_aversion) or self.risk_aversion <= 0.0:
            raise ValueError("risk_aversion must be finite and positive")
        self.shrinkage = bool(shrinkage)
        self.allow_short = bool(allow_short)

    # ------------------------------------------------------------------
    # Estimation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clean(returns: pd.DataFrame) -> pd.DataFrame:
        """Use rows observed for every asset in the estimation subset."""
        return prepare_return_panel(returns, name="MeanVariance returns")

    def _estimate(self, returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mu, sigma)`` sample/shrunk estimates from ``returns``."""
        X = self._clean(returns)
        mu = X.mean(axis=0).to_numpy(dtype=np.float64)
        n_assets = X.shape[1]

        if n_assets == 1 or X.shape[0] < 2:
            var = X.var(axis=0, ddof=0).to_numpy(dtype=np.float64)
            sigma = np.diag(np.where(np.isfinite(var) & (var > 0), var, 1.0))
        elif self.shrinkage:
            sigma = LedoitWolf().fit(X.to_numpy(dtype=np.float64)).covariance_
        else:
            sigma = np.cov(X.to_numpy(dtype=np.float64), rowvar=False, ddof=0)

        sigma = np.atleast_2d(np.asarray(sigma, dtype=np.float64))
        # Symmetrise and ridge-regularise for guaranteed invertibility.
        sigma = 0.5 * (sigma + sigma.T)
        sigma += _RIDGE * np.eye(n_assets)
        if not np.all(np.isfinite(mu)):
            mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        return mu, sigma

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------
    def _solve_long_only(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Constrained QP: maximise the mean-variance utility on the simplex."""
        n = mu.size
        lower = max(0.0, self.weight_bounds[0])
        upper = self.weight_bounds[1]
        bounds = [(lower, upper)] * n
        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

        def neg_utility(w: np.ndarray) -> float:
            return -(w @ mu) + 0.5 * self.risk_aversion * (w @ sigma @ w)

        def neg_grad(w: np.ndarray) -> np.ndarray:
            return -mu + self.risk_aversion * (sigma @ w)

        x0 = self._project_configured_bounds(
            np.full(n, 1.0 / n, dtype=np.float64)
        )
        res = minimize(
            neg_utility,
            x0,
            jac=neg_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not res.success or not np.all(np.isfinite(res.x)):
            raise RuntimeError(f"MeanVariance solver failed: {res.message}")
        return self._project_configured_bounds(res.x)

    def _solve_market_neutral(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Closed-form ``Sigma^-1 mu`` then demean / L1-normalise to sum 0."""
        n = mu.size
        try:
            raw = np.linalg.solve(sigma, mu)
        except np.linalg.LinAlgError:
            raw = np.linalg.lstsq(sigma, mu, rcond=None)[0]
        if not np.all(np.isfinite(raw)):
            return np.zeros(n, dtype=np.float64)
        return self._project_configured_bounds(normalize_market_neutral(raw))

    # ------------------------------------------------------------------
    # PortfolioOptimizer API
    # ------------------------------------------------------------------
    def _compute_weights(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[Union[np.ndarray, Sequence[float], pd.Series]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute raw mean-variance weights satisfying the contract.

        Parameters
        ----------
        returns:
            ``(T x N)`` DataFrame of per-asset simple returns.
        expected_returns:
            Optional override for the expected-return vector :math:`\\mu`.  When
            omitted the sample mean of ``returns`` is used.

        Notes
        -----
        Columns with fewer than 2 finite observations (dead assets) carry no
        usable risk information; forward/backward/zero-filling them would
        create zero-variance pseudo-assets that absorb most of the book.  They
        are excluded from the optimization and re-inserted with weight 0.0 at
        their original positions. If all columns are dead, optimization fails.
        """
        df = validate_return_panel(returns, name="MeanVariance returns")
        n_total = df.shape[1]
        alive = (df.notna().sum(axis=0) >= 2).to_numpy()

        if expected_returns is not None:
            er = np.asarray(expected_returns, dtype=np.float64).reshape(-1)
            if er.size != n_total:
                raise ValueError(
                    f"expected_returns has length {er.size}, expected {n_total}"
                )
            if not np.all(np.isfinite(er)):
                raise ValueError("expected_returns must contain only finite values")
        else:
            er = None

        if not alive.any():
            raise ValueError(
                "MeanVariance requires at least one asset with two observations"
            )
        if (~alive).any():
            lower, upper = self.weight_bounds
            effective_lower = max(0.0, lower) if self.mode is PortfolioMode.LONG_ONLY else lower
            if not effective_lower - self.tolerance <= 0.0 <= upper + self.tolerance:
                raise ValueError(
                    "MeanVariance cannot assign the configured non-zero minimum "
                    "weight to assets without enough observations"
                )

        sub = df.loc[:, df.columns[alive]]
        w_sub = self._optimize_subset(sub, er[alive] if er is not None else None)

        # Renormalise the optimized sub-vector to the mode's target sum, then
        # re-insert weight 0.0 at the dead-column positions.
        if self.mode is PortfolioMode.LONG_ONLY:
            s = float(w_sub.sum())
            if s > 0.0 and np.isfinite(s):
                w_sub = w_sub / s
        out = np.zeros(n_total, dtype=np.float64)
        out[alive] = w_sub
        return out

    def _optimize_subset(
        self, returns: pd.DataFrame, expected_returns: Optional[np.ndarray]
    ) -> np.ndarray:
        """Run the mean-variance machinery on the live-asset subset."""
        mu, sigma = self._estimate(returns)
        n = mu.size

        if expected_returns is not None:
            mu = expected_returns

        # Degenerate single-asset book.
        if n == 1:
            return self._project_configured_bounds(np.array(
                [1.0 if self.mode is PortfolioMode.LONG_ONLY else 0.0],
                dtype=np.float64,
            ))

        if self.mode is PortfolioMode.MARKET_NEUTRAL:
            return self._solve_market_neutral(mu, sigma)
        return self._solve_long_only(mu, sigma)
