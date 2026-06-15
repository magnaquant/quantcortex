"""Black-Litterman portfolio optimizer.

This module implements the Black-Litterman (1992) asset-allocation model, which
blends a *market-equilibrium prior* on expected returns with an investor's
subjective *views*, producing a posterior return vector that is then fed into a
mean-variance optimizer.

Why Black-Litterman?
--------------------
Naive Markowitz optimization on raw historical means is notoriously unstable:
small changes in the (noisy) sample mean produce wildly different, often
extreme, portfolios.  Black-Litterman anchors the optimization to the returns
*implied by the market portfolio* (reverse optimization) and then nudges them
only in the directions the investor has explicit, confidence-weighted views on.
The result is well-behaved, intuitive allocations.

The math
--------
Let ``Sigma`` be the asset covariance matrix (estimated here with Ledoit-Wolf
shrinkage), ``delta`` the risk-aversion coefficient and ``w_mkt`` the market-cap
(or equal) weights.

* **Equilibrium / prior returns (reverse optimization)**::

      Pi = delta * Sigma * w_mkt

* **Views** are expressed as ``P * E[R] = Q + eps``, ``eps ~ N(0, Omega)`` where

    - ``P`` is a ``(K x N)`` pick matrix (each row a view over the assets),
    - ``Q`` is the ``(K,)`` vector of expected returns for those views,
    - ``Omega`` is the ``(K x K)`` diagonal uncertainty of the views, expressed in
      the *same variance units* as the prior.  With per-view confidences
      ``c  in  (0, 1)`` we use the Idzorek-style scaling::

          Omega = diag( ((1 - c) / c) * diag(P (tau*Sigma) P') )

      so ``c -> 1`` makes a view (almost) certain and ``c -> 0`` makes it
      irrelevant.  Without confidences we default to the He-Litterman choice
      ``Omega = diag(diag(P (tau*Sigma) P'))``.

* **Posterior expected returns** (the Black-Litterman "master formula")::

      E[R] = [(tau*Sigma)^-1 + P' Omega^-1 P]^-1 * [(tau*Sigma)^-1 Pi + P' Omega^-1 Q]

  With no views this collapses to the prior, ``E[R] = Pi``.

* **Mean-variance optimal weights**::

      w is proportional to (delta*Sigma)^-1 * E[R]

  which is finally projected onto the long-only simplex (clip at zero,
  renormalize) or made dollar-neutral, per the configured mode.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from portfolio.base import (
    PortfolioMode,
    PortfolioOptimizer,
    normalize_long_only,
    normalize_market_neutral,
)

__all__ = ["BlackLitterman"]


class BlackLitterman(PortfolioOptimizer):
    """Black-Litterman expected-return blending with mean-variance allocation.

    Parameters
    ----------
    mode:
        :data:`PortfolioMode.LONG_ONLY` (default) or
        :data:`PortfolioMode.MARKET_NEUTRAL`.
    risk_aversion:
        The risk-aversion coefficient ``delta`` used both for reverse optimization
        of the equilibrium prior and for the final mean-variance step.
    tau:
        Scalar ``tau`` weighting the uncertainty of the equilibrium prior relative
        to the views.  Typically small (0.01-0.05).
    **kw:
        Forwarded to :class:`~portfolio.base.PortfolioOptimizer`.
    """

    def __init__(
        self,
        mode: Union[PortfolioMode, str] = PortfolioMode.LONG_ONLY,
        *,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        **kw,
    ) -> None:
        super().__init__(mode, **kw)
        self.risk_aversion = float(risk_aversion)
        self.tau = float(tau)

    # ------------------------------------------------------------------ #
    # Covariance estimation.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
        """Ledoit-Wolf shrinkage estimate of the covariance matrix.

        Uses :class:`sklearn.covariance.LedoitWolf` when available, falling back
        to a simple sample covariance if estimation fails (e.g. too few rows).
        """
        x = returns.values.astype(np.float64)
        try:
            from sklearn.covariance import LedoitWolf

            cov = LedoitWolf().fit(x).covariance_
        except Exception:  # pragma: no cover - defensive fallback
            cov = np.cov(x, rowvar=False)
        cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
        # Symmetrize and lightly regularize for invertibility.
        cov = 0.5 * (cov + cov.T)
        eps = 1e-10 * np.trace(cov) / max(cov.shape[0], 1)
        cov += np.eye(cov.shape[0]) * eps
        return cov

    # ------------------------------------------------------------------ #
    # View / market-weight helpers.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _market_weights(
        market_weights: Optional[Union[np.ndarray, pd.Series, list]],
        columns: list,
    ) -> np.ndarray:
        """Resolve ``w_mkt``: supplied market-cap weights or equal weight."""
        n = len(columns)
        if market_weights is None:
            return np.full(n, 1.0 / n, dtype=np.float64)
        if isinstance(market_weights, pd.Series):
            w = market_weights.reindex(columns).values.astype(np.float64)
        else:
            w = np.asarray(market_weights, dtype=np.float64).ravel()
        if w.shape[0] != n or not np.all(np.isfinite(w)):
            raise ValueError("market_weights must be finite with one entry per asset")
        total = w.sum()
        if total <= 0.0:
            return np.full(n, 1.0 / n, dtype=np.float64)
        return w / total

    @staticmethod
    def _resolve_views(
        views: Optional[Union[np.ndarray, pd.DataFrame, list]],
        view_confidences,
        q,
        columns: list,
        tau_sigma: np.ndarray,
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Normalize the views payload into ``(P, Q, Omega)`` arrays.

        ``views`` may be:

        * a ``(K x N)`` pick matrix ``P`` (DataFrame columns aligned to assets,
          or a raw array), in which case ``q`` supplies the ``Q`` vector, or
        * ``None`` - meaning no views (the function returns ``None``).

        ``view_confidences`` is the per-view confidence used to build a diagonal
        ``Omega`` in the *same variance units as the prior* (Idzorek)::

            Omega = diag( ((1 - c) / c) * diag(P (tau*Sigma) P') )

        with ``c`` clipped away from 0 and 1; lower confidence -> larger view
        variance.  A dimensionless ``diag((1 - c) / c)`` would be O(1) against
        O(1e-5)-scale daily return variances, leaving views nearly inert.  If
        ``view_confidences`` is omitted, ``Omega`` defaults to the He-Litterman
        choice ``diag(diag(P (tau*Sigma) P'))`` (moderate, prior-scaled uncertainty
        per view).
        """
        if views is None:
            return None

        n = len(columns)
        if isinstance(views, pd.DataFrame):
            p = views.reindex(columns=columns).fillna(0.0).values.astype(np.float64)
        else:
            p = np.atleast_2d(np.asarray(views, dtype=np.float64))
        if p.shape[1] != n:
            raise ValueError(
                f"views pick matrix must have {n} columns (one per asset), "
                f"got shape {p.shape}"
            )
        k = p.shape[0]

        if q is None:
            raise ValueError("views supplied but no Q vector of view returns given")
        q_arr = np.asarray(q, dtype=np.float64).ravel()
        if q_arr.shape[0] != k:
            raise ValueError(
                f"Q must have one entry per view ({k}), got {q_arr.shape[0]}"
            )

        # Prior variance of each view portfolio: diag(P (tau*Sigma) P').  This puts
        # Omega on the same scale as the prior uncertainty, whatever the return
        # frequency of the inputs.
        view_prior_var = np.diag(p @ tau_sigma @ p.T).copy()
        # Guard against degenerate (zero-variance) view portfolios.
        view_prior_var = np.where(view_prior_var > 0.0, view_prior_var, 1e-12)
        if view_confidences is None:
            # He-Litterman default: Omega = diag(diag(P (tau*Sigma) P')).
            omega = np.diag(view_prior_var)
        else:
            c = np.asarray(view_confidences, dtype=np.float64).ravel()
            if c.shape[0] != k:
                raise ValueError(
                    f"view_confidences must have one entry per view ({k})"
                )
            c = np.clip(c, 1e-6, 1.0 - 1e-6)
            # Idzorek-style scaling: Omega = diag(((1 - c) / c) * diag(P (tau*Sigma) P')).
            omega = np.diag(((1.0 - c) / c) * view_prior_var)
        return p, q_arr, omega

    # ------------------------------------------------------------------ #
    # Linear algebra helper.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _safe_inv(mat: np.ndarray) -> np.ndarray:
        """Invert ``mat``, falling back to the pseudo-inverse if singular."""
        try:
            return np.linalg.inv(mat)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            return np.linalg.pinv(mat)

    # ------------------------------------------------------------------ #
    # Optimizer entry point.
    # ------------------------------------------------------------------ #
    def _compute_weights(
        self,
        returns: pd.DataFrame,
        views: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        view_confidences=None,
        q: Optional[Union[np.ndarray, list]] = None,
        market_weights: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute Black-Litterman weights.

        Parameters
        ----------
        returns:
            ``(T x N)`` DataFrame of per-asset returns.
        views:
            Optional ``(K x N)`` pick matrix ``P`` (DataFrame or array).  When
            ``None`` the posterior equals the equilibrium prior.
        view_confidences:
            Optional ``(K,)`` per-view confidences in ``(0, 1)`` used to build
            the diagonal view-uncertainty matrix ``Omega``.
        q:
            Optional ``(K,)`` vector of expected view returns ``Q``.  Required
            when ``views`` is provided.
        market_weights:
            Optional market-cap weights ``w_mkt`` (Series aligned to columns or
            array).  Defaults to equal weight.

        Returns
        -------
        numpy.ndarray
            Contract-valid weights of shape ``(N,)``.
        """
        if not isinstance(returns, pd.DataFrame):
            returns = pd.DataFrame(np.asarray(returns, dtype=float))

        columns = list(returns.columns)
        n = returns.shape[1]
        if n == 0:
            raise ValueError("Black-Litterman requires at least one asset")
        if n == 1:
            single = 1.0 if self.mode is PortfolioMode.LONG_ONLY else 0.0
            return np.array([single], dtype=np.float64)

        # Covariance (Ledoit-Wolf shrinkage) and market weights.
        sigma = self._ledoit_wolf_cov(returns)
        w_mkt = self._market_weights(market_weights, columns)

        # Equilibrium (prior) returns via reverse optimization: Pi = delta Sigma w_mkt.
        pi = self.risk_aversion * sigma @ w_mkt

        # Posterior expected returns.
        resolved = self._resolve_views(
            views, view_confidences, q, columns, self.tau * sigma
        )
        if resolved is None:
            posterior = pi
        else:
            p, q_arr, omega = resolved
            tau_sigma_inv = self._safe_inv(self.tau * sigma)
            omega_inv = self._safe_inv(omega)
            # A = (tau*Sigma)^-1 + P' Omega^-1 P ;  b = (tau*Sigma)^-1 Pi + P' Omega^-1 Q
            a_mat = tau_sigma_inv + p.T @ omega_inv @ p
            b_vec = tau_sigma_inv @ pi + p.T @ omega_inv @ q_arr
            posterior = self._safe_inv(a_mat) @ b_vec

        # Mean-variance optimal weights: w proportional to (delta*Sigma)^-1 E[R].
        raw = self._safe_inv(self.risk_aversion * sigma) @ posterior

        if not np.all(np.isfinite(raw)):
            # Degenerate solve - fall back to the market prior.
            raw = w_mkt.copy()

        # Project onto the configured mode's feasible set.
        if self.mode is PortfolioMode.LONG_ONLY:
            return normalize_long_only(raw)
        return normalize_market_neutral(raw)
