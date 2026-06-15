"""Barra-style factor-exposure limiting overlay.

In a Barra (multi-factor) risk model each asset carries a vector of factor
*loadings* (its sensitivity to style/industry/macro factors such as Value,
Momentum, Size, Beta...).  A portfolio's exposure to a factor is the
loading-weighted sum of its positions, ``loadings' * w``.  Concentrated,
unintended factor bets are a classic source of blow-ups, so risk managers cap
each factor exposure to a tolerance band ``[-max_exposure, +max_exposure]``.

This overlay measures portfolio factor exposures and, where any exceeds the
cap, *projects* the weights to pull the offending exposures back to their
limit while staying as close as possible (in the least-squares sense) to the
original weights.  Because it only ever removes/reduces a factor tilt it never
adds gross exposure, so it composes safely with the platform's exposure
contract.

Strictly causal: loadings and weights are point-in-time inputs; nothing here
looks ahead.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from portfolio.base import enforce_exposure_contract

__all__ = ["FactorExposureLimiter"]


class FactorExposureLimiter:
    """Cap portfolio factor exposures at ``+/-max_exposure`` (Barra-style).

    Parameters
    ----------
    max_exposure:
        Maximum absolute portfolio exposure permitted to any single factor.
    factors:
        Optional explicit list of factor names to police.  If ``None`` (the
        default) every column of the supplied loadings frame is policed.
    """

    def __init__(
        self,
        max_exposure: float = 0.2,
        *,
        factors: Optional[Sequence[str]] = None,
    ) -> None:
        if max_exposure < 0:
            raise ValueError("max_exposure must be non-negative.")
        self.max_exposure = float(max_exposure)
        self.factors = list(factors) if factors is not None else None
        self.last_exposures: Optional[pd.Series] = None

    # ------------------------------------------------------------------ #
    # Exposure measurement                                               #
    # ------------------------------------------------------------------ #
    def compute_exposures(
        self,
        weights: np.ndarray,
        loadings: pd.DataFrame,
    ) -> pd.Series:
        """Portfolio factor exposures ``loadings' * w``.

        Parameters
        ----------
        weights:
            1-D weight vector aligned to the rows (assets) of ``loadings``.
        loadings:
            ``(n_assets, n_factors)`` DataFrame indexed by asset, columns are
            factors.

        Returns
        -------
        pandas.Series
            Exposure per factor, indexed by factor name.
        """
        w = np.asarray(weights, dtype=np.float64).ravel()
        if loadings.shape[0] != w.size:
            raise ValueError(
                f"weights length {w.size} does not match loadings rows "
                f"{loadings.shape[0]}."
            )
        cols = self._selected_columns(loadings)
        L = loadings[cols].to_numpy(dtype=np.float64)
        exposures = L.T @ w
        return pd.Series(exposures, index=cols, name="factor_exposure")

    def _selected_columns(self, loadings: pd.DataFrame) -> list:
        """Resolve which factor columns to police for this call."""
        if self.factors is None:
            return list(loadings.columns)
        missing = [f for f in self.factors if f not in loadings.columns]
        if missing:
            raise ValueError(f"loadings missing requested factors: {missing}")
        return list(self.factors)

    # ------------------------------------------------------------------ #
    # Capping                                                            #
    # ------------------------------------------------------------------ #
    def apply(
        self,
        weights: np.ndarray,
        loadings: pd.DataFrame,
    ) -> np.ndarray:
        """Project ``weights`` so no policed factor exposure exceeds the cap.

        Method
        ------
        Let ``L`` be the ``(n_assets, k)`` loadings of the *offending* factors
        (those whose current exposure violates ``+/-max_exposure``) and ``e`` the
        current exposures of those factors.  We seek the minimal-norm adjustment
        ``delta`` to the weights that drives each offending exposure exactly to
        its signed cap ``t`` (``+max_exposure`` if the exposure was too high,
        ``-max_exposure`` if too low):

            minimise   ||delta||^2
            subject to L' (w + delta) = t          (i.e. L' delta = t - e)

        The closed-form least-squares (minimum-norm) solution is the projection
        onto the offending factor subspace::

            delta = L (L'L)^-1 (t - e)

        which subtracts exactly the component of the weights spanning the
        offending factor loadings needed to hit the cap, leaving the portfolio
        as close to the original as possible.

        Because the projection only constrains the *currently* offending
        factors, it can push OTHER policed factors beyond the cap.  We
        therefore **iterate**: after each projection the exposures are
        recomputed and all currently-offending factors are re-projected
        together, up to ``max_iter`` passes.  If the cap is still violated
        after the loop (e.g. mutually antagonistic loadings), we fall back to
        uniformly scaling the whole weight vector down until the worst
        ``|exposure|`` equals the cap - exposures are linear in ``w`` so this
        shrink is guaranteed feasible and direction-preserving.  Note the
        trade-off: the uniform shrink reduces the weight sum (the shortfall is
        implicitly held as cash) rather than preserving full investment.

        The result is then clipped to the ``[-1, 1]`` per-asset contract and
        validated via :func:`enforce_exposure_contract`.  The validator's
        gross-cap is sized to the input so a benign no-op still passes.
        """
        w = np.asarray(weights, dtype=np.float64).ravel()
        cols = self._selected_columns(loadings)
        L_all = loadings[cols].to_numpy(dtype=np.float64)

        exposures = L_all.T @ w
        self.last_exposures = pd.Series(exposures, index=cols, name="factor_exposure")

        cap = self.max_exposure
        tol = 1e-12
        in_gross = float(np.abs(w).sum())
        max_gross = max(1.0, in_gross) + 1e-9

        if not np.any(np.abs(exposures) > cap + tol):
            # Nothing to do; validate and return a clean copy.
            return enforce_exposure_contract(
                w, max_gross=max_gross, name="FactorExposureLimiter"
            )

        adjusted = w.copy()
        max_iter = 50
        for _ in range(max_iter):
            exposures = L_all.T @ adjusted
            offending = np.abs(exposures) > cap + tol
            if not np.any(offending):
                break

            L = L_all[:, offending]          # (n_assets, k)
            e = exposures[offending]         # (k,)
            # Signed target: pull each offending exposure to the nearer cap edge.
            t = np.sign(e) * cap             # (k,)

            gram = L.T @ L                   # (k, k)
            rhs = t - e                      # (k,)
            # Minimum-norm solution; lstsq handles rank-deficient / collinear
            # loadings gracefully.
            coef, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
            adjusted = adjusted + L @ coef

        # Guaranteed-feasible fallback: exposures are linear in the weights, so
        # uniformly shrinking the whole vector until the worst |exposure| hits
        # the cap always succeeds and preserves the allocation direction.
        exposures = L_all.T @ adjusted
        worst = float(np.max(np.abs(exposures))) if exposures.size else 0.0
        if worst > cap + tol:
            adjusted = adjusted * (cap / worst)

        # Neutralising a tilt can, in principle, nudge gross above the input.
        # An exposure overlay must never *add* gross, so if that happens we
        # rescale the whole vector back down to the input gross (this only
        # shrinks the exposures further, never re-introduces a violation).
        adj_gross = float(np.abs(adjusted).sum())
        if adj_gross > in_gross > 0.0:
            adjusted = adjusted * (in_gross / adj_gross)

        adjusted = np.clip(adjusted, -1.0, 1.0)
        return enforce_exposure_contract(
            adjusted, max_gross=max_gross, name="FactorExposureLimiter"
        )
