"""Kelly-criterion position sizing.

The Kelly criterion sizes a bet (or portfolio) to maximise the expected
*logarithm* of wealth - equivalently the long-run geometric growth rate.  Full
Kelly is growth-optimal but notoriously aggressive: it tolerates deep drawdowns
and is acutely sensitive to estimation error in the edge / covariance.  In
practice desks run **fractional Kelly** (typically 0.25-0.5x), trading a little
growth for a large reduction in volatility and drawdown.  This class therefore
multiplies every Kelly sizing by ``self.fraction`` and caps leverage at
``max_leverage``.

This is an exposure-scaling overlay: :meth:`apply` only ever rescales the gross
of an incoming weight vector, so it composes with the platform's exposure
contract.  All inputs are point-in-time estimates; nothing looks ahead.
"""

from __future__ import annotations

import numpy as np

from portfolio.base import enforce_exposure_contract

__all__ = ["KellyCriterion"]


class KellyCriterion:
    """Fractional Kelly position sizing.

    Parameters
    ----------
    fraction:
        Kelly multiplier in ``(0, 1]``.  ``1.0`` is full Kelly; ``0.25``-``0.5``
        is the standard prudent range.
    max_leverage:
        Upper bound on the leverage produced by :meth:`apply` (the lower bound
        is 0 - Kelly never recommends a negative scaling of a directional bet).
    """

    def __init__(self, fraction: float = 0.5, *, max_leverage: float = 1.0) -> None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must lie in (0, 1].")
        if max_leverage <= 0:
            raise ValueError("max_leverage must be positive.")
        self.fraction = float(fraction)
        self.max_leverage = float(max_leverage)
        self.last_scale: float | None = None

    # ------------------------------------------------------------------ #
    # Scalar / discrete Kelly                                            #
    # ------------------------------------------------------------------ #
    def kelly_fraction(self, edge: float, odds: float) -> float:
        """Classic Kelly fraction ``f* = edge / odds`` (times ``self.fraction``).

        ``edge`` is the expected net gain per unit staked and ``odds`` the
        net payoff per unit on a win (``b`` in the canonical ``f = (bp - q)/b``
        form, where ``edge = bp - q``).
        """
        if odds == 0:
            raise ValueError("odds must be non-zero.")
        return self.fraction * (edge / odds)

    def kelly_continuous(self, mean: float, var: float) -> float:
        """Continuous Kelly ``f* = mean / var`` (times ``self.fraction``).

        For a continuously-compounded return with drift ``mean`` and variance
        ``var`` the growth-optimal leverage is ``mean / var``.
        """
        if var <= 0:
            raise ValueError("var must be positive.")
        return self.fraction * (mean / var)

    # ------------------------------------------------------------------ #
    # Multivariate Kelly                                                 #
    # ------------------------------------------------------------------ #
    def kelly_vector(
        self,
        expected_returns: np.ndarray,
        cov: np.ndarray,
    ) -> np.ndarray:
        """Growth-optimal Kelly portfolio ``f* = Sigma^-1 mu`` (times ``fraction``).

        Parameters
        ----------
        expected_returns:
            Vector of expected returns ``mu``, shape ``(n_assets,)``.
        cov:
            Covariance matrix ``Sigma``, shape ``(n_assets, n_assets)``.

        Returns
        -------
        numpy.ndarray
            The (fractional) Kelly allocation.  This is *not* normalised - its
            gross is the leverage Kelly prescribes; downstream layers decide how
            to deploy it.
        """
        mu = np.asarray(expected_returns, dtype=np.float64).ravel()
        sigma = np.asarray(cov, dtype=np.float64)
        if sigma.shape != (mu.size, mu.size):
            raise ValueError(
                f"cov shape {sigma.shape} incompatible with expected_returns "
                f"length {mu.size}."
            )
        # Solve Sigma f = mu (more stable than forming the explicit inverse);
        # lstsq tolerates a singular / near-singular covariance.
        f_star, *_ = np.linalg.lstsq(sigma, mu, rcond=None)
        return self.fraction * f_star

    # ------------------------------------------------------------------ #
    # Overlay                                                            #
    # ------------------------------------------------------------------ #
    def apply(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov: np.ndarray,
    ) -> np.ndarray:
        """Scale ``weights`` by the fractional-Kelly leverage of that direction.

        For a fixed weight *direction* ``w`` the growth-optimal leverage is the
        continuous-Kelly result applied to the portfolio's own drift and
        variance::

            scale = clip( fraction * (w'mu) / (w'Sigmaw), 0, max_leverage )

        i.e. we treat the ``w``-portfolio as a single synthetic asset with mean
        ``w'mu`` and variance ``w'Sigmaw`` and Kelly-size *that*.

        The requested scalar scale is then capped so no element of the scaled
        book exceeds the ``[-1, 1]`` per-asset contract:
        ``effective_scale = min(scale, 1 / max|w_i|)``.  The capped scale is
        applied *unclipped*, preserving the allocation proportions (per-asset
        clipping would silently distort them and make the realized gross differ
        from the prescribed scale).  ``last_scale`` reports the EFFECTIVE
        (possibly capped) scale.

        A non-positive expected return (or non-positive variance) yields a zero
        scale - Kelly declines to take a bet with no edge.
        """
        w = np.asarray(weights, dtype=np.float64).ravel()
        mu = np.asarray(expected_returns, dtype=np.float64).ravel()
        sigma = np.asarray(cov, dtype=np.float64)
        if mu.size != w.size:
            raise ValueError(
                f"expected_returns length {mu.size} does not match weights "
                f"length {w.size}."
            )
        if sigma.shape != (w.size, w.size):
            raise ValueError(
                f"cov shape {sigma.shape} incompatible with weights length "
                f"{w.size}."
            )

        port_mean = float(w @ mu)
        port_var = float(w @ sigma @ w)

        if port_var <= 0.0 or port_mean <= 0.0:
            scale = 0.0
        else:
            raw = self.fraction * (port_mean / port_var)
            scale = float(np.clip(raw, 0.0, self.max_leverage))

        # Cap the scalar so no element leaves the [-1, 1] per-asset contract;
        # the capped scale is applied unclipped to preserve proportions.
        max_abs = float(np.max(np.abs(w))) if w.size else 0.0
        if max_abs > 0.0:
            scale = min(scale, 1.0 / max_abs)

        self.last_scale = scale

        scaled = w * scale
        # Gross of the scaled book is in_gross * scale. Size the cap to the
        # larger of the input gross and the levered gross so both a de-risk
        # and a legitimate lever-up pass.
        in_gross = float(np.abs(w).sum())
        max_gross = max(1.0, in_gross, in_gross * scale) + 1e-9
        return enforce_exposure_contract(
            scaled, max_gross=max_gross, name="KellyCriterion"
        )
