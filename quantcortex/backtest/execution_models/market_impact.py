"""Almgren-Chriss market-impact execution model.

This module implements the market-impact cost structure of Almgren & Chriss
(2000), "Optimal Execution of Portfolio Transactions", *Journal of Risk* 3(2),
5-39.  The model decomposes the price an order pays into two components:

* **Permanent impact** -- a lasting shift in the equilibrium price caused by the
  information content of the trade.  It scales (here) linearly with the
  participation ``q = qty / ADV``::

      permanent = gamma * sigma * q

* **Temporary impact** -- a transient liquidity premium paid for demanding
  immediacy, which decays once trading stops.  It is modelled as a (concave)
  power of participation::

      temporary = eta * sigma * sign(q) * |q| ** alpha

  with ``alpha = 0.5`` reproducing the common square-root liquidity cost.

The realised fill price for an order on a bar is the bar close pushed in the
direction of the trade by the sum of permanent and temporary impact (expressed
as a fraction of price)::

    fill = close * (1 + sign(qty) * (|permanent| + |temporary|))

The closed-form *optimal execution trajectory* of Almgren-Chriss (the schedule
that minimises a mean-variance combination of impact cost and timing risk) is
provided as :meth:`AlmgrenChriss.optimal_execution_trajectory`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from quantcortex.backtest.execution_models.ideal_fill import ExecutionModel

__all__ = ["AlmgrenChriss"]

# Default volatility used when none is supplied / derivable.  Expressed as a
# per-bar return standard deviation (~2% is a typical daily equity figure).
_DEFAULT_SIGMA = 0.02
# Default participation exponent: 0.5 gives the canonical square-root law.
_DEFAULT_TEMPORARY_EXPONENT = 0.5


class AlmgrenChriss(ExecutionModel):
    """Permanent + temporary market-impact fill model (Almgren-Chriss, 2000).

    Parameters
    ----------
    eta:
        Temporary-impact coefficient.  Multiplies ``sigma * participation **
        temporary_exponent``.  Default ``0.142`` (an empirically motivated
        order of magnitude).
    gamma:
        Permanent-impact coefficient.  Multiplies ``sigma * participation``.
        Default ``0.314``.
    volatility:
        Optional default per-bar return volatility ``sigma`` used when a bar /
        keyword does not provide one.  ``None`` falls back to a 2% default.
    daily_volume:
        Optional default ADV (shares) used when neither the call nor the bar
        supplies a volume.  ``None`` means the model expects ``adv`` per call.
    temporary_exponent:
        Power applied to participation in the temporary-impact term.  Default
        ``0.5`` (square-root impact).
    """

    def __init__(
        self,
        eta: float = 0.142,
        gamma: float = 0.314,
        volatility: Optional[float] = None,
        daily_volume: Optional[float] = None,
        temporary_exponent: float = _DEFAULT_TEMPORARY_EXPONENT,
    ) -> None:
        if eta < 0 or gamma < 0:
            raise ValueError("eta and gamma must be non-negative.")
        if temporary_exponent <= 0:
            raise ValueError("temporary_exponent must be positive.")
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.volatility = None if volatility is None else float(volatility)
        self.daily_volume = None if daily_volume is None else float(daily_volume)
        self.temporary_exponent = float(temporary_exponent)

    # ------------------------------------------------------------------ #
    # Impact components (fractional price moves)
    # ------------------------------------------------------------------ #
    def _participation(self, qty: float, adv: float) -> float:
        """Signed participation ``qty / adv`` (NaN-safe, returns 0 if adv<=0)."""
        if adv is None or not np.isfinite(adv) or adv <= 0:
            return 0.0
        return float(qty) / float(adv)

    def temporary_impact(self, qty: float, adv: float, sigma: float) -> float:
        """Temporary price impact as a fraction of price.

        ``eta * sigma * sign(q) * |q| ** temporary_exponent`` with
        ``q = qty / adv``.  Returns a *signed* fraction (positive for buys).
        """
        q = self._participation(qty, adv)
        if q == 0.0:
            return 0.0
        return float(self.eta * sigma * np.sign(q) * np.abs(q) ** self.temporary_exponent)

    def permanent_impact(self, qty: float, adv: float, sigma: float) -> float:
        """Permanent price impact as a fraction of price.

        ``gamma * sigma * q`` with ``q = qty / adv`` (signed, linear in
        participation).
        """
        q = self._participation(qty, adv)
        if q == 0.0:
            return 0.0
        return float(self.gamma * sigma * q)

    # ------------------------------------------------------------------ #
    # Fill
    # ------------------------------------------------------------------ #
    def _resolve_sigma(self, bar: "pd.Series", kw: dict) -> float:
        sigma = kw.get("sigma")
        if sigma is None and "volatility" in bar.index and pd.notna(bar["volatility"]):
            sigma = bar["volatility"]
        if sigma is None:
            sigma = self.volatility
        if sigma is None:
            sigma = _DEFAULT_SIGMA
        return float(sigma)

    def _resolve_adv(self, bar: "pd.Series", kw: dict) -> Optional[float]:
        adv = kw.get("adv")
        if adv is None:
            adv = kw.get("volume")
        if adv is None and "volume" in bar.index and pd.notna(bar["volume"]):
            adv = bar["volume"]
        if adv is None:
            adv = self.daily_volume
        return None if adv is None else float(adv)

    def fill(
        self,
        symbol: str,
        target_qty: float,
        bar: "pd.Series",
        **kw,
    ) -> float:
        """Return the close pushed by permanent + temporary impact.

        Uses ``sigma`` and ``adv`` from (in priority order) the keyword
        arguments, the bar (``volatility`` / ``volume`` columns), and finally
        the model's configured defaults.

        The model charges the FULL permanent + temporary impact on this
        one-shot fill -- a deliberately conservative convention (the canonical
        Almgren-Chriss accounting embeds only ~half the permanent impact in
        the average execution price of a schedule).
        """
        close = float(bar["close"])
        if target_qty == 0:
            return close

        sigma = self._resolve_sigma(bar, kw)
        adv = self._resolve_adv(bar, kw)

        perm = self.permanent_impact(target_qty, adv, sigma)
        temp = self.temporary_impact(target_qty, adv, sigma)
        direction = 1.0 if target_qty > 0 else -1.0
        total_impact = direction * (abs(perm) + abs(temp))
        return float(close * (1.0 + total_impact))

    # ------------------------------------------------------------------ #
    # Optimal execution trajectory (closed form)
    # ------------------------------------------------------------------ #
    def optimal_execution_trajectory(
        self,
        X: float,
        T: float,
        n: int = 50,
        *,
        sigma: float = _DEFAULT_SIGMA,
        risk_aversion: float = 1e-6,
        eta: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> "pd.DataFrame":
        """Closed-form Almgren-Chriss optimal liquidation schedule.

        Liquidating ``X`` shares over horizon ``T`` in ``n`` equal intervals
        (``tau = T / n``), the variance-minimising holdings trajectory under a
        linear temporary-impact specification is

        .. math::

            x_j = X \\, \\frac{\\sinh(\\kappa (T - t_j))}{\\sinh(\\kappa T)},

        where the urgency parameter :math:`\\kappa` solves

        .. math::

            2 \\, (\\cosh(\\kappa \\tau) - 1) = \\tilde\\eta \\, \\lambda \\,
            \\sigma^2 \\, \\tau^2, \\qquad
            \\tilde\\eta = \\eta - \\tfrac{1}{2} \\gamma \\tau .

        Parameters
        ----------
        X:
            Total shares to liquidate (positive) or acquire (negative sign just
            flips the schedule).
        T:
            Execution horizon (in the same time units as ``sigma``).
        n:
            Number of equally spaced trading intervals.
        sigma:
            Per-unit-time volatility.
        risk_aversion:
            Mean-variance risk-aversion ``lambda``.  Larger => faster (front
            loaded) liquidation; ``->0`` => the linear/uniform (TWAP) schedule.
        eta, gamma:
            Override the instance temporary / permanent impact coefficients.

        Returns
        -------
        pandas.DataFrame
            Columns ``time`` (interval endpoints), ``holdings`` ``x_j`` and
            ``trade`` -- the shares executed during the interval *starting* at
            ``t_j`` (row ``j`` holds ``x_j - x_{j+1}``); the final row's
            ``trade`` is ``0`` since no interval starts at ``T``.
            ``holdings`` runs from ``X`` down to ``0``.
        """
        if T <= 0 or n < 1:
            raise ValueError("T must be > 0 and n >= 1.")
        eta_v = self.eta if eta is None else float(eta)
        gamma_v = self.gamma if gamma is None else float(gamma)

        tau = T / n
        eta_tilde = eta_v - 0.5 * gamma_v * tau
        times = np.arange(0, n + 1) * tau

        # Solve for kappa.  When risk_aversion->0 (or eta_tilde<=0) fall back to
        # the uniform TWAP schedule.
        if risk_aversion <= 0 or eta_tilde <= 0:
            holdings = X * (1.0 - times / T)
        else:
            kappa_tilde_sq = risk_aversion * sigma ** 2 / eta_tilde
            # cosh(kappa*tau) = 1 + 0.5 * kappa_tilde^2 * tau^2
            cosh_arg = 1.0 + 0.5 * kappa_tilde_sq * tau ** 2
            kappa = float(np.arccosh(cosh_arg)) / tau
            sinh_kT = np.sinh(kappa * T)
            if sinh_kT == 0:
                holdings = X * (1.0 - times / T)
            else:
                holdings = X * np.sinh(kappa * (T - times)) / sinh_kT

        holdings[0] = X
        holdings[-1] = 0.0
        trades = -np.diff(holdings, prepend=X)  # x_{j-1} - x_j per interval
        return pd.DataFrame(
            {"time": times, "holdings": holdings, "trade": np.append(trades[1:], 0.0)}
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"AlmgrenChriss(eta={self.eta}, gamma={self.gamma}, "
            f"temporary_exponent={self.temporary_exponent})"
        )
