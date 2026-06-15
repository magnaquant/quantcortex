"""Factor signal-decay analysis.

A factor's edge is rarely instantaneous: it predicts returns over some
horizon and the predictive power fades as the horizon lengthens. This module
quantifies that decay by computing the Information Coefficient (IC) of a
factor against forward returns at a range of holding lags, and by measuring
the autocorrelation/half-life of the resulting signal.

Conventions
-----------
* ``factor`` and ``returns`` are date x symbol panels. ``returns[t]`` is the
  *single-period* return realised over the period ending at ``t`` (a standard
  return series).
* The **L-period-ahead forward return** for a decision made at date ``t`` is
  the cumulative return from ``t`` to ``t + L`` (for ``L >= 1``), built by
  compounding forward single-period returns and then shifting back to ``t``.
  This is strictly causal: ``factor[t]`` is correlated only with returns that
  occur on or after ``t``. The decay profile therefore starts at lag 1 (the
  immediate tradeable payoff); lag 0 would be the contemporaneous return
  realised over the period *ending* at ``t``, which a decision taken at the
  close of ``t`` could never capture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantcortex.alpha.validation.alphalens_report import (
    compute_information_coefficient,
)


class FactorDecay:
    """Measure how a factor's IC decays across forward-return horizons.

    Parameters
    ----------
    ic_method:
        Correlation method passed through to the IC computation
        (``"spearman"`` default, i.e. rank IC).
    """

    def __init__(self, ic_method: str = "spearman") -> None:
        self.ic_method = ic_method

    # ------------------------------------------------------------------
    # Forward-return construction
    # ------------------------------------------------------------------
    @staticmethod
    def _forward_return(returns: pd.DataFrame, lag: int) -> pd.DataFrame:
        """Causal forward return aligned to the decision date.

        Returns the cumulative compounded return over the next ``lag``
        periods (``lag >= 1``), indexed at the decision date ``t``.  Lag 0 is
        deliberately unsupported: it would be the return realised over the
        period ending at ``t``, which is not a tradeable payoff for a
        decision taken at the close of ``t``.
        """
        if lag < 1:
            raise ValueError("lag must be >= 1 (lag 0 is not tradeable)")
        # Cumulative gross return over a trailing window of `lag` periods...
        gross = (1.0 + returns).rolling(window=lag, min_periods=lag).apply(
            np.prod, raw=True
        )
        cum = gross - 1.0
        # ...then shift back by `lag` so the window starting at t is at t.
        return cum.shift(-lag)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------
    def compute(
        self,
        factor: pd.DataFrame,
        returns: pd.DataFrame,
        max_lag: int = 10,
    ) -> pd.DataFrame:
        """IC decay profile across forward horizons ``1..max_lag``.

        Lag 1 is the immediate tradeable payoff (the one-period return after
        the decision date); the profile starts there because the lag-0
        contemporaneous return is already realised and would report an
        in-sample, non-tradeable IC.

        Parameters
        ----------
        factor, returns:
            Date x symbol panels (see module docstring for return semantics).
        max_lag:
            Largest forward horizon, in periods (default 10).

        Returns
        -------
        pandas.DataFrame
            Indexed by ``lag`` (1..``max_lag``) with columns ``ic_mean``,
            ``ic_std`` and ``icir`` (= ``ic_mean / ic_std``).
        """
        if max_lag < 1:
            raise ValueError("max_lag must be >= 1")

        rows = {}
        for lag in range(1, max_lag + 1):
            fwd = self._forward_return(returns, lag)
            ic = compute_information_coefficient(
                factor, fwd, method=self.ic_method
            ).dropna()
            if len(ic):
                mean = float(ic.mean())
                std = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
                icir = mean / std if std and not np.isnan(std) else np.nan
            else:
                mean = std = icir = np.nan
            rows[lag] = {"ic_mean": mean, "ic_std": std, "icir": icir}

        out = pd.DataFrame.from_dict(rows, orient="index")
        out.index.name = "lag"
        return out[["ic_mean", "ic_std", "icir"]]

    # ------------------------------------------------------------------
    # Persistence diagnostics
    # ------------------------------------------------------------------
    def ic_autocorrelation(self, ic: pd.Series, max_lag: int = 10) -> pd.Series:
        """Autocorrelation of an IC time series at lags ``1..max_lag``.

        High autocorrelation means the factor's daily predictive signal is
        persistent (slowly decaying); near-zero autocorrelation means each
        period's IC is essentially independent.

        Parameters
        ----------
        ic:
            An IC time series (e.g. from
            :func:`compute_information_coefficient`).
        max_lag:
            Maximum autocorrelation lag (default 10).

        Returns
        -------
        pandas.Series
            Autocorrelation indexed by lag ``1..max_lag``.
        """
        if max_lag < 1:
            raise ValueError("max_lag must be >= 1")
        clean = ic.dropna()
        ac = {lag: clean.autocorr(lag=lag) for lag in range(1, max_lag + 1)}
        out = pd.Series(ac, dtype=float)
        out.index.name = "lag"
        out.name = "ic_autocorr"
        return out

    def half_life(self, decay: pd.DataFrame) -> float:
        """Estimate the lag at which mean IC decays to half its lag-1 value.

        We anchor on the lag-1 mean IC (the typical one-period-ahead signal)
        and find the smallest lag ``L`` at which ``|ic_mean(L)|`` falls to or
        below half of ``|ic_mean(1)|``. Linear interpolation between the
        bracketing lags gives a fractional half-life. If the IC never decays
        that far within the available lags, ``inf`` is returned; if the
        reference IC is non-positive or unavailable, ``nan`` is returned.

        Parameters
        ----------
        decay:
            The frame returned by :meth:`compute` (must contain ``ic_mean``
            and lags including 1).

        Returns
        -------
        float
            Estimated half-life in periods.
        """
        if "ic_mean" not in decay.columns or 1 not in decay.index:
            return float("nan")

        series = decay["ic_mean"].abs()
        ref = series.loc[1]
        if pd.isna(ref) or ref <= 0:
            return float("nan")
        target = ref / 2.0

        lags = [lag for lag in decay.index if lag >= 1]
        prev_lag = 1
        prev_val = series.loc[1]
        for lag in lags:
            if lag == 1:
                continue
            val = series.loc[lag]
            if pd.isna(val):
                continue
            if val <= target:
                # Interpolate between prev_lag (above target) and lag (at/below).
                if prev_val == val:
                    return float(lag)
                frac = (prev_val - target) / (prev_val - val)
                return float(prev_lag + frac * (lag - prev_lag))
            prev_lag, prev_val = lag, val
        return float("inf")

    # ------------------------------------------------------------------
    # Plotting (optional, lazy import)
    # ------------------------------------------------------------------
    def plot(self, decay: pd.DataFrame):
        """Plot the IC decay profile (mean IC with +/- 1 std band).

        Parameters
        ----------
        decay:
            The frame returned by :meth:`compute`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt  # lazy import

        fig, ax = plt.subplots(figsize=(8, 4.5))
        lags = decay.index.to_numpy()
        mean = decay["ic_mean"].to_numpy(dtype=float)
        std = decay["ic_std"].to_numpy(dtype=float)

        ax.plot(lags, mean, marker="o", color="C0", label="mean IC")
        ax.fill_between(lags, mean - std, mean + std, color="C0", alpha=0.2,
                        label="+/- 1 std")
        ax.axhline(0.0, color="black", lw=0.6, alpha=0.6)
        ax.set_title("Factor IC decay")
        ax.set_xlabel("forward-return lag (periods)")
        ax.set_ylabel("Information Coefficient")
        ax.legend(loc="best")
        fig.tight_layout()
        return fig
