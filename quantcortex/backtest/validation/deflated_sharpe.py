"""Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

When a researcher tries *many* strategy configurations and reports the best,
the maximum in-sample Sharpe ratio is upward-biased by selection: even pure
noise produces an impressive-looking winner if you draw enough candidates.
The **Deflated Sharpe Ratio (DSR)** corrects for this by asking the right
question - *given that I selected the best of ``n_trials`` candidates, what is
the probability that the true Sharpe is greater than a benchmark?* - and by
accounting for the non-normality (skewness and kurtosis) of the return stream.

The core statistic is the **Probabilistic Sharpe Ratio (PSR)**

.. math::

    \\mathrm{PSR}(\\mathrm{SR}_0) = \\Phi\\!\\left[
        \\frac{(\\widehat{\\mathrm{SR}} - \\mathrm{SR}_0)\\,\\sqrt{T-1}}
             {\\sqrt{1 - \\gamma_3\\,\\widehat{\\mathrm{SR}}
                       + \\frac{\\gamma_4 - 1}{4}\\,\\widehat{\\mathrm{SR}}^2}}
    \\right]

where

* :math:`\\widehat{\\mathrm{SR}}` - the observed, **per-observation** Sharpe
  ratio (NOT annualized);
* :math:`\\mathrm{SR}_0` - the benchmark Sharpe to beat;
* :math:`T` - the number of return observations;
* :math:`\\gamma_3` - the skewness of the returns;
* :math:`\\gamma_4` - the **non-excess** kurtosis of the returns (so a normal
  distribution has :math:`\\gamma_4 = 3`); and
* :math:`\\Phi` - the standard-normal CDF.

The DSR is the PSR evaluated at the benchmark
:math:`\\mathrm{SR}_0 = \\mathrm{SR}_{\\text{benchmark}} +
\\mathbb{E}[\\max_n \\mathrm{SR}]`, i.e. the *expected maximum* Sharpe under the
null hypothesis that none of the ``n_trials`` candidates has skill.  The
expected maximum of ``N`` independent Sharpe estimates with variance ``V`` is
approximated (Bailey & López de Prado, 2014, eq. 5) by

.. math::

    \\mathbb{E}[\\max_n \\mathrm{SR}] \\approx \\sqrt{V}\\,\\left[
        (1 - \\gamma)\\,\\Phi^{-1}\\!\\left(1 - \\tfrac{1}{N}\\right)
        + \\gamma\\,\\Phi^{-1}\\!\\left(1 - \\tfrac{1}{N e}\\right)
    \\right]

with :math:`\\gamma \\approx 0.5772` the Euler-Mascheroni constant.

References
----------
Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
*Journal of Portfolio Management*, 40(5), 94-107.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "sharpe_ratio",
    "expected_max_sharpe",
    "compute_dsr",
    "probabilistic_sharpe_ratio",
]

# Euler-Mascheroni constant.
EULER_MASCHERONI = 0.5772156649015329


def _clean_returns(returns: pd.Series) -> np.ndarray:
    """Coerce a return series to a 1-D float array, dropping NaNs."""
    arr = np.asarray(pd.Series(returns).dropna(), dtype=float)
    return arr


def sharpe_ratio(
    returns: pd.Series, periods_per_year: Optional[float] = None
) -> float:
    """Per-observation Sharpe ratio (optionally annualized).

    Parameters
    ----------
    returns:
        Periodic returns (already excess of the risk-free rate if you want a
        risk-free-adjusted Sharpe).
    periods_per_year:
        If given, the per-observation Sharpe is multiplied by
        ``sqrt(periods_per_year)`` to annualize it.  If ``None`` (the default),
        the **raw per-observation** Sharpe is returned - this is the quantity
        required by the PSR/DSR formulas.

    Returns
    -------
    float
        The Sharpe ratio, or ``nan`` if it is undefined (fewer than two
        observations or zero dispersion).
    """
    arr = _clean_returns(returns)
    if arr.size < 2:
        return float("nan")
    sd = arr.std(ddof=1)
    if not np.isfinite(sd) or sd == 0.0:
        return float("nan")
    sr = arr.mean() / sd
    if periods_per_year is not None:
        sr *= math.sqrt(periods_per_year)
    return float(sr)


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum of ``n_trials`` independent Sharpe estimates.

    Implements Bailey & López de Prado (2014), eq. 5::

        E[max SR] = sqrt(V) * [ (1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e)) ]

    where ``V`` is the cross-sectional variance of the Sharpe estimates, ``N``
    the number of trials, ``g`` the Euler-Mascheroni constant, and ``Z`` the
    standard-normal inverse CDF (ppf).

    Parameters
    ----------
    n_trials:
        Number of independent strategy configurations tried (``N``).  Must be
        at least 1.
    sr_variance:
        Variance ``V`` of the Sharpe-ratio estimates across trials.

    Returns
    -------
    float
        The expected maximum Sharpe under the null.  Returns ``0.0`` when
        ``n_trials <= 1`` (no selection effect) or when the variance is
        non-positive / non-finite.
    """
    n = int(n_trials)
    if n <= 1:
        return 0.0
    if not np.isfinite(sr_variance) or sr_variance <= 0.0:
        return 0.0
    g = EULER_MASCHERONI
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(math.sqrt(sr_variance) * ((1.0 - g) * z1 + g * z2))


def _sr_estimate_variance(sr: float, skew: float, kurt: float, n_obs: int) -> float:
    """Variance of the Sharpe-ratio estimator (Lo, 2002; Mertens, 2002).

    .. math::

        \\widehat{V}[\\mathrm{SR}] = \\frac{1}{T-1}\\left(
            1 - \\gamma_3 \\mathrm{SR} + \\frac{\\gamma_4 - 1}{4}\\mathrm{SR}^2
        \\right)

    using non-excess kurtosis :math:`\\gamma_4`.  This is the default
    cross-trial variance used to compute the expected maximum Sharpe when the
    caller does not supply one.
    """
    if n_obs < 2:
        return float("nan")
    base = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    return base / (n_obs - 1)


def probabilistic_sharpe_ratio(
    returns: pd.Series, sr_benchmark: float = 0.0
) -> float:
    """Probabilistic Sharpe Ratio - the DSR with a single trial.

    The PSR is the probability that the true (population) Sharpe exceeds
    ``sr_benchmark``, given the observed per-observation Sharpe and the higher
    moments of the return stream.  It is the special case ``n_trials = 1`` of
    the DSR (no selection-bias deflation).

    Parameters
    ----------
    returns:
        Periodic returns.
    sr_benchmark:
        Per-observation benchmark Sharpe to beat (default 0).

    Returns
    -------
    float
        A probability in ``[0, 1]``, or ``nan`` if undefined.
    """
    return _psr(returns, sr_benchmark)


def _psr(returns: pd.Series, sr_zero: float) -> float:
    """Core PSR evaluation against an arbitrary benchmark ``sr_zero``."""
    arr = _clean_returns(returns)
    t = arr.size
    if t < 2:
        return float("nan")

    sr = sharpe_ratio(pd.Series(arr))  # per-observation
    if not np.isfinite(sr):
        return float("nan")

    skew = float(stats.skew(arr, bias=False))
    kurt = float(stats.kurtosis(arr, fisher=False, bias=False))  # non-excess

    # Denominator: standard error of the Sharpe estimator (scaled by sqrt(T-1)).
    denom_sq = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if not np.isfinite(denom_sq) or denom_sq <= 0.0:
        # Degenerate variance: a positive edge over the benchmark is certain,
        # a negative one impossible.
        if sr > sr_zero:
            return 1.0
        if sr < sr_zero:
            return 0.0
        return 0.5

    numerator = (sr - sr_zero) * math.sqrt(t - 1)
    z = numerator / math.sqrt(denom_sq)
    return float(stats.norm.cdf(z))


def compute_dsr(
    returns: pd.Series,
    n_trials: int,
    sr_benchmark: float = 0.0,
    sr_variance: Optional[float] = None,
) -> float:
    """Deflated Sharpe Ratio.

    Computes the observed per-observation Sharpe, its skewness and (non-excess)
    kurtosis, the number of observations ``T``, and the deflated benchmark

    .. math::

        \\mathrm{SR}_0 = \\mathrm{SR}_{\\text{benchmark}}
            + \\mathbb{E}[\\max_n \\mathrm{SR}]

    then evaluates the PSR against that benchmark.  The result is the
    probability that the strategy's *true* Sharpe is positive (exceeds
    ``sr_benchmark``) **after** correcting for having selected the best of
    ``n_trials`` candidates and for non-normal returns.

    Parameters
    ----------
    returns:
        Periodic returns of the selected (best) strategy.
    n_trials:
        Number of strategy configurations tried before selecting this one.
        ``n_trials = 1`` reduces the DSR to the PSR.
    sr_benchmark:
        Per-observation Sharpe the strategy must beat *before* deflation
        (default 0).
    sr_variance:
        Variance of the Sharpe estimates across trials, used in
        :func:`expected_max_sharpe`.  If ``None`` (default) it is estimated
        from this return stream via the Lo/Mertens Sharpe-estimator variance,
        which equals ``1/(T-1)`` for normal returns.

    Returns
    -------
    float
        The DSR, a probability in ``[0, 1]``.  Returns ``nan`` when the Sharpe
        is undefined (``T < 2`` or zero dispersion).
    """
    arr = _clean_returns(returns)
    t = arr.size
    if t < 2:
        return float("nan")

    sr = sharpe_ratio(pd.Series(arr))
    if not np.isfinite(sr):
        return float("nan")

    skew = float(stats.skew(arr, bias=False))
    kurt = float(stats.kurtosis(arr, fisher=False, bias=False))  # non-excess

    if sr_variance is None:
        sr_variance = _sr_estimate_variance(sr, skew, kurt, t)
        if not np.isfinite(sr_variance) or sr_variance <= 0.0:
            sr_variance = 1.0 / (t - 1)

    sr_zero = sr_benchmark + expected_max_sharpe(n_trials, sr_variance)
    return _psr(pd.Series(arr), sr_zero)
