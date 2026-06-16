"""Alphalens-style single-factor evaluation, implemented from scratch.

This module provides the core diagnostics one wants when evaluating a
cross-sectional alpha factor, without depending on the (heavy, partly
unmaintained) ``alphalens`` package:

* **Information Coefficient (IC):** the per-date cross-sectional rank
  correlation between the factor and forward returns. The IC time series, its
  mean, its information ratio (ICIR) and a t-statistic summarise predictive
  power.
* **Quantile returns:** sorting names into quantiles by factor value each
  date and measuring the mean forward return per quantile, plus the
  top-minus-bottom long/short spread.
* **Factor turnover:** how much the top-quantile membership churns from one
  period to the next (a proxy for the trading cost of harvesting the signal).

Inputs are date x symbol panels (``factor`` and ``forward_returns``) that are
assumed to already be aligned in time: ``forward_returns[t]`` is the return
*earned over the period beginning at* ``t`` (i.e. it is the realised payoff to
acting on ``factor[t]``). NaNs are dropped pairwise, per date, so a symbol
missing either a factor value or a forward return on a given date is simply
excluded from that date's cross-section.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------
# Free functions
# ----------------------------------------------------------------------
def _check_unique_dates(*panels: pd.DataFrame) -> None:
    """Raise a clear error when a panel's date index contains duplicates.

    The per-date loops below use ``.loc[date]``, which returns a DataFrame
    (not a row) for a duplicated label and would otherwise fail deep inside
    the loop with an opaque "truth value of a Series is ambiguous" error.
    Duplicate dates are a data problem the caller should fix (e.g. via
    ``df[~df.index.duplicated(keep="last")]``).
    """
    for panel in panels:
        if panel.index.has_duplicates:
            dupes = panel.index[panel.index.duplicated()].unique()[:5]
            raise ValueError(
                "factor/forward_returns index contains duplicate dates "
                f"(e.g. {list(dupes)}); deduplicate before computing "
                "cross-sectional statistics"
            )


def _validate_panel(panel: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must use a DatetimeIndex")
    if panel.index.hasnans:
        raise ValueError(f"{name} index must contain valid timestamps")
    if not panel.index.is_monotonic_increasing:
        raise ValueError(f"{name} dates must be sorted in increasing order")
    if panel.columns.has_duplicates:
        raise ValueError(f"{name} columns must be unique")
    _check_unique_dates(panel)
    numeric = panel.apply(pd.to_numeric, errors="coerce")
    if (numeric.isna() & panel.notna()).any(axis=None):
        raise ValueError(f"{name} contains non-numeric observations")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError(f"{name} contains infinite observations")
    return numeric.astype(float)


def _validate_quantiles(quantiles: int) -> int:
    if isinstance(quantiles, bool) or int(quantiles) != quantiles or quantiles < 2:
        raise ValueError("quantiles must be an integer >= 2")
    return int(quantiles)


def _newey_west_tstat(values: pd.Series, max_lag: int | None = None) -> float:
    """HAC t-statistic for a potentially autocorrelated mean."""
    clean = values.dropna().to_numpy(dtype=float)
    n = clean.size
    if n < 2:
        return float("nan")
    if max_lag is None:
        max_lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    max_lag = max(0, min(int(max_lag), n - 1))
    demeaned = clean - clean.mean()
    long_run_variance = float(np.dot(demeaned, demeaned) / n)
    for lag in range(1, max_lag + 1):
        covariance = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    variance_of_mean = max(long_run_variance, 0.0) / n
    if variance_of_mean <= 0.0:
        return float("nan")
    return float(clean.mean() / np.sqrt(variance_of_mean))


def compute_information_coefficient(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """Per-date cross-sectional Information Coefficient.

    For each date ``t`` we compute the correlation between ``factor.loc[t]``
    and ``forward_returns.loc[t]`` across symbols, after dropping any symbol
    that is NaN in either. With ``method="spearman"`` this is the rank IC.

    Parameters
    ----------
    factor, forward_returns:
        Date x symbol panels. They are aligned on their common dates and
        columns before computation.
    method:
        ``"spearman"`` (rank IC, default) or ``"pearson"``.

    Returns
    -------
    pandas.Series
        IC indexed by date. Dates with fewer than two valid pairs (or zero
        variance) yield NaN.
    """
    method = method.lower()
    if method not in ("spearman", "pearson"):
        raise ValueError("method must be 'spearman' or 'pearson'")
    fac = _validate_panel(factor, "factor")
    fwd = _validate_panel(forward_returns, "forward_returns")
    fac, fwd = fac.align(fwd, join="inner")
    ics = pd.Series(index=fac.index, dtype=float)

    for date in fac.index:
        x = fac.loc[date]
        y = fwd.loc[date]
        mask = x.notna() & y.notna()
        if mask.sum() < 2:
            ics.loc[date] = np.nan
            continue
        xv = x[mask].to_numpy(dtype=float)
        yv = y[mask].to_numpy(dtype=float)
        # Zero variance -> correlation undefined.
        if np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
            ics.loc[date] = np.nan
            continue
        if method == "spearman":
            coef, _ = stats.spearmanr(xv, yv)
        else:
            coef, _ = stats.pearsonr(xv, yv)
        ics.loc[date] = coef

    ics.name = f"ic_{method}"
    return ics


def _quantile_labels(row: pd.Series, quantiles: int) -> pd.Series:
    """Assign integer quantile labels 1..q to a single cross-section.

    Uses rank-based binning (robust to ties and to non-uniform value
    distributions). Rows with fewer valid observations than ``quantiles`` are
    returned all-NaN.
    """
    valid = row.dropna()
    if len(valid) < quantiles:
        return pd.Series(np.nan, index=row.index)
    ranks = valid.rank(method="average")
    try:
        labels = pd.qcut(ranks, quantiles, labels=False, duplicates="drop") + 1
    except ValueError:
        return pd.Series(np.nan, index=row.index)
    if labels.nunique() != quantiles:
        return pd.Series(np.nan, index=row.index)
    out = pd.Series(np.nan, index=row.index, dtype=float)
    out.loc[valid.index] = labels.astype(float)
    return out


def quantile_returns(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    quantiles: int = 5,
) -> pd.DataFrame:
    """Mean forward return by factor quantile, averaged across dates.

    Each date the cross-section is bucketed into ``quantiles`` equal-count
    groups by factor value (1 = lowest factor, ``quantiles`` = highest). The
    mean forward return within each bucket is computed per date, then averaged
    over time.

    Parameters
    ----------
    factor, forward_returns:
        Date x symbol panels.
    quantiles:
        Number of buckets (default 5).

    Returns
    -------
    pandas.DataFrame
        Single-column (``"mean_return"``) frame indexed by quantile label
        ``1..quantiles`` plus a final row ``"long_short"`` holding the
        top-minus-bottom spread.
    """
    quantiles = _validate_quantiles(quantiles)
    fac = _validate_panel(factor, "factor")
    fwd = _validate_panel(forward_returns, "forward_returns")
    fac, fwd = fac.align(fwd, join="inner")
    per_date = []  # list of Series indexed by quantile label, one per date

    for date in fac.index:
        labels = _quantile_labels(fac.loc[date], quantiles)
        rets = fwd.loc[date]
        frame = pd.DataFrame({"q": labels, "r": rets}).dropna()
        if frame.empty:
            continue
        per_date.append(frame.groupby("q")["r"].mean())

    if not per_date:
        idx = list(range(1, quantiles + 1)) + ["long_short"]
        return pd.DataFrame({"mean_return": [np.nan] * len(idx)}, index=idx)

    # Rows = dates, columns = quantile labels.
    by_date = pd.DataFrame(per_date)
    mean_by_q = by_date.mean(axis=0)
    mean_by_q = mean_by_q.reindex(range(1, quantiles + 1))

    out = mean_by_q.to_frame(name="mean_return")
    top, bottom = quantiles, 1
    if top in by_date.columns and bottom in by_date.columns:
        long_short = (by_date[top] - by_date[bottom]).dropna().mean()
    else:
        long_short = np.nan
    out.loc["long_short"] = long_short
    out.index.name = "quantile"
    return out


def factor_turnover(factor: pd.DataFrame, quantiles: int = 5) -> pd.Series:
    """Period-over-period turnover of the top-quantile membership.

    On each date we identify the set of names in the highest factor quantile.
    Turnover is the fraction of the *current* top-quantile set that was not in
    the *previous* date's top-quantile set (Jaccard-style churn), i.e.
    ``|S_t \\ S_{t-1}| / |S_t|``.

    Parameters
    ----------
    factor:
        Date x symbol panel.
    quantiles:
        Number of buckets used to define "top quantile" (default 5).

    Returns
    -------
    pandas.Series
        Turnover in ``[0, 1]`` indexed by date. The first date is NaN (no
        prior set to compare against).
    """
    quantiles = _validate_quantiles(quantiles)
    factor = _validate_panel(factor, "factor")

    top_sets: Dict[pd.Timestamp, set] = {}
    for date in factor.index:
        labels = _quantile_labels(factor.loc[date], quantiles)
        members = set(labels.index[labels == quantiles])
        top_sets[date] = members

    turnover = pd.Series(index=factor.index, dtype=float)
    prev: set = set()
    first = True
    for date in factor.index:
        current = top_sets[date]
        if first or not current:
            turnover.loc[date] = np.nan
        else:
            new_names = current - prev
            turnover.loc[date] = len(new_names) / len(current)
        prev = current if current else prev
        if current:
            first = False
    turnover.name = "turnover"
    return turnover


# ----------------------------------------------------------------------
# Report object
# ----------------------------------------------------------------------
class AlphalensReport:
    """Bundle the IC / quantile / turnover diagnostics for one factor.

    Parameters
    ----------
    factor, forward_returns:
        Aligned date x symbol panels. ``forward_returns[t]`` is the payoff to
        acting on ``factor[t]``.
    quantiles:
        Number of quantile buckets (default 5).
    ic_method:
        Correlation method for the IC (``"spearman"`` default).
    """

    def __init__(
        self,
        factor: pd.DataFrame,
        forward_returns: pd.DataFrame,
        quantiles: int = 5,
        ic_method: str = "spearman",
    ) -> None:
        if not isinstance(factor, pd.DataFrame) or not isinstance(
            forward_returns, pd.DataFrame
        ):
            raise TypeError("factor and forward_returns must be DataFrames")
        quantiles = _validate_quantiles(quantiles)
        self.factor = factor
        self.forward_returns = forward_returns
        self.quantiles = quantiles
        self.ic_method = ic_method
        self._results: Dict[str, object] | None = None

    def compute(self) -> Dict[str, object]:
        """Compute and cache the full set of diagnostics.

        Returns
        -------
        dict
            Keys: ``ic`` (Series), ``ic_mean``, ``icir``, ``ic_tstat``,
            ``turnover`` (Series), ``quantile_returns`` (DataFrame),
            ``long_short_return`` (float), ``hit_rate`` (float).
        """
        ic = compute_information_coefficient(
            self.factor, self.forward_returns, method=self.ic_method
        )
        ic_clean = ic.dropna()

        ic_mean = float(ic_clean.mean()) if len(ic_clean) else np.nan
        ic_std = float(ic_clean.std(ddof=1)) if len(ic_clean) > 1 else np.nan
        icir = ic_mean / ic_std if ic_std and not np.isnan(ic_std) else np.nan

        # Keep the iid statistic for comparison, but use a HAC estimate as the
        # headline because daily IC series are commonly autocorrelated.
        n = len(ic_clean)
        if n > 1 and ic_std and not np.isnan(ic_std):
            ic_tstat_naive = ic_mean / (ic_std / np.sqrt(n))
        else:
            ic_tstat_naive = np.nan
        ic_tstat = _newey_west_tstat(ic_clean)

        # Hit rate: fraction of dates with positive IC (directional accuracy).
        hit_rate = float((ic_clean > 0).mean()) if n else np.nan

        qret = quantile_returns(
            self.factor, self.forward_returns, quantiles=self.quantiles
        )
        long_short = (
            float(qret.loc["long_short", "mean_return"])
            if "long_short" in qret.index
            else np.nan
        )

        turnover = factor_turnover(self.factor, quantiles=self.quantiles)

        self._results = {
            "ic": ic,
            "ic_mean": ic_mean,
            "icir": icir,
            "ic_tstat": float(ic_tstat) if not np.isnan(ic_tstat) else np.nan,
            "ic_tstat_naive": (
                float(ic_tstat_naive) if not np.isnan(ic_tstat_naive) else np.nan
            ),
            "turnover": turnover,
            "quantile_returns": qret,
            "long_short_return": long_short,
            "hit_rate": hit_rate,
        }
        return self._results

    def _ensure(self) -> Dict[str, object]:
        if self._results is None:
            self.compute()
        if self._results is None:
            raise RuntimeError("factor analysis did not produce results")
        return self._results

    def summary(self) -> str:
        """Return a human-readable, fixed-width summary table."""
        r = self._ensure()
        qret: pd.DataFrame = r["quantile_returns"]  # type: ignore[assignment]
        turnover: pd.Series = r["turnover"]  # type: ignore[assignment]
        mean_turnover = float(turnover.dropna().mean()) if len(turnover.dropna()) else np.nan

        def fmt(x: object) -> str:
            try:
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return "      n/a"
                return f"{float(x):9.4f}"
            except (TypeError, ValueError):
                return f"{x!s:>9}"

        lines = []
        lines.append("=" * 46)
        lines.append("           Alphalens-style factor report")
        lines.append("=" * 46)
        lines.append(f"  IC method            : {self.ic_method}")
        lines.append(f"  Quantiles            : {self.quantiles}")
        lines.append("-" * 46)
        lines.append(f"  IC mean              : {fmt(r['ic_mean'])}")
        lines.append(f"  IC IR (mean/std)     : {fmt(r['icir'])}")
        lines.append(f"  IC t-stat (Newey-West): {fmt(r['ic_tstat'])}")
        lines.append(f"  IC t-stat (iid)      : {fmt(r['ic_tstat_naive'])}")
        lines.append(f"  IC hit rate          : {fmt(r['hit_rate'])}")
        lines.append(f"  Long-short return    : {fmt(r['long_short_return'])}")
        lines.append(f"  Mean top-q turnover  : {fmt(mean_turnover)}")
        lines.append("-" * 46)
        lines.append("  Mean return by quantile:")
        for q, row in qret.iterrows():
            label = "long_short" if q == "long_short" else f"Q{q}"
            lines.append(f"    {label:>10} : {fmt(row['mean_return'])}")
        lines.append("=" * 46)
        return "\n".join(lines)

    def plot(self):
        """Plot cumulative IC and mean return by quantile.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt  # lazy import
        from matplotlib.ticker import PercentFormatter

        from quantcortex.backtest.metrics.plotting import (
            INK,
            NEGATIVE_RED,
            REFERENCE_BLUE,
            plot_style_context,
            style_axis,
        )

        r = self._ensure()
        ic: pd.Series = r["ic"]  # type: ignore[assignment]
        qret: pd.DataFrame = r["quantile_returns"]  # type: ignore[assignment]

        with plot_style_context("notebook"):
            fig, (ax_ic, ax_q) = plt.subplots(1, 2, figsize=(11, 4.2))

            cum_ic = ic.fillna(0.0).cumsum()
            ax_ic.plot(cum_ic.index, cum_ic.values, color=REFERENCE_BLUE, lw=1.5)
            ax_ic.axhline(0.0, color=INK, lw=0.8)
            ax_ic.set_title(f"Cumulative IC (mean={r['ic_mean']:.4f})")
            ax_ic.set_xlabel("Date")
            ax_ic.set_ylabel("Cumulative IC")
            style_axis(ax_ic, grid="y")

            bars = qret.drop(index="long_short", errors="ignore")
            values = bars["mean_return"].to_numpy(dtype=float)
            colors = [
                REFERENCE_BLUE if value >= 0.0 else NEGATIVE_RED
                for value in values
            ]
            ax_q.bar([str(q) for q in bars.index], values, color=colors)
            ax_q.axhline(0.0, color=INK, lw=0.8)
            ax_q.set_title("Mean forward return by quantile")
            ax_q.set_xlabel("Quantile")
            ax_q.set_ylabel("Mean return")
            ax_q.yaxis.set_major_formatter(PercentFormatter(1.0))
            style_axis(ax_q, grid="y")

            fig.tight_layout()
        return fig
