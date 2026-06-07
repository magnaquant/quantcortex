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

    fac, fwd = factor.align(forward_returns, join="inner")
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
    ranks = valid.rank(method="first")
    try:
        labels = pd.qcut(ranks, quantiles, labels=False, duplicates="drop") + 1
    except ValueError:
        return pd.Series(np.nan, index=row.index)
    out = pd.Series(np.nan, index=row.index)
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
    if quantiles < 2:
        raise ValueError("quantiles must be >= 2")

    fac, fwd = factor.align(forward_returns, join="inner")
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
    if top in mean_by_q.index and bottom in mean_by_q.index:
        long_short = mean_by_q.loc[top] - mean_by_q.loc[bottom]
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
    if quantiles < 2:
        raise ValueError("quantiles must be >= 2")

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
        if quantiles < 2:
            raise ValueError("quantiles must be >= 2")
        self.factor = factor
        self.forward_returns = forward_returns
        self.quantiles = int(quantiles)
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

        # t-stat of mean IC = mean / (std / sqrt(n)) = ICIR * sqrt(n).
        n = len(ic_clean)
        if n > 1 and ic_std and not np.isnan(ic_std):
            ic_tstat = ic_mean / (ic_std / np.sqrt(n))
        else:
            ic_tstat = np.nan

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
            "turnover": turnover,
            "quantile_returns": qret,
            "long_short_return": long_short,
            "hit_rate": hit_rate,
        }
        return self._results

    def _ensure(self) -> Dict[str, object]:
        if self._results is None:
            self.compute()
        assert self._results is not None
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
        lines.append(f"  IC t-stat            : {fmt(r['ic_tstat'])}")
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

        r = self._ensure()
        ic: pd.Series = r["ic"]  # type: ignore[assignment]
        qret: pd.DataFrame = r["quantile_returns"]  # type: ignore[assignment]

        fig, (ax_ic, ax_q) = plt.subplots(1, 2, figsize=(12, 4.5))

        cum_ic = ic.fillna(0.0).cumsum()
        ax_ic.plot(cum_ic.index, cum_ic.values, color="C0", lw=1.4)
        ax_ic.axhline(0.0, color="black", lw=0.6, alpha=0.6)
        ax_ic.set_title(f"Cumulative IC (mean={r['ic_mean']:.4f})")
        ax_ic.set_xlabel("date")
        ax_ic.set_ylabel("cumulative IC")

        bars = qret.drop(index="long_short", errors="ignore")
        ax_q.bar(
            [str(q) for q in bars.index],
            bars["mean_return"].values,
            color="C1",
        )
        ax_q.axhline(0.0, color="black", lw=0.6, alpha=0.6)
        ax_q.set_title("Mean forward return by quantile")
        ax_q.set_xlabel("quantile")
        ax_q.set_ylabel("mean return")

        fig.tight_layout()
        return fig
