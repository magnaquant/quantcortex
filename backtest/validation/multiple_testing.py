"""Multiple-testing corrections for factor and strategy mining.

Why factor mining *requires* multiple-testing correction
--------------------------------------------------------
A single hypothesis test controls the probability of a false positive at the
chosen significance level ``alpha`` (e.g. 5%).  But quantitative research
rarely tests one hypothesis: a factor zoo is built by screening hundreds or
thousands of candidate signals against the same price history.  If you run
``m`` independent tests at ``alpha = 0.05``, the expected number of *spurious*
"discoveries" is ``0.05 * m`` — test 200 useless factors and you expect ~10 to
look significant by pure chance.  Reporting the best of many backtests without
correction is the statistical equivalent of p-hacking, and it is the single
biggest driver of backtest overfitting in published factor research.

Multiple-testing procedures restore a meaningful error guarantee:

* **Bonferroni** controls the *family-wise error rate* (FWER) — the
  probability of *any* false positive — by testing each hypothesis at
  ``alpha / m``.  Simple and valid under any dependence, but very
  conservative.
* **Benjamini-Hochberg (BH)** controls the *false discovery rate* (FDR) — the
  expected fraction of false positives *among the rejections* — and is far
  more powerful.  Valid under independence or positive dependence (PRDS).
* **Benjamini-Hochberg-Yekutieli (BHY)** controls the FDR under *arbitrary*
  dependence by inflating the threshold with the harmonic penalty
  ``c(m) = sum_{i=1}^{m} 1/i``.  Strategy returns are heavily cross-correlated,
  so BHY is the prudent default for factor mining.

Each function returns ``(reject, adjusted)`` where ``reject`` is a boolean mask
(``adjusted <= alpha``) and ``adjusted`` are the adjusted p-values, comparable
directly against ``alpha`` and monotone in the raw p-values.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "bonferroni",
    "benjamini_hochberg",
    "bhy_correction",
    "MultipleTestingReport",
]


def _as_pvalue_array(pvalues: Sequence[float]) -> np.ndarray:
    """Validate and coerce p-values to a 1-D float array."""
    arr = np.asarray(pvalues, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("pvalues must be non-empty")
    if np.any(~np.isfinite(arr)):
        raise ValueError("pvalues must all be finite")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("pvalues must lie in [0, 1]")
    return arr


def bonferroni(
    pvalues: Sequence[float], alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Bonferroni FWER correction.

    Adjusted p-value is ``min(1, m * p)``; reject where adjusted ``<= alpha``.

    Returns
    -------
    (reject, adjusted):
        Boolean rejection mask and adjusted p-values, in input order.
    """
    p = _as_pvalue_array(pvalues)
    m = p.size
    adjusted = np.minimum(p * m, 1.0)
    reject = adjusted <= alpha
    return reject, adjusted


def benjamini_hochberg(
    pvalues: Sequence[float], alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction (independent / positively dependent).

    Sort p-values ascending ``p_(1) <= ... <= p_(m)``.  The BH-adjusted p-value
    (a.k.a. the q-value) is the enforced-monotone cumulative minimum, from the
    largest rank down, of ``p_(i) * m / i``.

    Returns
    -------
    (reject, adjusted):
        Boolean rejection mask (``adjusted <= alpha``) and adjusted p-values,
        in input order.
    """
    p = _as_pvalue_array(pvalues)
    return _step_up(p, alpha, penalty=1.0)


def bhy_correction(
    pvalues: Sequence[float], alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg-Yekutieli FDR correction (arbitrary dependence).

    Identical to :func:`benjamini_hochberg` but with the threshold inflated by
    the harmonic penalty ``c(m) = sum_{i=1}^{m} 1/i``, which makes the FDR
    guarantee valid under *any* dependence structure.  Consequently the
    BHY-adjusted p-values are always ``>=`` the BH-adjusted p-values.

    Returns
    -------
    (reject, adjusted):
        Boolean rejection mask and adjusted p-values, in input order.
    """
    p = _as_pvalue_array(pvalues)
    m = p.size
    cm = float(np.sum(1.0 / np.arange(1, m + 1)))  # harmonic number c(m)
    return _step_up(p, alpha, penalty=cm)


def _step_up(
    p: np.ndarray, alpha: float, penalty: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Shared BH/BHY step-up engine.

    ``penalty`` is 1 for BH and the harmonic number ``c(m)`` for BHY.  The
    adjusted p-value for rank ``i`` (1-based) is ``p_(i) * m * penalty / i``,
    made monotone via a cumulative minimum from the largest rank downward and
    clipped to ``[0, 1]``.
    """
    m = p.size
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    ranks = np.arange(1, m + 1, dtype=float)

    raw_adjusted = ranked * m * penalty / ranks
    # Enforce monotonicity: q_(i) = min(q_(i), q_(i+1), ...).
    monotone = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    monotone = np.minimum(monotone, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = monotone
    reject = adjusted <= alpha
    return reject, adjusted


class MultipleTestingReport:
    """Side-by-side comparison of multiple-testing corrections.

    Parameters
    ----------
    pvalues:
        Raw p-values, one per tested factor / strategy.
    labels:
        Optional names for each test (defaults to ``test_0``, ``test_1`` ...).
    """

    def __init__(
        self,
        pvalues: Sequence[float],
        labels: Optional[Sequence[str]] = None,
    ) -> None:
        self.pvalues = _as_pvalue_array(pvalues)
        m = self.pvalues.size
        if labels is None:
            self.labels: List[str] = [f"test_{i}" for i in range(m)]
        else:
            labels = list(labels)
            if len(labels) != m:
                raise ValueError(
                    f"labels length {len(labels)} != number of pvalues {m}"
                )
            self.labels = [str(x) for x in labels]

    def compare(self, alpha: float = 0.05) -> pd.DataFrame:
        """Return a DataFrame of raw/BH/BHY/Bonferroni p-values and rejections.

        Columns: ``raw_pvalue``, ``bh_adjusted``, ``bh_reject``,
        ``bhy_adjusted``, ``bhy_reject``, ``bonferroni_adjusted``,
        ``bonferroni_reject``.  Indexed by ``labels``.
        """
        bh_reject, bh_adj = benjamini_hochberg(self.pvalues, alpha)
        bhy_reject, bhy_adj = bhy_correction(self.pvalues, alpha)
        bonf_reject, bonf_adj = bonferroni(self.pvalues, alpha)

        return pd.DataFrame(
            {
                "raw_pvalue": self.pvalues,
                "bh_adjusted": bh_adj,
                "bh_reject": bh_reject,
                "bhy_adjusted": bhy_adj,
                "bhy_reject": bhy_reject,
                "bonferroni_adjusted": bonf_adj,
                "bonferroni_reject": bonf_reject,
            },
            index=pd.Index(self.labels, name="test"),
        )

    def summary(self, alpha: float = 0.05) -> str:
        """Human-readable summary of how many discoveries survive each method."""
        df = self.compare(alpha)
        m = len(df)
        lines = [
            f"Multiple-testing report ({m} tests, alpha={alpha:g})",
            "-" * 56,
            f"  Raw significant (no correction): "
            f"{int((df['raw_pvalue'] <= alpha).sum())}",
            f"  Benjamini-Hochberg (FDR):        {int(df['bh_reject'].sum())}",
            f"  Benjamini-Hochberg-Yekutieli:    {int(df['bhy_reject'].sum())}",
            f"  Bonferroni (FWER):               "
            f"{int(df['bonferroni_reject'].sum())}",
            "-" * 56,
            "  Note: factor mining over many candidates inflates false",
            "  positives; prefer BHY when strategy returns are correlated.",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"MultipleTestingReport(n_tests={self.pvalues.size})"
