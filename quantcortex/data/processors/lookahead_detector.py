"""Look-ahead bias detector - scans a feature matrix for future-data leakage.

Look-ahead bias is the single most damaging backtesting pitfall: a feature that
secretly encodes information from ``t+k`` will produce a beautiful in-sample
Sharpe that evaporates live.  This module provides *static* (no-model) scans of
a feature ``DataFrame`` that flag the structural fingerprints of leakage:

1. **Trailing-NaN fingerprint.**  A strictly-causal feature (rolling means,
   lagged returns, ...) can only be undefined at the *start* of a series while
   the window warms up.  A feature built with ``series.shift(-k)`` instead runs
   out of *future* data and is undefined at the *end*.  A column with NaNs
   concentrated at its tail is therefore a prime leakage suspect.

2. **Future-aligned cross-correlation.**  Given a reference series (e.g. close
   price), a causal feature correlates best with the reference at lag ``>= 0``
   (its own past).  If a feature instead correlates best with a *future* lag of
   the reference, it is leaking that future value.

3. **Near-perfect target correlation.**  Given a (forward) target, a feature
   whose correlation with that target is implausibly close to 1.0 almost
   certainly *is* (a transform of) the target.

The detector *flags* - it errs toward surfacing suspects.  The companion
``quantcortex/backtest/validation/lookahead_audit.py`` performs the dynamic,
model-based audit (re-run with shifted features).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "LeakageFinding",
    "LookaheadReport",
    "LookaheadDetector",
    "LookaheadViolationError",
]


class LookaheadViolationError(AssertionError):
    """Raised by :meth:`LookaheadDetector.assert_clean` when leakage is found."""


@dataclass
class LeakageFinding:
    column: str
    reason: str
    severity: str  # "high" | "medium"
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.severity}] {self.column}: {self.reason} ({self.detail})"


@dataclass
class LookaheadReport:
    findings: List[LeakageFinding] = field(default_factory=list)

    @property
    def flagged_columns(self) -> List[str]:
        # Preserve first-seen order, de-duplicated.
        seen: dict[str, None] = {}
        for f in self.findings:
            seen.setdefault(f.column, None)
        return list(seen)

    @property
    def is_clean(self) -> bool:
        return len(self.findings) == 0

    def __bool__(self) -> bool:
        # Truthy when leakage was detected.
        return not self.is_clean

    def summary(self) -> str:
        if self.is_clean:
            return "No look-ahead leakage detected."
        lines = [f"Detected {len(self.findings)} potential leak(s):"]
        lines += [f"  - {f}" for f in self.findings]
        return "\n".join(lines)


class LookaheadDetector:
    """Static scanner for look-ahead bias in a feature ``DataFrame``."""

    def __init__(
        self,
        *,
        corr_threshold: float = 0.90,
        target_corr_threshold: float = 0.985,
        max_lag: int = 5,
        trailing_nan_ratio: float = 0.0,
        min_obs: int = 20,
    ) -> None:
        self.corr_threshold = float(corr_threshold)
        self.target_corr_threshold = float(target_corr_threshold)
        self.max_lag = int(max_lag)
        # A column trips the trailing-NaN check when it has strictly more
        # trailing than leading NaNs (ratio compares against this slack).
        self.trailing_nan_ratio = float(trailing_nan_ratio)
        self.min_obs = int(min_obs)

    # ------------------------------------------------------------------ #
    # individual checks
    # ------------------------------------------------------------------ #
    @staticmethod
    def _leading_nans(col: pd.Series) -> int:
        mask = col.isna().to_numpy()
        if not mask.any():
            return 0
        # number of NaNs before the first valid value
        first_valid = np.argmax(~mask) if (~mask).any() else len(mask)
        return int(first_valid)

    @staticmethod
    def _trailing_nans(col: pd.Series) -> int:
        mask = col.isna().to_numpy()
        if not mask.any():
            return 0
        rev = mask[::-1]
        first_valid = np.argmax(~rev) if (~rev).any() else len(rev)
        return int(first_valid)

    def _check_trailing_nans(self, df: pd.DataFrame) -> List[LeakageFinding]:
        findings: List[LeakageFinding] = []
        for col in df.columns:
            series = df[col]
            if not pd.api.types.is_numeric_dtype(series):
                continue
            trailing = self._trailing_nans(series)
            leading = self._leading_nans(series)
            if trailing > leading and trailing > 0:
                findings.append(
                    LeakageFinding(
                        column=str(col),
                        reason="trailing NaNs exceed leading NaNs",
                        severity="high",
                        detail=(
                            f"{trailing} trailing vs {leading} leading NaN(s) - "
                            "fingerprint of a forward shift(-k)"
                        ),
                    )
                )
        return findings

    def _check_reference_alignment(
        self, df: pd.DataFrame, reference: pd.Series
    ) -> List[LeakageFinding]:
        findings: List[LeakageFinding] = []
        ref = pd.Series(reference).astype(float)
        for col in df.columns:
            series = df[col]
            if not pd.api.types.is_numeric_dtype(series):
                continue
            best_lag, best_corr = 0, 0.0
            for lag in range(-self.max_lag, self.max_lag + 1):
                # positive lag -> reference's past; negative -> reference's future
                shifted = ref.shift(lag)
                pair = pd.concat([series, shifted], axis=1).dropna()
                if len(pair) < self.min_obs:
                    continue
                a = pair.iloc[:, 0].to_numpy()
                b = pair.iloc[:, 1].to_numpy()
                if a.std() == 0 or b.std() == 0:
                    continue
                corr = abs(float(np.corrcoef(a, b)[0, 1]))
                if corr > best_corr:
                    best_corr, best_lag = corr, lag
            if best_lag < 0 and best_corr >= self.corr_threshold:
                findings.append(
                    LeakageFinding(
                        column=str(col),
                        reason="best correlation with a FUTURE lag of reference",
                        severity="high",
                        detail=(
                            f"|corr|={best_corr:.3f} at lag {best_lag} "
                            "(reference shifted into the future)"
                        ),
                    )
                )
        return findings

    def _check_target_leakage(
        self, df: pd.DataFrame, target: pd.Series
    ) -> List[LeakageFinding]:
        findings: List[LeakageFinding] = []
        tgt = pd.Series(target).astype(float)
        for col in df.columns:
            series = df[col]
            if not pd.api.types.is_numeric_dtype(series):
                continue
            pair = pd.concat([series, tgt], axis=1).dropna()
            if len(pair) < self.min_obs:
                continue
            a, b = pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
            if a.std() == 0 or b.std() == 0:
                continue
            corr = abs(float(np.corrcoef(a, b)[0, 1]))
            if corr >= self.target_corr_threshold:
                findings.append(
                    LeakageFinding(
                        column=str(col),
                        reason="near-perfect correlation with target",
                        severity="high",
                        detail=f"|corr|={corr:.4f} >= {self.target_corr_threshold}",
                    )
                )
        return findings

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def scan(
        self,
        features: pd.DataFrame,
        *,
        target: Optional[pd.Series] = None,
        reference: Optional[pd.Series] = None,
    ) -> LookaheadReport:
        """Run all applicable checks and return a :class:`LookaheadReport`."""
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        report = LookaheadReport()
        report.findings.extend(self._check_trailing_nans(features))
        if reference is not None:
            report.findings.extend(
                self._check_reference_alignment(features, reference)
            )
        if target is not None:
            report.findings.extend(self._check_target_leakage(features, target))
        return report

    def assert_clean(
        self,
        features: pd.DataFrame,
        *,
        target: Optional[pd.Series] = None,
        reference: Optional[pd.Series] = None,
    ) -> None:
        """Raise :class:`LookaheadViolationError` if any leakage is detected."""
        report = self.scan(features, target=target, reference=reference)
        if not report.is_clean:
            raise LookaheadViolationError(report.summary())


def scan_features(features: pd.DataFrame, **kwargs) -> LookaheadReport:
    """Module-level convenience wrapper."""
    detector = LookaheadDetector()
    return detector.scan(features, **kwargs)
