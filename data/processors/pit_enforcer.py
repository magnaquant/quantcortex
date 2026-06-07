"""Point-in-time (PIT) enforcement for fundamentals — anti-lookahead guardrail.

Look-ahead bias via fundamentals is one of the most common and most damaging
research pitfalls.  A naive join attaches a company's Q1 financials to dates
*inside* Q1, but those numbers were not public until the earnings announcement
(often weeks after the period closed and after analysts could trade on them).

The cardinal rule enforced here:

    A feature computed on ``feature_date`` may only use fundamentals whose
    ``announcement_date`` is on or before ``feature_date``.

We deliberately key everything off ``announcement_date`` (the date the data
became public), **never** ``period_end`` (the accounting period close).  Using
``period_end`` would leak data that the market did not yet have, inflating
backtest performance.  ``period_end`` is only used as a sanity check: an
announcement cannot precede the close of the period it reports on.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["PITViolationError", "PITEnforcer"]


class PITViolationError(Exception):
    """Raised when fundamentals data would leak future information."""


class PITEnforcer:
    """Validate and apply point-in-time semantics to fundamentals data."""

    def enforce(
        self,
        fundamentals: pd.DataFrame,
        feature_date_col: str = "feature_date",
        announcement_col: str = "announcement_date",
        symbol_col: str = "symbol",
    ) -> pd.DataFrame:
        """Assert PIT correctness for every row of ``fundamentals``.

        Checks that ``announcement_date <= feature_date`` on every row (a feature
        may only use data that was already public on the date it is computed)
        and, when a ``period_end`` column is present, that
        ``announcement_date >= period_end`` (data cannot be announced before the
        reporting period closes).

        Returns the (unmodified) frame on success so the call can be chained.

        Raises
        ------
        PITViolationError
            On the first offending row, naming the ticker and the conflicting
            dates.
        """
        for col in (feature_date_col, announcement_col, symbol_col):
            if col not in fundamentals.columns:
                raise PITViolationError(f"missing required column '{col}'")

        df = fundamentals.copy()
        feature_dates = pd.to_datetime(df[feature_date_col])
        announce_dates = pd.to_datetime(df[announcement_col])

        # Rule 1: a feature cannot use data that was not yet public on the
        # date the feature is computed (announcement must be on/before feature).
        bad = announce_dates > feature_dates
        if bad.any():
            i = bad.idxmax()
            sym = df.loc[i, symbol_col]
            raise PITViolationError(
                f"PIT violation for {sym!r}: announcement_date "
                f"{pd.Timestamp(announce_dates.loc[i]).date()} is after "
                f"feature_date {pd.Timestamp(feature_dates.loc[i]).date()} "
                f"(would use data not yet public)"
            )

        # Rule 2 (sanity): an announcement cannot precede the period close.
        if "period_end" in df.columns:
            period_end = pd.to_datetime(df["period_end"])
            early = announce_dates < period_end
            if early.any():
                i = early.idxmax()
                sym = df.loc[i, symbol_col]
                raise PITViolationError(
                    f"PIT violation for {sym!r}: announcement_date "
                    f"{pd.Timestamp(announce_dates.loc[i]).date()} is before "
                    f"period_end {pd.Timestamp(period_end.loc[i]).date()} "
                    f"(data announced before the period closed)"
                )

        return fundamentals

    def as_of(
        self,
        fundamentals: pd.DataFrame,
        as_of_date,
        announcement_col: str = "announcement_date",
    ) -> pd.DataFrame:
        """Return a point-in-time slice known on ``as_of_date``.

        Keeps only rows whose ``announcement_date <= as_of_date`` — i.e. exactly
        the fundamentals a researcher could have observed at that moment.
        """
        if announcement_col not in fundamentals.columns:
            raise PITViolationError(f"missing required column '{announcement_col}'")
        ts = pd.Timestamp(as_of_date)
        announce = pd.to_datetime(fundamentals[announcement_col])
        return fundamentals.loc[announce <= ts].copy()

    def point_in_time_merge(
        self,
        features: pd.DataFrame,
        fundamentals: pd.DataFrame,
        on: str = "symbol",
        feature_date_col: str = "feature_date",
        announcement_col: str = "announcement_date",
    ) -> pd.DataFrame:
        """As-of merge attaching the latest *known* fundamentals per feature row.

        For each row of ``features`` (keyed by ``on`` and dated by
        ``feature_date_col``) this attaches the most recent fundamentals record
        whose ``announcement_date <= feature_date``.  The match uses
        ``announcement_date`` — the public-availability date — and **never**
        ``period_end``, which is why the result is free of look-ahead bias.

        Implemented with :func:`pandas.merge_asof` (backward direction) grouped by
        the join key.
        """
        if feature_date_col not in features.columns:
            raise PITViolationError(f"features missing column '{feature_date_col}'")
        if announcement_col not in fundamentals.columns:
            raise PITViolationError(
                f"fundamentals missing column '{announcement_col}'"
            )
        if on not in features.columns or on not in fundamentals.columns:
            raise PITViolationError(f"both frames must contain join key '{on}'")

        left = features.copy()
        right = fundamentals.copy()
        left[feature_date_col] = pd.to_datetime(left[feature_date_col])
        right[announcement_col] = pd.to_datetime(right[announcement_col])

        # merge_asof requires both keys sorted by the time column.
        left = left.sort_values(feature_date_col, kind="mergesort")
        right = right.sort_values(announcement_col, kind="mergesort")

        merged = pd.merge_asof(
            left,
            right,
            left_on=feature_date_col,
            right_on=announcement_col,
            by=on,
            direction="backward",
            allow_exact_matches=True,
        )
        return merged.reset_index(drop=True)
