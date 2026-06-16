"""Point-in-time (PIT) enforcement for fundamentals - anti-lookahead guardrail.

Look-ahead bias via fundamentals is one of the most common and most damaging
research pitfalls.  A naive join attaches a company's Q1 financials to dates
*inside* Q1, but those numbers were not public until the earnings announcement
(often weeks after the period closed and after analysts could trade on them).

The default cardinal rule enforced here:

    A feature computed on ``feature_date`` may only use fundamentals whose
    ``announcement_date`` is strictly before ``feature_date``.

Strict-before is deliberate for date-only data: equal midnight timestamps do
not prove that a filing was available before the feature or closing trade was
formed. Callers with observed intraday timestamps may opt into exact matches.

We deliberately key everything off ``announcement_date`` (the date the data
became public), **never** ``period_end`` (the accounting period close).  Using
``period_end`` would leak data that the market did not yet have, inflating
backtest performance.  ``period_end`` is only used as a sanity check: an
announcement cannot precede the close of the period it reports on.
"""

from __future__ import annotations

from numbers import Number

import pandas as pd

__all__ = ["PITViolationError", "PITEnforcer"]


class PITViolationError(Exception):
    """Raised when fundamentals data would leak future information."""


class PITEnforcer:
    """Validate and apply point-in-time semantics to fundamentals data."""

    def __init__(self, *, allow_exact_matches: bool = False) -> None:
        if not isinstance(allow_exact_matches, bool):
            raise TypeError("allow_exact_matches must be a boolean")
        self.allow_exact_matches = allow_exact_matches

    @staticmethod
    def _parse_dates(values: pd.Series, column: str) -> pd.Series:
        """Parse a date column and reject missing or invalid observations."""
        if any(
            isinstance(value, Number) and not pd.isna(value) for value in values
        ):
            raise PITViolationError(
                f"column '{column}' contains numeric epoch dates"
            )
        parsed = pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None)
        missing = parsed.isna()
        if missing.any():
            rows = missing[missing].index.tolist()
            raise PITViolationError(
                f"column '{column}' contains missing or invalid dates at rows {rows}"
            )
        return parsed

    @staticmethod
    def _validate_join_key(values: pd.Series, column: str) -> None:
        if any(not isinstance(value, str) for value in values):
            raise PITViolationError(
                f"column '{column}' must contain string join keys"
            )
        missing = values.isna() | values.astype(str).str.strip().eq("")
        if missing.any():
            rows = missing[missing].index.tolist()
            raise PITViolationError(
                f"column '{column}' contains missing join keys at rows {rows}"
            )

    def enforce(
        self,
        fundamentals: pd.DataFrame,
        feature_date_col: str = "feature_date",
        announcement_col: str = "announcement_date",
        symbol_col: str = "symbol",
    ) -> pd.DataFrame:
        """Assert PIT correctness for every row of ``fundamentals``.

        By default checks that ``announcement_date < feature_date`` on every row.
        With ``allow_exact_matches=True``, exact timestamps are also accepted.
        This ensures a feature only uses data already public when it is computed
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
        self._validate_join_key(df[symbol_col], symbol_col)
        feature_dates = self._parse_dates(df[feature_date_col], feature_date_col)
        announce_dates = self._parse_dates(df[announcement_col], announcement_col)

        # Rule 1: a feature cannot use data that was not yet public on the
        # date the feature is computed (announcement must be on/before feature).
        bad = (
            announce_dates > feature_dates
            if self.allow_exact_matches
            else announce_dates >= feature_dates
        )
        if bad.any():
            pos = int(bad.to_numpy().nonzero()[0][0])
            sym = df.iloc[pos][symbol_col]
            raise PITViolationError(
                f"PIT violation for {sym!r}: announcement_date "
                f"{pd.Timestamp(announce_dates.iloc[pos]).date()} is after "
                f"feature_date {pd.Timestamp(feature_dates.iloc[pos])} "
                f"(would use data not yet public under the configured "
                f"exact-match policy)"
            )

        # Rule 2 (sanity): an announcement cannot precede the period close.
        if "period_end" in df.columns:
            period_end = self._parse_dates(df["period_end"], "period_end")
            early = announce_dates < period_end
            if early.any():
                pos = int(early.to_numpy().nonzero()[0][0])
                sym = df.iloc[pos][symbol_col]
                raise PITViolationError(
                    f"PIT violation for {sym!r}: announcement_date "
                    f"{pd.Timestamp(announce_dates.iloc[pos]).date()} is before "
                    f"period_end {pd.Timestamp(period_end.iloc[pos]).date()} "
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

        Uses strict-before matching by default; exact timestamps require
        ``allow_exact_matches=True`` on the enforcer.
        """
        if announcement_col not in fundamentals.columns:
            raise PITViolationError(f"missing required column '{announcement_col}'")
        if isinstance(as_of_date, (bool, Number)):
            raise PITViolationError("as_of_date must not be a numeric epoch")
        ts = pd.to_datetime(as_of_date, errors="coerce", utc=True)
        if not isinstance(ts, pd.Timestamp) or pd.isna(ts):
            raise PITViolationError("as_of_date is missing or invalid")
        ts = ts.tz_localize(None)
        announce = self._parse_dates(
            fundamentals[announcement_col], announcement_col
        )
        known = announce <= ts if self.allow_exact_matches else announce < ts
        return fundamentals.loc[known].copy()

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
        whose announcement timestamp precedes the feature timestamp. The match uses
        ``announcement_date`` - the public-availability date - and **never**
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
        self._validate_join_key(left[on], on)
        self._validate_join_key(right[on], on)
        left[feature_date_col] = self._parse_dates(
            left[feature_date_col], feature_date_col
        )
        right[announcement_col] = self._parse_dates(
            right[announcement_col], announcement_col
        )
        duplicate_records = right.duplicated([on, announcement_col], keep=False)
        if duplicate_records.any():
            rows = duplicate_records[duplicate_records].index.tolist()
            raise PITViolationError(
                "point_in_time_merge requires at most one fundamentals record "
                f"per ({on}, {announcement_col}); reshape tidy fields to one "
                f"row before merging (duplicate rows {rows})"
            )
        if "period_end" in right.columns:
            right["period_end"] = self._parse_dates(right["period_end"], "period_end")
            early = right[announcement_col] < right["period_end"]
            if early.any():
                rows = early[early].index.tolist()
                raise PITViolationError(
                    "fundamentals contain announcement dates before period end "
                    f"at rows {rows}"
                )

        # merge_asof requires both keys sorted by the time column. Preserve the
        # feature frame's input order because callers commonly align the result
        # positionally with another panel.
        order_col = "__pit_input_order__"
        while order_col in left.columns or order_col in right.columns:
            order_col = f"_{order_col}"
        left[order_col] = range(len(left))
        left = left.sort_values(feature_date_col, kind="mergesort")
        right = right.sort_values(announcement_col, kind="mergesort")

        merged = pd.merge_asof(
            left,
            right,
            left_on=feature_date_col,
            right_on=announcement_col,
            by=on,
            direction="backward",
            allow_exact_matches=self.allow_exact_matches,
        )
        return (
            merged.sort_values(order_col, kind="mergesort")
            .drop(columns=[order_col])
            .reset_index(drop=True)
        )
