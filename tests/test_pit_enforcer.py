"""Tests for point-in-time enforcement (the fundamentals anti-lookahead guard).

The default rule is strict-before matching so a date-only announcement is not
silently assumed available at the start of that same date. Exact matches are an
explicit opt-in for observed intraday timestamps.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantcortex.data.processors.pit_enforcer import PITEnforcer, PITViolationError


def test_enforce_passes_when_announcement_precedes_feature_date():
    df = pd.DataFrame({
        "symbol": ["AAA", "BBB"],
        "feature_date": ["2024-03-01", "2024-03-01"],
        "announcement_date": ["2024-02-15", "2024-01-30"],  # both already public
    })
    out = PITEnforcer().enforce(df)
    # returns the (unmodified) frame so calls can chain
    pd.testing.assert_frame_equal(out, df)


def test_enforce_rejects_future_announcement():
    # announcement_date is AFTER feature_date -> would use data not yet public
    df = pd.DataFrame({
        "symbol": ["AAA"],
        "feature_date": ["2024-01-01"],
        "announcement_date": ["2024-02-15"],
    })
    with pytest.raises(PITViolationError, match="not yet public"):
        PITEnforcer().enforce(df)


def test_exact_announcement_timestamp_requires_explicit_opt_in():
    df = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "feature_date": ["2024-02-15"],
            "announcement_date": ["2024-02-15"],
        }
    )
    with pytest.raises(PITViolationError, match="exact-match policy"):
        PITEnforcer().enforce(df)
    pd.testing.assert_frame_equal(
        PITEnforcer(allow_exact_matches=True).enforce(df),
        df,
    )

    features = pd.DataFrame(
        {"symbol": ["AAA"], "feature_date": ["2024-02-15"]}
    )
    fundamentals = pd.DataFrame(
        {"symbol": ["AAA"], "announcement_date": ["2024-02-15"], "eps": [1.0]}
    )
    strict = PITEnforcer().point_in_time_merge(features, fundamentals)
    exact = PITEnforcer(allow_exact_matches=True).point_in_time_merge(
        features, fundamentals
    )
    assert pd.isna(strict.loc[0, "eps"])
    assert exact.loc[0, "eps"] == 1.0


def test_enforce_rejects_announcement_before_period_end():
    # Sanity rule: data cannot be announced before the reporting period closes.
    df = pd.DataFrame({
        "symbol": ["AAA"],
        "feature_date": ["2024-05-01"],     # well after announcement, so Rule 1 is fine
        "announcement_date": ["2024-03-15"],
        "period_end": ["2024-03-31"],       # announced 03-15 < close 03-31 -> impossible
    })
    with pytest.raises(PITViolationError, match="before the period closed|period_end"):
        PITEnforcer().enforce(df)


def test_enforce_requires_columns():
    df = pd.DataFrame({"symbol": ["AAA"], "feature_date": ["2024-01-01"]})  # no announcement_date
    with pytest.raises(PITViolationError, match="missing required column"):
        PITEnforcer().enforce(df)


def test_as_of_keeps_only_already_announced_rows():
    df = pd.DataFrame({
        "symbol": ["A", "A", "A"],
        "announcement_date": ["2024-01-10", "2024-02-10", "2024-03-10"],
        "eps": [1, 2, 3],
    })
    out = PITEnforcer().as_of(df, "2024-02-15")
    # only the 01-10 and 02-10 announcements were public by 02-15
    assert list(out["eps"]) == [1, 2]


def test_point_in_time_merge_attaches_latest_known_not_future():
    features = pd.DataFrame({"symbol": ["A"], "feature_date": ["2024-02-01"]})
    fundamentals = pd.DataFrame({
        "symbol": ["A", "A"],
        "announcement_date": ["2024-01-15", "2024-03-01"],
        "eps": [10, 20],
    })
    merged = PITEnforcer().point_in_time_merge(features, fundamentals)
    # 10 was public by 2024-02-01; 20 (announced 2024-03-01) must NOT leak back.
    assert merged.loc[0, "eps"] == 10


def test_point_in_time_merge_is_nan_before_any_announcement():
    features = pd.DataFrame({"symbol": ["A"], "feature_date": ["2024-01-01"]})
    fundamentals = pd.DataFrame({
        "symbol": ["A"],
        "announcement_date": ["2024-01-15"],  # not public until after the feature date
        "eps": [10],
    })
    merged = PITEnforcer().point_in_time_merge(features, fundamentals)
    assert pd.isna(merged.loc[0, "eps"])
