"""Tests for the static look-ahead bias detector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcortex.data.processors.lookahead_detector import (
    LookaheadDetector,
    LookaheadViolationError,
)


@pytest.fixture
def feature_frame():
    """A feature frame with one strictly-causal and one leaking column."""
    rng = np.random.default_rng(7)
    n = 200
    idx = pd.bdate_range("2021-01-01", periods=n)
    base = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, n)), index=idx)

    df = pd.DataFrame(index=idx)
    # causal: a lagged rolling mean (only past data; NaNs at the START)
    df["causal_mom"] = base.rolling(5).mean().shift(1)
    # causal: a simple lagged return
    df["causal_ret"] = base.pct_change().shift(1)
    # LEAK: tomorrow's value used today (NaN at the END; future-aligned)
    df["leaky_future"] = base.shift(-1)
    return df, base


def test_detector_flags_future_looking_column(feature_frame):
    df, base = feature_frame
    detector = LookaheadDetector()
    report = detector.scan(df, reference=base)

    assert not report.is_clean
    assert "leaky_future" in report.flagged_columns


def test_detector_clears_causal_columns(feature_frame):
    df, base = feature_frame
    detector = LookaheadDetector()
    report = detector.scan(df, reference=base)

    assert "causal_mom" not in report.flagged_columns
    assert "causal_ret" not in report.flagged_columns


def test_trailing_nan_fingerprint_alone_detects_leak(feature_frame):
    df, _ = feature_frame
    detector = LookaheadDetector()
    # No reference / target -> only the trailing-NaN check runs, yet the
    # forward-shift leak (trailing NaN) is still caught.
    report = detector.scan(df[["causal_mom", "leaky_future"]])
    assert "leaky_future" in report.flagged_columns
    assert "causal_mom" not in report.flagged_columns


def test_near_perfect_target_correlation_flagged():
    rng = np.random.default_rng(1)
    n = 150
    idx = pd.bdate_range("2021-01-01", periods=n)
    target = pd.Series(rng.normal(0, 0.02, n), index=idx)
    df = pd.DataFrame(
        {
            "honest": pd.Series(rng.normal(0, 1, n), index=idx),
            "is_the_target": target * 3.0,  # perfectly correlated transform
        }
    )
    detector = LookaheadDetector()
    report = detector.scan(df, target=target)
    assert "is_the_target" in report.flagged_columns
    assert "honest" not in report.flagged_columns


def test_assert_clean_raises(feature_frame):
    df, base = feature_frame
    detector = LookaheadDetector()
    with pytest.raises(LookaheadViolationError):
        detector.assert_clean(df, reference=base)
