from __future__ import annotations

import pandas as pd
import pytest

from quantcortex.alpha.validation.factor_decay import FactorDecay
from quantcortex.backtest.validation.lookahead_audit import LookaheadAudit
from quantcortex.backtest.validation.multiple_testing import (
    MultipleTestingReport,
    benjamini_hochberg,
    bhy_correction,
    bonferroni,
)
from quantcortex.backtest.validation.survivorship_check import (
    SurvivorshipValidator,
    survivorship_check,
)
from quantcortex.data.universe.base import PITMembership, Universe


class _Universe(Universe):
    name = "test"

    def membership(self) -> PITMembership:
        return PITMembership(
            pd.DataFrame(
                {
                    "symbol": ["AAA", "BRK-B"],
                    "start_date": ["2020-01-01", "2020-01-01"],
                    "end_date": [None, None],
                }
            )
        )


def test_multiple_testing_matches_hand_calculated_adjustments():
    pvalues = [0.01, 0.04, 0.03]
    bh_reject, bh_adjusted = benjamini_hochberg(pvalues, alpha=0.05)
    by_reject, by_adjusted = bhy_correction(pvalues, alpha=0.05)
    bonf_reject, bonf_adjusted = bonferroni(pvalues, alpha=0.05)

    assert bh_adjusted == pytest.approx([0.03, 0.04, 0.04])
    assert bh_reject.tolist() == [True, True, True]
    assert by_adjusted == pytest.approx([0.055, 0.0733333333, 0.0733333333])
    assert not by_reject.any()
    assert bonf_adjusted == pytest.approx([0.03, 0.12, 0.09])
    assert bonf_reject.tolist() == [True, False, False]


def test_multiple_testing_rejects_ambiguous_inputs():
    with pytest.raises(ValueError, match="one-dimensional"):
        benjamini_hochberg([[0.01, 0.02]])
    with pytest.raises(ValueError, match="boolean"):
        bonferroni([True, 0.1])
    with pytest.raises(ValueError, match="unique"):
        MultipleTestingReport([0.1, 0.2], labels=["same", "same"])


def test_lookahead_audit_requires_ordered_dates_and_isolates_callback_mutation():
    dates = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
    features = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=dates)

    def mutating_score(frame: pd.DataFrame) -> float:
        score = float(frame["x"].sum(skipna=True))
        frame.iloc[:, :] = -999.0
        return score

    result = LookaheadAudit(mutating_score).run(features, shift=1)
    assert result["base_score"] == pytest.approx(6.0)
    assert result["shifted_score"] == pytest.approx(3.0)
    assert features["x"].tolist() == [1.0, 2.0, 3.0]

    with pytest.raises(ValueError, match="sorted"):
        LookaheadAudit(mutating_score).run(features.iloc[::-1])


def test_factor_decay_uses_future_returns_and_validates_half_life():
    dates = pd.date_range("2024-01-01", periods=4)
    returns = pd.DataFrame({"A": [0.0, 0.10, 0.20, 0.30]}, index=dates)
    forward = FactorDecay._forward_return(returns, lag=2)
    assert forward.loc[dates[0], "A"] == pytest.approx(1.10 * 1.20 - 1.0)

    decay = pd.DataFrame(
        {"ic_mean": [0.10, 0.075, 0.025]}, index=pd.Index([1, 2, 3], name="lag")
    )
    assert FactorDecay().half_life(decay) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="sorted"):
        FactorDecay().half_life(decay.iloc[[0, 2, 1]])


def test_survivorship_validation_normalizes_symbols_and_rejects_bad_inputs():
    universe = _Universe()
    result = survivorship_check(universe, [" aaa ", "BRK.B"], "2024-01-01")
    assert result["ok"]

    with pytest.raises(ValueError, match="non-empty"):
        survivorship_check(universe, [None], "2024-01-01")
    with pytest.raises(TypeError, match="boolean"):
        SurvivorshipValidator(universe, strict="yes")
