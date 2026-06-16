"""Look-ahead / data-leakage audit via feature time-shifting.

Look-ahead bias is the use of information that would not have been available at
decision time - the most pernicious backtest pitfall because it is invisible in
the equity curve (it just looks like a great strategy). This module provides a
behavioural sensitivity diagnostic that treats the backtest as a black box. It
cannot prove that a strategy is clean or leaky.

The audit logic
---------------
Staling features can reveal timing sensitivity, but persistent legitimate
signals may degrade little and noisy signals may improve by chance. The
``suspicious`` flag therefore means "inspect this result," not "leakage found":

    suspicious := shifted_score >= base_score - tol

The forward-shift variant deliberately supplies future values. A performance
jump measures sensitivity to unavailable information; it does not identify
where a production pipeline leaks.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

__all__ = ["LookaheadAudit"]

BacktestFn = Callable[[pd.DataFrame], float]


class LookaheadAudit:
    """Detect look-ahead bias by perturbing the timing of input features.

    Parameters
    ----------
    backtest_fn:
        A callable mapping a feature :class:`~pandas.DataFrame` to a single
        scalar performance metric (higher is better, e.g. a Sharpe ratio).
        It must be deterministic given its input for the audit to be
        meaningful.
    """

    def __init__(self, backtest_fn: BacktestFn) -> None:
        if not callable(backtest_fn):
            raise TypeError("backtest_fn must be callable")
        self.backtest_fn = backtest_fn

    @staticmethod
    def _score(value: object) -> float:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("backtest_fn returned a boolean instead of a score")
        try:
            score = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("backtest_fn must return a scalar numeric score") from exc
        if not np.isfinite(score):
            raise ValueError("backtest_fn returned a non-finite score")
        return score

    @staticmethod
    def _validate_features(features: pd.DataFrame, shift: int) -> None:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        if features.empty or features.shape[1] == 0:
            raise ValueError("features must be a non-empty DataFrame")
        if not isinstance(features.index, pd.DatetimeIndex):
            raise TypeError("features must use a DatetimeIndex")
        if features.index.hasnans:
            raise ValueError("features index must contain valid timestamps")
        if not features.index.is_monotonic_increasing:
            raise ValueError("features dates must be sorted in increasing order")
        if features.index.has_duplicates or features.columns.has_duplicates:
            raise ValueError("features index and columns must be unique")
        if shift >= len(features):
            raise ValueError("shift must be smaller than the number of feature rows")

    def run(
        self,
        features: pd.DataFrame,
        shift: int = 1,
        n_random: int = 0,
        tol: float = 0.0,
        random_state: Optional[int] = None,
    ) -> Dict[str, object]:
        """Run the staleness audit (shift features into the *past*).

        Scores the strategy on the real features (``base_score``) and on
        features shifted forward in index by ``shift`` periods, which hands the
        model **stale** data.  A legitimate strategy should lose performance;
        if it does not, the strategy is flagged as suspicious.

        Parameters
        ----------
        features:
            The feature matrix passed to ``backtest_fn``.
        shift:
            Number of periods to shift the features into the past (default 1).
        n_random:
            If ``> 0``, also score the strategy on ``n_random`` row-shuffled
            copies of the features to estimate a noise-floor (``random_mean``
            / ``random_std``); a base score that does not clear this floor is
            additional evidence of leakage or absence of edge.
        tol:
            Tolerance for the degradation test.  ``suspicious`` is ``True``
            when ``shifted_score >= base_score - tol``.
        random_state:
            Seed for the row-shuffle permutations.

        Returns
        -------
        dict
            ``{base_score, shifted_score, delta, suspicious, shift, tol,
            [random_mean, random_std, random_scores]}`` where
            ``delta = base_score - shifted_score`` (positive = healthy
            degradation).
        """
        if isinstance(shift, bool) or int(shift) != shift or shift < 1:
            raise ValueError("shift must be a positive integer")
        shift = int(shift)
        self._validate_features(features, shift)
        if isinstance(n_random, bool) or int(n_random) != n_random or n_random < 0:
            raise ValueError("n_random must be a non-negative integer")
        n_random = int(n_random)
        if isinstance(tol, (bool, np.bool_)) or not np.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be finite and non-negative")
        if random_state is not None and (
            isinstance(random_state, (bool, np.bool_))
            or not isinstance(random_state, (int, np.integer))
            or random_state < 0
        ):
            raise ValueError("random_state must be a non-negative integer or None")

        base_score = self._score(self.backtest_fn(features.copy(deep=True)))
        shifted_score = self._score(
            self.backtest_fn(features.shift(shift).copy(deep=True))
        )
        delta = base_score - shifted_score

        result: Dict[str, object] = {
            "base_score": base_score,
            "shifted_score": shifted_score,
            "delta": delta,
            "suspicious": bool(shifted_score >= base_score - tol),
            "shift": shift,
            "tol": tol,
        }

        if n_random > 0:
            rng = np.random.default_rng(random_state)
            random_scores = []
            for _ in range(n_random):
                perm = rng.permutation(features.index.to_numpy())
                shuffled = features.reindex(perm)
                shuffled.index = features.index
                random_scores.append(
                    self._score(self.backtest_fn(shuffled.copy(deep=True)))
                )
            arr = np.asarray(random_scores, dtype=float)
            result["random_scores"] = arr.tolist()
            result["random_mean"] = float(arr.mean())
            result["random_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0

        return result

    def run_forward(
        self,
        features: pd.DataFrame,
        shift: int = 1,
        tol: float = 0.0,
    ) -> Dict[str, object]:
        """Run the future-leakage audit (shift features into the *future*).

        Feeds the strategy ``features.shift(-shift)`` - values it could not
        have observed at decision time. A performance jump is a sensitivity
        warning, not proof that the unshifted pipeline leaks:

            suspicious := forward_score > base_score + tol

        Returns
        -------
        dict
            ``{base_score, forward_score, delta, suspicious, shift, tol}`` with
            ``delta = forward_score - base_score`` (positive = performance rose
            when fed the future, the warning condition).
        """
        if isinstance(shift, bool) or int(shift) != shift or shift < 1:
            raise ValueError("shift must be a positive integer")
        shift = int(shift)
        self._validate_features(features, shift)
        if isinstance(tol, (bool, np.bool_)) or not np.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be finite and non-negative")

        base_score = self._score(self.backtest_fn(features.copy(deep=True)))
        forward_score = self._score(
            self.backtest_fn(features.shift(-shift).copy(deep=True))
        )
        delta = forward_score - base_score

        return {
            "base_score": base_score,
            "forward_score": forward_score,
            "delta": delta,
            "suspicious": bool(forward_score > base_score + tol),
            "shift": shift,
            "tol": tol,
        }
