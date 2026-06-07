"""Look-ahead / data-leakage audit via feature time-shifting.

Look-ahead bias is the use of information that would not have been available at
decision time — the most pernicious backtest pitfall because it is invisible in
the equity curve (it just looks like a great strategy).  This module provides a
*behavioural* test for it that treats the backtest as a black box.

The audit logic
---------------
A legitimate strategy extracts its edge from the **timing** of its features: it
acts on information that is fresh *as of* the decision date.  If we deliberately
feed it **stale** features (``features.shift(shift)``, i.e. yesterday's values
presented as today's), a genuine strategy must do meaningfully *worse* — its
information advantage has been blunted.

A *leaky* strategy, by contrast, is not really using the timing of the
features; it is exploiting some artefact (e.g. a feature that already encodes
the future, or an index that secretly aligns with the target).  Shifting such
features does **not** degrade performance — the score holds up or even improves.
That non-degradation is the red flag:

    suspicious := shifted_score >= base_score - tol

The forward-shift variant feeds the strategy features from the *future*
(``features.shift(-shift)``).  No honest strategy can benefit from data it could
not have seen; if performance *jumps* when given the future, the harness is
leaking look-ahead information.
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
        score = float(value)  # type: ignore[arg-type]
        if not np.isfinite(score):
            raise ValueError("backtest_fn returned a non-finite score")
        return score

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
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        if shift < 1:
            raise ValueError("shift must be a positive integer")

        base_score = self._score(self.backtest_fn(features))
        shifted_score = self._score(self.backtest_fn(features.shift(shift)))
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
            for _ in range(int(n_random)):
                perm = rng.permutation(features.index.to_numpy())
                shuffled = features.reindex(perm)
                shuffled.index = features.index
                random_scores.append(self._score(self.backtest_fn(shuffled)))
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

        Feeds the strategy ``features.shift(-shift)`` — values it could not
        have observed at decision time.  No honest strategy benefits from the
        future, so a performance **jump** is a direct leakage signal:

            suspicious := forward_score > base_score + tol

        Returns
        -------
        dict
            ``{base_score, forward_score, delta, suspicious, shift, tol}`` with
            ``delta = forward_score - base_score`` (positive = performance rose
            when fed the future, the warning condition).
        """
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        if shift < 1:
            raise ValueError("shift must be a positive integer")

        base_score = self._score(self.backtest_fn(features))
        forward_score = self._score(self.backtest_fn(features.shift(-shift)))
        delta = forward_score - base_score

        return {
            "base_score": base_score,
            "forward_score": forward_score,
            "delta": delta,
            "suspicious": bool(forward_score > base_score + tol),
            "shift": shift,
            "tol": tol,
        }
