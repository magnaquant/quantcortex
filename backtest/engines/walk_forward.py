"""Walk-forward cross-validation splitter for time-series strategy research.

:class:`WalkForwardOptimizer` produces successive ``(train, test)`` index folds
that walk forward through time, never letting test data precede its training
data.  It supports the two canonical schemes plus the leakage controls from
Lopez de Prado, *Advances in Financial Machine Learning* (2018):

* **expanding** -- the training window always starts at index 0 and its end
  grows fold over fold (anchored / cumulative).  Uses all history available so
  far.
* **rolling** -- the training window is a fixed-length block of ``train_size``
  bars that slides forward each fold (recent-history only; adapts to regime
  change at the cost of throwing away old data).

Two leakage controls separate train from test:

* **embargo** -- ``embargo_gap`` bars are dropped *between* the (purged) train
  end and the test start.  This prevents serial-correlation leakage from
  observations that straddle the boundary.
* **purge** -- because labels are forward-looking over ``label_horizon`` bars,
  the last ``label_horizon`` training samples have label windows that overlap
  the embargo/test region.  They are *purged* (removed) so no training label
  uses information from the test period.

The total separation between the last usable training index and the first test
index is therefore ``label_horizon + embargo_gap`` bars.
"""

from __future__ import annotations

from typing import Iterator, Tuple, Union

import numpy as np
import pandas as pd

__all__ = ["WalkForwardOptimizer"]

IndexLike = Union[int, "pd.Index", np.ndarray, list]


class WalkForwardOptimizer:
    """Generate purged & embargoed walk-forward train/test splits.

    Parameters
    ----------
    train_size:
        Number of bars in the training window.  In ``rolling`` mode this is the
        fixed window length; in ``expanding`` mode it is the size of the first
        window (the window then grows).
    test_size:
        Number of bars in each test window.
    embargo_gap:
        Number of bars to drop between the (purged) training end and the test
        start.  Default ``21``.
    mode:
        ``"expanding"`` (anchored, growing train) or ``"rolling"`` (fixed-size
        sliding train).  Default ``"expanding"``.
    label_horizon:
        Forward-return label horizon in bars.  The last ``label_horizon``
        training samples are purged so their label windows do not overlap the
        test region.  Default ``1``.
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        *,
        embargo_gap: int = 21,
        mode: str = "expanding",
        label_horizon: int = 1,
    ) -> None:
        if train_size < 1:
            raise ValueError("train_size must be >= 1.")
        if test_size < 1:
            raise ValueError("test_size must be >= 1.")
        if embargo_gap < 0:
            raise ValueError("embargo_gap must be >= 0.")
        if label_horizon < 0:
            raise ValueError("label_horizon must be >= 0.")
        if mode not in ("expanding", "rolling"):
            raise ValueError("mode must be 'expanding' or 'rolling'.")
        self.train_size = int(train_size)
        self.test_size = int(test_size)
        self.embargo_gap = int(embargo_gap)
        self.mode = mode
        self.label_horizon = int(label_horizon)

    @staticmethod
    def _resolve_n(index_or_n: IndexLike) -> int:
        """Return the number of positional samples for an int or index/array."""
        if isinstance(index_or_n, (int, np.integer)):
            return int(index_or_n)
        return len(index_or_n)

    def n_splits(self, index_or_n: IndexLike) -> int:
        """Number of folds produced for ``n`` samples (or a given index)."""
        n = self._resolve_n(index_or_n)
        gap = self.embargo_gap
        count = 0
        train_end = self.train_size  # exclusive end of the raw training block
        while True:
            test_start = train_end + gap
            test_end = test_start + self.test_size
            if test_end > n:
                break
            count += 1
            train_end += self.test_size
        return count

    def split(
        self, index_or_n: IndexLike
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` positional integer arrays per fold.

        The test window walks forward by ``test_size`` bars each fold until the
        data is exhausted.  Training indices are purged of their last
        ``label_horizon`` samples and an ``embargo_gap`` is left before the test
        window begins.

        Parameters
        ----------
        index_or_n:
            Either the integer sample count ``n`` or a pandas ``Index`` / array
            / list whose *positional* indices are used.

        Yields
        ------
        (train_idx, test_idx):
            Tuples of 1-D ``numpy`` integer arrays.
        """
        n = self._resolve_n(index_or_n)
        gap = self.embargo_gap

        train_end = self.train_size  # exclusive end of the raw training block
        while True:
            test_start = train_end + gap
            test_end = test_start + self.test_size
            if test_end > n:
                break

            if self.mode == "expanding":
                train_start = 0
            else:  # rolling
                train_start = train_end - self.train_size

            # Purge the last `label_horizon` training samples whose forward
            # label windows would reach into the embargo / test region.
            purged_train_end = train_end - self.label_horizon
            if purged_train_end <= train_start:
                # No training samples survive purging for this fold; advance.
                train_end += self.test_size
                continue

            train_idx = np.arange(train_start, purged_train_end, dtype=np.int64)
            test_idx = np.arange(test_start, test_end, dtype=np.int64)
            yield train_idx, test_idx

            train_end += self.test_size

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"WalkForwardOptimizer(train_size={self.train_size}, "
            f"test_size={self.test_size}, embargo_gap={self.embargo_gap}, "
            f"mode='{self.mode}', label_horizon={self.label_horizon})"
        )
