"""Corporate-action adjustment, validation, and unadjusted-data detection.

Splits and dividends distort raw price series: an unhandled 2:1 split looks like
a -50% overnight crash and a missed dividend looks like an unexplained gap.
This module builds back-adjusted prices from a full history of corporate actions,
validates that a vendor's adjusted series is consistent with the provided actions,
and heuristically flags symbols whose raw prices appear never to have been
adjusted at all.

Corporate-action processing is inherently full-history (not strictly causal): the
back-adjustment factor on an early date depends on *all* future splits/dividends,
which is correct because adjustment is only ever applied to historical data for
analysis, never used as a forward-looking trading signal.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = ["AdjustmentError", "AdjustmentValidator"]


class AdjustmentError(Exception):
    """Raised when an adjusted series is inconsistent with corporate actions."""


class AdjustmentValidator:
    """Apply and validate corporate-action adjustments on OHLCV data."""

    # OHLCV price columns that should be scaled by the adjustment factor.
    _PRICE_COLS = ("open", "high", "low", "close")
    _VOLUME_COLS = ("volume",)

    def _split_factors(
        self, index: pd.DatetimeIndex, splits: Optional[pd.Series]
    ) -> pd.Series:
        """Cumulative *backward* split factor aligned to ``index``.

        A split ratio ``r`` on date ``d`` (e.g. ``r=2`` for a 2:1 split) scales
        every price strictly *before* ``d`` by ``1/r``.  The returned factor on
        each date is the product of ``1/r`` over all splits occurring after that
        date.
        """
        factor = pd.Series(1.0, index=index)
        if splits is None or len(splits) == 0:
            return factor
        splits = pd.Series(splits).copy()
        splits.index = pd.to_datetime(splits.index)
        splits = splits[(splits.values != 0) & (splits.values != 1.0)]
        for split_date, ratio in splits.items():
            if ratio == 0:
                continue
            # Prices strictly before the split date are divided by the ratio.
            mask = index < pd.Timestamp(split_date)
            factor[mask] = factor[mask] / float(ratio)
        return factor

    def _dividend_factors(
        self,
        index: pd.DatetimeIndex,
        close: pd.Series,
        dividends: Optional[pd.Series],
    ) -> pd.Series:
        """Cumulative *backward* dividend factor aligned to ``index``.

        A cash dividend ``D`` with ex-date ``d`` and prior close ``C`` scales
        every price strictly before ``d`` by ``(1 - D/C)``.  Factors compound
        multiplicatively across all dividends after each date.
        """
        factor = pd.Series(1.0, index=index)
        if dividends is None or len(dividends) == 0:
            return factor
        dividends = pd.Series(dividends).copy()
        dividends.index = pd.to_datetime(dividends.index)
        dividends = dividends[dividends.values != 0]
        for ex_date, amount in dividends.items():
            ex_ts = pd.Timestamp(ex_date)
            prior = close[close.index < ex_ts]
            if len(prior) == 0:
                continue
            prior_close = float(prior.iloc[-1])
            if prior_close <= 0:
                continue
            ratio = 1.0 - float(amount) / prior_close
            if ratio <= 0:
                continue
            mask = index < ex_ts
            factor[mask] = factor[mask] * ratio
        return factor

    def apply_adjustments(
        self,
        ohlcv: pd.DataFrame,
        splits: Optional[pd.Series] = None,
        dividends: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Return ``ohlcv`` with back-adjusted prices.

        Computes cumulative backward split and dividend factors over the full
        history and multiplies them into the price columns.  Volume is divided by
        the split factor (split-adjusted shares).  An ``adj_close`` column is
        always added; existing price columns are adjusted in place on the copy.

        Parameters
        ----------
        ohlcv:
            DataFrame indexed by date with any of ``open/high/low/close/volume``.
        splits:
            Series indexed by split date with the split ratio (``2`` for 2:1).
        dividends:
            Series indexed by ex-dividend date with the cash amount per share.
        """
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            df = ohlcv.copy()
            df.index = pd.to_datetime(df.index)
        else:
            df = ohlcv.copy()
        df = df.sort_index()
        index = df.index

        if "close" not in df.columns:
            raise AdjustmentError("apply_adjustments requires a 'close' column")
        raw_close = df["close"].astype(float)

        split_factor = self._split_factors(index, splits)
        div_factor = self._dividend_factors(index, raw_close, dividends)
        total_factor = split_factor * div_factor

        for col in self._PRICE_COLS:
            if col in df.columns:
                df[col] = df[col].astype(float) * total_factor
        for col in self._VOLUME_COLS:
            if col in df.columns:
                # Inverse split factor restores share counts to current basis.
                df[col] = df[col].astype(float) / split_factor

        df["adj_close"] = raw_close * total_factor
        df["adj_factor"] = total_factor
        return df

    def validate_adjustment(
        self,
        raw_close: pd.Series,
        adj_close: pd.Series,
        splits: Optional[pd.Series] = None,
        dividends: Optional[pd.Series] = None,
        tol: float = 1e-3,
    ) -> bool:
        """Validate a vendor adjusted series against corporate actions.

        The implied per-date adjustment factor is ``adj_close / raw_close``.  It
        is compared against the factor reconstructed from ``splits`` and
        ``dividends``.  Both factors are normalized to ``1.0`` on the last common
        date (vendors anchor adjustment at the latest date) before comparison.

        Raises
        ------
        AdjustmentError
            If the relative discrepancy exceeds ``tol`` on any date.
        """
        raw = pd.Series(raw_close).astype(float).copy()
        adj = pd.Series(adj_close).astype(float).copy()
        raw.index = pd.to_datetime(raw.index)
        adj.index = pd.to_datetime(adj.index)
        common = raw.index.intersection(adj.index).sort_values()
        if len(common) == 0:
            raise AdjustmentError("raw_close and adj_close share no dates")
        raw = raw.reindex(common)
        adj = adj.reindex(common)

        with np.errstate(divide="ignore", invalid="ignore"):
            implied = adj / raw
        if implied.isna().any() or (raw <= 0).any():
            raise AdjustmentError("non-positive or missing raw close prices")

        expected = self._split_factors(common, splits) * self._dividend_factors(
            common, raw, dividends
        )

        # Normalize both to 1.0 at the last common date for comparison.
        implied_n = implied / implied.iloc[-1]
        expected_n = expected / expected.iloc[-1]

        rel_err = (implied_n - expected_n).abs() / expected_n.abs().clip(lower=1e-12)
        worst = rel_err.max()
        if worst > tol:
            bad_date = rel_err.idxmax()
            raise AdjustmentError(
                f"adjusted close inconsistent with corporate actions on "
                f"{pd.Timestamp(bad_date).date()}: implied factor "
                f"{implied_n.loc[bad_date]:.6f} vs expected "
                f"{expected_n.loc[bad_date]:.6f} (rel err {worst:.6f} > tol {tol})"
            )
        return True

    def detect_unadjusted(
        self, prices: pd.DataFrame, jump_threshold: float = 0.30
    ) -> Dict[str, List[pd.Timestamp]]:
        """Flag symbols whose raw prices look like they were never adjusted.

        Scans overnight returns per symbol and flags dates where the magnitude of
        the move is consistent with an unhandled split: a large drop (e.g. a 2:1
        split shows as roughly ``-50%``) or a large jump (e.g. a 1:2 reverse split
        shows as roughly ``+100%``).  Returns a mapping ``symbol -> [dates]`` for
        symbols with at least one suspicious move.

        Parameters
        ----------
        prices:
            Wide DataFrame indexed by date, one column per symbol of close prices.
        jump_threshold:
            Minimum absolute overnight return to flag (default ``0.30``).  Ordinary
            daily equity moves are far smaller, so this isolates split-sized jumps.
        """
        df = prices.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        flagged: Dict[str, List[pd.Timestamp]] = {}
        for symbol in df.columns:
            series = df[symbol].astype(float)
            rets = series.pct_change()
            # A clean split shows a near-exact fractional jump; require both a
            # large magnitude and that it is "split-shaped" (not just volatile).
            suspicious = rets[rets.abs() >= jump_threshold]
            dates: List[pd.Timestamp] = []
            for date, ret in suspicious.items():
                ratio = 1.0 + ret  # price_today / price_yesterday
                # Distance to the nearest plausible integer split ratio (and its
                # inverse), e.g. 1:2 (0.5), 1:3 (0.333), 2:1 (2.0), 3:1 (3.0).
                candidates = []
                for n in (2, 3, 4, 5, 7, 10):
                    candidates.append(1.0 / n)  # forward split (price drops)
                    candidates.append(float(n))  # reverse split (price jumps)
                nearest = min(candidates, key=lambda c: abs(c - ratio))
                if abs(nearest - ratio) / nearest <= 0.10:
                    dates.append(pd.Timestamp(date))
            if dates:
                flagged[str(symbol)] = dates
        return flagged
