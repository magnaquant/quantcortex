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

    @staticmethod
    def _normalize_actions(
        actions: Optional[pd.Series], name: str, *, strictly_positive: bool
    ) -> Optional[pd.Series]:
        if actions is None or len(actions) == 0:
            return None
        series = pd.Series(actions).copy()
        index = pd.to_datetime(series.index, errors="coerce", utc=True)
        if index.isna().any():
            raise AdjustmentError(f"{name} contains invalid event dates")
        series.index = index.tz_localize(None)
        if series.index.has_duplicates:
            raise AdjustmentError(f"{name} contains duplicate event dates")
        series = pd.to_numeric(series, errors="coerce")
        if series.isna().any() or not np.all(np.isfinite(series.to_numpy())):
            raise AdjustmentError(f"{name} must contain only finite numeric values")
        if strictly_positive and (series <= 0.0).any():
            raise AdjustmentError(f"{name} ratios must be strictly positive")
        if not strictly_positive and (series < 0.0).any():
            raise AdjustmentError(f"{name} amounts must be non-negative")
        return series.sort_index()

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
        splits = self._normalize_actions(splits, "splits", strictly_positive=True)
        if splits is None:
            return factor
        splits = splits[splits != 1.0]
        for split_date, ratio in splits.items():
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
        dividends = self._normalize_actions(
            dividends, "dividends", strictly_positive=False
        )
        if dividends is None:
            return factor
        dividends = dividends[dividends != 0.0]
        for ex_date, amount in dividends.items():
            ex_ts = pd.Timestamp(ex_date)
            prior = close[close.index < ex_ts]
            if len(prior) == 0:
                raise AdjustmentError(
                    f"dividend on {ex_ts.date()} has no prior close in the input"
                )
            prior_close = float(prior.iloc[-1])
            if not np.isfinite(prior_close) or prior_close <= 0:
                raise AdjustmentError("dividend prior close must be finite and positive")
            ratio = 1.0 - float(amount) / prior_close
            if ratio <= 0:
                raise AdjustmentError(
                    f"dividend on {ex_ts.date()} is not smaller than prior close"
                )
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
        if not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty:
            raise AdjustmentError("ohlcv must be a non-empty DataFrame")
        df = ohlcv.copy()
        index = pd.to_datetime(df.index, errors="coerce", utc=True)
        if index.isna().any():
            raise AdjustmentError("ohlcv contains invalid timestamps")
        df.index = index.tz_localize(None)
        df = df.sort_index()
        if df.index.has_duplicates:
            raise AdjustmentError("ohlcv contains duplicate timestamps")
        index = df.index

        if "close" not in df.columns:
            raise AdjustmentError("apply_adjustments requires a 'close' column")
        numeric_cols = [
            col for col in (*self._PRICE_COLS, *self._VOLUME_COLS) if col in df.columns
        ]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        if df[numeric_cols].isna().any().any() or not np.all(
            np.isfinite(df[numeric_cols].to_numpy(dtype=float))
        ):
            raise AdjustmentError("ohlcv numeric fields must be finite and complete")
        for col in self._PRICE_COLS:
            if col in df.columns and (df[col] <= 0.0).any():
                raise AdjustmentError(f"{col} prices must be strictly positive")
        if "volume" in df.columns and (df["volume"] < 0.0).any():
            raise AdjustmentError("volume must be non-negative")
        if {"high", "low"} <= set(df.columns):
            comparable = [col for col in ("open", "low", "close") if col in df.columns]
            if (df["high"] < df[comparable].max(axis=1)).any():
                raise AdjustmentError("high is below another observed price field")
            comparable = [col for col in ("open", "high", "close") if col in df.columns]
            if (df["low"] > df[comparable].min(axis=1)).any():
                raise AdjustmentError("low is above another observed price field")
        raw_close = df["close"].copy()

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
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be finite and non-negative")
        raw = pd.to_numeric(pd.Series(raw_close).copy(), errors="coerce")
        adj = pd.to_numeric(pd.Series(adj_close).copy(), errors="coerce")
        raw.index = pd.to_datetime(raw.index, errors="coerce", utc=True).tz_localize(
            None
        )
        adj.index = pd.to_datetime(adj.index, errors="coerce", utc=True).tz_localize(
            None
        )
        if raw.index.isna().any() or adj.index.isna().any():
            raise AdjustmentError("raw_close and adj_close require valid timestamps")
        if raw.index.has_duplicates or adj.index.has_duplicates:
            raise AdjustmentError("raw_close and adj_close require unique timestamps")
        raw = raw.sort_index()
        adj = adj.sort_index()
        if not raw.index.equals(adj.index):
            raise AdjustmentError(
                "raw_close and adj_close must have identical timestamp coverage"
            )
        common = raw.index
        if len(common) == 0:
            raise AdjustmentError("raw_close and adj_close must not be empty")

        with np.errstate(divide="ignore", invalid="ignore"):
            implied = adj / raw
        if (
            implied.isna().any()
            or not np.all(np.isfinite(implied.to_numpy()))
            or (raw <= 0).any()
            or (adj <= 0).any()
        ):
            raise AdjustmentError("raw and adjusted closes must be finite and positive")

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
        if not isinstance(prices, pd.DataFrame) or prices.empty:
            raise ValueError("prices must be a non-empty DataFrame")
        if prices.columns.has_duplicates or prices.shape[1] == 0:
            raise ValueError("prices must have unique, non-empty columns")
        df = prices.copy()
        index = pd.to_datetime(df.index, errors="coerce", utc=True)
        if index.isna().any():
            raise ValueError("prices contain invalid timestamps")
        df.index = index.tz_localize(None)
        df = df.sort_index()
        if df.index.has_duplicates:
            raise ValueError("prices timestamps must be unique")
        if not np.isfinite(jump_threshold) or jump_threshold <= 0.0:
            raise ValueError("jump_threshold must be finite and positive")

        df = df.apply(pd.to_numeric, errors="coerce")
        values = df.to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("prices must not contain infinite values")
        if (df.notna() & (df <= 0.0)).any(axis=None):
            raise ValueError("observed prices must be strictly positive")

        flagged: Dict[str, List[pd.Timestamp]] = {}
        for symbol in df.columns:
            series = df[symbol]
            rets = series.pct_change(fill_method=None)
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
