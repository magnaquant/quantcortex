"""Microsoft qlib Alpha158 feature set, reimplemented from scratch.

This module provides :class:`Alpha158`, a pure ``pandas``/``numpy`` implementation
of the canonical `qlib <https://github.com/microsoft/qlib>`_ ``Alpha158`` handler.
No qlib dependency is required.

Every value at bar ``t`` uses data no later than ``t``. A close-derived feature
is therefore available only after that bar closes and must be executed on a
later bar; the feature code alone does not enforce execution timing.

The feature families and their naming follow qlib conventions:

* ``KBAR`` (9 features, windowless) -- candle geometry.
* ``PRICE`` (4 features, windowless) -- normalized open/high/low/vwap vs close.
* ``ROLLING`` families over each window (ROC, MA, STD, BETA, RSQR, RESI, MAX,
  MIN, QTLU, QTLD, RANK, RSV, IMAX, IMIN, IMXD, CORR, CORD, CNTP, CNTN, CNTD,
  SUMP, SUMN, SUMD, VMA, VSTD, WVMA, VSUMP, VSUMN, VSUMD).

With the default five windows ``(5, 10, 20, 30, 60)`` this yields
``9 + 4 + 29 * 5 = 158`` features, matching qlib's count exactly.

Parity notes
------------
* ``IMAX``/``IMIN`` follow qlib's 1-based convention: the feature is
  ``(argmax + 1) / w`` (resp. ``argmin``), where ``argmax`` is the 0-based
  position of the extremum within the trailing window (oldest = 0).
* Rolling features here require a *full* window (``min_periods=w``) whereas
  qlib uses ``min_periods=1``. This is a deliberate stricter-warmup deviation:
  partially-filled windows are reported as ``NaN`` instead of being computed
  on fewer observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["Alpha158"]

# Small constant used to avoid division by zero, matching qlib's convention.
_EPS: float = 1e-12


class Alpha158:
    """Compute the qlib Alpha158 feature set for a single instrument.

    Parameters
    ----------
    windows:
        Lookback window sizes (in bars) used by the rolling feature families.
        Defaults to ``(5, 10, 20, 30, 60)`` which reproduces qlib's 158-feature
        layout.

    Notes
    -----
    The instance is stateless across :meth:`compute` calls; ``windows`` is the
    only configuration held on the object.
    """

    # Rolling family suffixes that depend on a window. Used for name generation.
    _ROLLING_FAMILIES: tuple[str, ...] = (
        "ROC",
        "MA",
        "STD",
        "BETA",
        "RSQR",
        "RESI",
        "MAX",
        "MIN",
        "QTLU",
        "QTLD",
        "RANK",
        "RSV",
        "IMAX",
        "IMIN",
        "IMXD",
        "CORR",
        "CORD",
        "CNTP",
        "CNTN",
        "CNTD",
        "SUMP",
        "SUMN",
        "SUMD",
        "VMA",
        "VSTD",
        "WVMA",
        "VSUMP",
        "VSUMN",
        "VSUMD",
    )

    _KBAR_NAMES: tuple[str, ...] = (
        "KMID",
        "KLEN",
        "KMID2",
        "KUP",
        "KUP2",
        "KLOW",
        "KLOW2",
        "KSFT",
        "KSFT2",
    )

    _PRICE_NAMES: tuple[str, ...] = ("OPEN0", "HIGH0", "LOW0", "VWAP0")

    def __init__(self, windows: tuple[int, ...] = (5, 10, 20, 30, 60)) -> None:
        if not windows:
            raise ValueError("`windows` must contain at least one window size.")
        if any((not isinstance(w, (int, np.integer))) or w < 2 for w in windows):
            raise ValueError("All windows must be integers >= 2.")
        if len(windows) != len(set(windows)):
            raise ValueError("window sizes must be unique")
        self.windows: tuple[int, ...] = tuple(int(w) for w in windows)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def feature_names(self) -> list[str]:
        """Return the ordered list of feature column names produced by ``compute``."""
        names: list[str] = list(self._KBAR_NAMES)
        names.extend(self._PRICE_NAMES)
        for fam in self._ROLLING_FAMILIES:
            for w in self.windows:
                names.append(f"{fam}{w}")
        return names

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute the Alpha158 feature matrix for a single symbol.

        Parameters
        ----------
        ohlcv:
            Price/volume history for a single instrument, indexed by a
            :class:`~pandas.DatetimeIndex` and containing the lowercase columns
            ``open``, ``high``, ``low``, ``close`` and ``volume``. An optional
            ``vwap`` column is used if present; otherwise VWAP is approximated as
            ``(high + low + close) / 3``.

        Returns
        -------
        pandas.DataFrame
            A frame indexed identically to ``ohlcv`` whose columns are the 158
            Alpha158 features (see :meth:`feature_names`). Infinities are
            replaced with ``NaN``. Warmup rows (shorter than the relevant
            window) contain ``NaN`` and are intentionally **not** filled, so the
            causal/leakage boundary is preserved.
        """
        if not isinstance(ohlcv, pd.DataFrame):
            raise TypeError("ohlcv must be a pandas DataFrame")
        if ohlcv.empty:
            raise ValueError("ohlcv must not be empty")
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            raise TypeError("ohlcv must use a DatetimeIndex")
        if (
            ohlcv.index.hasnans
            or ohlcv.index.has_duplicates
            or not ohlcv.index.is_monotonic_increasing
        ):
            raise ValueError("ohlcv index must be unique, valid, and increasing")
        if ohlcv.columns.has_duplicates:
            raise ValueError("ohlcv columns must be unique")
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(ohlcv.columns)
        if missing:
            raise ValueError(f"`ohlcv` is missing required columns: {sorted(missing)}")

        # Work on float64 copies to avoid mutating the caller's frame and to keep
        # numerical stability in rolling sums.
        columns = ["open", "high", "low", "close", "volume"]
        if "vwap" in ohlcv.columns:
            columns.append("vwap")
        numeric = ohlcv.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("ohlcv inputs must be finite and complete")
        if (numeric[["open", "high", "low", "close"]] <= 0.0).any(axis=None):
            raise ValueError("OHLC prices must be strictly positive")
        if (numeric["volume"] < 0.0).any():
            raise ValueError("volume must be non-negative")
        if (numeric["high"] < numeric[["open", "low", "close"]].max(axis=1)).any():
            raise ValueError("high is below another OHLC field")
        if (numeric["low"] > numeric[["open", "high", "close"]].min(axis=1)).any():
            raise ValueError("low is above another OHLC field")
        if "vwap" in numeric and (numeric["vwap"] <= 0.0).any():
            raise ValueError("vwap must be strictly positive")

        open_ = numeric["open"].astype("float64")
        high = numeric["high"].astype("float64")
        low = numeric["low"].astype("float64")
        close = numeric["close"].astype("float64")
        volume = numeric["volume"].astype("float64")
        if "vwap" in ohlcv.columns:
            vwap = numeric["vwap"].astype("float64")
        else:
            vwap = (high + low + close) / 3.0

        features: dict[str, pd.Series] = {}

        self._add_kbar(features, open_, high, low, close)
        self._add_price(features, open_, high, low, close, vwap)
        self._add_rolling(features, high, low, close, volume)

        result = pd.DataFrame(features, index=ohlcv.index)
        # Preserve canonical ordering.
        result = result[self.feature_names()]

        # Replace +/- inf produced by extreme ratios with NaN. Do NOT fill: the
        # NaN warmup region is the causal boundary and must stay empty.
        result = result.replace([np.inf, -np.inf], np.nan)
        return result

    # ------------------------------------------------------------------ #
    # KBAR family (9 features, windowless)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _add_kbar(
        features: dict[str, pd.Series],
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> None:
        hl = high - low
        max_oc = np.maximum(open_, close)
        min_oc = np.minimum(open_, close)

        features["KMID"] = (close - open_) / open_
        features["KLEN"] = hl / open_
        features["KMID2"] = (close - open_) / (hl + _EPS)
        features["KUP"] = (high - max_oc) / open_
        features["KUP2"] = (high - max_oc) / (hl + _EPS)
        features["KLOW"] = (min_oc - low) / open_
        features["KLOW2"] = (min_oc - low) / (hl + _EPS)
        features["KSFT"] = (2.0 * close - high - low) / open_
        features["KSFT2"] = (2.0 * close - high - low) / (hl + _EPS)

    # ------------------------------------------------------------------ #
    # PRICE family (4 features, windowless)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _add_price(
        features: dict[str, pd.Series],
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        vwap: pd.Series,
    ) -> None:
        features["OPEN0"] = open_ / close
        features["HIGH0"] = high / close
        features["LOW0"] = low / close
        features["VWAP0"] = vwap / close

    # ------------------------------------------------------------------ #
    # ROLLING families
    # ------------------------------------------------------------------ #
    def _add_rolling(
        self,
        features: dict[str, pd.Series],
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> None:
        # Pre-compute series reused across windows.
        close_prev = close.shift(1)
        price_chg = close - close_prev  # close - Ref(close, 1)
        abs_price_chg = price_chg.abs()
        up_move = price_chg.clip(lower=0.0)  # max(diff, 0)
        down_move = (-price_chg).clip(lower=0.0)  # max(-diff, 0)

        log_volume = np.log(volume + 1.0)
        close_ret = close / close_prev  # for CORD
        log_vol_ret = np.log(volume / volume.shift(1) + 1.0)  # for CORD

        vol_prev = volume.shift(1)
        vol_chg = volume - vol_prev
        abs_vol_chg = vol_chg.abs()
        vol_up = vol_chg.clip(lower=0.0)
        vol_down = (-vol_chg).clip(lower=0.0)

        # |pct change of close| * volume, used by WVMA (volume-weighted volatility).
        wv = (close / close_prev - 1.0).abs() * volume

        for w in self.windows:
            # --- price level features -------------------------------------
            features[f"ROC{w}"] = close.shift(w) / close
            features[f"MA{w}"] = close.rolling(w).mean() / close
            features[f"STD{w}"] = close.rolling(w).std() / close

            beta, rsqr, resi = self._rolling_regression(close, w)
            features[f"BETA{w}"] = beta / close
            features[f"RSQR{w}"] = rsqr
            features[f"RESI{w}"] = resi / close

            features[f"MAX{w}"] = high.rolling(w).max() / close
            features[f"MIN{w}"] = low.rolling(w).min() / close
            features[f"QTLU{w}"] = close.rolling(w).quantile(0.8) / close
            features[f"QTLD{w}"] = close.rolling(w).quantile(0.2) / close
            features[f"RANK{w}"] = self._rolling_rank(close, w)

            roll_low = low.rolling(w).min()
            roll_high = high.rolling(w).max()
            features[f"RSV{w}"] = (close - roll_low) / (roll_high - roll_low + _EPS)

            # qlib parity: 1-based position, i.e. (argmax + 1) / w.
            imax = (self._rolling_idxmax(high, w) + 1.0) / w
            imin = (self._rolling_idxmin(low, w) + 1.0) / w
            features[f"IMAX{w}"] = imax
            features[f"IMIN{w}"] = imin
            features[f"IMXD{w}"] = imax - imin

            features[f"CORR{w}"] = self._rolling_corr(close, log_volume, w)
            features[f"CORD{w}"] = self._rolling_corr(close_ret, log_vol_ret, w)

            # --- up/down day counts ---------------------------------------
            comparison_valid = close_prev.notna()
            cntp = (
                (close > close_prev)
                .astype("float64")
                .where(comparison_valid)
                .rolling(w)
                .mean()
            )
            cntn = (
                (close < close_prev)
                .astype("float64")
                .where(comparison_valid)
                .rolling(w)
                .mean()
            )
            features[f"CNTP{w}"] = cntp
            features[f"CNTN{w}"] = cntn
            features[f"CNTD{w}"] = cntp - cntn

            # --- up/down move sums ----------------------------------------
            sum_abs = abs_price_chg.rolling(w).sum()
            sump = up_move.rolling(w).sum() / (sum_abs + _EPS)
            sumn = down_move.rolling(w).sum() / (sum_abs + _EPS)
            features[f"SUMP{w}"] = sump
            features[f"SUMN{w}"] = sumn
            features[f"SUMD{w}"] = sump - sumn

            # --- volume features ------------------------------------------
            features[f"VMA{w}"] = volume.rolling(w).mean() / (volume + _EPS)
            features[f"VSTD{w}"] = volume.rolling(w).std() / (volume + _EPS)
            features[f"WVMA{w}"] = wv.rolling(w).std() / (wv.rolling(w).mean() + _EPS)

            vol_sum_abs = abs_vol_chg.rolling(w).sum()
            vsump = vol_up.rolling(w).sum() / (vol_sum_abs + _EPS)
            vsumn = vol_down.rolling(w).sum() / (vol_sum_abs + _EPS)
            features[f"VSUMP{w}"] = vsump
            features[f"VSUMN{w}"] = vsumn
            features[f"VSUMD{w}"] = vsump - vsumn

    # ------------------------------------------------------------------ #
    # Rolling helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rolling_regression(
        series: pd.Series, window: int
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Causal rolling OLS of ``series`` on an integer time index.

        For each bar ``t`` a regression ``y = a + b * x`` is fit over the trailing
        ``window`` observations, where ``x = 0, 1, ..., window - 1`` (oldest to
        newest). Returns ``(slope, r_squared, residual_at_last_point)``.

        The slope/intercept are computed in closed form via rolling sums (no
        per-row Python loop), giving an O(n) vectorized implementation.
        """
        n = float(window)
        # Independent variable: fixed positions within the window.
        x = np.arange(window, dtype="float64")
        sum_x = x.sum()
        sum_x2 = (x * x).sum()
        denom = n * sum_x2 - sum_x * sum_x  # variance term of x (* n)

        roll = series.rolling(window)
        sum_y = roll.sum()
        # sum of x_i * y_i over the window: weighted rolling sum.
        sum_xy = series.rolling(window).apply(
            lambda arr: float(np.dot(x, arr)), raw=True
        )
        sum_y2 = (series * series).rolling(window).sum()

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        # Predicted value at the last (most recent) point: x = window - 1.
        x_last = n - 1.0
        y_pred_last = intercept + slope * x_last
        residual = series - y_pred_last

        # R^2 = 1 - SS_res / SS_tot, computed from rolling moments.
        # SS_tot = sum_y2 - (sum_y^2)/n
        # SS_res = sum((y - (a + b x))^2)
        #        = sum_y2 - 2a*sum_y - 2b*sum_xy + n*a^2 + 2ab*sum_x + b^2*sum_x2
        ss_tot = sum_y2 - (sum_y * sum_y) / n
        ss_res = (
            sum_y2
            - 2.0 * intercept * sum_y
            - 2.0 * slope * sum_xy
            + n * intercept * intercept
            + 2.0 * intercept * slope * sum_x
            + slope * slope * sum_x2
        )
        # Guard against tiny negative values from floating point cancellation.
        ss_res = ss_res.clip(lower=0.0)
        rsqr = 1.0 - ss_res / (ss_tot + _EPS)
        # A zero-variance window has no defined R^2 (the regression explains
        # nothing of nothing); mask it to NaN instead of reporting 1.0.
        rsqr = rsqr.where(ss_tot > _EPS)

        return slope, rsqr, residual

    @staticmethod
    def _rolling_rank(series: pd.Series, window: int) -> pd.Series:
        """Trailing time-series percentile rank of the current value within ``window``.

        Returns, for each bar, the fraction of the trailing ``window`` observations
        (including the current bar) that are less than or equal to the current
        value. Equivalent to qlib's ``Rank`` operator.
        """

        def _rank(arr: np.ndarray) -> float:
            current = arr[-1]
            return float((arr <= current).sum()) / float(len(arr))

        return series.rolling(window).apply(_rank, raw=True)

    @staticmethod
    def _rolling_idxmax(series: pd.Series, window: int) -> pd.Series:
        """Position (0..window-1, oldest..newest) of the max within the trailing window."""
        return series.rolling(window).apply(
            lambda arr: float(np.argmax(arr)), raw=True
        )

    @staticmethod
    def _rolling_idxmin(series: pd.Series, window: int) -> pd.Series:
        """Position (0..window-1, oldest..newest) of the min within the trailing window."""
        return series.rolling(window).apply(
            lambda arr: float(np.argmin(arr)), raw=True
        )

    @staticmethod
    def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
        """Causal rolling Pearson correlation between two aligned series."""
        return a.rolling(window).corr(b)


# ---------------------------------------------------------------------- #
# Feature count sanity check (qlib Alpha158 == 158 with 5 windows):
#   KBAR (9) + PRICE (4) + 29 rolling families * 5 windows (145) == 158
# ---------------------------------------------------------------------- #
if 9 + 4 + len(Alpha158._ROLLING_FAMILIES) * 5 != 158:
    raise RuntimeError("Alpha158 default layout must produce exactly 158 features")


def _self_test() -> None:
    """Build a synthetic random-walk OHLCV frame and validate :class:`Alpha158`."""
    rng = np.random.default_rng(42)
    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rets = rng.normal(0.0, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    spread = np.abs(rng.normal(0.0, 0.5, size=n)) + 0.1
    open_ = close * (1.0 + rng.normal(0.0, 0.003, size=n))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(1_000, 10_000, size=n).astype("float64")

    ohlcv = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )

    model = Alpha158()
    feats = model.compute(ohlcv)

    if list(feats.columns) != model.feature_names():
        raise RuntimeError("Alpha158 self-test found a column-order mismatch")
    if len(feats.columns) != 158:
        raise RuntimeError(
            f"Alpha158 self-test expected 158 features, got {len(feats.columns)}"
        )

    # After the warmup (longest window), no column should be entirely NaN.
    warmup = max(model.windows) + 2
    tail = feats.iloc[warmup:]
    all_nan = [c for c in tail.columns if tail[c].isna().all()]
    if all_nan:
        raise RuntimeError(f"Alpha158 columns entirely NaN after warmup: {all_nan}")

    # No infinities should remain.
    if np.isinf(feats.to_numpy(dtype="float64")).any():
        raise RuntimeError("Alpha158 self-test found infinite values")

    print(f"Alpha158 self-test passed: {len(feats.columns)} features.")


if __name__ == "__main__":
    _self_test()
