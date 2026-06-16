"""Validation helpers for explicit cash-account return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["align_cash_returns"]


def align_cash_returns(
    cash_returns: pd.Series | None,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Align per-period simple cash returns to a backtest price index.

    ``cash_returns[t]`` is the return earned by the cash account from the
    preceding bar close through bar ``t``. ``None`` means a zero-return cash
    account. Missing observations fail closed rather than being filled
    implicitly.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("cash-return alignment requires a DatetimeIndex")
    if index.hasnans or index.has_duplicates:
        raise ValueError("cash-return target index must be unique and valid")
    if cash_returns is None:
        return pd.Series(0.0, index=index, name="cash_return", dtype="float64")
    if not isinstance(cash_returns, pd.Series):
        raise TypeError("cash_returns must be a pandas Series or None")
    if not isinstance(cash_returns.index, pd.DatetimeIndex):
        raise TypeError("cash_returns must use a DatetimeIndex")
    if cash_returns.index.hasnans or cash_returns.index.has_duplicates:
        raise ValueError("cash_returns index must contain unique valid timestamps")

    series = cash_returns.copy()
    if series.index.tz is not None:
        series.index = series.index.tz_convert("UTC").tz_localize(None)
    series = pd.to_numeric(series.sort_index(), errors="coerce").astype("float64")
    aligned = series.reindex(index)
    if aligned.isna().any():
        missing = aligned.index[aligned.isna()]
        preview = ", ".join(timestamp.strftime("%Y-%m-%d") for timestamp in missing[:3])
        raise ValueError(
            "cash_returns must cover every price bar without missing values"
            + (f"; first missing dates: {preview}" if preview else "")
        )
    values = aligned.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("cash_returns must contain only finite values")
    if np.any(values <= -1.0):
        raise ValueError("cash_returns must be greater than -100%")
    aligned.name = cash_returns.name or "cash_return"
    return aligned
