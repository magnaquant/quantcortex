"""Validated point-in-time transforms for quarterly fundamental data."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from quantcortex.data.providers.base import canonical_fundamental_field

_REQUIRED_COLUMNS = ("symbol", "period_end", "announcement_date", "field", "value")


def validate_fundamentals(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Return a canonical, validated copy of a tidy fundamentals frame."""
    if not isinstance(fundamentals, pd.DataFrame):
        raise TypeError("fundamentals must be a pandas DataFrame")
    missing = [col for col in _REQUIRED_COLUMNS if col not in fundamentals.columns]
    if missing:
        raise ValueError(f"fundamentals missing required columns: {missing}")
    if fundamentals.empty:
        raise ValueError("fundamentals must not be empty")

    frame = fundamentals.loc[:, _REQUIRED_COLUMNS].copy()
    for column in ("symbol", "field"):
        invalid = frame[column].isna() | frame[column].astype(str).str.strip().eq("")
        if invalid.any():
            raise ValueError(f"fundamentals contain invalid {column} values")
    frame["symbol"] = frame["symbol"].astype(str).str.strip()
    frame["field"] = frame["field"].map(canonical_fundamental_field)

    for column in ("period_end", "announcement_date"):
        parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
        if parsed.isna().any():
            raise ValueError(f"fundamentals contain invalid {column} values")
        frame[column] = parsed.dt.tz_localize(None)

    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    values = frame["value"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("fundamental values must be finite numbers")
    if (frame["announcement_date"] < frame["period_end"]).any():
        raise ValueError("announcement_date must not precede period_end")

    keys = ["symbol", "field", "period_end", "announcement_date"]
    if frame.duplicated(keys).any():
        raise ValueError("fundamentals contain duplicate report observations")
    return frame.sort_values(
        ["announcement_date", "symbol", "field", "period_end"],
        kind="stable",
    ).reset_index(drop=True)


def panel_axes(
    fundamentals: pd.DataFrame,
    index: pd.Index | None = None,
    columns: pd.Index | None = None,
    *,
    allow_exact_matches: bool = False,
) -> tuple[pd.DatetimeIndex, pd.Index]:
    """Resolve and validate output axes for a factor panel."""
    if not isinstance(allow_exact_matches, bool):
        raise TypeError("allow_exact_matches must be a boolean")
    if index is None:
        resolved_index = pd.DatetimeIndex(
            sorted(fundamentals["announcement_date"].unique())
        )
        if not allow_exact_matches:
            resolved_index = resolved_index + pd.Timedelta(1, unit="ns")
    else:
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError("factor panel index must be a DatetimeIndex")
        if index.hasnans or index.has_duplicates:
            raise ValueError("factor panel index must contain unique valid dates")
        resolved_index = index
        if resolved_index.tz is not None:
            resolved_index = resolved_index.tz_convert("UTC").tz_localize(None)
        resolved_index = resolved_index.sort_values()

    if columns is None:
        resolved_columns = pd.Index(sorted(fundamentals["symbol"].unique()))
    else:
        if columns.has_duplicates:
            raise ValueError("factor panel columns must be unique")
        if any(not isinstance(col, str) or not col.strip() for col in columns):
            raise ValueError("factor panel columns must be non-empty symbols")
        resolved_columns = pd.Index(columns)
    return resolved_index, resolved_columns


def pit_panel(
    fundamentals: pd.DataFrame,
    field: str,
    index: pd.DatetimeIndex,
    columns: pd.Index,
    *,
    mode: Literal["latest", "ttm", "average_balance"] = "latest",
    allow_exact_matches: bool = False,
) -> pd.DataFrame:
    """Build a PIT panel from report vintages without restatement lookahead.

    Date-only announcement records are available strictly after their timestamp
    by default. Set ``allow_exact_matches=True`` only when the source provides
    an observed release timestamp that is known to precede the feature time.
    """
    if not isinstance(allow_exact_matches, bool):
        raise TypeError("allow_exact_matches must be a boolean")
    canonical_field = canonical_fundamental_field(field)
    subset = fundamentals.loc[fundamentals["field"] == canonical_field]
    if subset.empty:
        return pd.DataFrame(index=index, columns=columns, dtype=float)

    events: dict[str, pd.Series] = {}
    for symbol, symbol_rows in subset.groupby("symbol", sort=False):
        by_period: dict[pd.Timestamp, tuple[pd.Timestamp, float]] = {}
        values: dict[pd.Timestamp, float] = {}
        for announcement, announced_rows in symbol_rows.groupby(
            "announcement_date", sort=True
        ):
            for row in announced_rows.itertuples(index=False):
                current = by_period.get(row.period_end)
                if current is None or row.announcement_date >= current[0]:
                    by_period[row.period_end] = (
                        row.announcement_date,
                        float(row.value),
                    )
            available_at = pd.Timestamp(announcement)
            if not allow_exact_matches:
                available_at += pd.Timedelta(1, unit="ns")
            values[available_at] = _snapshot_value(by_period, mode)
        events[str(symbol)] = pd.Series(values, dtype=float)

    event_frame = pd.DataFrame(events).sort_index()
    full_index = event_frame.index.union(index).sort_values()
    return (
        event_frame.reindex(full_index)
        .ffill()
        .reindex(index=index, columns=columns)
        .astype(float)
    )


def available_mean(panels: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Mean aligned panels while requiring at least one available component."""
    if not panels:
        raise ValueError("at least one factor panel is required")
    total = panels[0].copy()
    count = panels[0].notna().astype(int)
    for panel in panels[1:]:
        total = total.add(panel, fill_value=0.0)
        count = count.add(panel.notna().astype(int), fill_value=0)
    return total.divide(count.where(count > 0))


def _snapshot_value(
    by_period: dict[pd.Timestamp, tuple[pd.Timestamp, float]],
    mode: Literal["latest", "ttm", "average_balance"],
) -> float:
    observations = sorted(
        ((period, value) for period, (_, value) in by_period.items()),
        key=lambda item: item[0],
    )
    if not observations:
        return np.nan
    if mode == "latest":
        return observations[-1][1]
    if mode == "ttm":
        if len(observations) < 4:
            return np.nan
        last_four = observations[-4:]
        periods = pd.DatetimeIndex(period for period, _ in last_four)
        gaps = np.array(
            [(periods[pos] - periods[pos - 1]).days for pos in range(1, 4)]
        )
        if (gaps < 45).any() or (gaps > 150).any():
            return np.nan
        return float(sum(value for _, value in last_four))
    if mode == "average_balance":
        current_period, current_value = observations[-1]
        candidates = [
            (period, value)
            for period, value in observations[:-1]
            if 300 <= (current_period - period).days <= 430
        ]
        if not candidates:
            return np.nan
        prior_value = candidates[-1][1]
        return float((current_value + prior_value) / 2.0)
    raise ValueError(f"unsupported PIT panel mode {mode!r}")
