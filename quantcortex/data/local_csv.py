"""Validated loaders for owner-supplied local market data."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from quantcortex.data.providers.base import OHLCV_COLUMNS

__all__ = [
    "LocalDataError",
    "load_ohlcv_csv",
    "load_price_matrix",
    "sha256_file",
]


class LocalDataError(ValueError):
    """Raised when a local market-data file violates the expected schema."""


def _read_dated_csv(path: str | Path) -> tuple[Path, pd.DataFrame]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise LocalDataError(f"data file does not exist: {resolved}")

    try:
        with resolved.open(encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
    except StopIteration as exc:
        raise LocalDataError(f"CSV is empty: {resolved}") from exc
    except Exception as exc:
        raise LocalDataError(f"could not read CSV header {resolved}: {exc}") from exc
    if len(header) != len(set(header)):
        raise LocalDataError(f"duplicate CSV columns in {resolved}")

    try:
        frame = pd.read_csv(resolved)
    except Exception as exc:
        raise LocalDataError(f"could not read CSV {resolved}: {exc}") from exc

    if "date" not in frame.columns:
        raise LocalDataError(f"CSV must contain a 'date' column: {resolved}")

    try:
        dates = pd.to_datetime(frame.pop("date"), errors="raise", utc=True)
    except Exception as exc:
        raise LocalDataError(f"invalid date values in {resolved}: {exc}") from exc

    index = pd.DatetimeIndex(dates.dt.tz_convert(None), name="date")
    if index.hasnans:
        raise LocalDataError(f"empty date values in {resolved}")
    if index.has_duplicates:
        duplicates = index[index.duplicated()].unique().strftime("%Y-%m-%d").tolist()
        raise LocalDataError(f"duplicate dates in {resolved}: {duplicates}")

    frame.index = index
    return resolved, frame.sort_index()


def _slice(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start is not None:
        parsed_start = pd.to_datetime(start, errors="coerce", utc=True)
        if pd.isna(parsed_start):
            raise LocalDataError("start must be a valid timestamp")
        frame = frame.loc[frame.index >= parsed_start.tz_localize(None)]
    if end is not None:
        parsed_end = pd.to_datetime(end, errors="coerce", utc=True)
        if pd.isna(parsed_end):
            raise LocalDataError("end must be a valid timestamp")
        frame = frame.loc[frame.index <= parsed_end.tz_localize(None)]
    if frame.empty:
        raise LocalDataError("no rows remain after applying the requested date window")
    return frame


def _numeric(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    try:
        numeric = frame.apply(pd.to_numeric, errors="raise").astype("float64")
    except Exception as exc:
        raise LocalDataError(f"non-numeric market data in {path}: {exc}") from exc
    if np.isinf(numeric.to_numpy()).any():
        raise LocalDataError(f"infinite market data value in {path}")
    return numeric


def load_price_matrix(
    path: str | Path,
    symbols: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    max_ffill: int | None = 5,
) -> pd.DataFrame:
    """Load a wide adjusted-close CSV indexed by a required ``date`` column.

    Missing prices are forward-filled for at most ``max_ffill`` rows. Set it to
    ``None`` to disable filling; unlimited filling is intentionally unsupported
    because it can keep stale or delisted assets alive indefinitely.
    """
    resolved, frame = _read_dated_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.has_duplicates or any(not column for column in frame.columns):
        raise LocalDataError(f"symbol columns must be unique and non-empty in {resolved}")

    if symbols is not None:
        if isinstance(symbols, str):
            raise LocalDataError("symbols must be a sequence, not a string")
        required = list(symbols)
        if any(
            not isinstance(symbol, str) or not symbol.strip() for symbol in required
        ) or len(required) != len(set(required)):
            raise LocalDataError("symbols must contain unique non-empty strings")
        required = [symbol.strip() for symbol in required]
        missing = [symbol for symbol in required if symbol not in frame.columns]
        if missing:
            raise LocalDataError(f"missing symbol columns in {resolved}: {missing}")
        frame = frame.loc[:, required]
    if frame.shape[1] == 0:
        raise LocalDataError(f"price CSV has no symbol columns: {resolved}")

    frame = _numeric(frame, resolved)
    if (frame <= 0).any(axis=None):
        raise LocalDataError(f"prices must be strictly positive in {resolved}")

    if max_ffill is not None:
        if (
            isinstance(max_ffill, bool)
            or int(max_ffill) != max_ffill
            or max_ffill < 0
        ):
            raise LocalDataError("max_ffill must be a non-negative integer or None")
        frame = frame.ffill(limit=int(max_ffill)) if max_ffill > 0 else frame
    frame = _slice(frame, start, end).dropna(how="any")
    if frame.empty:
        raise LocalDataError("no complete price rows remain after forward-filling")
    return frame


def load_ohlcv_csv(
    path: str | Path,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load a single-symbol OHLCV CSV in the canonical provider schema."""
    resolved, frame = _read_dated_csv(path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if frame.columns.has_duplicates:
        raise LocalDataError(f"duplicate OHLCV columns in {resolved}")

    missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise LocalDataError(f"missing OHLCV columns in {resolved}: {missing}")

    frame = _numeric(frame.loc[:, OHLCV_COLUMNS], resolved)
    if frame.isna().any(axis=None):
        raise LocalDataError(f"OHLCV values may not be empty in {resolved}")
    if (frame[["open", "high", "low", "close", "adj_close"]] <= 0).any(axis=None):
        raise LocalDataError(f"OHLCV prices must be strictly positive in {resolved}")
    if (frame["volume"] < 0).any():
        raise LocalDataError(f"OHLCV volume must be non-negative in {resolved}")
    if (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any():
        raise LocalDataError(f"OHLCV high is below another price field in {resolved}")
    if (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).any():
        raise LocalDataError(f"OHLCV low is above another price field in {resolved}")

    return _slice(frame, start, end)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
