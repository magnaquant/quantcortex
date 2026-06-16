"""Abstract investable-universe interface with point-in-time membership.

Survivorship bias is one of the seven enforced pitfalls: a backtest that uses
*today's* index constituents for *all* history silently deletes every company
that went bankrupt or was delisted, inflating returns.  Every universe here is
therefore queried **as of a date** - :meth:`Universe.constituents` returns the
membership that was true on that date, not the membership today.
"""

from __future__ import annotations

import abc
from numbers import Number
from typing import List, Optional, Union

import pandas as pd

__all__ = ["Universe", "PITMembership"]

DateLike = Union[str, pd.Timestamp]


class PITMembership:
    """A point-in-time membership table.

    Holds rows of ``(symbol, start_date, end_date)`` where ``end_date`` may be
    ``NaT`` for currently-active members. Intervals are half-open:
    ``[start_date, end_date)``. A constituent removed effective on date ``d``
    is therefore not a member on ``d``.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        required = {"symbol", "start_date", "end_date"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"PITMembership missing columns: {missing}")
        if frame.empty:
            raise ValueError("PITMembership requires at least one interval")
        df = frame.copy()
        if any(not isinstance(value, str) for value in df["symbol"]):
            raise ValueError("PITMembership symbols must be strings")
        for column in ("start_date", "end_date"):
            if any(
                isinstance(value, Number) and not pd.isna(value)
                for value in df[column]
            ):
                raise ValueError(
                    f"PITMembership {column} values must not be numeric epochs"
                )
        df["symbol"] = (
            df["symbol"]
            .astype("string")
            .str.strip()
            .str.upper()
            .str.replace(".", "-", regex=False)
        )
        df["start_date"] = pd.to_datetime(df["start_date"], utc=True).dt.tz_localize(
            None
        )
        df["end_date"] = pd.to_datetime(df["end_date"], utc=True).dt.tz_localize(
            None
        )
        if df["symbol"].isna().any() or (df["symbol"] == "").any():
            raise ValueError("PITMembership symbols must be non-empty strings")
        if df["start_date"].isna().any():
            raise ValueError("PITMembership start_date values must be valid")
        bad_interval = df["end_date"].notna() & (
            df["end_date"] <= df["start_date"]
        )
        if bad_interval.any():
            raise ValueError("PITMembership requires start_date < end_date")
        if df.duplicated(["symbol", "start_date", "end_date"]).any():
            raise ValueError("PITMembership contains duplicate intervals")

        df = df.sort_values(["symbol", "start_date", "end_date"], na_position="last")
        for symbol, group in df.groupby("symbol", sort=False):
            previous_end = None
            for row in group.itertuples(index=False):
                if previous_end is None:
                    previous_end = row.end_date
                    continue
                if pd.isna(previous_end) or row.start_date < previous_end:
                    raise ValueError(
                        f"PITMembership contains overlapping intervals for {symbol!r}"
                    )
                previous_end = row.end_date
        self._frame = df.reset_index(drop=True)
        self._coverage_start = pd.Timestamp(self._frame["start_date"].min())

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy()

    @property
    def coverage_start(self) -> pd.Timestamp:
        """Earliest date for which this membership table has any coverage."""
        return self._coverage_start

    def members_asof(self, as_of: DateLike) -> List[str]:
        if isinstance(as_of, (bool, Number)):
            raise ValueError("as_of must be a date-like scalar, not a numeric epoch")
        ts = pd.to_datetime(as_of, errors="coerce", utc=True)
        if not isinstance(ts, pd.Timestamp) or pd.isna(ts):
            raise ValueError("as_of must be a valid timestamp")
        ts = ts.tz_localize(None)
        if ts < self._coverage_start:
            raise ValueError(
                f"as_of predates membership coverage starting "
                f"{self._coverage_start.date()}"
            )
        df = self._frame
        active = (df["start_date"] <= ts) & (
            df["end_date"].isna() | (df["end_date"] > ts)
        )
        return sorted(df.loc[active, "symbol"].unique().tolist())

    def is_member(self, symbol: str, as_of: DateLike) -> bool:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        return symbol.strip().upper().replace(".", "-") in set(
            self.members_asof(as_of)
        )

    def all_symbols(self) -> List[str]:
        return sorted(self._frame["symbol"].unique().tolist())


class Universe(abc.ABC):
    """Abstract investable universe."""

    name: str = "base"

    @abc.abstractmethod
    def membership(self) -> PITMembership:
        """Return the point-in-time membership table for this universe."""
        raise NotImplementedError

    def constituents(self, as_of: Optional[DateLike] = None) -> List[str]:
        """Symbols that were members ``as_of`` the given date (default: now)."""
        as_of = (
            as_of
            if as_of is not None
            else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        )
        return self.membership().members_asof(as_of)

    def is_member(self, symbol: str, as_of: DateLike) -> bool:
        return self.membership().is_member(symbol, as_of)

    def all_symbols(self) -> List[str]:
        """Every symbol represented in the membership history."""
        return self.membership().all_symbols()
