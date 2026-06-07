"""Abstract investable-universe interface with point-in-time membership.

Survivorship bias is one of the seven enforced pitfalls: a backtest that uses
*today's* index constituents for *all* history silently deletes every company
that went bankrupt or was delisted, inflating returns.  Every universe here is
therefore queried **as of a date** — :meth:`Universe.constituents` returns the
membership that was true on that date, not the membership today.
"""

from __future__ import annotations

import abc
from typing import List, Optional, Union

import pandas as pd

__all__ = ["Universe", "PITMembership"]

DateLike = Union[str, pd.Timestamp]


class PITMembership:
    """A point-in-time membership table.

    Holds rows of ``(symbol, start_date, end_date)`` where ``end_date`` may be
    ``NaT`` for currently-active members, and answers membership queries as of
    any date.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        required = {"symbol", "start_date", "end_date"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"PITMembership missing columns: {missing}")
        df = frame.copy()
        df["start_date"] = pd.to_datetime(df["start_date"])
        df["end_date"] = pd.to_datetime(df["end_date"])
        self._frame = df.reset_index(drop=True)

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    def members_asof(self, as_of: DateLike) -> List[str]:
        ts = pd.Timestamp(as_of)
        df = self._frame
        active = (df["start_date"] <= ts) & (
            df["end_date"].isna() | (df["end_date"] >= ts)
        )
        return sorted(df.loc[active, "symbol"].unique().tolist())

    def is_member(self, symbol: str, as_of: DateLike) -> bool:
        return symbol in set(self.members_asof(as_of))

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
        as_of = as_of if as_of is not None else pd.Timestamp.utcnow().normalize()
        return self.membership().members_asof(as_of)

    def is_member(self, symbol: str, as_of: DateLike) -> bool:
        return self.membership().is_member(symbol, as_of)

    def all_symbols(self) -> List[str]:
        """Every symbol that was *ever* a member (for survivorship-safe loads)."""
        return self.membership().all_symbols()
