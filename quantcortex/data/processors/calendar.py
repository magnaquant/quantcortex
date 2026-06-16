"""Trading calendar with a pure-offline US-equity fallback.

The preferred backend is :mod:`pandas_market_calendars` (an optional dependency).
When it is not installed, :class:`TradingCalendar` falls back to a self-contained
US-equity ("NYSE") calendar built from business days minus a generated set of
US market holidays.  The fallback is fully offline and requires no network or
extra packages, so calendar logic always works in tests and CI.

The fallback models the standard NYSE holiday schedule including weekend
observance shifts (a holiday falling on Saturday is observed the preceding
Friday; one falling on Sunday is observed the following Monday) and Juneteenth,
which became an exchange holiday in 2022. New Year's Day is special-cased per
NYSE convention: when January 1 falls on a Saturday it is simply not observed
(December 31 of the prior year remains a trading day).

The fallback is a regular-holiday calendar, not an authoritative historical
exchange schedule. It does not model unscheduled closures or early closes; use
``pandas_market_calendars`` or an exchange-licensed calendar when those matter.
"""

from __future__ import annotations

import datetime as _dt
from typing import Union

import pandas as pd

__all__ = [
    "TradingCalendar",
    "first_session_each_week",
    "last_session_each_month",
]

DateLike = Union[str, _dt.date, _dt.datetime, pd.Timestamp]
_US_EQUITY_EXCHANGES = {"NYSE", "XNYS", "NASDAQ", "XNAS"}


def _validate_session_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Validate an observed-session index used to derive rebalance dates."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("sessions must be a DatetimeIndex")
    if index.hasnans:
        raise ValueError("sessions must contain valid timestamps")
    if index.has_duplicates:
        raise ValueError("sessions must not contain duplicate timestamps")
    if not index.is_monotonic_increasing:
        raise ValueError("sessions must be sorted in increasing order")
    return index


def _session_periods(index: pd.DatetimeIndex, frequency: str) -> pd.PeriodIndex:
    """Return calendar periods without changing timezone-aware local dates."""
    local = index.tz_localize(None) if index.tz is not None else index
    return local.to_period(frequency)


def first_session_each_week(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the first observed session in each Saturday-through-Friday week.

    Deriving the schedule from observed sessions handles Monday exchange
    holidays correctly: Tuesday becomes that week's rebalance decision date.
    """
    sessions = _validate_session_index(sessions)
    periods = _session_periods(sessions, "W-FRI")
    return sessions[~periods.duplicated(keep="first")]


def last_session_each_month(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the last observed session in each calendar month."""
    sessions = _validate_session_index(sessions)
    periods = _session_periods(sessions, "M")
    return sessions[~periods.duplicated(keep="last")]


def _normalize_date(value: DateLike) -> pd.Timestamp:
    """Return a valid timezone-naive calendar date, preserving local date."""
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError("date must be a valid timestamp")
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _easter(year: int) -> _dt.date:
    """Return the Gregorian Easter Sunday date for ``year`` (Anonymous algorithm)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month = (h + ll - 7 * m + 114) // 31
    day = ((h + ll - 7 * m + 114) % 31) + 1
    return _dt.date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> _dt.date:
    """Return the ``n``-th ``weekday`` (Mon=0) of ``month`` in ``year``."""
    first = _dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + _dt.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> _dt.date:
    """Return the last ``weekday`` (Mon=0) of ``month`` in ``year``."""
    if month == 12:
        last = _dt.date(year, 12, 31)
    else:
        last = _dt.date(year, month + 1, 1) - _dt.timedelta(days=1)
    offset = (last.weekday() - weekday) % 7
    return last - _dt.timedelta(days=offset)


def _observed(date: _dt.date) -> _dt.date:
    """Apply weekend-observance shift for a fixed-date holiday."""
    if date.weekday() == 5:  # Saturday -> observed Friday
        return date - _dt.timedelta(days=1)
    if date.weekday() == 6:  # Sunday -> observed Monday
        return date + _dt.timedelta(days=1)
    return date


def _us_market_holidays(year: int) -> set[_dt.date]:
    """Generate the set of US-equity market holidays for a calendar ``year``."""
    holidays: set[_dt.date] = set()

    # New Year's Day. NYSE convention: when Jan 1 falls on a Saturday the
    # holiday is *not* observed (Dec 31 of the prior year stays a trading
    # day); a Sunday Jan 1 is observed the following Monday.
    new_years = _dt.date(year, 1, 1)
    if new_years.weekday() == 6:  # Sunday -> observed Monday
        holidays.add(new_years + _dt.timedelta(days=1))
    elif new_years.weekday() != 5:  # Saturday -> not observed
        holidays.add(new_years)
    # Martin Luther King Jr. Day: 3rd Monday of January (since 1998).
    if year >= 1998:
        holidays.add(_nth_weekday(year, 1, 0, 3))
    # Washington's Birthday / Presidents Day: 3rd Monday of February.
    holidays.add(_nth_weekday(year, 2, 0, 3))
    # Good Friday: Friday before Easter Sunday.
    holidays.add(_easter(year) - _dt.timedelta(days=2))
    # Memorial Day: last Monday of May.
    holidays.add(_last_weekday(year, 5, 0))
    # Juneteenth National Independence Day: first US exchange closure in 2022.
    if year >= 2022:
        holidays.add(_observed(_dt.date(year, 6, 19)))
    # Independence Day (observed).
    holidays.add(_observed(_dt.date(year, 7, 4)))
    # Labor Day: 1st Monday of September.
    holidays.add(_nth_weekday(year, 9, 0, 1))
    # Thanksgiving Day: 4th Thursday of November.
    holidays.add(_nth_weekday(year, 11, 3, 4))
    # Christmas Day (observed).
    holidays.add(_observed(_dt.date(year, 12, 25)))

    return holidays


class TradingCalendar:
    """Exchange trading calendar.

    Uses :mod:`pandas_market_calendars` when available; otherwise falls back to
    a built-in offline US-equity calendar (business days minus generated US
    market holidays with weekend observance).

    Parameters
    ----------
    exchange:
        Exchange code (e.g. ``"NYSE"``).  Passed through to
        ``pandas_market_calendars`` when that backend is active.  The offline
        fallback supports only NYSE/Nasdaq aliases because it models the shared
        US-equity holiday schedule. Unknown exchanges raise instead of silently
        receiving NYSE dates.
    """

    def __init__(self, exchange: str = "NYSE") -> None:
        if not isinstance(exchange, str) or not exchange.strip():
            raise ValueError("exchange must be a non-empty string")
        self.exchange = exchange.strip()
        self._mcal = None
        try:
            import pandas_market_calendars as mcal  # lazy optional import
        except ImportError:
            if self.exchange.upper() not in _US_EQUITY_EXCHANGES:
                raise ValueError(
                    f"offline calendar does not support exchange {self.exchange!r}"
                )
        else:
            try:
                self._mcal = mcal.get_calendar(self.exchange)
            except Exception as exc:
                raise ValueError(f"unknown exchange {self.exchange!r}") from exc

    @property
    def using_fallback(self) -> bool:
        """``True`` when the offline built-in calendar is in use."""
        return self._mcal is None

    def _holidays_in_range(self, start: pd.Timestamp, end: pd.Timestamp) -> set[_dt.date]:
        # Generate one extra year on each side: an observed holiday can land
        # in the calendar year adjacent to its nominal one (e.g. a Sunday
        # Dec 25 observed the following Monday, Jan 1 of the next year... or a
        # Sunday Jan 1 nominally belonging to the next year's generation).
        holidays: set[_dt.date] = set()
        for year in range(start.year - 1, end.year + 2):
            holidays |= _us_market_holidays(year)
        return holidays

    def sessions(self, start: DateLike, end: DateLike) -> pd.DatetimeIndex:
        """Return all trading sessions in ``[start, end]`` (inclusive)."""
        start_ts = _normalize_date(start)
        end_ts = _normalize_date(end)
        if start_ts > end_ts:
            return pd.DatetimeIndex([])

        if self._mcal is not None:
            sched = self._mcal.schedule(start_date=start_ts, end_date=end_ts)
            return pd.DatetimeIndex(sched.index).normalize()

        # Offline fallback: business days minus generated holidays.
        bdays = pd.bdate_range(start=start_ts, end=end_ts)
        holidays = self._holidays_in_range(start_ts, end_ts)
        holiday_ts = {pd.Timestamp(h) for h in holidays}
        sessions = [d for d in bdays if d not in holiday_ts]
        return pd.DatetimeIndex(sessions)

    def is_trading_day(self, date: DateLike) -> bool:
        """Return ``True`` if ``date`` is a trading session."""
        ts = _normalize_date(date)
        if self._mcal is not None:
            sched = self._mcal.schedule(start_date=ts, end_date=ts)
            return len(sched.index) > 0
        if ts.weekday() >= 5:
            return False
        # Generate adjacent years too: observed holidays can cross year ends.
        return ts.date() not in self._holidays_in_range(ts, ts)

    def next_session(self, date: DateLike) -> pd.Timestamp:
        """Return the first trading session strictly after ``date``."""
        ts = _normalize_date(date)
        probe = ts + pd.Timedelta(days=1)
        # Look ahead in 30-day windows until a session is found.
        for _ in range(24):
            window = self.sessions(probe, probe + pd.Timedelta(days=30))
            if len(window) > 0:
                return window[0]
            probe = probe + pd.Timedelta(days=31)
        raise ValueError(f"No trading session found after {ts.date()}")

    def previous_session(self, date: DateLike) -> pd.Timestamp:
        """Return the last trading session strictly before ``date``."""
        ts = _normalize_date(date)
        probe = ts - pd.Timedelta(days=1)
        for _ in range(24):
            window = self.sessions(probe - pd.Timedelta(days=30), probe)
            if len(window) > 0:
                return window[-1]
            probe = probe - pd.Timedelta(days=31)
        raise ValueError(f"No trading session found before {ts.date()}")

    def n_sessions_between(self, a: DateLike, b: DateLike) -> int:
        """Return the number of trading sessions in ``[a, b]`` inclusive.

        If ``a > b`` the count is returned as a negative number.
        """
        a_ts = _normalize_date(a)
        b_ts = _normalize_date(b)
        if a_ts <= b_ts:
            return int(len(self.sessions(a_ts, b_ts)))
        return -int(len(self.sessions(b_ts, a_ts)))
