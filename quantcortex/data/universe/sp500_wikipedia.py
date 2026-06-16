"""Reconstruct point-in-time S&P 500 membership from Wikipedia.

The opt-in demo subset in
:mod:`quantcortex.data.universe.sp500_universe` is survivorship-biased. This
module fetches Wikipedia's *current constituents* table together with its dated
*change log* (index additions and removals) and reconstructs an approximate
point-in-time membership so that
``members_asof(date)`` returns the names that were actually in the index then -
including companies later dropped (acquired, bankrupt, demoted).

Reconstruction
--------------
* Each current member contributes one *open* interval ``[date_added, NaT)`` using
  Wikipedia's "Date added" (the start of its current continuous tenure).
* Each removed name in the change log contributes a *closed* interval ending on
  its removal date and starting at the matching prior addition. When the
  addition predates the available log, reconstruction starts at the log's first
  effective date rather than inventing earlier membership.

Limitations (documented, not hidden): the change log only reaches back a finite
number of years, so membership before the log's earliest entry is unsupported.
The reconstruction preserves fully closed prior cycles for companies that were
removed and later re-added. This is a free research source and a material
upgrade over a static subset, not a substitute for a licensed index-history
feed. Network + ``lxml`` are required (imported lazily).
"""

from __future__ import annotations

import io
import urllib.request
from collections import defaultdict
from typing import Optional, Tuple

import pandas as pd

from quantcortex.data.universe.base import PITMembership

__all__ = ["WIKI_SP500_URL", "fetch_sp500_tables", "build_pit_membership"]

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_USER_AGENT = "quantcortex-research (https://github.com/magnaquant/quantcortex)"


def _normalize_ticker(t: object) -> Optional[str]:
    """Wikipedia uses BRK.B; most data APIs use BRK-B."""
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return None
    s = str(t).strip().upper()
    if not s or s in {"NAN", "-"}:
        return None
    return s.replace(".", "-")


def fetch_sp500_tables(url: str = WIKI_SP500_URL, timeout: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(current_constituents, change_log)`` DataFrames from Wikipedia."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted URL)
        html = resp.read().decode("utf-8", "replace")
    try:
        tables = pd.read_html(io.StringIO(html))
    except ImportError as exc:  # pragma: no cover - missing parser
        raise ImportError(
            "parsing Wikipedia tables needs lxml (pip install lxml)"
        ) from exc

    current = next((t for t in tables if "Symbol" in [str(c) for c in t.columns]), None)
    if current is None:
        raise RuntimeError("could not locate the S&P 500 constituents table")

    # The change log has a two-level header (Added/Removed -> Ticker/Security).
    changes = None
    for t in tables:
        flat = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        if any("Added" in f for f in flat) and any("Removed" in f for f in flat):
            changes = t
            break
    if changes is None:
        changes = pd.DataFrame()
    return current, changes


def build_pit_membership(
    current: pd.DataFrame,
    changes: pd.DataFrame,
    *,
    floor: Optional[pd.Timestamp] = None,
) -> PITMembership:
    """Reconstruct half-open membership intervals from Wikipedia tables.

    ``floor`` may explicitly narrow the supported history. When omitted, the
    earliest valid change-log date is used for tenures whose additions predate
    the log. Queries earlier than that inferred boundary should not be treated
    as complete index membership.
    """
    if not isinstance(current, pd.DataFrame) or current.empty:
        raise ValueError("current constituents table must be a non-empty DataFrame")
    if not isinstance(changes, pd.DataFrame):
        raise TypeError("changes must be a DataFrame")
    if changes.empty:
        raise ValueError(
            "change log must be non-empty for point-in-time reconstruction"
        )

    events = defaultdict(list)
    change_dates: list[pd.Timestamp] = []
    if not changes.empty:
        ch = changes.copy()
        cols = {}
        for c in ch.columns:
            label = " ".join(map(str, c)) if isinstance(c, tuple) else str(c)
            low = label.lower()
            words = low.split()
            if "effective date" in low or (words and set(words) == {"date"}):
                cols["date"] = c
            elif "added" in low and "ticker" in low:
                cols["added"] = c
            elif "removed" in low and "ticker" in low:
                cols["removed"] = c
        if changes.shape[0] and not {"date", "added", "removed"} <= set(cols):
            raise ValueError("could not identify date/added/removed change-log columns")
        for _, row in ch.iterrows():
            d = pd.to_datetime(row[cols["date"]], errors="coerce", utc=True)
            if pd.isna(d):
                continue
            d = pd.Timestamp(d).tz_localize(None).normalize()
            change_dates.append(d)
            add_t = _normalize_ticker(row[cols["added"]])
            rem_t = _normalize_ticker(row[cols["removed"]])
            if add_t:
                events[add_t].append((d, "add"))
            if rem_t:
                events[rem_t].append((d, "remove"))

    if not change_dates:
        raise ValueError("change log contains no valid effective dates")
    earliest_change = min(change_dates)

    if floor is not None:
        coverage_start = pd.to_datetime(floor, errors="coerce", utc=True)
        if pd.isna(coverage_start):
            raise ValueError("floor must be a valid timestamp")
        coverage_start = coverage_start.tz_localize(None).normalize()
        if coverage_start < earliest_change:
            raise ValueError(
                "floor cannot precede the earliest available change-log date"
            )
    else:
        coverage_start = earliest_change

    def closed_intervals(
        ticker: str, cutoff: Optional[pd.Timestamp] = None
    ) -> list[dict[str, object]]:
        prior_events = [
            (date, action)
            for date, action in events.get(ticker, [])
            if cutoff is None
            or date < cutoff
            or (date == cutoff and action == "remove")
        ]
        # Close a prior tenure before opening a new one on the same date.
        prior_events.sort(key=lambda item: (item[0], 0 if item[1] == "remove" else 1))
        out: list[dict[str, object]] = []
        open_start: Optional[pd.Timestamp] = None
        for date, action in prior_events:
            if action == "add":
                if open_start is None:
                    open_start = date
                continue
            start = open_start if open_start is not None else coverage_start
            start = max(start, coverage_start)
            if start < date:
                out.append(
                    {"symbol": ticker, "start_date": start, "end_date": date}
                )
            open_start = None
        return out

    # ---- current members: prior closed cycles + current open tenure ----
    cur = current.copy()
    cur.columns = [str(c) for c in cur.columns]
    sym_col = "Symbol"
    if sym_col not in cur.columns:
        raise ValueError("current constituents table has no Symbol column")
    added_col = next((c for c in cur.columns if c.lower().startswith("date added")), None)
    current_syms: dict[str, pd.Timestamp] = {}
    for _, row in cur.iterrows():
        sym = _normalize_ticker(row[sym_col])
        if sym is None:
            continue
        start = (
            pd.to_datetime(row[added_col], errors="coerce", utc=True)
            if added_col
            else pd.NaT
        )
        if pd.isna(start):
            adds = [date for date, action in events.get(sym, []) if action == "add"]
            start = max(adds) if adds else coverage_start
        else:
            start = pd.Timestamp(start).tz_localize(None)
        start = max(pd.Timestamp(start).normalize(), coverage_start)
        if sym in current_syms:
            raise ValueError(f"duplicate current constituent {sym!r}")
        current_syms[sym] = start

    rows: list[dict[str, object]] = []
    for symbol, start in current_syms.items():
        rows.extend(closed_intervals(symbol, cutoff=start))
        rows.append({"symbol": symbol, "start_date": start, "end_date": pd.NaT})

    # ---- non-current names: emit only closed, source-supported tenures ----
    for symbol in sorted(set(events) - set(current_syms)):
        rows.extend(closed_intervals(symbol))

    frame = pd.DataFrame(rows, columns=["symbol", "start_date", "end_date"])
    return PITMembership(frame)
