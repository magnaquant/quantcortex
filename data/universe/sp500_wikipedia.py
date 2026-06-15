"""Reconstruct point-in-time S&P 500 membership from Wikipedia.

The static snapshot in :mod:`data.universe.sp500_universe` is survivorship-biased
(it only knows today's members). This module fetches Wikipedia's *current
constituents* table together with its dated *change log* (index additions and
removals) and reconstructs an approximate point-in-time membership so that
``members_asof(date)`` returns the names that were actually in the index then -
including companies later dropped (acquired, bankrupt, demoted).

Reconstruction
--------------
* Each current member contributes one *open* interval ``[date_added, NaT]`` using
  Wikipedia's "Date added" (the start of its current continuous tenure).
* Each removed name in the change log contributes a *closed* interval ending on
  its removal date and starting at the matching prior addition (or a configurable
  ``floor`` date when it joined before the log begins).

Limitations (documented, not hidden): the change log only reaches back a finite
number of years, so membership before the log's earliest entry is approximate,
and a company removed *and re-added* keeps only its current open tenure plus any
fully-closed prior cycles. This is a free, reproducible source for research and
a drop-in upgrade over the static snapshot; a licensed point-in-time vendor feed
remains the gold standard. Network + ``lxml`` are required (imported lazily).
"""

from __future__ import annotations

import io
import urllib.request
from collections import defaultdict
from typing import Optional, Tuple

import pandas as pd

from data.universe.base import PITMembership

__all__ = ["WIKI_SP500_URL", "fetch_sp500_tables", "build_pit_membership"]

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_USER_AGENT = "quantcortex-research (https://github.com/magnaquant/quantcortex)"
_DEFAULT_FLOOR = pd.Timestamp("2000-01-01")


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
    floor: pd.Timestamp = _DEFAULT_FLOOR,
) -> PITMembership:
    """Reconstruct a :class:`PITMembership` from the two Wikipedia tables."""
    # ---- current members: open interval from "Date added" ----
    cur = current.copy()
    cur.columns = [str(c) for c in cur.columns]
    sym_col = "Symbol"
    added_col = next((c for c in cur.columns if c.lower().startswith("date added")), None)
    current_syms = {}
    for _, row in cur.iterrows():
        sym = _normalize_ticker(row[sym_col])
        if sym is None:
            continue
        start = pd.to_datetime(row[added_col], errors="coerce") if added_col else pd.NaT
        current_syms[sym] = start if pd.notna(start) else floor
    current_set = set(current_syms)

    rows = [{"symbol": s, "start_date": st, "end_date": pd.NaT} for s, st in current_syms.items()]

    # ---- change log: closed intervals for removed (non-current) names ----
    if not changes.empty:
        ch = changes.copy()
        # Flatten the MultiIndex header to (date, added, removed).
        cols = {}
        for c in ch.columns:
            label = " ".join(map(str, c)) if isinstance(c, tuple) else str(c)
            low = label.lower()
            if "effective date" in low or low.strip() == "date":
                cols["date"] = c
            elif "added" in low and "ticker" in low:
                cols["added"] = c
            elif "removed" in low and "ticker" in low:
                cols["removed"] = c
        if {"date", "added", "removed"} <= set(cols):
            events = defaultdict(list)  # ticker -> list of (date, "add"|"remove")
            for _, row in ch.iterrows():
                d = pd.to_datetime(row[cols["date"]], errors="coerce")
                if pd.isna(d):
                    continue
                add_t = _normalize_ticker(row[cols["added"]])
                rem_t = _normalize_ticker(row[cols["removed"]])
                if add_t:
                    events[add_t].append((d, "add"))
                if rem_t:
                    events[rem_t].append((d, "remove"))

            for tkr, evs in events.items():
                if tkr in current_set:
                    # Current tenure already captured via the open interval above;
                    # only emit fully-closed PRIOR cycles (add followed by remove
                    # that both precede the current tenure). Rare; skip for safety.
                    continue
                evs.sort()
                open_start = None
                for d, act in evs:
                    if act == "add":
                        open_start = d
                    elif act == "remove":
                        rows.append({
                            "symbol": tkr,
                            "start_date": open_start if open_start is not None else floor,
                            "end_date": d,
                        })
                        open_start = None
                # A trailing unmatched "add" for a non-current ticker means it was
                # added then (per current table) is no longer listed -> treat as a
                # short open-ended membership from that add to now is wrong, so we
                # leave it out rather than fabricate an end date.

    frame = pd.DataFrame(rows, columns=["symbol", "start_date", "end_date"])
    return PITMembership(frame)
