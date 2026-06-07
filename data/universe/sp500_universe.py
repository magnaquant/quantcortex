"""S&P 500 investable universe.

.. warning::
   The bundled membership is a **static snapshot** of large-cap S&P 500 names,
   all marked active from 2010-01-01 with no end date.  It is intended for demos
   and smoke tests only.  Using a static present-day membership for historical
   research introduces **survivorship bias**: companies that were dropped from
   the index (acquired, bankrupt, demoted) are missing, which inflates backtest
   returns.

   For survivorship-safe research, supply a real point-in-time constituents file
   via ``SP500Universe(membership_csv="path/to/constituents.csv")``.  The CSV
   must have columns ``symbol,start_date,end_date`` (``end_date`` empty/NaT means
   the symbol is currently a member), so that membership can be queried correctly
   as of any historical date.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from data.universe.base import PITMembership, Universe

__all__ = ["SP500Universe", "STATIC_SP500"]

# Representative large-cap S&P 500 tickers (static snapshot; not point-in-time).
STATIC_SP500: List[str] = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "BRK-B", "JPM",
    "TSLA", "LLY", "V", "UNH", "XOM", "MA", "JNJ", "AVGO", "PG", "HD", "COST",
    "MRK", "ABBV", "CVX", "PEP", "ADBE", "KO", "WMT", "CRM", "BAC", "MCD",
    "ACN", "CSCO", "NFLX", "ABT", "TMO", "LIN", "ORCL", "DIS", "WFC", "INTC",
    "AMD", "DHR", "VZ", "TXN", "QCOM", "PM", "INTU", "CAT", "NEE", "IBM",
    "GE", "UNP", "AMGN", "LOW", "SPGI", "HON", "BA", "RTX", "GS", "ISRG",
    "PFE", "NOW", "BKNG", "AXP", "ELV", "DE", "BLK", "PLD", "SBUX", "MDT",
    "GILD", "ADI", "MMC", "C", "AMAT", "T", "CB", "LMT", "SYK", "TJX",
]


class SP500Universe(Universe):
    """S&P 500 universe backed by a CSV or a bundled static snapshot."""

    name = "sp500"

    def __init__(self, membership_csv: Optional[str] = None) -> None:
        """Initialize the universe.

        Parameters
        ----------
        membership_csv:
            Optional path to a point-in-time constituents CSV with columns
            ``symbol,start_date,end_date``.  When omitted, the bundled static
            snapshot is used (see module docstring for the survivorship caveat).
        """
        self._membership_csv = membership_csv
        self._membership: Optional[PITMembership] = None

    def _build_static(self) -> PITMembership:
        frame = pd.DataFrame(
            {
                "symbol": STATIC_SP500,
                "start_date": pd.Timestamp("2010-01-01"),
                "end_date": pd.NaT,
            }
        )
        return PITMembership(frame)

    def _load_csv(self, path: str) -> PITMembership:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"membership CSV not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        return PITMembership(frame)

    def membership(self) -> PITMembership:
        """Return the point-in-time membership table."""
        if self._membership is None:
            if self._membership_csv is not None:
                self._membership = self._load_csv(self._membership_csv)
            else:
                self._membership = self._build_static()
        return self._membership
