"""Cross-sectional value factor.

Combines three classic valuation ratios into a single composite value score:

* Earnings yield (E/P), the reciprocal of the price-to-earnings ratio.
* Book-to-price (B/P), the reciprocal of the price-to-book ratio.
* EBITDA-to-enterprise-value (EBITDA/EV), the reciprocal of the EV/EBITDA
  multiple.

Each ratio is expressed so that *cheaper is higher* (a high earnings yield
means a low P/E, i.e. a cheap stock). The composite is the equal-weighted
average of the three individual cross-sectional z-scores, so a higher composite
score identifies more attractively valued names.

Fundamental data are treated point-in-time (PIT): a fundamental value only
becomes usable on its ``announcement_date``, and on any given date we use the
most recently announced (forward-filled) value. This prevents look-ahead bias
from period-end values that were not yet public.

Cross-sectional normalization helpers are shared with the other classical
factor modules via the private :mod:`alpha.factors.classical._cross_section`
module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantcortex.alpha.factors.classical._cross_section import (
    cross_sectional_rank,
    cross_sectional_zscore,
    validate_prices,
)

_REQUIRED_COLUMNS = ("symbol", "period_end", "announcement_date", "field", "value")


class ValueFactor:
    """Composite cross-sectional value factor (earnings yield, B/P, EBITDA/EV)."""

    # The three sub-factors are combined with equal weight in :meth:`compute`.

    # ------------------------------------------------------------------
    # Public composite
    # ------------------------------------------------------------------
    def compute(self, fundamentals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute the composite value factor panel.

        Parameters
        ----------
        fundamentals:
            Tidy frame with columns
            ``[symbol, period_end, announcement_date, field, value]``. Relevant
            ``field`` values are ``earnings``, ``book_value``, ``ebitda``,
            ``enterprise_value`` and ``shares_outstanding``.
        prices:
            Adjusted close prices indexed by date, columns are symbols.

        Returns
        -------
        pandas.DataFrame
            Composite value factor panel (cross-sectional z-score average),
            indexed like ``prices``, higher = cheaper = more attractive.
        """
        prices = self._validate_prices(prices)

        ey = self.earnings_yield(fundamentals, prices)
        bp = self.book_to_price(fundamentals, prices)
        ee = self.ev_to_ebitda(fundamentals, prices)

        z_ey = self.cross_sectional_zscore(ey)
        z_bp = self.cross_sectional_zscore(bp)
        z_ee = self.cross_sectional_zscore(ee)

        # Equal-weighted average of the available sub-factor z-scores. Using a
        # nan-aware mean lets a symbol still receive a composite score on dates
        # where one sub-factor is missing, rather than dropping it entirely.
        composite = pd.concat([z_ey, z_bp, z_ee]).groupby(level=0).mean()
        composite = composite.reindex(index=prices.index, columns=prices.columns)
        return composite

    # ------------------------------------------------------------------
    # Individual sub-factors (cheap = high)
    # ------------------------------------------------------------------
    def earnings_yield(self, fundamentals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """Earnings yield E/P = earnings / market_cap (reciprocal of P/E).

        Market capitalisation is reconstructed point-in-time from
        ``shares_outstanding`` and the contemporaneous price, so the ratio is
        comparable across symbols regardless of share count differences.
        """
        prices = self._validate_prices(prices)
        earnings = self._pit_panel(fundamentals, "earnings", prices)
        shares = self._pit_panel(fundamentals, "shares_outstanding", prices)
        market_cap = prices.multiply(shares)
        return self._safe_ratio(earnings, market_cap)

    def book_to_price(self, fundamentals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """Book-to-price B/P = book_value / market_cap (reciprocal of P/B)."""
        prices = self._validate_prices(prices)
        book = self._pit_panel(fundamentals, "book_value", prices)
        shares = self._pit_panel(fundamentals, "shares_outstanding", prices)
        market_cap = prices.multiply(shares)
        return self._safe_ratio(book, market_cap)

    def ev_to_ebitda(self, fundamentals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """EBITDA yield = EBITDA / enterprise_value (reciprocal of EV/EBITDA)."""
        prices = self._validate_prices(prices)
        ebitda = self._pit_panel(fundamentals, "ebitda", prices)
        ev = self._pit_panel(fundamentals, "enterprise_value", prices)
        return self._safe_ratio(ebitda, ev)

    # ------------------------------------------------------------------
    # Cross-sectional normalization (shared via _cross_section)
    # ------------------------------------------------------------------
    cross_sectional_zscore = staticmethod(cross_sectional_zscore)
    rank = staticmethod(cross_sectional_rank)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    _validate_prices = staticmethod(validate_prices)

    @staticmethod
    def _safe_ratio(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
        """Element-wise ratio guarding against division by non-positive denoms."""
        denom = denominator.where(denominator > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = numerator.divide(denom)
        return ratio.replace([np.inf, -np.inf], np.nan)

    @classmethod
    def _pit_panel(
        cls, fundamentals: pd.DataFrame, field: str, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Build a point-in-time, forward-filled panel for one fundamental field.

        The value for ``(date, symbol)`` is the most recently *announced* value
        for that field as of ``date`` (using ``announcement_date``, never
        ``period_end``), aligned onto the trading calendar of ``prices``.
        """
        cls._validate_fundamentals(fundamentals)
        sub = fundamentals.loc[fundamentals["field"] == field,
                               ["symbol", "announcement_date", "value"]].copy()
        if sub.empty:
            return pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

        sub["announcement_date"] = pd.to_datetime(sub["announcement_date"])
        # Stable sort so that announcements tied on date keep their input
        # order and drop_duplicates(keep="last") retains the latest row.
        sub = sub.sort_values("announcement_date", kind="stable")
        # If multiple announcements share a date for one symbol, keep the last.
        sub = sub.drop_duplicates(subset=["announcement_date", "symbol"], keep="last")

        wide = sub.pivot(index="announcement_date", columns="symbol", values="value")
        # Reindex onto the union of announcement dates and trading dates, ffill
        # (so a value is carried forward only after it is announced), then
        # restrict to the price calendar. Values are NaN before first announce.
        full_index = wide.index.union(prices.index)
        wide = wide.reindex(full_index).sort_index().ffill()
        wide = wide.reindex(index=prices.index, columns=prices.columns)
        return wide.astype(float)

    @staticmethod
    def _validate_fundamentals(fundamentals: pd.DataFrame) -> None:
        if not isinstance(fundamentals, pd.DataFrame):
            raise TypeError("fundamentals must be a pandas DataFrame")
        missing = [c for c in _REQUIRED_COLUMNS if c not in fundamentals.columns]
        if missing:
            raise ValueError(f"fundamentals missing required columns: {missing}")
