"""Cross-sectional quality factor.

Combines three classic measures of business quality into a single composite:

* Return on equity (ROE) = net_income / book_value. Higher is better.
* Gross margin = gross_profit / revenue. Higher is better.
* Accruals = (net_income - operating_cashflow) / total_assets (Sloan, 1996).
  High accruals indicate low-quality, less cash-backed earnings, so the
  accrual measure is *sign-flipped* before entering the composite, making low
  accruals score high.

Each sub-factor is converted to a cross-sectional z-score and the composite is
their equal-weighted average, so a higher composite score identifies
higher-quality companies.

Fundamental data are handled point-in-time (PIT): a value is only usable from
its ``announcement_date`` onward, forward-filled to the most recently announced
figure, which avoids look-ahead bias.

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
)

_REQUIRED_COLUMNS = ("symbol", "period_end", "announcement_date", "field", "value")


class QualityFactor:
    """Composite cross-sectional quality factor (ROE, gross margin, accruals)."""

    # The three sub-factors are combined with equal weight in :meth:`compute`.

    # ------------------------------------------------------------------
    # Public composite
    # ------------------------------------------------------------------
    def compute(self, fundamentals: pd.DataFrame) -> pd.DataFrame:
        """Compute the composite quality factor panel.

        Parameters
        ----------
        fundamentals:
            Tidy frame with columns
            ``[symbol, period_end, announcement_date, field, value]``. Relevant
            ``field`` values are ``net_income``, ``book_value``,
            ``gross_profit``, ``revenue``, ``operating_cashflow`` and
            ``total_assets``.

        Returns
        -------
        pandas.DataFrame
            Composite quality factor panel indexed by the announcement-aware
            daily calendar implied by the fundamentals, columns are symbols.
            Higher = higher quality = more attractive.
        """
        self._validate_fundamentals(fundamentals)
        index, columns = self._panel_axes(fundamentals)

        roe = self.roe(fundamentals, index, columns)
        gm = self.gross_margin(fundamentals, index, columns)
        # accruals() already returns the sign-flipped (higher = better) measure.
        acc = self.accruals(fundamentals, index, columns)

        z_roe = self.cross_sectional_zscore(roe)
        z_gm = self.cross_sectional_zscore(gm)
        z_acc = self.cross_sectional_zscore(acc)

        composite = pd.concat([z_roe, z_gm, z_acc]).groupby(level=0).mean()
        composite = composite.reindex(index=index, columns=columns)
        return composite

    # ------------------------------------------------------------------
    # Individual sub-factors (higher = better quality)
    # ------------------------------------------------------------------
    def roe(
        self,
        fundamentals: pd.DataFrame,
        index: pd.Index | None = None,
        columns: pd.Index | None = None,
    ) -> pd.DataFrame:
        """Return on equity = net_income / book_value (higher = better)."""
        index, columns = self._resolve_axes(fundamentals, index, columns)
        net_income = self._pit_panel(fundamentals, "net_income", index, columns)
        book = self._pit_panel(fundamentals, "book_value", index, columns)
        return self._safe_ratio(net_income, book)

    def gross_margin(
        self,
        fundamentals: pd.DataFrame,
        index: pd.Index | None = None,
        columns: pd.Index | None = None,
    ) -> pd.DataFrame:
        """Gross margin = gross_profit / revenue (higher = better)."""
        index, columns = self._resolve_axes(fundamentals, index, columns)
        gross_profit = self._pit_panel(fundamentals, "gross_profit", index, columns)
        revenue = self._pit_panel(fundamentals, "revenue", index, columns)
        return self._safe_ratio(gross_profit, revenue)

    def accruals(
        self,
        fundamentals: pd.DataFrame,
        index: pd.Index | None = None,
        columns: pd.Index | None = None,
    ) -> pd.DataFrame:
        """Sign-flipped accruals (higher = better quality).

        Raw accruals are ``(net_income - operating_cashflow) / total_assets``.
        High accruals signal earnings not backed by cash and predict lower
        future returns, so the returned panel is the *negative* of raw accruals
        such that low-accrual (high-quality) firms score high.
        """
        index, columns = self._resolve_axes(fundamentals, index, columns)
        net_income = self._pit_panel(fundamentals, "net_income", index, columns)
        ocf = self._pit_panel(fundamentals, "operating_cashflow", index, columns)
        assets = self._pit_panel(fundamentals, "total_assets", index, columns)
        raw_accruals = self._safe_ratio(net_income - ocf, assets)
        return -raw_accruals

    # ------------------------------------------------------------------
    # Cross-sectional normalization (shared via _cross_section)
    # ------------------------------------------------------------------
    cross_sectional_zscore = staticmethod(cross_sectional_zscore)
    rank = staticmethod(cross_sectional_rank)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_ratio(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
        """Element-wise ratio guarding against division by non-positive denoms."""
        denom = denominator.where(denominator > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = numerator.divide(denom)
        return ratio.replace([np.inf, -np.inf], np.nan)

    @classmethod
    def _panel_axes(cls, fundamentals: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
        """Derive the (date_index, symbol_columns) for the output panel.

        The date index is the sorted set of distinct announcement dates; the
        columns are the sorted set of distinct symbols.
        """
        dates = pd.to_datetime(fundamentals["announcement_date"]).dropna().unique()
        index = pd.DatetimeIndex(sorted(dates))
        columns = pd.Index(sorted(fundamentals["symbol"].dropna().unique()))
        return index, columns

    @classmethod
    def _resolve_axes(
        cls,
        fundamentals: pd.DataFrame,
        index: pd.Index | None,
        columns: pd.Index | None,
    ) -> tuple[pd.Index, pd.Index]:
        cls._validate_fundamentals(fundamentals)
        if index is None or columns is None:
            d_index, d_columns = cls._panel_axes(fundamentals)
            index = d_index if index is None else index
            columns = d_columns if columns is None else columns
        return index, columns

    @classmethod
    def _pit_panel(
        cls,
        fundamentals: pd.DataFrame,
        field: str,
        index: pd.Index,
        columns: pd.Index,
    ) -> pd.DataFrame:
        """Build a point-in-time, forward-filled panel for one fundamental field.

        The value for ``(date, symbol)`` is the most recently *announced* value
        for that field as of ``date`` (keyed on ``announcement_date``, never
        ``period_end``), aligned onto ``index``/``columns``.
        """
        sub = fundamentals.loc[fundamentals["field"] == field,
                               ["symbol", "announcement_date", "value"]].copy()
        if sub.empty:
            return pd.DataFrame(index=index, columns=columns, dtype=float)

        sub["announcement_date"] = pd.to_datetime(sub["announcement_date"])
        # Stable sort so that announcements tied on date keep their input
        # order and drop_duplicates(keep="last") retains the latest row.
        sub = sub.sort_values("announcement_date", kind="stable")
        sub = sub.drop_duplicates(subset=["announcement_date", "symbol"], keep="last")

        wide = sub.pivot(index="announcement_date", columns="symbol", values="value")
        full_index = wide.index.union(pd.DatetimeIndex(index))
        wide = wide.reindex(full_index).sort_index().ffill()
        wide = wide.reindex(index=index, columns=columns)
        return wide.astype(float)

    @staticmethod
    def _validate_fundamentals(fundamentals: pd.DataFrame) -> None:
        if not isinstance(fundamentals, pd.DataFrame):
            raise TypeError("fundamentals must be a pandas DataFrame")
        missing = [c for c in _REQUIRED_COLUMNS if c not in fundamentals.columns]
        if missing:
            raise ValueError(f"fundamentals missing required columns: {missing}")
