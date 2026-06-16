"""Point-in-time cross-sectional quality factor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantcortex.alpha.factors.classical._cross_section import (
    cross_sectional_rank,
    cross_sectional_zscore,
)
from quantcortex.alpha.factors.classical._fundamentals import (
    available_mean,
    panel_axes,
    pit_panel,
    validate_fundamentals,
)


class QualityFactor:
    """Composite ROE, gross-margin, and cash-backed-earnings factor.

    Income and cash-flow measures are trailing-four-quarter sums. Balance-sheet
    denominators use the average of the latest announced balance and the
    corresponding balance roughly one year earlier. Report revisions enter only
    strictly after their announcement timestamps by default.
    """

    def compute(
        self,
        fundamentals: pd.DataFrame,
        *,
        index: pd.DatetimeIndex | None = None,
        columns: pd.Index | None = None,
    ) -> pd.DataFrame:
        """Compute the composite on availability timestamps or a supplied calendar."""
        frame = validate_fundamentals(fundamentals)
        index, columns = panel_axes(frame, index, columns)
        panels = [
            self.cross_sectional_zscore(self._roe(frame, index, columns)),
            self.cross_sectional_zscore(self._gross_margin(frame, index, columns)),
            self.cross_sectional_zscore(self._accruals(frame, index, columns)),
        ]
        return available_mean(panels)

    def roe(
        self,
        fundamentals: pd.DataFrame,
        index: pd.DatetimeIndex | None = None,
        columns: pd.Index | None = None,
    ) -> pd.DataFrame:
        """Return TTM net income divided by average book equity."""
        frame = validate_fundamentals(fundamentals)
        index, columns = panel_axes(frame, index, columns)
        return self._roe(frame, index, columns)

    def gross_margin(
        self,
        fundamentals: pd.DataFrame,
        index: pd.DatetimeIndex | None = None,
        columns: pd.Index | None = None,
    ) -> pd.DataFrame:
        """Return TTM gross profit divided by TTM revenue."""
        frame = validate_fundamentals(fundamentals)
        index, columns = panel_axes(frame, index, columns)
        return self._gross_margin(frame, index, columns)

    def accruals(
        self,
        fundamentals: pd.DataFrame,
        index: pd.DatetimeIndex | None = None,
        columns: pd.Index | None = None,
    ) -> pd.DataFrame:
        """Return sign-flipped TTM accruals divided by average total assets."""
        frame = validate_fundamentals(fundamentals)
        index, columns = panel_axes(frame, index, columns)
        return self._accruals(frame, index, columns)

    @staticmethod
    def _roe(
        fundamentals: pd.DataFrame,
        index: pd.DatetimeIndex,
        columns: pd.Index,
    ) -> pd.DataFrame:
        net_income = pit_panel(
            fundamentals, "net_income", index, columns, mode="ttm"
        )
        average_book = pit_panel(
            fundamentals, "book_value", index, columns, mode="average_balance"
        )
        return QualityFactor._safe_ratio(net_income, average_book)

    @staticmethod
    def _gross_margin(
        fundamentals: pd.DataFrame,
        index: pd.DatetimeIndex,
        columns: pd.Index,
    ) -> pd.DataFrame:
        gross_profit = pit_panel(
            fundamentals, "gross_profit", index, columns, mode="ttm"
        )
        revenue = pit_panel(fundamentals, "revenue", index, columns, mode="ttm")
        return QualityFactor._safe_ratio(gross_profit, revenue)

    @staticmethod
    def _accruals(
        fundamentals: pd.DataFrame,
        index: pd.DatetimeIndex,
        columns: pd.Index,
    ) -> pd.DataFrame:
        net_income = pit_panel(
            fundamentals, "net_income", index, columns, mode="ttm"
        )
        operating_cashflow = pit_panel(
            fundamentals, "operating_cashflow", index, columns, mode="ttm"
        )
        average_assets = pit_panel(
            fundamentals, "total_assets", index, columns, mode="average_balance"
        )
        raw_accruals = QualityFactor._safe_ratio(
            net_income - operating_cashflow, average_assets
        )
        return -raw_accruals

    @staticmethod
    def _safe_ratio(
        numerator: pd.DataFrame, denominator: pd.DataFrame
    ) -> pd.DataFrame:
        denom = denominator.where(denominator > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = numerator.divide(denom)
        return ratio.replace([np.inf, -np.inf], np.nan)

    cross_sectional_zscore = staticmethod(cross_sectional_zscore)
    rank = staticmethod(cross_sectional_rank)
