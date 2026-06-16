"""Point-in-time cross-sectional value factor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantcortex.alpha.factors.classical._cross_section import (
    cross_sectional_rank,
    cross_sectional_zscore,
    validate_prices,
)
from quantcortex.alpha.factors.classical._fundamentals import (
    available_mean,
    pit_panel,
    validate_fundamentals,
)


class ValueFactor:
    """Composite earnings-yield, book-to-price, and EBITDA-yield factor.

    Flow numerators use the latest four announced quarterly reports. Report
    revisions become visible strictly after their announcement timestamps by
    default, avoiding same-date assumptions for date-only feeds.
    """

    def compute(
        self,
        fundamentals: pd.DataFrame,
        prices: pd.DataFrame,
        *,
        market_caps: pd.DataFrame | None = None,
        prices_are_unadjusted: bool = False,
    ) -> pd.DataFrame:
        """Compute the composite value panel.

        ``prices`` must be raw, split-unadjusted closes when market capitalisation
        is derived from reported shares. Alternatively, pass a point-in-time
        daily ``market_caps`` panel. Adjusted prices and historical raw shares
        are not on the same share basis and must not be multiplied silently.
        """
        frame = validate_fundamentals(fundamentals)
        prices = self._validate_prices(prices)
        market_cap = self._resolve_market_cap(
            frame,
            prices,
            market_caps=market_caps,
            prices_are_unadjusted=prices_are_unadjusted,
        )
        panels = [
            self.cross_sectional_zscore(
                self._earnings_yield(frame, prices, market_cap)
            ),
            self.cross_sectional_zscore(
                self._book_to_price(frame, prices, market_cap)
            ),
            self.cross_sectional_zscore(self._ebitda_yield(frame, prices)),
        ]
        return available_mean(panels)

    def earnings_yield(
        self,
        fundamentals: pd.DataFrame,
        prices: pd.DataFrame,
        *,
        market_caps: pd.DataFrame | None = None,
        prices_are_unadjusted: bool = False,
    ) -> pd.DataFrame:
        """Return TTM net income divided by current market capitalisation."""
        frame = validate_fundamentals(fundamentals)
        prices = self._validate_prices(prices)
        market_cap = self._resolve_market_cap(
            frame,
            prices,
            market_caps=market_caps,
            prices_are_unadjusted=prices_are_unadjusted,
        )
        return self._earnings_yield(frame, prices, market_cap)

    def book_to_price(
        self,
        fundamentals: pd.DataFrame,
        prices: pd.DataFrame,
        *,
        market_caps: pd.DataFrame | None = None,
        prices_are_unadjusted: bool = False,
    ) -> pd.DataFrame:
        """Return latest announced book value divided by current market cap."""
        frame = validate_fundamentals(fundamentals)
        prices = self._validate_prices(prices)
        market_cap = self._resolve_market_cap(
            frame,
            prices,
            market_caps=market_caps,
            prices_are_unadjusted=prices_are_unadjusted,
        )
        return self._book_to_price(frame, prices, market_cap)

    def ebitda_yield(
        self, fundamentals: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Return TTM EBITDA divided by latest announced enterprise value."""
        frame = validate_fundamentals(fundamentals)
        prices = self._validate_prices(prices)
        return self._ebitda_yield(frame, prices)

    def ev_to_ebitda(
        self, fundamentals: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Return latest announced enterprise value divided by TTM EBITDA."""
        frame = validate_fundamentals(fundamentals)
        prices = self._validate_prices(prices)
        ebitda = pit_panel(
            frame, "ebitda", prices.index, prices.columns, mode="ttm"
        )
        enterprise_value = pit_panel(
            frame,
            "enterprise_value",
            prices.index,
            prices.columns,
            mode="latest",
        )
        return self._safe_ratio(enterprise_value, ebitda)

    @staticmethod
    def _earnings_yield(
        fundamentals: pd.DataFrame,
        prices: pd.DataFrame,
        market_cap: pd.DataFrame,
    ) -> pd.DataFrame:
        earnings = pit_panel(
            fundamentals,
            "net_income",
            prices.index,
            prices.columns,
            mode="ttm",
        )
        return ValueFactor._safe_ratio(earnings, market_cap)

    @staticmethod
    def _book_to_price(
        fundamentals: pd.DataFrame,
        prices: pd.DataFrame,
        market_cap: pd.DataFrame,
    ) -> pd.DataFrame:
        book = pit_panel(
            fundamentals, "book_value", prices.index, prices.columns, mode="latest"
        )
        return ValueFactor._safe_ratio(book, market_cap)

    @staticmethod
    def _ebitda_yield(
        fundamentals: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        ebitda = pit_panel(
            fundamentals, "ebitda", prices.index, prices.columns, mode="ttm"
        )
        enterprise_value = pit_panel(
            fundamentals,
            "enterprise_value",
            prices.index,
            prices.columns,
            mode="latest",
        )
        return ValueFactor._safe_ratio(ebitda, enterprise_value)

    @staticmethod
    def _resolve_market_cap(
        fundamentals: pd.DataFrame,
        prices: pd.DataFrame,
        *,
        market_caps: pd.DataFrame | None,
        prices_are_unadjusted: bool,
    ) -> pd.DataFrame:
        if market_caps is not None:
            validated = validate_prices(market_caps)
            return validated.reindex(index=prices.index, columns=prices.columns)
        if not isinstance(prices_are_unadjusted, bool):
            raise TypeError("prices_are_unadjusted must be a boolean")
        if not prices_are_unadjusted:
            raise ValueError(
                "pass point-in-time market_caps or explicitly set "
                "prices_are_unadjusted=True before deriving market cap"
            )
        shares = pit_panel(
            fundamentals,
            "shares_outstanding",
            prices.index,
            prices.columns,
            mode="latest",
        )
        shares = shares.where(shares > 0.0)
        return prices.multiply(shares)

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
    _validate_prices = staticmethod(validate_prices)
