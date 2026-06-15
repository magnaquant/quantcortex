"""Strategy base class - wires the full quantcortex pipeline.

A strategy realises the platform's core equation::

    w_t = R_t( T_t( A_t( S_t( X_{<=t} ) ) ) )
          +risk+ +time+ +alloc+ +select+

* **S - select**:   turn point-in-time data into alpha *scores* per symbol.
* **A - allocate**: turn scores into a contract-valid weight vector via a
  :class:`~portfolio.base.PortfolioOptimizer`.
* **T - timing**:   scale exposure by regime / trend overlays.
* **R - risk**:     apply drawdown / vol / VaR overlays.

Timing and risk overlays are supplied as ``callable(weights, ctx) -> weights``
so the pipeline stays decoupled from each overlay's individual signature; a
strategy author binds the right context with a small lambda/partial.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from portfolio.base import (
    PortfolioMode,
    PortfolioOptimizer,
    enforce_exposure_contract,
    normalize_long_only,
    normalize_market_neutral,
)

__all__ = ["StrategyContext", "RebalanceResult", "Strategy", "Overlay"]

# An overlay maps (weights, context) -> weights.
Overlay = Callable[[np.ndarray, "StrategyContext"], np.ndarray]


@dataclass
class StrategyContext:
    """Everything the pipeline needs to compute weights at one point in time."""

    as_of: pd.Timestamp
    prices: pd.DataFrame  # wide price panel (dates x symbols), data <= as_of
    returns: pd.DataFrame  # wide simple-returns panel, data <= as_of
    universe: Optional[List[str]] = None
    extra: Dict[str, object] = field(default_factory=dict)

    def asset_returns(self, symbols: Sequence[str]) -> pd.DataFrame:
        cols = [s for s in symbols if s in self.returns.columns]
        return self.returns[cols]


@dataclass
class RebalanceResult:
    as_of: pd.Timestamp
    symbols: List[str]
    scores: pd.Series
    allocation_weights: np.ndarray  # post-allocation, pre-overlay
    target_weights: np.ndarray  # final, post timing+risk
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def as_series(self) -> pd.Series:
        return pd.Series(self.target_weights, index=self.symbols, name=self.as_of)


class Strategy(abc.ABC):
    """Abstract base class wiring the Select -> Allocate -> Time -> Risk pipeline."""

    def __init__(
        self,
        optimizer: PortfolioOptimizer,
        *,
        timing_overlays: Sequence[Overlay] = (),
        risk_overlays: Sequence[Overlay] = (),
        mode: PortfolioMode = PortfolioMode.LONG_ONLY,
        max_gross: float = 1.0,
        name: Optional[str] = None,
    ) -> None:
        self.optimizer = optimizer
        self.timing_overlays: List[Overlay] = list(timing_overlays)
        self.risk_overlays: List[Overlay] = list(risk_overlays)
        self.mode = PortfolioMode.coerce(mode)
        self.max_gross = float(max_gross)
        self._name = name or type(self).__name__

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------ #
    # S - selection (must be implemented by concrete strategies)
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def select(self, ctx: StrategyContext) -> pd.Series:
        """Return alpha scores indexed by the *selected* symbols.

        The index of the returned Series defines the assets that proceed to the
        allocation step; the values are the alpha signal (higher = more
        attractive).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # A - allocation
    # ------------------------------------------------------------------ #
    def allocate(self, scores: pd.Series, ctx: StrategyContext) -> np.ndarray:
        """Map alpha scores to contract-valid weights.

        Default behaviour delegates to the optimizer using the selected assets'
        return history (the scores select the assets; the optimizer sets the
        weights).  Strategies whose allocation is *score-driven* (weight
        proportional to alpha score) should override this and call
        :meth:`scores_to_weights`.
        """
        symbols = list(scores.index)
        sub_returns = ctx.asset_returns(symbols)
        if sub_returns.shape[1] != len(symbols) or sub_returns.empty:
            # Fall back to a score-driven allocation when return history is
            # unavailable/misaligned for some selected names.
            return self.scores_to_weights(scores)
        return self.optimizer.optimize(sub_returns)

    def scores_to_weights(self, scores: pd.Series) -> np.ndarray:
        """Helper: turn raw scores into a contract-valid weight vector."""
        if self.mode is PortfolioMode.MARKET_NEUTRAL:
            return normalize_market_neutral(scores.to_numpy())
        return normalize_long_only(scores.to_numpy())

    # ------------------------------------------------------------------ #
    # T + R - overlays
    # ------------------------------------------------------------------ #
    def apply_overlays(self, weights: np.ndarray, ctx: StrategyContext) -> np.ndarray:
        w = np.asarray(weights, dtype=np.float64)
        for overlay in self.timing_overlays:
            w = np.asarray(overlay(w, ctx), dtype=np.float64)
        for overlay in self.risk_overlays:
            w = np.asarray(overlay(w, ctx), dtype=np.float64)
        return w

    # ------------------------------------------------------------------ #
    # full pipeline
    # ------------------------------------------------------------------ #
    def rebalance(self, ctx: StrategyContext) -> RebalanceResult:
        scores = self.select(ctx)
        if scores.empty:
            symbols: List[str] = []
            final = np.array([], dtype=np.float64)
            return RebalanceResult(ctx.as_of, symbols, scores, final, final)

        symbols = list(scores.index)
        alloc = self.allocate(scores, ctx)
        overlaid = self.apply_overlays(alloc, ctx)
        final = enforce_exposure_contract(
            overlaid, max_gross=self.max_gross + 1e-9, name=self.name
        )
        return RebalanceResult(
            as_of=ctx.as_of,
            symbols=symbols,
            scores=scores,
            allocation_weights=alloc,
            target_weights=final,
            diagnostics={"n_assets": len(symbols)},
        )

    # ------------------------------------------------------------------ #
    # weight path generation (consumed by backtest engines)
    # ------------------------------------------------------------------ #
    def build_context(
        self, prices: pd.DataFrame, as_of: pd.Timestamp
    ) -> StrategyContext:
        px = prices.loc[:as_of]
        rets = px.pct_change().dropna(how="all")
        return StrategyContext(as_of=as_of, prices=px, returns=rets)

    def generate_weights(
        self,
        prices: pd.DataFrame,
        rebalance_dates: Sequence[pd.Timestamp],
    ) -> pd.DataFrame:
        """Return a (rebalance_dates x symbols) target-weight panel.

        Each row is the final pipeline output as of that date.  This panel is
        the canonical hand-off to the backtest engines.
        """
        rows: Dict[pd.Timestamp, pd.Series] = {}
        for dt in rebalance_dates:
            ctx = self.build_context(prices, pd.Timestamp(dt))
            if ctx.returns.empty:
                continue
            result = self.rebalance(ctx)
            if result.symbols:
                rows[pd.Timestamp(dt)] = result.as_series()
        if not rows:
            return pd.DataFrame(columns=prices.columns)
        panel = pd.DataFrame(rows).T.reindex(columns=prices.columns).fillna(0.0)
        panel.index.name = "date"
        return panel
