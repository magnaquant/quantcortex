"""Strategy base class - wires the full quantcortex pipeline.

A strategy realises the platform's core equation::

    w_t = R_t( T_t( A_t( S_t( X_{<=t} ) ) ) )
          +risk+ +time+ +alloc+ +select+

* **S - select**:   turn point-in-time data into alpha *scores* per symbol.
* **A - allocate**: turn scores into a contract-valid weight vector via a
  :class:`~quantcortex.portfolio.base.PortfolioOptimizer`.
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

from quantcortex.portfolio.base import (
    PortfolioMode,
    PortfolioOptimizer,
    enforce_exposure_contract,
    enforce_weight_contract,
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

    def __post_init__(self) -> None:
        as_of = pd.to_datetime(self.as_of, errors="coerce", utc=True)
        if pd.isna(as_of):
            raise ValueError("as_of must be a valid timestamp")
        self.as_of = as_of.tz_localize(None)
        for name in ("prices", "returns"):
            frame = getattr(self, name)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame")
            if not isinstance(frame.index, pd.DatetimeIndex):
                raise TypeError(f"{name} must use a DatetimeIndex")
            if frame.index.hasnans or frame.index.has_duplicates:
                raise ValueError(f"{name} index must contain unique valid timestamps")
            if frame.columns.has_duplicates:
                raise ValueError(f"{name} columns must be unique")
            if any(
                not isinstance(symbol, str) or not symbol.strip()
                for symbol in frame.columns
            ):
                raise ValueError(f"{name} columns must be non-empty symbols")
            normalized = frame.copy()
            if normalized.index.tz is not None:
                normalized.index = normalized.index.tz_convert("UTC").tz_localize(None)
            normalized = normalized.sort_index()
            if not normalized.empty and normalized.index[-1] > self.as_of:
                raise ValueError(f"{name} contains observations after as_of")
            values = normalized.to_numpy(dtype=np.float64)
            if np.isinf(values).any():
                raise ValueError(f"{name} must not contain infinite values")
            if name == "prices" and (
                normalized.notna() & (normalized <= 0.0)
            ).any(axis=None):
                raise ValueError("observed prices must be strictly positive")
            setattr(self, name, normalized)
        if self.universe is not None:
            if len(self.universe) != len(set(self.universe)) or any(
                not isinstance(symbol, str) or not symbol.strip()
                for symbol in self.universe
            ):
                raise ValueError("universe must contain unique non-empty symbols")
        if not isinstance(self.extra, dict):
            raise TypeError("extra must be a dictionary")

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
        if isinstance(max_gross, (bool, np.bool_)):
            raise TypeError("max_gross must be numeric, not boolean")
        try:
            self.max_gross = float(max_gross)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("max_gross must be numeric") from exc
        if self.optimizer.mode is not self.mode:
            raise ValueError(
                f"optimizer mode {self.optimizer.mode.value!r} does not match "
                f"strategy mode {self.mode.value!r}"
            )
        if not np.isfinite(self.max_gross) or self.max_gross < 0.0:
            raise ValueError("max_gross must be finite and non-negative")
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
            raise ValueError(
                "selected assets require aligned, non-empty return history"
            )
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
        if not isinstance(scores, pd.Series):
            raise TypeError("select() must return a pandas Series")
        if scores.index.has_duplicates:
            raise ValueError("selected symbols must be unique")
        if any(
            not isinstance(symbol, str) or not symbol.strip()
            for symbol in scores.index
        ):
            raise ValueError("selected symbols must be non-empty strings")
        unknown = [
            symbol
            for symbol in scores.index
            if symbol not in ctx.prices.columns or symbol not in ctx.returns.columns
        ]
        if unknown:
            raise ValueError(f"selected symbols are absent from strategy data: {unknown}")
        if not np.all(np.isfinite(scores.to_numpy(dtype=float))):
            raise ValueError("alpha scores must contain only finite values")
        if scores.empty:
            symbols: List[str] = []
            final = np.array([], dtype=np.float64)
            return RebalanceResult(ctx.as_of, symbols, scores, final, final)

        symbols = list(scores.index)
        alloc = enforce_weight_contract(
            self.allocate(scores, ctx), mode=self.mode, name=f"{self.name}.allocate"
        )
        if alloc.size != len(symbols):
            raise ValueError(
                f"allocation length {alloc.size} does not match {len(symbols)} symbols"
            )
        overlaid = self.apply_overlays(alloc, ctx)
        if overlaid.ndim != 1 or overlaid.size != len(symbols):
            raise ValueError("overlays must preserve the allocation vector shape")
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
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame")
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("prices must use a DatetimeIndex")
        if prices.index.hasnans or prices.index.has_duplicates:
            raise ValueError("prices index must contain unique valid timestamps")
        normalized = prices.copy()
        if normalized.index.tz is not None:
            normalized.index = normalized.index.tz_convert("UTC").tz_localize(None)
        normalized = normalized.sort_index()
        decision_time = pd.to_datetime(as_of, errors="coerce", utc=True)
        if pd.isna(decision_time):
            raise ValueError("as_of must be a valid timestamp")
        decision_time = decision_time.tz_localize(None)
        px = normalized.loc[normalized.index <= decision_time]
        rets = px.pct_change(fill_method=None).dropna(how="all")
        return StrategyContext(as_of=decision_time, prices=px, returns=rets)

    def generate_weights(
        self,
        prices: pd.DataFrame,
        rebalance_dates: Sequence[pd.Timestamp],
    ) -> pd.DataFrame:
        """Return a (rebalance_dates x symbols) target-weight panel.

        Each row is the final pipeline output as of that date.  This panel is
        the canonical hand-off to the backtest engines.
        """
        decisions = pd.to_datetime(list(rebalance_dates), errors="coerce", utc=True)
        if decisions.isna().any():
            raise ValueError("rebalance_dates must contain valid timestamps")
        decisions = decisions.tz_localize(None)
        if decisions.has_duplicates:
            raise ValueError("rebalance_dates must be unique")
        rows: Dict[pd.Timestamp, pd.Series] = {}
        current_weights = pd.Series(0.0, index=prices.columns, dtype=np.float64)
        for dt in decisions.sort_values():
            ctx = self.build_context(prices, dt)
            if ctx.returns.empty:
                continue
            # Stateful allocators may require the current target to model
            # turnover. Live callers should supply actual marked-to-market
            # weights in StrategyContext.extra instead.
            ctx.extra["current_weights"] = current_weights.copy()
            result = self.rebalance(ctx)
            if result.symbols:
                target = result.as_series()
                rows[pd.Timestamp(dt)] = target
                current_weights = target.reindex(prices.columns, fill_value=0.0)
            else:
                # An empty selection is an explicit flat target. Omitting the
                # row would cause engines to forward-fill a stale invested book.
                current_weights = pd.Series(
                    0.0, index=prices.columns, dtype=np.float64
                )
                rows[pd.Timestamp(dt)] = current_weights.copy()
        if not rows:
            return pd.DataFrame(columns=prices.columns)
        panel = pd.DataFrame(rows).T.reindex(columns=prices.columns).fillna(0.0)
        panel = panel.sort_index()
        panel.index.name = "date"
        return panel
