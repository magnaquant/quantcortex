"""Macro-regime asset-class rotation strategy.

:class:`MacroTimingStrategy` rotates across broad asset classes (equities,
bonds, commodities, defensives) by ranking each class on its trailing
risk-adjusted momentum (return / volatility) and holding the strongest classes.
A macro/market regime classifier (:class:`~timing.hmm_regime.HMMRegime`) gates
the selection: in a bear regime the strategy retreats to the defensive class
(or the fewest, safest names), while in calmer regimes it holds the leaders.

Within the selected names, capital is allocated by the configured optimizer
(risk parity / inverse vol by default), so risk - not dollars - is balanced
across the held assets.

Everything is strictly causal: momentum, volatility and the regime label at
date ``t`` use only data observed on or before ``t``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from quantcortex.portfolio.base import PortfolioMode
from quantcortex.portfolio.risk_parity import RiskParity
from quantcortex.strategies.base_strategy import Strategy, StrategyContext
from quantcortex.timing.hmm_regime import BEAR, SIDEWAYS, HMMRegime

__all__ = ["MacroTimingStrategy"]

_TRADING_DAYS = 252.0


class MacroTimingStrategy(Strategy):
    """Macro-regime gated asset-class rotation.

    Parameters
    ----------
    optimizer:
        Within-selection allocator; defaults to :class:`RiskParity`.
    top_classes:
        Number of asset classes held in a non-bear regime.
    mom_lookback, mom_gap:
        Formation window and skip for risk-adjusted momentum.
    regime:
        Enable the macro/market regime gate.
    **kw:
        Forwarded to :class:`~strategies.base_strategy.Strategy`.
    """

    #: Asset-class proxy ETFs.  Tolerant of whichever symbols are present.
    GROUPS: Dict[str, List[str]] = {
        "equities": ["SPY", "QQQ"],
        "bonds": ["TLT", "IEF"],
        "commodities": ["GLD", "DBC"],
        "defensive": ["XLP", "XLU"],
    }
    #: Class preferred when the regime turns bearish.
    DEFENSIVE_CLASS: str = "defensive"

    def __init__(
        self,
        *,
        optimizer=None,
        top_classes: int = 2,
        mom_lookback: int = 126,
        mom_gap: int = 21,
        regime: bool = True,
        **kw,
    ) -> None:
        optimizer = optimizer if optimizer is not None else RiskParity()
        super().__init__(optimizer, mode=PortfolioMode.LONG_ONLY, **kw)
        self.top_classes = int(top_classes)
        self.mom_lookback = int(mom_lookback)
        self.mom_gap = int(mom_gap)
        self.regime_enabled = bool(regime)
        self._hmm = HMMRegime(n_states=3)

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    def select(self, ctx: StrategyContext) -> pd.Series:
        returns = ctx.returns
        if returns.empty:
            return pd.Series(dtype=float)

        present = {
            name: [s for s in members if s in returns.columns]
            for name, members in self.GROUPS.items()
        }
        present = {n: m for n, m in present.items() if m}
        if not present:
            return pd.Series(dtype=float)

        # Risk-adjusted momentum (return / vol) per asset class.
        class_score = self._class_momentum(returns, present)
        ranked = sorted(class_score, key=lambda n: class_score[n], reverse=True)

        regime = self._macro_regime(ctx)

        if regime == BEAR:
            # Retreat to defensives (or the single best class if absent).
            if self.DEFENSIVE_CLASS in present:
                chosen = [self.DEFENSIVE_CLASS]
            else:
                chosen = ranked[:1]
        elif regime == SIDEWAYS:
            chosen = ranked[: max(1, self.top_classes - 1)]
        else:  # BULL or unknown
            chosen = ranked[: max(1, self.top_classes)]

        members = [s for c in chosen for s in present[c]]
        members = list(dict.fromkeys(members))
        if not members:
            return pd.Series(dtype=float)

        # Scores: per-member risk-adjusted momentum (positive => attractive).
        member_scores = self._member_momentum(returns, members)
        member_scores = member_scores.dropna()
        if member_scores.empty:
            return pd.Series(0.0, index=members, dtype=float)
        return member_scores

    # ------------------------------------------------------------------ #
    # Allocation: optimizer over selected names (risk parity by default)
    # ------------------------------------------------------------------ #
    def allocate(self, scores: pd.Series, ctx: StrategyContext) -> np.ndarray:
        symbols = list(scores.index)
        sub_returns = ctx.asset_returns(symbols)
        if sub_returns.shape[1] != len(symbols) or sub_returns.empty:
            return self.scores_to_weights(scores)
        try:
            return self.optimizer.optimize(sub_returns)
        except Exception:
            return self.scores_to_weights(scores)

    # ------------------------------------------------------------------ #
    # Signal helpers
    # ------------------------------------------------------------------ #
    def _class_momentum(
        self, returns: pd.DataFrame, present: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """Risk-adjusted momentum (mean/std of returns) per asset class."""
        lb = min(self.mom_lookback, len(returns))
        gap = min(self.mom_gap, max(0, lb - 2))
        scores: Dict[str, float] = {}
        for name, members in present.items():
            grp = returns[members].mean(axis=1)
            window = grp.iloc[-lb:]
            if gap > 0 and len(window) > gap:
                window = window.iloc[:-gap]
            window = window.dropna()
            if window.empty:
                scores[name] = 0.0
                continue
            sd = float(window.std(ddof=0))
            mu = float(window.mean())
            ra = mu / sd if sd > 0 and np.isfinite(sd) else 0.0
            scores[name] = ra if np.isfinite(ra) else 0.0
        return scores

    def _member_momentum(
        self, returns: pd.DataFrame, members: List[str]
    ) -> pd.Series:
        """Per-member risk-adjusted momentum used as the alpha score."""
        lb = min(self.mom_lookback, len(returns))
        gap = min(self.mom_gap, max(0, lb - 2))
        out: Dict[str, float] = {}
        for sym in members:
            r = returns[sym].iloc[-lb:]
            if gap > 0 and len(r) > gap:
                r = r.iloc[:-gap]
            r = r.dropna()
            if r.empty:
                out[sym] = np.nan
                continue
            sd = float(r.std(ddof=0))
            mu = float(r.mean())
            ra = mu / sd if sd > 0 and np.isfinite(sd) else 0.0
            out[sym] = ra if np.isfinite(ra) else 0.0
        return pd.Series(out, dtype=float)

    def _macro_regime(self, ctx: StrategyContext) -> Optional[int]:
        """Classify the prevailing macro/market regime (or ``None`` if gated off)."""
        if not self.regime_enabled:
            return None

        feats = self._regime_features(ctx)
        if feats is None or len(feats) < 60:
            return None
        try:
            self._hmm.fit(feats)
            return int(self._hmm.current_regime(feats))
        except Exception:
            return None

    def _regime_features(self, ctx: StrategyContext) -> Optional[pd.DataFrame]:
        """Build the (returns, realized_vol, vix) feature frame for the HMM.

        Prefers an explicit macro feature frame in ``ctx.extra['macro']`` (a
        :class:`~alpha.feature_engineering.macro_features.MacroFeatures` output
        or compatible frame); otherwise derives features from the market return.
        """
        extra = ctx.extra if isinstance(ctx.extra, dict) else {}
        macro = extra.get("macro")
        if isinstance(macro, pd.DataFrame) and not macro.empty:
            cols = {c.lower(): c for c in macro.columns}
            if {"returns", "realized_vol", "vix"}.issubset(cols.keys()):
                return macro.rename(columns={cols[k]: k for k in cols}).loc[
                    :, ["returns", "realized_vol", "vix"]
                ]

        # Market return: benchmark proxy if present, else cross-sectional mean.
        mkt = self._market_return_series(ctx)
        if len(mkt) < 5:
            return None
        return HMMRegime._features_from_returns(mkt)

    def _market_return_series(self, ctx: StrategyContext) -> pd.Series:
        """Equity-proxy market return, falling back to the cross-sectional mean."""
        returns = ctx.returns
        for proxy in ("SPY", "QQQ"):
            if proxy in returns.columns:
                return returns[proxy].dropna()
        return returns.mean(axis=1).dropna()
