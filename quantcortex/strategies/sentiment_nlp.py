"""News-sentiment overlay on a momentum alpha.

:class:`SentimentNLPStrategy` blends a classical cross-sectional momentum signal
with a news/earnings sentiment signal derived from an explicitly configured
FinBERT or finance-lexicon backend.  The combined alpha is::

    score = (1 - sentiment_weight) * z(momentum) + sentiment_weight * z(sentiment)

where ``z`` is a cross-sectional z-score.  The sentiment input is either:

* ``ctx.extra['news']`` - a long DataFrame of ``(date, symbol, headline)`` rows
  aggregated *causally* (only headlines ``<= as_of``) via
  :meth:`~quantcortex.alpha.factors.nlp.news_scorer.NewsScorer.aggregate_daily`; or
* ``ctx.extra['sentiment']`` - a pre-computed wide ``date x symbol`` sentiment
  panel whose timestamp contract is supplied by the caller.

If no sentiment source is available the strategy falls back to pure momentum.
The book holds the top half of names by combined score, weighted by positive
score. Intraday news must be cut off or delayed consistently with the intended
execution time.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from quantcortex.alpha.factors.classical.momentum import MomentumFactor
from quantcortex.alpha.factors.nlp.finbert_sentiment import FinBERTSentiment
from quantcortex.alpha.factors.nlp.news_scorer import NewsScorer
from quantcortex.portfolio.base import PortfolioMode
from quantcortex.portfolio.equal_weight import EqualWeight
from quantcortex.strategies.base_strategy import Strategy, StrategyContext

__all__ = ["SentimentNLPStrategy"]


class SentimentNLPStrategy(Strategy):
    """Momentum alpha with a configured news/earnings sentiment overlay.

    Parameters
    ----------
    optimizer:
        Within-selection allocator; defaults to :class:`EqualWeight`.
    base_lookback:
        Formation window (trading days) for the momentum base signal.
    base_gap:
        Most-recent days skipped in the momentum window.
    sentiment_weight:
        Blend weight on the sentiment z-score in ``[0, 1]``.
    half_life:
        Recency half-life (days) for causal news aggregation.
    sentiment:
        Optional configured sentiment scorer. The default uses the deterministic
        lexicon backend; pass ``FinBERTSentiment(backend="transformers")`` to
        require FinBERT.
    **kw:
        Forwarded to :class:`~quantcortex.strategies.base_strategy.Strategy`.
    """

    def __init__(
        self,
        *,
        optimizer=None,
        base_lookback: int = 126,
        base_gap: int = 21,
        sentiment_weight: float = 0.5,
        half_life: float = 3.0,
        sentiment: Optional[FinBERTSentiment] = None,
        **kw,
    ) -> None:
        optimizer = optimizer if optimizer is not None else EqualWeight()
        super().__init__(optimizer, mode=PortfolioMode.LONG_ONLY, **kw)
        if (
            isinstance(sentiment_weight, (bool, np.bool_))
            or not np.isfinite(sentiment_weight)
            or not 0.0 <= sentiment_weight <= 1.0
        ):
            raise ValueError("sentiment_weight must be in [0, 1]")
        for name, value in (
            ("base_lookback", base_lookback),
            ("base_gap", base_gap),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
            ):
                raise TypeError(f"{name} must be an integer")
        self.base_lookback = int(base_lookback)
        self.base_gap = int(base_gap)
        if self.base_lookback <= 0:
            raise ValueError("base_lookback must be positive")
        if self.base_gap < 0:
            raise ValueError("base_gap must be non-negative")
        if self.base_gap >= self.base_lookback:
            raise ValueError("base_gap must be smaller than base_lookback")
        self.sentiment_weight = float(sentiment_weight)
        configured_sentiment = (
            sentiment
            if sentiment is not None
            else FinBERTSentiment(backend="lexicon")
        )
        self._scorer = NewsScorer(
            sentiment=configured_sentiment, half_life=half_life
        )

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    def select(self, ctx: StrategyContext) -> pd.Series:
        prices = ctx.prices
        if prices.shape[1] == 0 or len(prices) < 2:
            return pd.Series(dtype=float)

        mom_z = self._momentum_zscore(prices)
        if mom_z.dropna().empty:
            return pd.Series(dtype=float)

        sent_z = self._sentiment_zscore(ctx, prices.columns)

        if sent_z is None or sent_z.dropna().empty:
            # No usable sentiment -> pure momentum.
            return self._actionable_scores(mom_z)

        combined = self._blend(mom_z, sent_z)
        combined = combined.dropna()
        if combined.empty:
            return self._actionable_scores(mom_z)
        return self._actionable_scores(combined)

    # ------------------------------------------------------------------ #
    # Allocation: top half by combined score, weighted by positive score
    # ------------------------------------------------------------------ #
    def allocate(self, scores: pd.Series, ctx: StrategyContext) -> np.ndarray:
        clean = scores.dropna()
        if clean.empty:
            raise ValueError("sentiment allocation requires non-empty scores")

        n_keep = max(1, int(np.ceil(len(clean) / 2)))
        chosen = clean.sort_values(ascending=False).head(n_keep).index

        # Weight by positive (shifted) score within the kept names.
        masked = pd.Series(0.0, index=scores.index, dtype=float)
        masked.loc[chosen] = scores.loc[chosen]
        if float(scores.loc[chosen].clip(lower=0).sum()) == 0.0:
            raise ValueError("sentiment allocation requires a positive score")
        return self.scores_to_weights(masked)

    @staticmethod
    def _actionable_scores(scores: pd.Series) -> pd.Series:
        clean = scores.dropna().astype(float)
        if clean.empty or not (clean > 0.0).any():
            return pd.Series(dtype=float)
        return clean

    # ------------------------------------------------------------------ #
    # Signal helpers
    # ------------------------------------------------------------------ #
    def _momentum_zscore(self, prices: pd.DataFrame) -> pd.Series:
        """Latest cross-sectional momentum z-score per symbol."""
        lb = min(self.base_lookback, len(prices) - 1)
        if lb <= 1:
            ret = (prices.iloc[-1] / prices.iloc[0] - 1.0)
            return self._zscore(ret.astype(float))
        gap = min(self.base_gap, lb - 1)
        factor = MomentumFactor(lookback=lb, gap=gap)
        panel = factor.compute(prices)
        z_panel = factor.cross_sectional_zscore(panel)
        last = z_panel.iloc[-1]
        if last.dropna().empty:
            ret = (prices.iloc[-1] / prices.iloc[0] - 1.0)
            return self._zscore(ret.astype(float))
        return last.astype(float)

    def _sentiment_zscore(
        self, ctx: StrategyContext, symbols: pd.Index
    ) -> Optional[pd.Series]:
        """Latest cross-sectional sentiment z-score per symbol (causal)."""
        extra = ctx.extra if isinstance(ctx.extra, dict) else {}

        # Pre-computed wide panel takes precedence if supplied.
        panel = extra.get("sentiment")
        if isinstance(panel, pd.DataFrame) and not panel.empty:
            if not isinstance(panel.index, pd.DatetimeIndex):
                raise ValueError("sentiment panel must use a DatetimeIndex")
            if panel.index.hasnans or not panel.index.is_unique:
                raise ValueError("sentiment panel index must contain unique valid dates")
            if not panel.index.is_monotonic_increasing:
                raise ValueError("sentiment panel dates must be sorted in increasing order")
            if panel.columns.has_duplicates or any(
                not isinstance(symbol, str) or not symbol.strip()
                for symbol in panel.columns
            ):
                raise ValueError(
                    "sentiment panel columns must be unique non-empty symbols"
                )
            panel = panel.copy()
            panel.index = pd.to_datetime(panel.index, utc=True).tz_convert(None)
            panel.columns = pd.Index([symbol.strip() for symbol in panel.columns])
            if panel.columns.has_duplicates:
                raise ValueError(
                    "sentiment symbols must remain unique after whitespace trimming"
                )
            numeric = panel.apply(pd.to_numeric, errors="coerce")
            if (numeric.isna() & panel.notna()).any(axis=None):
                raise ValueError("sentiment panel contains non-numeric observations")
            if np.isinf(numeric.to_numpy(dtype=float)).any():
                raise ValueError("sentiment panel contains infinite observations")
            panel = numeric
            causal = panel.loc[panel.index <= ctx.as_of]
            if not causal.empty:
                latest = causal.iloc[-1].reindex(symbols)
                return self._zscore(latest.astype(float))

        news = extra.get("news")
        if isinstance(news, pd.DataFrame) and not news.empty:
            row = self._scorer.aggregate_daily(
                news,
                time_decay=True,
                lookback_days=int(self._scorer.half_life * 7),
                as_of=ctx.as_of,
            )
            if row is None or row.empty:
                return None
            latest = row.iloc[-1].reindex(symbols)
            if latest.dropna().empty:
                return None
            return self._zscore(latest.astype(float))

        return None

    def _blend(self, mom_z: pd.Series, sent_z: pd.Series) -> pd.Series:
        """Weighted blend of the two z-scores over their union of symbols."""
        idx = mom_z.index.union(sent_z.index)
        m = mom_z.reindex(idx)
        s = sent_z.reindex(idx)
        sw = self.sentiment_weight
        # Where a side is missing, lean fully on the other to avoid dropping names.
        blended = pd.Series(index=idx, dtype=float)
        for sym in idx:
            mv, sv = m.get(sym), s.get(sym)
            if pd.notna(mv) and pd.notna(sv):
                blended[sym] = (1.0 - sw) * mv + sw * sv
            elif pd.notna(mv):
                blended[sym] = mv
            elif pd.notna(sv):
                blended[sym] = sv
        return blended

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        """Cross-sectional z-score of a single cross-section, robust to ties."""
        s = series.astype(float)
        valid = s.dropna()
        if valid.empty:
            return s
        mu = float(valid.mean())
        sd = float(valid.std(ddof=0))
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(0.0, index=s.index, dtype=float).where(s.notna())
        return (s - mu) / sd
