"""GBDT cross-sectional momentum strategy with walk-forward refitting.

:class:`MomentumMLStrategy` learns a cross-sectional return predictor with a
gradient-boosted decision tree (:class:`~alpha.factors.ml.gbdt_factor.GBDTFactor`).
At each rebalance it builds a strictly-causal feature matrix from the price
panel (multi-horizon momentum, trailing volatility, recent return and
moving-average ratios) - optionally enriched with Alpha158 features when an
OHLCV panel is supplied in ``ctx.extra['ohlcv']`` - and predicts an alpha score
per symbol.  The model is refit on a trailing window of stacked
``(date, symbol)`` samples no more often than ``refit_freq`` and cached on the
instance between rebalances.

The top ``top_quantile`` of symbols by predicted score are held, equal-weighted
(or routed through the optimizer).  A cold start with insufficient history falls
back to a plain 12-1 :class:`~alpha.factors.classical.momentum.MomentumFactor`.

All features and labels are causal: a feature on date ``t`` uses only prices
``<= t`` and the training label is the *historical* next-period cross-sectional
return, so no future information ever enters the fitted model or the score.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from alpha.factors.classical.momentum import MomentumFactor
from alpha.factors.ml.gbdt_factor import GBDTFactor
from portfolio.base import PortfolioMode
from portfolio.equal_weight import EqualWeight
from strategies.base_strategy import Strategy, StrategyContext

__all__ = ["MomentumMLStrategy"]

# Momentum horizons (trading days) used to build the price-based feature matrix.
_MOM_HORIZONS = (21, 63, 126, 252)
# Moving-average ratio windows (price / MA - 1).
_MA_WINDOWS = (21, 63, 126)
# Number of label-forming periods that must be reserved at the tail of history.
_LABEL_HORIZON = 21


class MomentumMLStrategy(Strategy):
    """Cross-sectional momentum strategy powered by a GBDT predictor.

    Parameters
    ----------
    optimizer:
        Within-selection allocator; defaults to :class:`EqualWeight`.
    top_quantile:
        Fraction of the cross-section (by predicted score) held each rebalance.
    refit_freq:
        Pandas period alias governing the *minimum* spacing between model
        refits (``"Q"`` -> quarterly).  The fitted model and its refit date are
        cached on the instance.
    lookback, gap:
        Formation window and skip used by the cold-start momentum fallback.
    max_train:
        Maximum number of trailing *dates* contributing training samples.
    **kw:
        Forwarded to :class:`~strategies.base_strategy.Strategy`.
    """

    def __init__(
        self,
        *,
        optimizer=None,
        top_quantile: float = 0.3,
        refit_freq: str = "Q",
        lookback: int = 252,
        gap: int = 21,
        max_train: int = 756,
        **kw,
    ) -> None:
        optimizer = optimizer if optimizer is not None else EqualWeight()
        super().__init__(optimizer, mode=PortfolioMode.LONG_ONLY, **kw)
        if not 0.0 < top_quantile <= 1.0:
            raise ValueError("top_quantile must be in (0, 1]")
        self.top_quantile = float(top_quantile)
        self.refit_freq = str(refit_freq)
        self.lookback = int(lookback)
        self.gap = int(gap)
        self.max_train = int(max_train)
        self.label_horizon = _LABEL_HORIZON

        # Cached model state (walk-forward refit).
        self._model: Optional[GBDTFactor] = None
        self._last_refit: Optional[pd.Timestamp] = None
        self._feature_cols: Optional[List[str]] = None

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    def select(self, ctx: StrategyContext) -> pd.Series:
        prices = ctx.prices
        if prices.shape[1] == 0:
            return pd.Series(dtype=float)

        ohlcv = None
        if isinstance(ctx.extra, dict):
            maybe = ctx.extra.get("ohlcv")
            if isinstance(maybe, dict) and maybe:
                ohlcv = maybe

        # Build the full feature panel (long, MultiIndexed by (date, symbol)).
        feature_panel = self._build_feature_panel(prices, ohlcv)
        current = self._current_features(prices, ohlcv)

        # Cold start: not enough history for a labelled training set.
        min_history = max(_MOM_HORIZONS) + self.label_horizon + 30
        if (
            feature_panel is None
            or current is None
            or current.empty
            or len(prices) < min_history
        ):
            return self._cold_start(prices)

        # (Re)fit the model on a trailing window if the refit cadence elapsed.
        self._maybe_refit(feature_panel, ctx.as_of)
        if self._model is None or self._feature_cols is None:
            return self._cold_start(prices)

        try:
            X = current.reindex(columns=self._feature_cols)
            preds = self._model.predict(X)
        except Exception:
            return self._cold_start(prices)

        scores = pd.Series(preds, index=current.index, dtype=float).dropna()
        if scores.empty:
            return self._cold_start(prices)
        return scores

    # ------------------------------------------------------------------ #
    # Allocation: hold the top quantile, equal-weight (or via optimizer)
    # ------------------------------------------------------------------ #
    def allocate(self, scores: pd.Series, ctx: StrategyContext) -> np.ndarray:
        """Select the top ``top_quantile`` of symbols and weight them.

        The returned vector is aligned to ``scores.index`` (the contract the
        base pipeline expects): non-selected names receive zero weight.
        """
        clean = scores.dropna()
        if clean.empty:
            n = len(scores)
            return np.full(n, 1.0 / n, dtype=np.float64) if n else np.array([])

        n_keep = max(1, int(np.ceil(len(clean) * self.top_quantile)))
        chosen = clean.sort_values(ascending=False).head(n_keep).index

        sub_returns = ctx.asset_returns(list(chosen))
        if (
            sub_returns.shape[1] == len(chosen)
            and not sub_returns.empty
            and not isinstance(self.optimizer, EqualWeight)
        ):
            chosen_w = self.optimizer.optimize(sub_returns)
            chosen_w = pd.Series(chosen_w, index=list(sub_returns.columns))
        else:
            chosen_w = pd.Series(1.0 / n_keep, index=chosen, dtype=float)

        weights = pd.Series(0.0, index=scores.index, dtype=float)
        weights.loc[chosen_w.index] = chosen_w.to_numpy()
        total = float(weights.sum())
        if total > 0:
            weights = weights / total
        return weights.to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Walk-forward refit
    # ------------------------------------------------------------------ #
    def _maybe_refit(self, feature_panel: pd.DataFrame, as_of: pd.Timestamp) -> None:
        as_of = pd.Timestamp(as_of)
        if self._model is not None and self._last_refit is not None:
            if not self._refit_due(self._last_refit, as_of):
                return

        labels = feature_panel["__label__"]
        feats = feature_panel.drop(columns=["__label__"])
        train = feats.join(labels.rename("__y__"), how="inner").dropna(how="any")
        if train.empty:
            return

        # Restrict to the most recent ``max_train`` distinct dates.
        dates = train.index.get_level_values(0)
        unique_dates = pd.Index(sorted(pd.unique(dates)))
        if len(unique_dates) > self.max_train:
            keep = unique_dates[-self.max_train :]
            train = train[dates.isin(keep)]
        if len(train) < 50:
            return

        feature_cols = [c for c in train.columns if c != "__y__"]
        model = GBDTFactor(model="auto")
        try:
            # Train on the cross-sectional rank of the forward return (robust to
            # heavy tails), matching GBDTFactor's recommended target transform.
            y = train.groupby(level=0)["__y__"].transform(
                lambda s: pd.Series(GBDTFactor.rank_scores(s.to_numpy()), index=s.index)
            )
            mask = y.notna()
            model.fit(train.loc[mask, feature_cols], y[mask])
        except Exception:
            return

        self._model = model
        self._feature_cols = feature_cols
        self._last_refit = as_of

    def _refit_due(self, last: pd.Timestamp, now: pd.Timestamp) -> bool:
        """True if ``now`` falls in a later ``refit_freq`` period than ``last``."""
        try:
            p_last = pd.Period(last, freq=self.refit_freq)
            p_now = pd.Period(now, freq=self.refit_freq)
            return p_now > p_last
        except Exception:
            return True

    # ------------------------------------------------------------------ #
    # Feature engineering (strictly causal)
    # ------------------------------------------------------------------ #
    def _build_feature_panel(
        self,
        prices: pd.DataFrame,
        ohlcv: Optional[Dict[str, pd.DataFrame]],
    ) -> Optional[pd.DataFrame]:
        """Build a long (date, symbol) feature+label panel from history.

        The label is the *next-period* cross-sectional return (causal: it is a
        historical realised return, lagged into the future only relative to the
        feature date, never beyond the as-of boundary).
        """
        feat = self._features_wide(prices, ohlcv)
        if feat is None or feat.empty:
            return None

        # Label: forward ``label_horizon`` return, shifted back so row t carries
        # the return realised over (t, t+h].  Tail rows with no realised future
        # become NaN and are dropped from training.
        fwd = prices.pct_change(self.label_horizon).shift(-self.label_horizon)
        label_long = fwd.stack(future_stack=True)
        label_long.index = label_long.index.set_names(["date", "symbol"])

        panel = feat.join(label_long.rename("__label__"), how="left")
        return panel

    def _features_wide(
        self,
        prices: pd.DataFrame,
        ohlcv: Optional[Dict[str, pd.DataFrame]],
    ) -> Optional[pd.DataFrame]:
        """Return a long (date, symbol) feature matrix for all history."""
        if ohlcv:
            return self._alpha158_features(prices, ohlcv)
        return self._price_features(prices)

    def _price_features(self, prices: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Multi-horizon momentum / vol / MA-ratio features (close-only)."""
        if prices.shape[1] == 0:
            return None
        ret = prices.pct_change()
        frames: Dict[str, pd.DataFrame] = {}

        for h in _MOM_HORIZONS:
            frames[f"mom_{h}"] = prices.shift(self.gap) / prices.shift(h) - 1.0
        for w in _MA_WINDOWS:
            ma = prices.rolling(w, min_periods=w).mean()
            frames[f"ma_ratio_{w}"] = prices / ma - 1.0
        frames["vol_63"] = ret.rolling(63, min_periods=20).std()
        frames["vol_21"] = ret.rolling(21, min_periods=10).std()
        frames["ret_5"] = prices.pct_change(5)
        frames["ret_21"] = prices.pct_change(21)

        long_frames = []
        for name, wide in frames.items():
            s = wide.stack(future_stack=True)
            s.index = s.index.set_names(["date", "symbol"])
            long_frames.append(s.rename(name))
        out = pd.concat(long_frames, axis=1)
        out = out.replace([np.inf, -np.inf], np.nan)
        return out

    def _alpha158_features(
        self, prices: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """Alpha158 features per symbol stacked into a long panel."""
        from alpha.feature_engineering.alpha158 import Alpha158

        engine = Alpha158()
        per_symbol = []
        for sym in prices.columns:
            df = ohlcv.get(sym)
            if df is None or df.empty:
                continue
            try:
                feats = engine.compute(df)
            except Exception:
                continue
            feats = feats.replace([np.inf, -np.inf], np.nan)
            feats = feats.reindex(prices.index)
            feats.index = pd.MultiIndex.from_product(
                [feats.index, [sym]], names=["date", "symbol"]
            )
            per_symbol.append(feats)
        if not per_symbol:
            return self._price_features(prices)
        out = pd.concat(per_symbol).sort_index()
        return out

    def _current_features(
        self,
        prices: pd.DataFrame,
        ohlcv: Optional[Dict[str, pd.DataFrame]],
    ) -> Optional[pd.DataFrame]:
        """Feature rows for the most recent (as-of) date, indexed by symbol."""
        feat = self._features_wide(prices, ohlcv)
        if feat is None or feat.empty:
            return None
        last_date = feat.index.get_level_values(0).max()
        current = feat.xs(last_date, level=0)
        # Drop symbols whose current features are entirely missing.
        current = current.dropna(how="all")
        return current

    # ------------------------------------------------------------------ #
    # Cold start
    # ------------------------------------------------------------------ #
    def _cold_start(self, prices: pd.DataFrame) -> pd.Series:
        """Plain 12-1 momentum scores when ML history is insufficient."""
        if len(prices) <= self.gap + 1 or prices.shape[1] == 0:
            return pd.Series(dtype=float)
        lb = min(self.lookback, len(prices) - 1)
        gap = min(self.gap, lb - 1) if lb > 1 else 0
        try:
            factor = MomentumFactor(lookback=lb, gap=gap)
            panel = factor.compute(prices)
        except Exception:
            return pd.Series(dtype=float)
        last = panel.iloc[-1].dropna()
        if last.empty:
            # Final fallback: simple total return.
            ret = (prices.iloc[-1] / prices.iloc[0] - 1.0).dropna()
            return ret.astype(float)
        return last.astype(float)
