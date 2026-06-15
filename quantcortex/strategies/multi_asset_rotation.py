"""Flagship multi-asset rotation strategy.

:class:`MultiAssetRotation` rotates between asset-class *groups* (growth, real
assets, defensive) using a two-stage cross-sectional signal and gates the book
with a macro regime overlay and a VIX (volatility) exposure scaler.

Signal construction
--------------------
1. **Group information ratio (IR).**  For each group we form an equal-weight
   return series of its members and measure its information ratio against the
   benchmark over ``ir_lookback`` days:
   ``IR = mean(group_return - benchmark_return) / std(group_return - benchmark_return)``
   (sample std, ``ddof=1``).  The ``top_n_groups`` groups with the highest IR
   are selected.

2. **Residual momentum within groups.**  For each member of a selected group we
   regress its returns on the benchmark (a trailing CAPM) and measure the
   momentum of the residual (idiosyncratic) return over ``mom_lookback`` days,
   excluding the most recent ``mom_gap`` days to skip short-term reversal.  The
   residual-momentum scores feed the allocator.

Overlays
--------
* **Regime gate** (timing overlay).  An :class:`~timing.hmm_regime.HMMRegime` is
  fit on the benchmark's return and realized vol; bear -> flat, sideways -> half,
  bull -> full exposure.  Disabled with ``regime=False`` or when the fit fails.
* **VIX scaler** (risk overlay).  A :class:`~timing.vix_scaler.VIXScaler` leans
  the book down when implied (or proxied realized) volatility is elevated.

Everything is strictly causal: every signal at date ``t`` uses only data
observed on or before ``t``.  The strategy is robust to a bare close panel of
the six ETFs (a VIX series in ``ctx.extra['vix']`` is optional).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from quantcortex.portfolio.base import PortfolioMode
from quantcortex.portfolio.equal_weight import EqualWeight
from quantcortex.strategies.base_strategy import Strategy, StrategyContext
from quantcortex.timing.hmm_regime import HMMRegime
from quantcortex.timing.vix_scaler import VIXScaler

__all__ = ["MultiAssetRotation"]

_TRADING_DAYS = 252.0


class MultiAssetRotation(Strategy):
    """Asset-class rotation with regime + VIX exposure gating.

    Parameters
    ----------
    optimizer:
        Portfolio optimizer used for the within-group allocation.  Unused by the
        default score-driven :meth:`allocate` override (weights are proportional
        to positive residual momentum); defaults to :class:`EqualWeight`.
    top_n_groups:
        Number of asset-class groups to hold each rebalance.
    ir_lookback:
        Trailing window (trading days) for the group information ratio.
    mom_lookback:
        Formation window (trading days) for residual momentum.
    mom_gap:
        Most-recent days skipped in the residual-momentum window.
    target_vix:
        VIX level at which the book runs at its natural exposure.
    regime:
        Enable the HMM regime timing gate.
    vix_scale:
        Enable the VIX risk overlay.
    **kw:
        Forwarded to :class:`~strategies.base_strategy.Strategy`.

    Notes
    -----
    The benchmark (QQQ) is *also* a member of the growth group (this
    composition is the product spec).  Because the QQQ leg of the group's
    active return is identically zero, the growth group's information ratio
    reduces to the IR of ``(VGT - QQQ) / 2`` -- a relative-to-benchmark
    spread that is different in character from the other groups' full active
    spreads.  Group selection is therefore, by design, relative to growth.

    The group IR uses the *sample* standard deviation (``ddof=1``) in its
    denominator.
    """

    #: Asset-class groups mapped to their member proxy ETFs.
    GROUPS: Dict[str, List[str]] = {
        "growth": ["QQQ", "VGT"],
        "real_assets": ["GLD", "TLT"],
        "defensive": ["SPY", "VIG"],
    }
    #: Benchmark used for active-return / CAPM calculations.
    BENCHMARK: str = "QQQ"

    def __init__(
        self,
        *,
        optimizer=None,
        top_n_groups: int = 2,
        ir_lookback: int = 126,
        mom_lookback: int = 126,
        mom_gap: int = 21,
        target_vix: float = 20.0,
        regime: bool = True,
        vix_scale: bool = True,
        **kw,
    ) -> None:
        optimizer = optimizer if optimizer is not None else EqualWeight()
        super().__init__(optimizer, mode=PortfolioMode.LONG_ONLY, **kw)
        self.top_n_groups = int(top_n_groups)
        self.ir_lookback = int(ir_lookback)
        self.mom_lookback = int(mom_lookback)
        self.mom_gap = int(mom_gap)
        self.regime_enabled = bool(regime)
        self.vix_scale_enabled = bool(vix_scale)

        self._hmm = HMMRegime(n_states=3)
        self._vix_scaler = VIXScaler(target_vix=float(target_vix))

        if self.regime_enabled:
            self.timing_overlays.append(self._regime_overlay)
        if self.vix_scale_enabled:
            self.risk_overlays.append(self._vix_overlay)

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    def select(self, ctx: StrategyContext) -> pd.Series:
        prices = ctx.prices
        returns = ctx.returns
        if returns.empty:
            return pd.Series(dtype=float)

        # Resolve the groups to whichever members are actually present.
        present_groups = {
            name: [s for s in members if s in returns.columns]
            for name, members in self.GROUPS.items()
        }
        present_groups = {n: m for n, m in present_groups.items() if m}
        if not present_groups:
            return pd.Series(dtype=float)

        bench = self.BENCHMARK if self.BENCHMARK in returns.columns else None
        if bench is None:
            # No benchmark -> fall back to plain (group-agnostic) momentum.
            return self._fallback_momentum(prices)

        bench_ret = returns[bench]

        # ---- Stage 1: rank groups by information ratio vs benchmark ----
        ir_lb = min(self.ir_lookback, len(returns))
        group_ir: Dict[str, float] = {}
        for name, members in present_groups.items():
            grp_ret = returns[members].mean(axis=1)
            active = (grp_ret - bench_ret).iloc[-ir_lb:].dropna()
            if active.empty:
                continue
            # Sample standard deviation (ddof=1): this is a sample IR.
            sd = float(active.std(ddof=1))
            ir = float(active.mean()) / sd if sd > 0 and np.isfinite(sd) else 0.0
            group_ir[name] = ir if np.isfinite(ir) else 0.0

        if not group_ir:
            return self._fallback_momentum(prices)

        ranked = sorted(group_ir, key=lambda n: group_ir[n], reverse=True)
        chosen = ranked[: max(1, self.top_n_groups)]
        members = [s for n in chosen for s in present_groups[n]]
        members = list(dict.fromkeys(members))  # de-dup, preserve order

        # ---- Stage 2: residual momentum of selected members ----
        scores = self._residual_momentum(prices[members], prices[bench])
        if scores is None or scores.dropna().empty:
            # CAPM unavailable (cold start) -> plain momentum over members.
            scores = self._plain_momentum(prices[members])

        scores = scores.reindex(members).dropna()
        if scores.empty:
            return self._fallback_momentum(prices)
        return scores

    # ------------------------------------------------------------------ #
    # Allocation: long-only weight proportional to positive residual momentum
    # ------------------------------------------------------------------ #
    def allocate(self, scores: pd.Series, ctx: StrategyContext) -> np.ndarray:
        """Weight each selected member by its positive residual momentum."""
        return self.scores_to_weights(scores)

    # ------------------------------------------------------------------ #
    # Signal helpers
    # ------------------------------------------------------------------ #
    def _residual_momentum(
        self, member_prices: pd.DataFrame, bench_prices: pd.Series
    ) -> Optional[pd.Series]:
        """Latest residual (CAPM) momentum per member, or ``None`` on cold start."""
        n = len(member_prices)
        if n <= self.mom_gap + 2:
            return None
        window = min(self.mom_lookback, max(2, n - self.mom_gap - 1))
        if window <= 1:
            return None

        member_ret = member_prices.pct_change()
        bench_ret = bench_prices.reindex(member_prices.index).pct_change()

        # Trailing-window CAPM beta over the most recent ``window`` observations.
        recent = member_ret.iloc[-window:]
        x = bench_ret.iloc[-window:]
        x_aligned = x.reindex(recent.index)
        x_var = float(x_aligned.var(ddof=0))
        if not np.isfinite(x_var) or x_var <= 0:
            return None

        residual_mom: Dict[str, float] = {}
        # Use the segment that excludes the most recent ``mom_gap`` days for the
        # momentum accumulation; betas use the full trailing window.
        for col in member_prices.columns:
            y = recent[col]
            mask = y.notna() & x_aligned.notna()
            if mask.sum() < 3:
                continue
            yc = y[mask]
            xc = x_aligned[mask]
            beta = float(np.cov(yc, xc, ddof=0)[0, 1] / x_var)
            alpha = float(yc.mean() - beta * xc.mean())
            resid = yc - (alpha + beta * xc)
            # Momentum = cumulative residual return up to t-gap.
            if self.mom_gap > 0 and len(resid) > self.mom_gap:
                resid = resid.iloc[: -self.mom_gap]
            if resid.empty:
                continue
            residual_mom[col] = float(resid.sum())

        if not residual_mom:
            return None
        return pd.Series(residual_mom)

    @staticmethod
    def _plain_momentum(member_prices: pd.DataFrame) -> pd.Series:
        """Simple trailing total return per member (cold-start fallback)."""
        if len(member_prices) < 2:
            return pd.Series(
                0.0, index=member_prices.columns, dtype=float
            )
        ret = member_prices.iloc[-1] / member_prices.iloc[0] - 1.0
        return ret.astype(float)

    def _fallback_momentum(self, prices: pd.DataFrame) -> pd.Series:
        """Plain momentum over the full available universe (last resort)."""
        if prices.shape[1] == 0 or len(prices) < 2:
            return pd.Series(dtype=float)
        mom = self._plain_momentum(prices).dropna()
        return mom

    # ------------------------------------------------------------------ #
    # Overlays
    # ------------------------------------------------------------------ #
    def _market_return_series(self, ctx: StrategyContext) -> pd.Series:
        """Benchmark return series, falling back to the cross-sectional mean."""
        if self.BENCHMARK in ctx.returns.columns:
            return ctx.returns[self.BENCHMARK].dropna()
        return ctx.returns.mean(axis=1).dropna()

    def _regime_overlay(
        self, weights: np.ndarray, ctx: StrategyContext
    ) -> np.ndarray:
        """HMM regime exposure gate (bear x0, sideways x0.5, bull x1)."""
        w = np.asarray(weights, dtype=np.float64)
        mkt = self._market_return_series(ctx)
        # Need enough history for a meaningful regime fit.
        if len(mkt) < 60:
            return w
        feats = HMMRegime._features_from_returns(mkt)
        try:
            self._hmm.fit(feats)
            return self._hmm.scale_weights(w, feats)
        except Exception:
            # Any fit/predict failure -> no gate (pass weights through).
            return w

    def _vix_overlay(self, weights: np.ndarray, ctx: StrategyContext) -> np.ndarray:
        """VIX exposure scaler; derives a realized-vol proxy if no VIX given."""
        w = np.asarray(weights, dtype=np.float64)
        vix = ctx.extra.get("vix") if isinstance(ctx.extra, dict) else None
        if vix is None:
            vix = self._realized_vol_proxy(ctx)
        if vix is None:
            return w
        try:
            return self._vix_scaler.apply(w, vix)
        except Exception:
            return w

    def _realized_vol_proxy(self, ctx: StrategyContext) -> Optional[float]:
        """Annualized 21-day realized vol of the benchmark, in VIX points."""
        mkt = self._market_return_series(ctx)
        if len(mkt) < 5:
            return None
        win = min(21, len(mkt))
        rv = float(mkt.iloc[-win:].std(ddof=0))
        if not np.isfinite(rv) or rv <= 0:
            return None
        return rv * np.sqrt(_TRADING_DAYS) * 100.0
