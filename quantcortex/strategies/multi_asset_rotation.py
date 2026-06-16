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
   positive residual-momentum scores feed the allocator; if none are positive,
   the strategy holds cash rather than switching to another signal.

Overlays
--------
* **Regime gate** (timing overlay). :class:`~quantcortex.timing.hmm_regime.HMMRegime` uses
  the explicitly configured backend (seeded GMM by default) on benchmark return
  and realized vol; bear -> flat, sideways -> half, bull -> full exposure.
  Insufficient history produces a flat book; model failures stop the run.
* **VIX scaler** (risk overlay).  A :class:`~quantcortex.timing.vix_scaler.VIXScaler` leans
  the book down when implied (or proxied realized) volatility is elevated.

Everything is strictly causal: every signal at date ``t`` uses only data
observed on or before ``t``.  The strategy is robust to a bare close panel of
the six ETFs (a VIX series in ``ctx.extra['vix']`` is optional).
"""

from __future__ import annotations

from typing import ClassVar, Dict, List, Optional

import numpy as np
import pandas as pd

from quantcortex.portfolio.base import PortfolioMode, project_bounded_sum
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
    max_position_weight:
        Maximum final weight for any selected ETF. If too few assets are
        selected to deploy the full book under this limit, residual capital
        remains in cash.
    regime:
        Enable the regime timing gate.
    regime_n_states, regime_covariance_type, regime_n_iter, regime_seed:
        Gaussian-mixture or HMM configuration used by the regime gate.
    regime_reg_covar:
        GMM covariance regularization. Ignored by the HMM backend.
    regime_feature_vol_lookback:
        Rolling realized-volatility window used in the regime feature matrix.
    vix_scale:
        Enable the VIX risk overlay.
    vix_floor, vix_cap:
        Lower and upper bounds for the inverse-volatility exposure multiplier.
    vix_proxy_lookback:
        Realized-volatility lookback used when no external VIX series is supplied.
    **kw:
        Forwarded to :class:`~quantcortex.strategies.base_strategy.Strategy`.

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
    GROUPS: ClassVar[Dict[str, List[str]]] = {
        "growth": ["QQQ", "VGT"],
        "real_assets": ["GLD", "TLT"],
        "defensive": ["SPY", "VIG"],
    }
    #: Benchmark used for active-return / CAPM calculations.
    BENCHMARK: str = "QQQ"
    DEFAULT_MAX_POSITION_WEIGHT: float = 0.60

    def __init__(
        self,
        *,
        optimizer=None,
        top_n_groups: int = 2,
        ir_lookback: int = 126,
        mom_lookback: int = 126,
        mom_gap: int = 21,
        target_vix: float = 20.0,
        max_position_weight: float = DEFAULT_MAX_POSITION_WEIGHT,
        regime: bool = True,
        regime_backend: str = "gmm",
        regime_n_states: int = 3,
        regime_covariance_type: str = "full",
        regime_n_iter: int = 100,
        regime_seed: int = 42,
        regime_reg_covar: float = 1e-5,
        regime_feature_vol_lookback: int = 20,
        vix_scale: bool = True,
        vix_floor: float = 0.3,
        vix_cap: float = 1.0,
        vix_proxy_lookback: int = 21,
        **kw,
    ) -> None:
        optimizer = optimizer if optimizer is not None else EqualWeight()
        super().__init__(optimizer, mode=PortfolioMode.LONG_ONLY, **kw)
        integer_parameters = {
            "top_n_groups": top_n_groups,
            "ir_lookback": ir_lookback,
            "mom_lookback": mom_lookback,
            "mom_gap": mom_gap,
            "regime_n_states": regime_n_states,
            "regime_n_iter": regime_n_iter,
            "regime_seed": regime_seed,
            "regime_feature_vol_lookback": regime_feature_vol_lookback,
            "vix_proxy_lookback": vix_proxy_lookback,
        }
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in integer_parameters.values()
        ):
            raise TypeError("integer strategy parameters must be integers")
        if not isinstance(regime, (bool, np.bool_)):
            raise TypeError("regime must be a boolean")
        if not isinstance(vix_scale, (bool, np.bool_)):
            raise TypeError("vix_scale must be a boolean")
        if isinstance(target_vix, (bool, np.bool_)):
            raise TypeError("target_vix must be numeric, not boolean")
        if isinstance(max_position_weight, (bool, np.bool_)):
            raise TypeError("max_position_weight must be numeric, not boolean")
        try:
            target_vix = float(target_vix)
            max_position_weight = float(max_position_weight)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(
                "target_vix and max_position_weight must be numeric"
            ) from exc
        if not np.isfinite(target_vix) or target_vix <= 0.0:
            raise ValueError("target_vix must be finite and positive")
        if (
            not np.isfinite(max_position_weight)
            or max_position_weight <= 0.0
            or max_position_weight > 1.0
        ):
            raise ValueError("max_position_weight must be in (0, 1]")
        self.top_n_groups = int(top_n_groups)
        self.ir_lookback = int(ir_lookback)
        self.mom_lookback = int(mom_lookback)
        self.mom_gap = int(mom_gap)
        self.max_position_weight = max_position_weight
        self.regime_enabled = bool(regime)
        self.vix_scale_enabled = bool(vix_scale)
        if not 1 <= self.top_n_groups <= len(self.GROUPS):
            raise ValueError(f"top_n_groups must be in [1, {len(self.GROUPS)}]")
        if self.ir_lookback < 2:
            raise ValueError("ir_lookback must be at least 2")
        if self.mom_lookback <= 0:
            raise ValueError("mom_lookback must be positive")
        if self.mom_gap < 0:
            raise ValueError("mom_gap must be non-negative")
        if self.mom_gap >= self.mom_lookback:
            raise ValueError("mom_gap must be smaller than mom_lookback")
        if regime_feature_vol_lookback < 2:
            raise ValueError("regime_feature_vol_lookback must be at least 2")
        if vix_proxy_lookback < 2:
            raise ValueError("vix_proxy_lookback must be at least 2")

        self.regime_feature_vol_lookback = int(regime_feature_vol_lookback)
        self.vix_proxy_lookback = int(vix_proxy_lookback)
        self._hmm = HMMRegime(
            n_states=regime_n_states,
            covariance_type=regime_covariance_type,
            n_iter=regime_n_iter,
            seed=regime_seed,
            reg_covar=regime_reg_covar,
            backend=regime_backend,
        )
        self._vix_scaler = VIXScaler(
            target_vix=target_vix,
            floor=vix_floor,
            cap=vix_cap,
        )

        if self.regime_enabled:
            self.timing_overlays.append(self._regime_overlay)
        if self.vix_scale_enabled:
            self.risk_overlays.append(self._vix_overlay)
        self.risk_overlays.append(self._position_limit_overlay)

    @property
    def required_history(self) -> int:
        """Price sessions needed before every default signal path is mature."""
        residual_momentum = self.mom_gap + 2 * self.mom_lookback + 1
        regime_prices = (
            max(61, self.regime_feature_vol_lookback + 1)
            if self.regime_enabled
            else 0
        )
        vix_proxy_prices = (
            self.vix_proxy_lookback + 1 if self.vix_scale_enabled else 0
        )
        ir_prices = self.ir_lookback + 1
        return max(residual_momentum, regime_prices, vix_proxy_prices, ir_prices)

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
        scores = scores[scores > 0.0]
        if scores.empty:
            return pd.Series(dtype=float)
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
        residual_history = self.mom_gap + 2 * self.mom_lookback + 1
        if n < residual_history:
            return None

        member_ret = member_prices.pct_change(fill_method=None)
        bench_ret = bench_prices.reindex(member_prices.index).pct_change(
            fill_method=None
        )

        # Estimate CAPM parameters on a window preceding the momentum
        # formation window. Estimating an intercept on the same window whose
        # residuals are summed would force the score to zero by construction.
        formation_end = len(member_ret) - self.mom_gap if self.mom_gap else len(member_ret)
        formation_start = max(1, formation_end - self.mom_lookback)
        estimation_start = formation_start - self.mom_lookback
        formation = member_ret.iloc[formation_start:formation_end]
        formation_bench = bench_ret.iloc[formation_start:formation_end]
        estimation = member_ret.iloc[estimation_start:formation_start]
        estimation_bench = bench_ret.iloc[estimation_start:formation_start]
        if (
            len(formation) != self.mom_lookback
            or len(estimation) != self.mom_lookback
        ):
            return None

        residual_mom: Dict[str, float] = {}
        for col in member_prices.columns:
            y_est = estimation[col]
            est_mask = y_est.notna() & estimation_bench.notna()
            if est_mask.sum() != len(estimation):
                continue
            yc = y_est[est_mask]
            xc = estimation_bench[est_mask]
            x_var = float(xc.var(ddof=0))
            if not np.isfinite(x_var) or x_var <= 0.0:
                continue
            beta = float(np.cov(yc, xc, ddof=0)[0, 1] / x_var)
            alpha = float(yc.mean() - beta * xc.mean())
            y_form = formation[col]
            form_mask = y_form.notna() & formation_bench.notna()
            if form_mask.sum() != len(formation):
                continue
            resid = y_form[form_mask] - (
                alpha + beta * formation_bench[form_mask]
            )
            if resid.empty:
                continue
            residual_mom[col] = float(resid.sum())

        if not residual_mom:
            return None
        return pd.Series(residual_mom)

    def _plain_momentum(self, member_prices: pd.DataFrame) -> pd.Series:
        """Simple trailing total return per member (cold-start fallback)."""
        end = len(member_prices) - 1 - self.mom_gap
        if end <= 0:
            return pd.Series(
                0.0, index=member_prices.columns, dtype=float
            )
        start = max(0, end - self.mom_lookback)
        ret = member_prices.iloc[end] / member_prices.iloc[start] - 1.0
        return ret.astype(float)

    def _fallback_momentum(self, prices: pd.DataFrame) -> pd.Series:
        """Plain momentum fallback when the benchmark/group signal is unavailable."""
        if prices.shape[1] == 0 or len(prices) < 2:
            return pd.Series(dtype=float)
        mom = self._plain_momentum(prices).dropna()
        return mom[mom > 0.0]

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
        """Regime exposure gate (bear x0, sideways x0.5, bull x1)."""
        w = np.asarray(weights, dtype=np.float64)
        mkt = self._market_return_series(ctx)
        # Need enough history for a meaningful regime fit.
        if len(mkt) < 60:
            return np.zeros_like(w)
        feats = HMMRegime._features_from_returns(
            mkt,
            realized_vol_lookback=self.regime_feature_vol_lookback,
        )
        self._hmm.fit(feats)
        return self._hmm.scale_weights(w, feats)

    def _vix_overlay(self, weights: np.ndarray, ctx: StrategyContext) -> np.ndarray:
        """VIX exposure scaler; derives a realized-vol proxy if no VIX given."""
        w = np.asarray(weights, dtype=np.float64)
        vix = ctx.extra.get("vix") if isinstance(ctx.extra, dict) else None
        if vix is None:
            vix = self._realized_vol_proxy(ctx)
        if vix is None:
            vix = np.nan
        return self._vix_scaler.apply(w, vix)

    def _position_limit_overlay(
        self, weights: np.ndarray, ctx: StrategyContext
    ) -> np.ndarray:
        """Cap final positions, redistributing only when the cap is feasible."""
        del ctx
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or not np.all(np.isfinite(w)):
            raise ValueError("position limit requires a finite one-dimensional vector")
        if np.any(w < -1e-12):
            raise ValueError("position limit received negative long-only weights")
        w = np.clip(w, 0.0, None)
        gross = float(w.sum())
        if gross <= 0.0:
            return np.zeros_like(w)
        if len(w) * self.max_position_weight < gross - 1e-12:
            return np.clip(w, 0.0, self.max_position_weight)
        return project_bounded_sum(
            w,
            target_sum=gross,
            lower=0.0,
            upper=self.max_position_weight,
        )

    def _realized_vol_proxy(self, ctx: StrategyContext) -> Optional[float]:
        """Annualized configured-lookback volatility, expressed in VIX points."""
        mkt = self._market_return_series(ctx)
        if len(mkt) < 2:
            return None
        win = min(self.vix_proxy_lookback, len(mkt))
        rv = float(mkt.iloc[-win:].std(ddof=0))
        if not np.isfinite(rv) or rv <= 0:
            return None
        return rv * np.sqrt(_TRADING_DAYS) * 100.0
