"""End-to-end deep-RL portfolio strategy.

:class:`DRLPortfolioStrategy` delegates allocation to a
:class:`~portfolio.drl_allocator.DRLAllocator` - a PPO agent that observes a
trailing window of asset returns and emits long-only weights via a softmax
policy.  The agent is (re)trained on a rolling multi-year window of returns
periodically (``retrain_freq``) and the trained model is cached on the instance.

Reward signal (handled inside :class:`DRLAllocator`'s environment):

    reward_t = log(1 + w_t . r_t)  -  txn_cost * turnover_t  -  var_penalty_t

i.e. the realised portfolio log-return, penalised for turnover (transaction
costs) and for cross-sectional position variance (a light risk adjustment) -
a risk-adjusted return net of trading costs.

Training requires ``gymnasium`` + ``stable-baselines3``.  When those optional
heavy libraries are absent, :meth:`DRLAllocator.train` is wrapped in a
``try/except`` and the allocator transparently falls back to its deterministic,
fully-offline policy (risk-adjusted-momentum softmax), so the strategy always
produces contract-valid weights offline.

Selection emits an equal score over the full present universe; the DRL allocator
performs the actual cross-asset allocation.  Everything is strictly causal: the
agent only ever sees returns ``<= as_of`` and is never trained on data beyond
the current rebalance date.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from portfolio.base import PortfolioMode
from portfolio.drl_allocator import DRLAllocator
from strategies.base_strategy import Strategy, StrategyContext

__all__ = ["DRLPortfolioStrategy"]


class DRLPortfolioStrategy(Strategy):
    """PPO end-to-end allocator with rolling multi-year retraining.

    Parameters
    ----------
    optimizer:
        A :class:`DRLAllocator`.  Defaults to ``DRLAllocator(window=60)``.
    train_window:
        Trailing window (trading days) of returns used to (re)train the agent.
        Defaults to ~3 years (756 days).
    retrain_freq:
        Pandas period alias governing the minimum spacing between retrains
        (``"Y"`` -> at most once per calendar year).
    **kw:
        Forwarded to :class:`~strategies.base_strategy.Strategy`.
    """

    def __init__(
        self,
        *,
        optimizer=None,
        train_window: int = 756,
        retrain_freq: str = "Y",
        **kw,
    ) -> None:
        optimizer = optimizer if optimizer is not None else DRLAllocator(window=60)
        super().__init__(optimizer, mode=PortfolioMode.LONG_ONLY, **kw)
        self.train_window = int(train_window)
        self.retrain_freq = str(retrain_freq)

        self._trained = False
        self._last_train: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------ #
    # Selection: equal score over the full present universe
    # ------------------------------------------------------------------ #
    def select(self, ctx: StrategyContext) -> pd.Series:
        symbols = [s for s in ctx.prices.columns if s in ctx.returns.columns]
        if not symbols:
            return pd.Series(dtype=float)
        return pd.Series(1.0, index=symbols, dtype=float)

    # ------------------------------------------------------------------ #
    # Allocation: rolling-window (re)train, then DRL optimize
    # ------------------------------------------------------------------ #
    def allocate(self, scores: pd.Series, ctx: StrategyContext) -> np.ndarray:
        symbols = list(scores.index)
        sub_returns = ctx.asset_returns(symbols)
        if sub_returns.empty or sub_returns.shape[1] != len(symbols):
            return self.scores_to_weights(scores)

        # Rolling training window (most recent ``train_window`` rows).
        train_returns = sub_returns.iloc[-self.train_window :]
        self._maybe_train(train_returns, ctx.as_of)

        try:
            weights = self.optimizer.optimize(sub_returns)
        except Exception:
            # Degenerate optimizer state -> safe equal-weight allocation.
            return self.scores_to_weights(scores)

        if len(weights) != len(symbols):
            return self.scores_to_weights(scores)
        return np.asarray(weights, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Rolling retrain (offline-safe)
    # ------------------------------------------------------------------ #
    def _maybe_train(self, train_returns: pd.DataFrame, as_of: pd.Timestamp) -> None:
        """(Re)train the DRL agent if the retrain cadence has elapsed.

        Training is attempted via ``DRLAllocator.train`` (which lazily imports
        SB3 / gymnasium).  Any failure - missing heavy libs, too little data -
        is swallowed: the allocator's deterministic offline policy is used
        instead, so allocation never breaks.
        """
        if not isinstance(self.optimizer, DRLAllocator):
            return

        as_of = pd.Timestamp(as_of)
        if self._trained and self._last_train is not None:
            if not self._retrain_due(self._last_train, as_of):
                return

        # Need more rows than the observation window to form an episode.
        if train_returns.shape[0] <= self.optimizer.window + 1:
            return

        try:
            self.optimizer.train(train_returns)
            self._trained = True
        except Exception:
            # SB3/gymnasium absent or training failed -> offline fallback.
            self._trained = False
        # Record the attempt regardless, so we respect the retrain cadence and
        # do not hammer training every single rebalance.
        self._last_train = as_of

    def _retrain_due(self, last: pd.Timestamp, now: pd.Timestamp) -> bool:
        """True if ``now`` falls in a later ``retrain_freq`` period than ``last``."""
        try:
            p_last = pd.Period(last, freq=self.retrain_freq)
            p_now = pd.Period(now, freq=self.retrain_freq)
            return p_now > p_last
        except Exception:
            return True
