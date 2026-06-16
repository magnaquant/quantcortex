"""End-to-end deep-RL portfolio strategy.

:class:`DRLPortfolioStrategy` delegates allocation to a
:class:`~quantcortex.portfolio.drl_allocator.DRLAllocator` - a PPO agent that observes a
trailing window of asset returns and the current portfolio weights, then emits
long-only weights via a softmax policy.  The agent is (re)trained on a rolling
multi-year window of returns periodically (``retrain_freq``), and the trained
model is cached on the instance.

Reward signal (handled inside :class:`DRLAllocator`'s environment):

    reward_t = log(1 + w_t . r_t)  -  txn_cost * turnover_t  -  var_penalty_t

i.e. the realised portfolio log-return, penalised for turnover (transaction
costs) and for trailing portfolio-return variance - a risk-adjusted return net
of trading costs.

Training requires ``gymnasium`` + ``stable-baselines3``. Missing dependencies
raise by default; callers may explicitly enable the documented deterministic
heuristic baseline with ``allow_heuristic_fallback=True``. Other training
failures always propagate.

Selection emits an equal score over the full present universe; the DRL allocator
performs the actual cross-asset allocation.  Everything is strictly causal: the
agent only ever sees returns ``<= as_of`` and is never trained on data beyond
the current rebalance date.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import numpy as np
import pandas as pd

from quantcortex.portfolio.base import PortfolioMode, enforce_exposure_contract
from quantcortex.portfolio.drl_allocator import DRLAllocator
from quantcortex.strategies.base_strategy import Strategy, StrategyContext

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
    allow_heuristic_fallback:
        Explicitly permit the non-DRL risk-adjusted-momentum baseline when the
        optional PPO stack is unavailable. Defaults to ``False``.
    **kw:
        Forwarded to :class:`~quantcortex.strategies.base_strategy.Strategy`.
    """

    def __init__(
        self,
        *,
        optimizer=None,
        train_window: int = 756,
        retrain_freq: str = "Y",
        allow_heuristic_fallback: bool = False,
        **kw,
    ) -> None:
        if not isinstance(allow_heuristic_fallback, (bool, np.bool_)):
            raise TypeError("allow_heuristic_fallback must be a boolean")
        optimizer = optimizer if optimizer is not None else DRLAllocator(
            window=60,
            untrained_policy=(
                "heuristic" if allow_heuristic_fallback else "error"
            ),
        )
        super().__init__(optimizer, mode=PortfolioMode.LONG_ONLY, **kw)
        if (
            isinstance(train_window, (bool, np.bool_))
            or not isinstance(train_window, (int, np.integer))
        ):
            raise TypeError("train_window must be an integer")
        self.train_window = int(train_window)
        self.retrain_freq = str(retrain_freq)
        self.allow_heuristic_fallback = bool(allow_heuristic_fallback)
        if self.train_window < 2:
            raise ValueError("train_window must be at least 2")
        try:
            pd.Period(pd.Timestamp("2000-01-01"), freq=self.retrain_freq)
        except Exception as exc:
            raise ValueError(f"invalid retrain_freq {self.retrain_freq!r}") from exc

        self._trained = False
        self._last_train: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------ #
    # Selection: equal score over the full present universe
    # ------------------------------------------------------------------ #
    def select(self, ctx: StrategyContext) -> pd.Series:
        if ctx.returns.empty:
            return pd.Series(dtype=float)
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
            raise ValueError("DRL allocation requires aligned, non-empty returns")

        # Rolling training window (most recent ``train_window`` rows).
        train_returns = sub_returns.iloc[-self.train_window :]
        self._maybe_train(train_returns, ctx.as_of)

        if isinstance(self.optimizer, DRLAllocator) and self.optimizer.model is not None:
            current = ctx.extra.get("current_weights")
            if current is None:
                raise ValueError(
                    "trained DRL strategy requires current_weights in StrategyContext.extra"
                )
            if isinstance(current, pd.Series):
                current = current.copy()
            elif isinstance(current, Mapping):
                current = pd.Series(dict(current))
            else:
                raise ValueError("current_weights must be a labeled numeric mapping")
            if current.index.has_duplicates:
                raise ValueError("current_weights index must be unique")
            if any(
                not isinstance(symbol, str) or not symbol.strip()
                for symbol in current.index
            ):
                raise ValueError("current_weights must use non-empty string symbols")
            if any(isinstance(value, (bool, np.bool_)) for value in current.array):
                raise ValueError("current_weights must be numeric, not boolean")
            try:
                current = current.astype(np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("current_weights must be a labeled numeric mapping") from exc
            current.index = pd.Index([symbol.strip() for symbol in current.index])
            if current.index.has_duplicates:
                raise ValueError(
                    "current_weights symbols must remain unique after trimming"
                )
            current_values = enforce_exposure_contract(
                current.to_numpy(dtype=np.float64),
                lower=0.0,
                upper=1.0,
                max_gross=1.0,
                name="DRLPortfolioStrategy.current_weights",
            )
            current = pd.Series(current_values, index=current.index, dtype=np.float64)
            outside = current.drop(index=symbols, errors="ignore")
            if (outside.abs() > 1e-12).any():
                raise ValueError(
                    "trained DRL strategy cannot ignore non-zero current weights "
                    f"outside its asset schema: {outside[outside.abs() > 1e-12].index.tolist()}"
                )
            previous = current.reindex(symbols, fill_value=0.0).to_numpy()
            weights = self.optimizer.optimize(
                sub_returns,
                previous_weights=previous,
            )
        else:
            weights = self.optimizer.optimize(sub_returns)

        if len(weights) != len(symbols):
            raise ValueError("DRL allocator returned a weight vector with wrong length")
        return np.asarray(weights, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Rolling retrain (offline-safe)
    # ------------------------------------------------------------------ #
    def _maybe_train(self, train_returns: pd.DataFrame, as_of: pd.Timestamp) -> None:
        """(Re)train the DRL agent if the retrain cadence has elapsed.

        Training is attempted via ``DRLAllocator.train`` (which lazily imports
        SB3 / gymnasium). Missing optional libraries propagate unless the caller
        explicitly enabled the deterministic heuristic baseline.
        """
        if not isinstance(self.optimizer, DRLAllocator):
            return

        as_of = pd.Timestamp(as_of)
        schema_changed = (
            self.optimizer.model is not None
            and self.optimizer._asset_names != tuple(train_returns.columns)
        )
        if self._trained and self._last_train is not None and not schema_changed:
            if not self._retrain_due(self._last_train, as_of):
                return

        # Need more rows than the observation window to form an episode.
        if train_returns.shape[0] <= self.optimizer.window + 1:
            return

        try:
            self.optimizer.train(train_returns)
            self._trained = True
        except ImportError:
            if not self.allow_heuristic_fallback:
                raise
            self._trained = False
        # Record the attempt regardless, so we respect the retrain cadence and
        # do not hammer training every single rebalance.
        self._last_train = as_of

    def _retrain_due(self, last: pd.Timestamp, now: pd.Timestamp) -> bool:
        """True if ``now`` falls in a later ``retrain_freq`` period than ``last``."""
        p_last = pd.Period(last, freq=self.retrain_freq)
        p_now = pd.Period(now, freq=self.retrain_freq)
        return p_now > p_last
