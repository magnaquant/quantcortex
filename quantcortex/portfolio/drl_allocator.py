"""Deep Reinforcement Learning (DRL) portfolio allocator.

This module provides :class:`DRLAllocator`, an end-to-end portfolio allocator
trained with Proximal Policy Optimization (PPO).  The agent observes a trailing
window of asset returns and emits a continuous action that is mapped - via a
softmax - onto the long-only simplex of portfolio weights.  The reward is the
realized portfolio log-return, penalized for transaction costs (turnover) and
for return variance (a simple risk adjustment).

Design goals
------------
* **Optional heavy dependencies.**  ``gymnasium`` and ``stable_baselines3`` are
  *not* required to import or use this module.  They are imported lazily, only
  inside :meth:`train` / :meth:`_make_env`.  If they are missing and the user
  calls :meth:`train`, a clear :class:`ImportError` with a ``pip`` hint is
  raised.
* **Always usable.**  Without a trained model (and therefore without SB3 /
  gymnasium installed), :meth:`_compute_weights` falls back to a deterministic,
  fully-offline heuristic policy - softmax of recent mean returns scaled by
  inverse volatility - that still yields contract-valid long-only weights.

The class subclasses :class:`~portfolio.base.PortfolioOptimizer`, so the public
:meth:`optimize` entry point validates the output against the weight contract.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from quantcortex.portfolio.base import PortfolioMode, PortfolioOptimizer

__all__ = ["DRLAllocator"]

_PIP_HINT = (
    "DRLAllocator.train() requires 'gymnasium' and 'stable-baselines3'. "
    "Install them with:  pip install gymnasium stable-baselines3"
)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D vector."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    if s <= 0.0 or not np.isfinite(s):
        return np.full(x.shape, 1.0 / x.size, dtype=np.float64)
    return e / s


class DRLAllocator(PortfolioOptimizer):
    """PPO-based end-to-end portfolio allocator with an offline fallback.

    Parameters
    ----------
    mode:
        Only :data:`PortfolioMode.LONG_ONLY` is supported (the action is mapped
        through a softmax, which is inherently long-only and fully invested).
    window:
        Number of trailing return rows used as the observation at each step.
    total_timesteps:
        Number of environment steps used by ``PPO.learn`` during :meth:`train`.
    transaction_cost:
        Proportional cost per unit of turnover, charged in the reward.
    seed:
        Random seed for reproducibility of the environment and PPO.
    **kw:
        Forwarded to :class:`~portfolio.base.PortfolioOptimizer`.
    """

    def __init__(
        self,
        mode: Union[PortfolioMode, str] = PortfolioMode.LONG_ONLY,
        *,
        window: int = 60,
        total_timesteps: int = 10_000,
        transaction_cost: float = 0.001,
        seed: int = 42,
        **kw,
    ) -> None:
        super().__init__(mode, **kw)
        if self.mode is not PortfolioMode.LONG_ONLY:
            raise ValueError("DRLAllocator only supports PortfolioMode.LONG_ONLY")
        self.window = int(window)
        self.total_timesteps = int(total_timesteps)
        self.transaction_cost = float(transaction_cost)
        self.seed = int(seed)
        self.model = None  # populated by .train()
        self._n_assets: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Environment factory (lazy-imports gymnasium).
    # ------------------------------------------------------------------ #
    def _make_env(self, returns: pd.DataFrame):
        """Build a fresh ``PortfolioEnv`` instance for ``returns``.

        ``gymnasium`` is imported here, so the class remains importable and
        usable (via the fallback policy) without it.
        """
        try:
            import gymnasium as gym
            from gymnasium import spaces
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(_PIP_HINT) from exc

        window = self.window
        transaction_cost = self.transaction_cost
        seed = self.seed
        data = np.asarray(returns.values, dtype=np.float64)
        n_steps, n_assets = data.shape

        class PortfolioEnv(gym.Env):
            """A single-episode portfolio-allocation environment.

            Observation
                The flattened trailing ``window`` of returns, shape
                ``(window * n_assets,)``.
            Action
                A continuous ``Box`` of shape ``(n_assets,)`` mapped to weights
                via softmax (long-only, sums to 1).
            Reward
                ``log(1 + w*r) - transaction_cost * turnover - var_penalty``
                where ``var_penalty`` is the cross-sectional variance of the
                positions (a light risk adjustment).
            """

            metadata = {"render_modes": []}

            def __init__(self) -> None:
                super().__init__()
                self.data = data
                self.window = window
                self.n_assets = n_assets
                self.transaction_cost = transaction_cost
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(window * n_assets,),
                    dtype=np.float32,
                )
                self.action_space = spaces.Box(
                    low=-10.0, high=10.0, shape=(n_assets,), dtype=np.float32
                )
                self._t = window
                self._prev_w = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)

            def _obs(self) -> np.ndarray:
                win = self.data[self._t - self.window : self._t]
                return win.astype(np.float32).ravel()

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self._t = self.window
                self._prev_w = np.full(
                    self.n_assets, 1.0 / self.n_assets, dtype=np.float64
                )
                return self._obs(), {}

            def step(self, action):
                w = _softmax(np.asarray(action, dtype=np.float64))
                r = self.data[self._t]
                port_ret = float(np.dot(w, r))
                turnover = float(np.abs(w - self._prev_w).sum())
                var_penalty = float(np.var(w))
                # Guard against log of a non-positive gross return.
                growth = max(1.0 + port_ret, 1e-8)
                reward = (
                    np.log(growth)
                    - self.transaction_cost * turnover
                    - var_penalty
                )
                self._prev_w = w
                self._t += 1
                terminated = self._t >= n_steps
                truncated = False
                return self._obs(), float(reward), terminated, truncated, {}

        env = PortfolioEnv()
        env.reset(seed=seed)
        return env

    # ------------------------------------------------------------------ #
    # Training (lazy-imports stable_baselines3).
    # ------------------------------------------------------------------ #
    def train(self, returns: pd.DataFrame) -> "DRLAllocator":
        """Train a PPO policy on ``returns`` and store it on ``self.model``.

        Parameters
        ----------
        returns:
            ``(T x N)`` DataFrame of per-asset returns.  ``T`` must exceed
            ``window`` so at least one transition is available.

        Returns
        -------
        DRLAllocator
            ``self`` (enables ``allocator.train(r).optimize(r)`` chaining).

        Raises
        ------
        ImportError
            If ``gymnasium`` / ``stable_baselines3`` are not installed.
        ValueError
            If there are too few rows to form an episode.
        """
        if not isinstance(returns, pd.DataFrame):
            returns = pd.DataFrame(np.asarray(returns, dtype=float))
        if returns.shape[0] <= self.window + 1:
            raise ValueError(
                f"Need more than window+1={self.window + 1} rows to train; "
                f"got {returns.shape[0]}"
            )

        try:
            from stable_baselines3 import PPO
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(_PIP_HINT) from exc

        env = self._make_env(returns)
        self._n_assets = returns.shape[1]
        model = PPO("MlpPolicy", env, seed=self.seed, verbose=0)
        model.learn(total_timesteps=self.total_timesteps)
        self.model = model
        return self

    # ------------------------------------------------------------------ #
    # Offline fallback policy.
    # ------------------------------------------------------------------ #
    def _fallback_weights(self, returns: pd.DataFrame) -> np.ndarray:
        """Deterministic offline allocation usable without SB3 / gymnasium.

        Scores each asset by its recent mean return scaled by inverse
        volatility (a risk-adjusted momentum signal) and maps the scores to
        long-only weights via softmax.  Always returns finite weights summing
        to 1.0.
        """
        window = min(self.window, returns.shape[0])
        recent = returns.iloc[-window:]
        mean = recent.mean(axis=0).values.astype(np.float64)
        vol = recent.std(axis=0).values.astype(np.float64)
        vol = np.where(np.isfinite(vol) & (vol > 0.0), vol, 1.0)
        score = np.nan_to_num(mean / vol, nan=0.0, posinf=0.0, neginf=0.0)
        return _softmax(score)

    # ------------------------------------------------------------------ #
    # Optimizer entry point.
    # ------------------------------------------------------------------ #
    def _compute_weights(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        """Compute portfolio weights.

        If a PPO model has been trained (:meth:`train`), the agent predicts an
        action from the most recent observation window and the action is mapped
        to weights via softmax.  Otherwise the deterministic offline fallback
        policy is used, so the allocator is fully functional without
        ``stable_baselines3`` / ``gymnasium`` installed.
        """
        if not isinstance(returns, pd.DataFrame):
            returns = pd.DataFrame(np.asarray(returns, dtype=float))

        n = returns.shape[1]
        if n == 0:
            raise ValueError("DRLAllocator requires at least one asset")
        if n == 1:
            return np.array([1.0], dtype=np.float64)

        if self.model is not None and returns.shape[0] >= self.window:
            obs = (
                returns.iloc[-self.window :]
                .values.astype(np.float32)
                .ravel()
            )
            action, _ = self.model.predict(obs, deterministic=True)
            return _softmax(np.asarray(action, dtype=np.float64))

        return self._fallback_weights(returns)
