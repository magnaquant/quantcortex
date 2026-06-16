"""Deep Reinforcement Learning (DRL) portfolio allocator.

This module provides :class:`DRLAllocator`, an end-to-end portfolio allocator
trained with Proximal Policy Optimization (PPO).  The agent observes a trailing
window of asset returns together with the current portfolio weights, then emits
a continuous action that is mapped - via a softmax - onto the long-only simplex
of portfolio weights.  The reward is the realized portfolio log-return,
penalized for transaction costs (turnover) and for trailing portfolio-return
variance (a simple risk adjustment).

Design goals
------------
* **Optional heavy dependencies.**  ``gymnasium`` and ``stable_baselines3`` are
  *not* required to import or use this module.  They are imported lazily, only
  inside :meth:`train` / :meth:`_make_env`.  If they are missing and the user
  calls :meth:`train`, a clear :class:`ImportError` with a ``pip`` hint is
  raised.
* **Explicit fallback.**  Without a trained model, allocation fails closed by
  default. Callers may explicitly request the deterministic heuristic policy
  with ``untrained_policy="heuristic"`` for offline demonstrations.

The class subclasses :class:`~quantcortex.portfolio.base.PortfolioOptimizer`, so the public
:meth:`optimize` entry point validates the output against the weight contract.
"""

from __future__ import annotations

from typing import ClassVar, Optional, Union

import numpy as np
import pandas as pd

from quantcortex.portfolio.base import (
    PortfolioMode,
    PortfolioOptimizer,
    enforce_exposure_contract,
)

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


def _portfolio_variance(returns: np.ndarray, weights: np.ndarray) -> float:
    """Return sample variance for a trailing portfolio-return window."""
    x = np.asarray(returns, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if x.ndim != 2 or x.shape[1] != w.size or x.shape[0] < 2:
        raise ValueError("portfolio variance needs a T x N window and N weights")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)):
        raise ValueError("portfolio variance inputs must be finite")
    return max(float(np.var(x @ w, ddof=1)), 0.0)


class DRLAllocator(PortfolioOptimizer):
    """PPO-based end-to-end portfolio allocator with an offline fallback.

    Parameters
    ----------
    mode:
        Only :data:`PortfolioMode.LONG_ONLY` is supported (the action is mapped
        through a softmax, which is inherently long-only and fully invested).
    window:
        Number of trailing return rows used in the observation at each step.
        The current portfolio weights are appended to that return window.
    total_timesteps:
        Number of environment steps used by ``PPO.learn`` during :meth:`train`.
    transaction_cost:
        Proportional cost per unit of turnover, charged in the reward.
    risk_aversion:
        Multiplier on trailing portfolio-return variance in the reward.
    seed:
        Random seed for reproducibility of the environment and PPO.
    untrained_policy:
        ``"error"`` (default) requires a trained PPO model. ``"heuristic"``
        explicitly selects the deterministic risk-adjusted-momentum baseline.
    **kw:
        Forwarded to :class:`~quantcortex.portfolio.base.PortfolioOptimizer`.
    """

    def __init__(
        self,
        mode: Union[PortfolioMode, str] = PortfolioMode.LONG_ONLY,
        *,
        window: int = 60,
        total_timesteps: int = 10_000,
        transaction_cost: float = 0.001,
        risk_aversion: float = 1.0,
        seed: int = 42,
        untrained_policy: str = "error",
        **kw,
    ) -> None:
        super().__init__(mode, **kw)
        if self.mode is not PortfolioMode.LONG_ONLY:
            raise ValueError("DRLAllocator only supports PortfolioMode.LONG_ONLY")
        if (
            isinstance(window, (bool, np.bool_))
            or not isinstance(window, (int, np.integer))
        ):
            raise TypeError("window must be an integer")
        if (
            isinstance(total_timesteps, (bool, np.bool_))
            or not isinstance(total_timesteps, (int, np.integer))
        ):
            raise TypeError("total_timesteps must be an integer")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        if untrained_policy not in {"error", "heuristic"}:
            raise ValueError("untrained_policy must be 'error' or 'heuristic'")
        if isinstance(transaction_cost, (bool, np.bool_)):
            raise TypeError("transaction_cost must be numeric, not boolean")
        if isinstance(risk_aversion, (bool, np.bool_)):
            raise TypeError("risk_aversion must be numeric, not boolean")
        self.window = int(window)
        self.total_timesteps = int(total_timesteps)
        self.transaction_cost = float(transaction_cost)
        self.risk_aversion = float(risk_aversion)
        self.seed = int(seed)
        self.untrained_policy = untrained_policy
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if not np.isfinite(self.transaction_cost) or self.transaction_cost < 0.0:
            raise ValueError("transaction_cost must be finite and non-negative")
        if not np.isfinite(self.risk_aversion) or self.risk_aversion < 0.0:
            raise ValueError("risk_aversion must be finite and non-negative")
        self.model = None  # populated by .train()
        self._n_assets: Optional[int] = None
        self._asset_names: Optional[tuple[object, ...]] = None

    # ------------------------------------------------------------------ #
    # Environment factory (lazy-imports gymnasium).
    # ------------------------------------------------------------------ #
    def _make_env(self, returns: pd.DataFrame):
        """Build a fresh ``PortfolioEnv`` instance for ``returns``.

        ``gymnasium`` is imported here, so the class remains importable and
        usable (via the explicitly selected fallback policy) without it.
        """
        try:
            import gymnasium as gym
            from gymnasium import spaces
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(_PIP_HINT) from exc

        window = self.window
        transaction_cost = self.transaction_cost
        risk_aversion = self.risk_aversion
        seed = self.seed
        project = self._project_configured_bounds
        data = np.asarray(returns.values, dtype=np.float64)
        n_steps, n_assets = data.shape

        class PortfolioEnv(gym.Env):
            """A single-episode portfolio-allocation environment.

            Observation
                The flattened trailing ``window`` of returns, shape
                ``(window * n_assets,)``, followed by the current portfolio
                weights, shape ``(n_assets,)``.
            Action
                A continuous ``Box`` of shape ``(n_assets,)`` mapped to weights
                via softmax (long-only, sums to 1).
            Reward
                ``log(1 + w*r) - transaction_cost * turnover - var_penalty``
                where ``var_penalty`` is trailing portfolio-return variance
                multiplied by ``risk_aversion``.
            """

            metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

            def __init__(self) -> None:
                super().__init__()
                self.data = data
                self.window = window
                self.n_assets = n_assets
                self.transaction_cost = transaction_cost
                self.risk_aversion = risk_aversion
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(window * n_assets + n_assets,),
                    dtype=np.float32,
                )
                self.action_space = spaces.Box(
                    low=-10.0, high=10.0, shape=(n_assets,), dtype=np.float32
                )
                self._t = window
                self._prev_w = np.zeros(n_assets, dtype=np.float64)

            def _obs(self) -> np.ndarray:
                win = self.data[self._t - self.window : self._t]
                return np.concatenate((win.ravel(), self._prev_w)).astype(np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self._t = self.window
                self._prev_w = np.zeros(self.n_assets, dtype=np.float64)
                return self._obs(), {}

            def step(self, action):
                w = project(_softmax(np.asarray(action, dtype=np.float64)))
                r = self.data[self._t]
                port_ret = float(np.dot(w, r))
                turnover = float(np.abs(w - self._prev_w).sum())
                trailing = self.data[self._t - self.window : self._t]
                var_penalty = self.risk_aversion * _portfolio_variance(trailing, w)
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
        self._validate_returns(returns)
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
        self._asset_names = tuple(returns.columns)
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
        return self._project_configured_bounds(_softmax(score))

    # ------------------------------------------------------------------ #
    # Optimizer entry point.
    # ------------------------------------------------------------------ #
    def _compute_weights(
        self,
        returns: pd.DataFrame,
        *,
        previous_weights=None,
        **kwargs,
    ) -> np.ndarray:
        """Compute portfolio weights.

        If a PPO model has been trained (:meth:`train`), the agent predicts an
        action from the most recent observation window and explicit current
        portfolio weights, then maps the action to weights via softmax.  A
        trained model therefore requires ``previous_weights``.  Otherwise an
        explicitly configured deterministic heuristic policy can be used
        without ``stable_baselines3`` / ``gymnasium`` installed.
        """
        if not isinstance(returns, pd.DataFrame):
            returns = pd.DataFrame(np.asarray(returns, dtype=float))
        self._validate_returns(returns)

        n = returns.shape[1]
        if n == 0:
            raise ValueError("DRLAllocator requires at least one asset")
        if n == 1:
            return self._project_configured_bounds(
                np.array([1.0], dtype=np.float64)
            )

        if self.model is not None:
            if returns.shape[0] < self.window:
                raise ValueError(
                    f"trained DRL policy needs at least {self.window} return rows"
                )
            if self._n_assets != n or self._asset_names != tuple(returns.columns):
                raise ValueError(
                    "trained DRL policy asset schema does not match current returns"
                )
            if previous_weights is None:
                raise ValueError(
                    "trained DRL policy requires explicit previous_weights"
                )
            previous = enforce_exposure_contract(
                previous_weights,
                lower=0.0,
                upper=1.0,
                max_gross=1.0,
                tolerance=self.tolerance,
                name="DRLAllocator.previous_weights",
            )
            if previous.size != n:
                raise ValueError(
                    "previous_weights length does not match current return assets"
                )
            obs = np.concatenate(
                (
                    returns.iloc[-self.window :]
                    .values.astype(np.float32)
                    .ravel(),
                    previous.astype(np.float32),
                )
            )
            action, _ = self.model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float64).ravel()
            if action.size != n or not np.all(np.isfinite(action)):
                raise ValueError("trained DRL policy returned an invalid action")
            return self._project_configured_bounds(_softmax(action))

        if self.untrained_policy == "heuristic":
            return self._fallback_weights(returns)
        raise RuntimeError(
            "DRLAllocator has no trained PPO model; call train() or explicitly "
            "construct it with untrained_policy='heuristic'"
        )

    @staticmethod
    def _validate_returns(returns: pd.DataFrame) -> None:
        if returns.ndim != 2 or returns.shape[0] == 0 or returns.shape[1] == 0:
            raise ValueError("DRLAllocator requires a non-empty T x N return panel")
        if returns.columns.has_duplicates:
            raise ValueError("DRLAllocator return columns must be unique")
        values = returns.to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("DRLAllocator returns must contain only finite values")
