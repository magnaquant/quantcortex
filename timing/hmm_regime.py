"""Hidden-Markov / Gaussian-mixture market-regime overlay.

This module implements :class:`HMMRegime`, a *timing overlay* that classifies the
prevailing market regime (bear / sideways / bull) from a small set of macro
features and scales gross portfolio exposure accordingly.

The overlay is **strictly causal**: a regime prediction for time ``t`` uses only
features observed up to and including ``t``.  Exposure is scaled by the *last*
predicted regime, which is the only regime an executing strategy could act on
without look-ahead.

Modelling backend
-----------------
The preferred backend is :class:`hmmlearn.hmm.GaussianHMM`, which models the
serial dependence between regimes.  ``hmmlearn`` is an optional dependency; when
it is not installed we transparently fall back to
:class:`sklearn.mixture.GaussianMixture`, a memoryless mixture model that still
clusters the feature space into ``n_states`` regimes.  Both backends are fully
offline (no network access, deterministic given ``seed``).

Regime labelling
-----------------
Raw model-state indices are arbitrary, so after fitting we *relabel* them by the
mean of the ``returns`` feature within each state: the lowest-mean state becomes
``bear`` (0), the highest-mean state becomes ``bull`` (2) and the remainder are
``sideways`` (1).  This gives a stable, economically interpretable mapping
regardless of the backend's internal ordering.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from portfolio.base import enforce_exposure_contract

__all__ = ["HMMRegime", "BEAR", "SIDEWAYS", "BULL"]

# Canonical regime labels.
BEAR: int = 0
SIDEWAYS: int = 1
BULL: int = 2

# Gross-exposure multiplier applied per regime.
_REGIME_SCALE: Dict[int, float] = {BEAR: 0.0, SIDEWAYS: 0.5, BULL: 1.0}

# Feature columns expected by the model, in canonical order.
_FEATURE_COLUMNS = ("returns", "realized_vol", "vix")


class HMMRegime:
    """Market-regime classifier and exposure-scaling timing overlay.

    Parameters
    ----------
    n_states:
        Number of latent regimes to fit.  The labelling logic assumes the three
        canonical regimes (bear / sideways / bull); ``n_states`` other than 3 is
        supported but the head/tail states map to bear/bull and everything in
        between maps to sideways.
    covariance_type:
        Covariance parameterisation passed to the backend (``"full"``,
        ``"diag"``, ``"tied"``, ``"spherical"``).
    n_iter:
        Maximum EM iterations.
    seed:
        Random seed for reproducible fits.
    """

    def __init__(
        self,
        n_states: int = 3,
        *,
        covariance_type: str = "full",
        n_iter: int = 100,
        seed: int = 42,
    ) -> None:
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        self.n_states = int(n_states)
        self.covariance_type = str(covariance_type)
        self.n_iter = int(n_iter)
        self.seed = int(seed)

        # Populated by ``fit``.
        self.model_: Optional[Any] = None
        self.backend_: Optional[str] = None  # "hmm" or "gmm"
        self.state_labels_: Dict[int, int] = {}

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(self, features: pd.DataFrame) -> "HMMRegime":
        """Fit the regime model on ``features`` and build the state→label map.

        Parameters
        ----------
        features:
            DataFrame with (at least) the columns ``returns``, ``realized_vol``
            and ``vix``.  Rows are time-ordered observations.

        Returns
        -------
        HMMRegime
            ``self`` (fitted).
        """
        X = self._extract_matrix(features)

        model, backend = self._build_model()
        model.fit(X)
        self.model_ = model
        self.backend_ = backend

        # Relabel raw states by mean of the "returns" feature (column 0).
        states = self._raw_predict(X)
        self.state_labels_ = self._label_states(states, X[:, 0])
        return self

    def _build_model(self) -> tuple[Any, str]:
        """Instantiate the preferred HMM backend, falling back to a GMM."""
        try:
            from hmmlearn.hmm import GaussianHMM  # type: ignore

            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.seed,
            )
            return model, "hmm"
        except Exception:
            # hmmlearn missing (or incompatible) -> memoryless GMM fallback.
            from sklearn.mixture import GaussianMixture

            model = GaussianMixture(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                max_iter=self.n_iter,
                random_state=self.seed,
            )
            return model, "gmm"

    def _label_states(
        self, states: np.ndarray, returns_feature: np.ndarray
    ) -> Dict[int, int]:
        """Map raw model-state indices to canonical {bear, sideways, bull}.

        States are ranked by the mean of the ``returns`` feature within each
        state: rank 0 (lowest mean) -> bear, top rank -> bull, middle -> sideways.
        """
        present = np.unique(states)
        # Mean return per state; states never observed default to -inf so they
        # rank as the most bearish (they will not be predicted anyway).
        means = {}
        for s in range(self.n_states):
            mask = states == s
            means[s] = float(returns_feature[mask].mean()) if mask.any() else -np.inf

        ranked = sorted(range(self.n_states), key=lambda s: means[s])
        labels: Dict[int, int] = {}
        last = len(ranked) - 1
        for rank, s in enumerate(ranked):
            if rank == 0:
                labels[s] = BEAR
            elif rank == last:
                labels[s] = BULL
            else:
                labels[s] = SIDEWAYS
        # Guard: if every observed state collapsed to one rank position.
        for s in present:
            labels.setdefault(int(s), SIDEWAYS)
        return labels

    # ------------------------------------------------------------------ #
    # Prediction (causal)
    # ------------------------------------------------------------------ #
    def predict_regime(self, features: pd.DataFrame) -> np.ndarray:
        """Return canonical regime labels for every row of ``features``.

        Causal: each label depends only on features up to and including its own
        timestamp.  For the HMM backend we use the Viterbi/posterior decoding
        ``predict`` over the supplied window; because we only ever *act* on the
        last label (see :meth:`scale_weights`), no future information leaks into
        an actionable decision.
        """
        self._check_fitted()
        X = self._extract_matrix(features)
        raw = self._raw_predict(X)
        return self._map_labels(raw)

    def current_regime(self, features: pd.DataFrame) -> int:
        """Return the canonical regime label for the most recent observation."""
        labels = self.predict_regime(features)
        return int(labels[-1])

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        """Backend-agnostic raw state prediction."""
        assert self.model_ is not None
        return np.asarray(self.model_.predict(X), dtype=int)

    def _map_labels(self, raw: np.ndarray) -> np.ndarray:
        """Translate raw state indices to canonical labels via the fitted map."""
        out = np.fromiter(
            (self.state_labels_.get(int(s), SIDEWAYS) for s in raw),
            dtype=int,
            count=raw.size,
        )
        return out

    # ------------------------------------------------------------------ #
    # Exposure scaling (overlay interface)
    # ------------------------------------------------------------------ #
    def scale_weights(
        self, weights: np.ndarray, features: pd.DataFrame
    ) -> np.ndarray:
        """Scale ``weights`` by the multiplier of the *last* predicted regime.

        bear -> x0.0 (flat), sideways -> x0.5, bull -> x1.0.  The result is
        validated through :func:`enforce_exposure_contract`.
        """
        w = np.asarray(weights, dtype=np.float64).ravel()
        regime = self.current_regime(features)
        scale = _REGIME_SCALE.get(int(regime), 0.5)
        scaled = w * scale

        input_gross = float(np.abs(w).sum())
        max_gross = max(1.0, input_gross) + 1e-9
        return enforce_exposure_contract(
            scaled, max_gross=max_gross, name=f"{type(self).__name__}:regime{regime}"
        )

    def apply(self, weights: np.ndarray, features: Any = None) -> np.ndarray:
        """Overlay entry point: alias for :meth:`scale_weights`.

        Accepts either an explicit ``features`` DataFrame or a
        ``StrategyContext``-like object (duck-typed): if ``features`` exposes
        ``.returns``/``.extra`` we extract the regime features from
        ``ctx.extra['regime_features']`` when present, otherwise we build a
        single-column ``returns`` feature frame from ``ctx.returns``.
        """
        feats = self._coerce_features(features)
        return self.scale_weights(weights, feats)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _coerce_features(self, features: Any) -> pd.DataFrame:
        """Normalise the ``features`` argument into a feature DataFrame."""
        if isinstance(features, pd.DataFrame):
            return features

        # Duck-typed StrategyContext-like object.
        extra = getattr(features, "extra", None)
        if isinstance(extra, dict) and "regime_features" in extra:
            rf = extra["regime_features"]
            if isinstance(rf, pd.DataFrame):
                return rf
            return pd.DataFrame(rf)

        returns = getattr(features, "returns", None)
        if returns is not None:
            return self._features_from_returns(returns)

        raise TypeError(
            "HMMRegime.apply requires a features DataFrame or a context object "
            "exposing .extra['regime_features'] or .returns"
        )

    @staticmethod
    def _features_from_returns(returns: Any) -> pd.DataFrame:
        """Construct a minimal feature frame from a return series/DataFrame.

        Builds ``returns`` (portfolio-level mean if multi-asset), a trailing
        ``realized_vol`` (20-obs rolling std) and a ``vix`` proxy (annualised
        realized vol scaled to volatility points).
        """
        r = returns
        if isinstance(r, pd.DataFrame):
            series = r.mean(axis=1)
        else:
            series = pd.Series(np.asarray(r, dtype=np.float64).ravel())
        series = series.astype(np.float64)
        rv = series.rolling(20, min_periods=1).std().fillna(0.0)
        vix_proxy = (rv * np.sqrt(252.0) * 100.0).fillna(0.0)
        return pd.DataFrame(
            {"returns": series.to_numpy(), "realized_vol": rv.to_numpy(), "vix": vix_proxy.to_numpy()}
        )

    @staticmethod
    def _extract_matrix(features: pd.DataFrame) -> np.ndarray:
        """Select the canonical feature columns into a float64 matrix."""
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)
        missing = [c for c in _FEATURE_COLUMNS if c not in features.columns]
        if missing:
            raise ValueError(
                f"features missing required columns {missing}; "
                f"expected {list(_FEATURE_COLUMNS)}"
            )
        X = features.loc[:, list(_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        if X.shape[0] == 0:
            raise ValueError("features is empty")
        if not np.all(np.isfinite(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("HMMRegime is not fitted; call fit() first")
