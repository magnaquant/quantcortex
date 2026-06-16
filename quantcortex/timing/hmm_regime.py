"""Hidden-Markov / Gaussian-mixture market-regime overlay.

This module implements :class:`HMMRegime`, a *timing overlay* that classifies the
prevailing market regime (bear / sideways / bull) from a small set of macro
features and scales gross portfolio exposure accordingly.

The actionable overlay is causal when the model is fit on data available at the
decision time and only the *last* decoded regime is used. Historical labels
returned from one full-window Viterbi decode are retrospective and must not be
treated as walk-forward labels for earlier timestamps.

Modelling backend
-----------------
The ``"hmm"`` backend models serial dependence between regimes and requires the
optional :mod:`hmmlearn` package. The core ``"gmm"`` backend is a memoryless
Gaussian mixture. ``"auto"`` selects HMM only when it can be imported; a model
fit failure never switches backend silently because these models have different
semantics. Both are offline and deterministic given a fixed environment.

Regime labelling
-----------------
Raw model-state indices are arbitrary, so after fitting we *relabel* them by the
mean of the ``returns`` feature within each state: the lowest-mean state becomes
``bear`` (0), the highest-mean state becomes ``bull`` (2) and the remainder are
``sideways`` (1).  This gives a stable, economically interpretable mapping
regardless of the backend's internal ordering.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from quantcortex.portfolio.base import enforce_exposure_contract

__all__ = ["HMMRegime", "BEAR", "SIDEWAYS", "BULL"]


def _single_threaded_blas():
    """Context manager pinning BLAS/OpenMP to one thread for deterministic fits.

    Returns a no-op context if ``threadpoolctl`` is unavailable.
    """
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=1)
    except ImportError:  # pragma: no cover - threadpoolctl ships with sklearn
        return contextlib.nullcontext()

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
    backend:
        ``"hmm"``, ``"gmm"``, or ``"auto"``. Strategies in this repository
        choose ``"gmm"`` explicitly so optional packages cannot change results.
    """

    def __init__(
        self,
        n_states: int = 3,
        *,
        covariance_type: str = "full",
        n_iter: int = 100,
        seed: int = 42,
        backend: str = "auto",
    ) -> None:
        if (
            isinstance(n_states, (bool, np.bool_))
            or not isinstance(n_states, (int, np.integer))
            or n_states < 2
        ):
            raise ValueError("n_states must be >= 2")
        if (
            isinstance(n_iter, (bool, np.bool_))
            or not isinstance(n_iter, (int, np.integer))
            or n_iter <= 0
        ):
            raise ValueError("n_iter must be a positive integer")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)
        ):
            raise ValueError("seed must be an integer")
        self.n_states = int(n_states)
        self.covariance_type = str(covariance_type)
        self.n_iter = int(n_iter)
        self.seed = int(seed)
        self.backend = str(backend)
        if self.covariance_type not in {"full", "diag", "tied", "spherical"}:
            raise ValueError("unsupported covariance_type")
        if self.backend not in {"auto", "hmm", "gmm"}:
            raise ValueError("backend must be 'auto', 'hmm', or 'gmm'")

        # Populated by ``fit``.
        self.model_: Optional[Any] = None
        self.backend_: Optional[str] = None  # "hmm" or "gmm"
        self.state_labels_: Dict[int, int] = {}

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(self, features: pd.DataFrame) -> "HMMRegime":
        """Fit the regime model on ``features`` and build the state->label map.

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

        # Reproducibility: a non-converged EM near a regime boundary is
        # sensitive to the float-reduction order of multithreaded BLAS, which
        # can flip a borderline classification run to run and make a backtest
        # non-deterministic.  Pinning BLAS to one thread for the (small, fast)
        # fit makes the reductions order-stable, so the same data always yields
        # the same regimes.  threadpoolctl ships with scikit-learn; if it is
        # somehow absent we degrade to a no-op context.
        model, backend = self._build_model()
        try:
            with _single_threaded_blas():
                model.fit(X)
                if not self._params_finite(model):
                    raise RuntimeError(f"{backend} fit produced non-finite parameters")
                states = np.asarray(model.predict(X), dtype=int)
        except Exception as exc:
            raise RuntimeError(f"HMMRegime {backend} backend failed") from exc

        self.model_ = model
        self.backend_ = backend
        self.state_labels_ = self._label_states(states, X[:, 0])
        return self

    def _build_model(self) -> tuple[Any, str]:
        """Construct exactly one requested regime model."""
        if self.backend in {"auto", "hmm"}:
            try:
                from hmmlearn.hmm import GaussianHMM  # type: ignore
            except (ImportError, OSError) as exc:
                if self.backend == "hmm":
                    raise ImportError(
                        "backend='hmm' requires a usable hmmlearn installation"
                    ) from exc
            else:
                return (
                    GaussianHMM(
                        n_components=self.n_states,
                        covariance_type=self.covariance_type,
                        n_iter=self.n_iter,
                        random_state=self.seed,
                    ),
                    "hmm",
                )

        from sklearn.mixture import GaussianMixture

        return (
            GaussianMixture(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                max_iter=self.n_iter,
                random_state=self.seed,
                reg_covar=1e-5,
            ),
            "gmm",
        )

    @staticmethod
    def _params_finite(model: Any) -> bool:
        """True if the fitted model's distribution parameters are all finite."""
        for attr in ("means_", "covars_", "covariances_", "transmat_", "startprob_"):
            val = getattr(model, attr, None)
            if val is not None and not np.all(np.isfinite(np.asarray(val))):
                return False
        return True

    def _label_states(
        self, states: np.ndarray, returns_feature: np.ndarray
    ) -> Dict[int, int]:
        """Map raw model-state indices to canonical {bear, sideways, bull}.

        Only states actually *observed* in the fit-time predictions are ranked,
        by the mean of the ``returns`` feature within each state: rank 0
        (lowest mean) -> bear, top rank -> bull, middle -> sideways.  States the
        model never predicted carry no return evidence, so they are mapped to
        the neutral SIDEWAYS label *after* the ranking (seeding them with a
        sentinel mean such as ``-inf`` would always win the bear rank and
        displace the genuinely lowest-mean observed state).

        If only a single state is observed it maps to SIDEWAYS: with one
        cluster there is no cross-sectional spread to provide evidence that
        the regime is bearish or bullish, so we stay neutral.
        """
        present = [int(s) for s in np.unique(states)]
        # Mean return per OBSERVED state; rank only what the model predicted.
        means = {s: float(returns_feature[states == s].mean()) for s in present}
        ranked = sorted(present, key=lambda s: means[s])

        labels: Dict[int, int] = {}
        if len(ranked) >= 2:
            last = len(ranked) - 1
            for rank, s in enumerate(ranked):
                if rank == 0:
                    labels[s] = BEAR
                elif rank == last:
                    labels[s] = BULL
                else:
                    labels[s] = SIDEWAYS
        # Single observed state (no spread evidence) and any never-observed
        # states default to the neutral SIDEWAYS label.
        for s in range(self.n_states):
            labels.setdefault(s, SIDEWAYS)
        return labels

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict_regime(self, features: pd.DataFrame) -> np.ndarray:
        """Return canonical regime labels for every row of ``features``.

        For the HMM backend this is a full-window Viterbi decode. Earlier labels
        can depend on later rows in the supplied window and are retrospective.
        Only the last label is suitable for an actionable as-of decision.
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
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w)):
            raise ValueError("weights must be a non-empty finite 1-D vector")
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
            if not isinstance(rf, pd.DataFrame):
                raise TypeError("ctx.extra['regime_features'] must be a DataFrame")
            if not isinstance(rf.index, pd.DatetimeIndex):
                raise TypeError(
                    "ctx.extra['regime_features'] must use a DatetimeIndex"
                )
            if rf.index.hasnans or rf.index.has_duplicates:
                raise ValueError(
                    "ctx.extra['regime_features'] index must contain unique valid timestamps"
                )
            frame = rf.copy()
            if frame.index.tz is not None:
                frame.index = frame.index.tz_convert("UTC").tz_localize(None)
            frame = frame.sort_index()
            as_of = pd.to_datetime(
                getattr(features, "as_of", None), errors="coerce", utc=True
            )
            if pd.isna(as_of):
                raise ValueError(
                    "context with regime_features must expose a valid as_of timestamp"
                )
            return frame.loc[frame.index <= as_of.tz_localize(None)]

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
            {
                "returns": series.to_numpy(),
                "realized_vol": rv.to_numpy(),
                "vix": vix_proxy.to_numpy(),
            },
            index=series.index,
        )

    @staticmethod
    def _extract_matrix(features: pd.DataFrame) -> np.ndarray:
        """Select the canonical feature columns into a float64 matrix."""
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)
        if features.columns.has_duplicates:
            raise ValueError("regime feature columns must be unique")
        if features.index.has_duplicates:
            raise ValueError("regime feature index must be unique")
        if isinstance(features.index, pd.DatetimeIndex):
            if features.index.hasnans:
                raise ValueError("regime feature index must contain valid timestamps")
            if not features.index.is_monotonic_increasing:
                raise ValueError("regime feature dates must be sorted")
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
            raise ValueError("regime features must contain only finite values")
        return X

    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("HMMRegime is not fitted; call fit() first")
