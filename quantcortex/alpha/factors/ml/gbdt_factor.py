"""Cross-sectional gradient-boosted decision tree (GBDT) return predictor.

Gradient-boosted decision trees are a useful baseline for tabular return
prediction because they capture nonlinearities and interactions without feature
scaling. Their performance is dataset- and validation-dependent; this module
does not assume that they outperform simpler models.

This module exposes :class:`GBDTFactor`, a thin seeded wrapper that
trains a GBDT to map a cross-section of firm characteristics ``X`` to a
forward-return target ``y`` (raw returns or, more robustly, cross-sectional
ranks). It transparently uses the best available boosting backend
(LightGBM -> XGBoost -> CatBoost) and *always* provides a fully functional
scikit-learn fallback (:class:`~sklearn.ensemble.HistGradientBoostingRegressor`)
so the class is usable offline with no optional dependencies installed.

Causality
---------
All backtesting helpers here are strictly causal: a prediction for date ``t``
is produced by a model fit only on data observed *before* ``t``. See
:meth:`GBDTFactor.fit_predict_cross_sectional`, which walks forward in time and
never trains on the cross-section it is scoring.

Optional dependencies (LightGBM, XGBoost, CatBoost) are imported lazily *inside*
methods so that importing this module only requires the standard scientific
stack (numpy, pandas, scipy, scikit-learn).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

# Backends in preference order. "auto" walks this list and uses the first one
# whose import succeeds, falling back to the always-available sklearn backend.
_BACKEND_PREFERENCE: tuple[str, ...] = ("lightgbm", "xgboost", "catboost", "sklearn")
_VALID_MODELS: frozenset[str] = frozenset({"auto", *_BACKEND_PREFERENCE})


class GBDTFactor:
    """Cross-sectional GBDT predictor of forward returns.

    The model learns ``y = f(X)`` where ``X`` is a cross-section of firm
    characteristics (one row per security on a given date) and ``y`` is the
    forward return (or its cross-sectional rank). Predictions are interpreted
    as *alpha scores*: a higher score means a more attractive security.

    Parameters
    ----------
    model:
        Boosting backend. One of ``{"auto", "lightgbm", "xgboost",
        "catboost", "sklearn"}``. ``"auto"`` tries LightGBM, then XGBoost,
        then CatBoost, then the scikit-learn fallback, using the first that
        imports successfully. An explicit name forces that backend (and raises
        if its package is not installed, except ``"sklearn"`` which is always
        available).
    random_state:
        Seed for reproducible training. Exact reproducibility also requires an
        explicit backend and fixed library versions; ``model="auto"`` is
        intentionally environment-dependent.
    **params:
        Backend-specific hyper-parameters forwarded to the underlying
        estimator constructor (e.g. ``n_estimators``, ``max_depth``,
        ``learning_rate``). Sensible defaults are supplied per backend.

    Backend choice is part of the model specification. ``model="auto"`` is
    convenient for exploration but environment-dependent; reproducible research
    should select a concrete backend and pin its version.
    """

    def __init__(
        self,
        model: str = "auto",
        random_state: int = 42,
        **params: Any,
    ) -> None:
        if model not in _VALID_MODELS:
            raise ValueError(
                f"model must be one of {sorted(_VALID_MODELS)}, got {model!r}"
            )
        self.model = model
        self.random_state = int(random_state)
        self.params: dict[str, Any] = dict(params)

        # Populated by fit().
        self.backend_: Optional[str] = None
        self.estimator_: Any = None
        self.feature_names_: Optional[list[str]] = None

    # ------------------------------------------------------------------
    # Backend construction (lazy imports live here)
    # ------------------------------------------------------------------
    def _build_estimator(self) -> tuple[str, Any]:
        """Resolve the backend and instantiate its estimator.

        Returns
        -------
        (backend_name, estimator)
            ``backend_name`` is the concrete backend actually selected.
        """
        if self.model == "auto":
            candidates: Sequence[str] = _BACKEND_PREFERENCE
        else:
            candidates = (self.model,)

        last_error: Optional[Exception] = None
        for name in candidates:
            try:
                return name, self._construct_backend(name)
            except (ImportError, OSError) as exc:
                # Missing packages and native-library load failures mean the
                # optional backend is unavailable. Constructor/configuration
                # errors are not caught because silently changing estimators
                # would hide a broken research specification.
                last_error = exc
                if self.model != "auto":
                    if isinstance(exc, ImportError):
                        raise ImportError(
                            f"Backend {name!r} requested but its package is not "
                            f"installed. Install it or use model='auto'/'sklearn'."
                        ) from exc
                    raise  # installed but unusable -> surface the underlying error
                continue
        # "auto" always ends at the sklearn fallback, which cannot fail to import.
        raise RuntimeError(  # pragma: no cover - defensive
            "No GBDT backend could be constructed."
        ) from last_error

    def _construct_backend(self, name: str) -> Any:
        """Instantiate a single backend by name (lazy-importing it)."""
        if name == "lightgbm":
            import lightgbm as lgb  # lazy

            defaults = dict(
                n_estimators=400,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=-1,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_lambda=1.0,
                n_jobs=1,
                verbosity=-1,
            )
            defaults.update(self.params)
            return lgb.LGBMRegressor(random_state=self.random_state, **defaults)

        if name == "xgboost":
            import xgboost as xgb  # lazy

            defaults = dict(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                n_jobs=1,
                verbosity=0,
                tree_method="hist",
            )
            defaults.update(self.params)
            return xgb.XGBRegressor(random_state=self.random_state, **defaults)

        if name == "catboost":
            from catboost import CatBoostRegressor  # lazy

            defaults = dict(
                iterations=400,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=3.0,
                verbose=False,
                allow_writing_files=False,
                thread_count=1,
            )
            defaults.update(self.params)
            return CatBoostRegressor(random_seed=self.random_state, **defaults)

        if name == "sklearn":
            # Always-available fallback. HistGradientBoostingRegressor is a
            # fast, LightGBM-style histogram boosting implementation that
            # natively handles NaNs.
            from sklearn.ensemble import HistGradientBoostingRegressor

            defaults = dict(
                max_iter=400,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
            )
            defaults.update(self.params)
            return HistGradientBoostingRegressor(
                random_state=self.random_state, **defaults
            )

        raise ValueError(f"Unknown backend {name!r}")  # pragma: no cover

    # ------------------------------------------------------------------
    # Core fit / predict
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GBDTFactor":
        """Fit the GBDT on a (cross-)sectional feature matrix.

        Parameters
        ----------
        X:
            Feature matrix, one row per observation (security/date), one
            column per characteristic.
        y:
            Target aligned to ``X`` -- forward cross-sectional returns or
            (more robustly) their cross-sectional ranks.

        Returns
        -------
        GBDTFactor
            ``self``, fitted.
        """
        X = self._validate_features(X)
        y_arr = self._validate_target(y, X.index)

        self.feature_names_ = list(X.columns)
        self.backend_, self.estimator_ = self._build_estimator()

        # Most backends fit on numpy arrays; pass values for backend neutrality.
        # The sklearn HistGradientBoosting backend tolerates NaNs natively.
        self.estimator_.fit(X.to_numpy(dtype=float), y_arr)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict alpha scores for a feature matrix.

        Parameters
        ----------
        X:
            Feature matrix with the same columns used in :meth:`fit`.

        Returns
        -------
        numpy.ndarray
            One alpha score per row (higher is more attractive).
        """
        if self.estimator_ is None:
            raise RuntimeError("GBDTFactor must be fitted before predict().")
        X = self._validate_features(X)
        if self.feature_names_ is not None:
            missing = [c for c in self.feature_names_ if c not in X.columns]
            if missing:
                raise ValueError(f"X is missing fitted feature columns: {missing}")
            X = X[self.feature_names_]
        preds = self.estimator_.predict(X.to_numpy(dtype=float))
        preds = np.asarray(preds, dtype=float).ravel()
        if preds.shape[0] != len(X) or not np.all(np.isfinite(preds)):
            raise ValueError("estimator returned malformed or non-finite predictions")
        return preds

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def feature_importance(self) -> pd.Series:
        """Return feature importances as a sorted ``pd.Series``.

        Importances are read from the underlying estimator
        (``feature_importances_`` for tree backends). They are returned
        indexed by feature name and sorted in descending order.

        Returns
        -------
        pandas.Series
            Importances indexed by feature name (descending).
        """
        if self.estimator_ is None:
            raise RuntimeError("GBDTFactor must be fitted before feature_importance().")

        importances = getattr(self.estimator_, "feature_importances_", None)
        if importances is None:
            # sklearn's HistGradientBoostingRegressor does not expose
            # feature_importances_; fall back to permutation importance on the
            # training-equivalent signal is out of scope here, so report NaNs.
            importances = np.full(
                len(self.feature_names_ or []), np.nan, dtype=float
            )
        importances = np.asarray(importances, dtype=float).ravel()

        names = self.feature_names_ or [f"f{i}" for i in range(len(importances))]
        series = pd.Series(importances, index=names, name="importance")
        return series.sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Evaluation / ranking helpers
    # ------------------------------------------------------------------
    @staticmethod
    def rank_scores(scores: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Convert raw alpha scores into cross-sectional ranks in ``[0, 1]``.

        NaNs are preserved (a NaN score yields a NaN rank). Ties receive their
        average rank.

        Parameters
        ----------
        scores:
            Raw alpha scores for one cross-section.

        Returns
        -------
        numpy.ndarray
            Normalized ranks in ``[0, 1]`` (highest score -> ~1.0).
        """
        arr = np.asarray(scores, dtype=float).ravel()
        out = np.full(arr.shape, np.nan, dtype=float)
        finite = np.isfinite(arr)
        n = int(finite.sum())
        if n == 0:
            return out
        if n == 1:
            out[finite] = 0.5
            return out
        ranks = stats.rankdata(arr[finite], method="average")
        out[finite] = (ranks - 1.0) / (n - 1.0)
        return out

    @staticmethod
    def ic_rank(
        scores: Union[np.ndarray, pd.Series],
        realized: Union[np.ndarray, pd.Series],
    ) -> float:
        """Spearman rank information coefficient (rank IC) between two series.

        The rank IC is the Spearman correlation between predicted alpha scores
        and subsequently realized returns -- the standard cross-sectional
        measure of predictive power. Pairs with a NaN on either side are
        dropped before the correlation is computed.

        Parameters
        ----------
        scores:
            Predicted alpha scores.
        realized:
            Realized forward returns aligned to ``scores``.

        Returns
        -------
        float
            Spearman correlation, or ``nan`` if it cannot be computed.
        """
        s = np.asarray(scores, dtype=float).ravel()
        r = np.asarray(realized, dtype=float).ravel()
        if s.shape != r.shape:
            raise ValueError("scores and realized must have the same length")
        mask = np.isfinite(s) & np.isfinite(r)
        if mask.sum() < 3:
            return float("nan")
        s_m, r_m = s[mask], r[mask]
        # Spearman is undefined if either side is constant.
        if np.ptp(s_m) == 0 or np.ptp(r_m) == 0:
            return float("nan")
        rho, _ = stats.spearmanr(s_m, r_m)
        return float(rho)

    # ------------------------------------------------------------------
    # Causal cross-sectional walk-forward convenience
    # ------------------------------------------------------------------
    def fit_predict_cross_sectional(
        self,
        features_panel: pd.DataFrame,
        forward_returns: pd.DataFrame,
        train_window: int,
        *,
        min_train_obs: int = 50,
        rank_target: bool = True,
        step: int = 1,
        purge: int = 0,
    ) -> pd.DataFrame:
        """Walk-forward cross-sectional prediction with optional purging.

        For each evaluation date ``t`` (after ``train_window + purge`` warm-up
        dates), the model is *re-fit* on every (security, date)
        observation whose date lies within the trailing window
        ``[t - train_window - purge, t - purge)`` -- i.e. strictly before
        ``t`` -- and then used to score the cross-section observed on ``t``.

        Training *feature* dates always precede ``t``, but training *labels*
        are forward returns whose measurement window extends beyond their own
        date: with a label horizon of ``h`` periods, the labels of the last
        ``h - 1`` training dates overlap the test date and leak future
        information unless they are purged. Set ``purge`` to at least the
        label horizon minus one (e.g. ``purge >= h - 1``) to exclude those
        dates from the end of the training window and keep the score panel
        free of label look-ahead. With the default ``purge=0`` the panel is
        only leak-free for one-period (``h = 1``) labels.

        Parameters
        ----------
        features_panel:
            Either a long DataFrame with a ``MultiIndex`` of
            ``(date, symbol)`` and one column per feature, or it must be
            convertible to that shape. The index level names are not required
            but the first level is treated as the date.
        forward_returns:
            Forward returns aligned to ``features_panel``. Accepts either a
            ``Series``/single-column ``DataFrame`` on the same
            ``(date, symbol)`` ``MultiIndex``, or a wide ``date x symbol``
            DataFrame which is stacked internally.
        train_window:
            Number of distinct trailing *dates* used for training at each step.
        min_train_obs:
            Minimum number of training rows required to fit; dates with fewer
            available historical observations are skipped (scored as NaN).
        rank_target:
            If ``True`` (default and recommended), the training target is the
            per-date cross-sectional rank of the forward return in ``[0, 1]``,
            which stabilizes learning against heavy-tailed returns.
        step:
            Stride (in dates) between successive re-fits. ``step=1`` re-fits
            every date; larger values trade freshness for speed.
        purge:
            Number of label-horizon dates to exclude from the *end* of the
            training window (default 0). Must be at least the label horizon
            minus one to prevent training labels from overlapping the test
            date (see above).

        Returns
        -------
        pandas.DataFrame
            Wide ``date x symbol`` panel of out-of-sample alpha scores. Dates
            in the ``train_window + purge`` warm-up window (and any skipped for
            insufficient data) are all-NaN rows.
        """
        if (
            isinstance(train_window, (bool, np.bool_))
            or int(train_window) != train_window
            or train_window <= 0
        ):
            raise ValueError("train_window must be a positive integer")
        if (
            isinstance(min_train_obs, (bool, np.bool_))
            or int(min_train_obs) != min_train_obs
            or min_train_obs <= 0
        ):
            raise ValueError("min_train_obs must be a positive integer")
        if isinstance(step, (bool, np.bool_)) or int(step) != step or step <= 0:
            raise ValueError("step must be a positive integer")
        if isinstance(purge, (bool, np.bool_)) or int(purge) != purge or purge < 0:
            raise ValueError("purge must be a non-negative integer")
        if not isinstance(rank_target, (bool, np.bool_)):
            raise TypeError("rank_target must be a boolean")

        feats = self._to_long_features(features_panel)
        target = self._to_long_target(forward_returns)

        # Align labels for training, but derive evaluation dates and test rows
        # from features alone. A forward label is not available at the live
        # tail and must never be required merely to produce a prediction.
        joined = feats.join(target.rename("__y__"), how="inner")
        if joined.empty:
            raise ValueError("features_panel and forward_returns do not overlap")

        feature_cols = list(feats.columns)
        train_dates_index = joined.index.get_level_values(0)
        feature_dates_index = feats.index.get_level_values(0)
        unique_dates = pd.Index(sorted(pd.unique(feature_dates_index)))
        symbols = pd.Index(sorted(pd.unique(feats.index.get_level_values(1))))

        # Pre-compute the per-date cross-sectional rank target if requested.
        if rank_target:
            joined["__y__"] = (
                joined.groupby(level=0)["__y__"]
                .transform(lambda s: pd.Series(self.rank_scores(s.to_numpy()), index=s.index))
            )

        out = pd.DataFrame(
            np.nan, index=unique_dates, columns=symbols, dtype=float
        )

        # A full train_window must remain after the most recent `purge` dates
        # are removed. Starting earlier silently changes the configured sample
        # size in the first purged folds.
        for i in range(train_window + purge, len(unique_dates), step):
            t = unique_dates[i]
            if i - purge <= 0:
                continue
            # Strictly < t, with the most recent `purge` dates excluded so
            # multi-period training labels cannot overlap the test date.
            train_dates = unique_dates[max(0, i - train_window - purge) : i - purge]

            train_mask = train_dates_index.isin(train_dates)
            train = joined.loc[train_mask]
            train = train.dropna(subset=feature_cols + ["__y__"], how="any")
            if len(train) < min_train_obs:
                continue

            test = feats.loc[feature_dates_index == t]
            test_feat = test[feature_cols].dropna(how="any")
            if test_feat.empty:
                continue

            # Fresh estimator each step to avoid leaking state across folds.
            self.fit(train[feature_cols], train["__y__"])
            preds = self.predict(test_feat)

            test_symbols = test_feat.index.get_level_values(1)
            out.loc[t, test_symbols] = preds

        return out

    # ------------------------------------------------------------------
    # Internal validation / reshaping
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_features(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if X.shape[1] == 0:
            raise ValueError("X must have at least one feature column")
        if X.columns.has_duplicates:
            raise ValueError("X feature columns must be unique")
        numeric = X.apply(pd.to_numeric, errors="raise")
        values = numeric.to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("X must not contain infinite values")
        return numeric

    @staticmethod
    def _validate_target(y: pd.Series, expected_index: pd.Index) -> np.ndarray:
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("y DataFrame must have exactly one column")
            y = y.iloc[:, 0]
        if isinstance(y, pd.Series) and not y.index.equals(expected_index):
            raise ValueError("y index must exactly match X index")
        arr = np.asarray(y, dtype=float).ravel()
        if arr.shape[0] != len(expected_index):
            raise ValueError(
                f"y has {arr.shape[0]} rows but X has {len(expected_index)}; "
                "they must match"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("y must contain only finite values")
        return arr

    @staticmethod
    def _to_long_features(panel: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(panel, pd.DataFrame):
            raise TypeError("features_panel must be a pandas DataFrame")
        if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels < 2:
            raise ValueError(
                "features_panel must have a (date, symbol) MultiIndex"
            )
        if panel.index.has_duplicates:
            raise ValueError("features_panel index must be unique")
        return panel

    @staticmethod
    def _to_long_target(returns: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        if isinstance(returns, pd.Series):
            if not isinstance(returns.index, pd.MultiIndex):
                raise ValueError(
                    "forward_returns Series must have a (date, symbol) MultiIndex"
                )
            if returns.index.has_duplicates:
                raise ValueError("forward_returns index must be unique")
            return returns.astype(float)
        if isinstance(returns, pd.DataFrame):
            if isinstance(returns.index, pd.MultiIndex):
                if returns.shape[1] != 1:
                    raise ValueError(
                        "MultiIndexed forward_returns DataFrame must have one column"
                    )
                if returns.index.has_duplicates:
                    raise ValueError("forward_returns index must be unique")
                return returns.iloc[:, 0].astype(float)
            # Wide date x symbol -> stack to long.
            if returns.index.has_duplicates or returns.columns.has_duplicates:
                raise ValueError("wide forward_returns axes must be unique")
            stacked = returns.stack()
            stacked.index = stacked.index.set_names(["date", "symbol"])
            return stacked.astype(float)
        raise TypeError("forward_returns must be a Series or DataFrame")
