"""Simple multi-layer perceptron (MLP) return-prediction baseline.

:class:`NeuralFactor` is a small, deterministic feed-forward neural network
used as a *baseline* against the gradient-boosted trees in
:mod:`quantcortex.alpha.factors.ml.gbdt_factor`. On tabular financial cross-sections GBDTs
typically win, but a shallow MLP is the standard sanity-check / ensemble member
(Gu, Kelly & Xiu, 2020): if a neural net cannot beat the trees, the trees are
the model to ship.

The network is intentionally tiny (a couple of hidden layers, ReLU
activations, MSE loss, Adam optimizer). It uses Torch when available and
also supports scikit-learn's :class:`~sklearn.neural_network.MLPRegressor` with
the same hidden-layer sizes. Backend choice is explicit because changing the
library changes the fitted model even with identical hyperparameters.

Inputs are standardized with :class:`~sklearn.preprocessing.StandardScaler`
(fit on the training data only -- causal), since neural nets, unlike trees, are
sensitive to feature scaling.

``import torch`` is performed lazily *inside* :meth:`NeuralFactor.fit` so that
merely importing this module needs only numpy/pandas/scipy/scikit-learn.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class NeuralFactor:
    """A shallow MLP regressor baseline for cross-sectional return prediction.

    Parameters
    ----------
    hidden:
        Sizes of the hidden layers, e.g. ``(64, 32)`` for two hidden layers.
    epochs:
        Number of training epochs (Torch backend) / max iterations
        (scikit-learn backend).
    lr:
        Learning rate for the Adam optimizer.
    batch_size:
        Mini-batch size for the Torch backend (ignored by the sklearn
        backend, which manages its own batching).
    weight_decay:
        L2 regularization strength (Adam ``weight_decay`` / sklearn ``alpha``).
    random_state:
        Seed for reproducible weight initialization and shuffling. Defaults to
        ``42`` for determinism.
    backend:
        ``"sklearn"`` (the deterministic default), ``"torch"``, or ``"auto"``.
        ``"auto"`` uses Torch when importable and otherwise uses sklearn; it is
        therefore environment-dependent. Runtime training failures never
        silently switch model classes.

    Notes
    -----
    This is a *baseline*. For tabular alpha signals, prefer
    :class:`~quantcortex.alpha.factors.ml.gbdt_factor.GBDTFactor`; use this model to
    benchmark the trees or as a diversifying ensemble member.
    """

    def __init__(
        self,
        hidden: Sequence[int] = (64, 32),
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        random_state: int = 42,
        backend: str = "sklearn",
    ) -> None:
        if not isinstance(hidden, Sequence) or isinstance(hidden, (str, bytes)):
            raise TypeError("hidden must be a sequence of positive integers")
        if not hidden or any(
            isinstance(h, (bool, np.bool_))
            or not isinstance(h, (int, np.integer))
            or h <= 0
            for h in hidden
        ):
            raise ValueError("hidden must contain positive integers")
        if (
            isinstance(epochs, (bool, np.bool_))
            or not isinstance(epochs, (int, np.integer))
            or epochs <= 0
        ):
            raise ValueError("epochs must be a positive integer")
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        if (
            isinstance(batch_size, (bool, np.bool_))
            or not isinstance(batch_size, (int, np.integer))
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if (
            isinstance(random_state, (bool, np.bool_))
            or not isinstance(random_state, (int, np.integer))
        ):
            raise ValueError("random_state must be an integer")
        if backend not in {"auto", "sklearn", "torch"}:
            raise ValueError("backend must be 'auto', 'sklearn', or 'torch'")

        self.hidden = tuple(int(h) for h in hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.random_state = int(random_state)
        self.backend = backend

        # Populated by fit().
        self.backend_: Optional[str] = None
        self.scaler_: Optional[StandardScaler] = None
        self.model_: object = None
        self.feature_names_: Optional[list[str]] = None
        self._impute_means_: Optional[np.ndarray] = None
        self._y_mean_: float = 0.0
        self._y_std_: float = 1.0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NeuralFactor":
        """Fit the MLP on a feature matrix and target.

        Parameters
        ----------
        X:
            Feature matrix, one row per observation.
        y:
            Target aligned to ``X`` (forward returns or cross-sectional ranks).

        Returns
        -------
        NeuralFactor
            ``self``, fitted.
        """
        X = self._validate_features(X)
        y_arr = self._validate_target(y, X.index)
        self.feature_names_ = list(X.columns)

        # Standardize inputs (fit on training data only -> causal).
        self.scaler_ = StandardScaler()
        # NaNs would break the net; impute column means after scaling fit.
        X_arr = X.to_numpy(dtype=float)
        col_means = np.nanmean(np.where(np.isfinite(X_arr), X_arr, np.nan), axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        X_filled = np.where(np.isfinite(X_arr), X_arr, col_means)
        self._impute_means_ = col_means
        X_scaled = self.scaler_.fit_transform(X_filled)

        if self.backend == "sklearn":
            self._fit_sklearn(X_scaled, y_arr)
            self.backend_ = "sklearn"
            return self

        try:
            import torch  # noqa: F401
        except (ImportError, OSError) as exc:
            if self.backend == "torch":
                raise ImportError(
                    "backend='torch' requested but Torch is unavailable"
                ) from exc
            self._fit_sklearn(X_scaled, y_arr)
            self.backend_ = "sklearn"
        else:
            self._fit_torch(X_scaled, y_arr)
            self.backend_ = "torch"
        return self

    def _fit_torch(self, X_scaled: np.ndarray, y_arr: np.ndarray) -> None:
        """Train a small Torch MLP with Adam + MSE (lazy torch import)."""
        import torch  # lazy
        from torch import nn

        # Standardize the target so MSE is well-scaled; undone at predict time.
        self._y_mean_ = float(np.mean(y_arr))
        self._y_std_ = float(np.std(y_arr)) or 1.0
        y_std = (y_arr - self._y_mean_) / self._y_std_

        # Model initialization uses Torch's process-global CPU RNG. fork_rng
        # restores the caller's state when fitting completes.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_state)
            n_features = X_scaled.shape[1]
            model = self._build_torch_mlp(nn, n_features)

            X_t = torch.as_tensor(X_scaled, dtype=torch.float32)
            y_t = torch.as_tensor(y_std, dtype=torch.float32).view(-1, 1)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            loss_fn = nn.MSELoss()

            n = X_t.shape[0]
            batch = min(self.batch_size, n)
            generator = torch.Generator().manual_seed(self.random_state)

            model.train()
            for _ in range(self.epochs):
                perm = torch.randperm(n, generator=generator)
                for start in range(0, n, batch):
                    idx = perm[start : start + batch]
                    optimizer.zero_grad()
                    pred = model(X_t[idx])
                    loss = loss_fn(pred, y_t[idx])
                    loss.backward()
                    optimizer.step()

        model.eval()
        self.model_ = model

    def _build_torch_mlp(self, nn, n_features: int):
        """Construct an ``nn.Sequential`` MLP with ReLU activations."""
        layers = []
        in_dim = n_features
        for h in self.hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def _fit_sklearn(self, X_scaled: np.ndarray, y_arr: np.ndarray) -> None:
        """Fallback: scikit-learn MLPRegressor with matching architecture."""
        from sklearn.neural_network import MLPRegressor

        # Target standardization kept consistent with the torch path so
        # predict() can invert it uniformly.
        self._y_mean_ = float(np.mean(y_arr))
        self._y_std_ = float(np.std(y_arr)) or 1.0
        y_std = (y_arr - self._y_mean_) / self._y_std_

        model = MLPRegressor(
            hidden_layer_sizes=self.hidden,
            activation="relu",
            solver="adam",
            learning_rate_init=self.lr,
            alpha=self.weight_decay,
            batch_size=min(self.batch_size, len(X_scaled)),
            max_iter=self.epochs,
            random_state=self.random_state,
            shuffle=True,
        )
        model.fit(X_scaled, y_std)
        self.model_ = model

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
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
        if self.model_ is None or self.scaler_ is None:
            raise RuntimeError("NeuralFactor must be fitted before predict().")
        X = self._validate_features(X)
        if self.feature_names_ is not None:
            missing = [c for c in self.feature_names_ if c not in X.columns]
            if missing:
                raise ValueError(f"X is missing fitted feature columns: {missing}")
            X = X[self.feature_names_]

        if self._impute_means_ is None:
            raise RuntimeError("NeuralFactor fitted state is missing imputation means")
        X_arr = X.to_numpy(dtype=float)
        X_filled = np.where(np.isfinite(X_arr), X_arr, self._impute_means_)
        X_scaled = self.scaler_.transform(X_filled)

        if self.backend_ == "torch":
            import torch  # lazy

            with torch.no_grad():
                preds = (
                    self.model_(torch.as_tensor(X_scaled, dtype=torch.float32))
                    .cpu()
                    .numpy()
                    .ravel()
                )
        elif self.backend_ == "sklearn":
            preds = np.asarray(self.model_.predict(X_scaled), dtype=float).ravel()
        else:
            raise RuntimeError(f"unknown fitted NeuralFactor backend {self.backend_!r}")

        # Undo target standardization.
        result = preds * self._y_std_ + self._y_mean_
        if result.shape != (len(X),):
            raise RuntimeError(
                f"NeuralFactor backend returned {result.size} predictions for "
                f"{len(X)} rows"
            )
        if not np.all(np.isfinite(result)):
            raise RuntimeError("NeuralFactor backend returned non-finite predictions")
        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_features(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must have at least one row and feature column")
        if X.columns.has_duplicates:
            raise ValueError("X feature columns must be unique")
        numeric = X.apply(pd.to_numeric, errors="raise")
        values = numeric.to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("X must not contain infinite values")
        if np.isnan(values).all(axis=0).any():
            raise ValueError("X must not contain an entirely missing feature column")
        return numeric

    @staticmethod
    def _validate_target(
        y: Union[pd.Series, pd.DataFrame], expected_index: pd.Index
    ) -> np.ndarray:
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
            raise ValueError("y contains non-finite values; clean the target first")
        return arr
