"""Time-series momentum (TSMOM) timing overlay.

:class:`TSMomentum` gates portfolio exposure on the sign of each asset's trailing
return over a ``lookback`` window.  The premise - well documented in the
cross-sectional and time-series momentum literature (Moskowitz, Ooi & Pedersen)
 -  is that an asset's own past return predicts the sign of its near-future
return.

Two modes:

* **long-flat** (``allow_short=False``, default): an asset keeps its weight when
  its trailing return is positive and is set to zero otherwise - the overlay can
  only *reduce* gross exposure.
* **long-short** (``allow_short=True``): an asset's weight is multiplied by the
  *sign* of its trailing return, so a long flips to a short when momentum turns
  negative.  This changes the net (sum) of the book but never increases gross
  exposure, so it still satisfies the exposure contract.

Causality
---------
The input is expected to include the contemporaneous bar ``t`` as its final
row. The trailing return deliberately drops that row and uses observations up
to ``t-1``, so the signal acting on bar ``t`` is observable before ``t``'s
return is realised.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import pandas as pd

from quantcortex.portfolio.base import enforce_exposure_contract

__all__ = ["TSMomentum"]


class TSMomentum:
    """Sign-of-trailing-return momentum overlay.

    Parameters
    ----------
    lookback:
        Number of trailing return observations used to form the momentum signal.
    allow_short:
        If ``False`` (default) weights are gated to long-or-flat
        (``max(sign, 0)``).  If ``True`` weights are multiplied by the full sign,
        flipping longs to shorts when momentum is negative.
    """

    def __init__(self, lookback: int = 21, *, allow_short: bool = False) -> None:
        if (
            isinstance(lookback, (bool, np.bool_))
            or not isinstance(lookback, (int, np.integer))
            or lookback < 1
        ):
            raise ValueError("lookback must be a positive integer")
        if not isinstance(allow_short, (bool, np.bool_)):
            raise TypeError("allow_short must be a boolean")
        self.lookback = int(lookback)
        self.allow_short = bool(allow_short)

    # ------------------------------------------------------------------ #
    def apply(
        self,
        weights: np.ndarray,
        returns: Union[pd.DataFrame, pd.Series, np.ndarray, Any] = None,
    ) -> np.ndarray:
        """Apply the momentum gate to ``weights``.

        Parameters
        ----------
        weights:
            Per-asset weight vector (1-D).
        returns:
            * 2-D (DataFrame / ``(T, n_assets)`` array): per-asset gating - each
              asset's weight is multiplied by the sign of *its own* trailing
              return.
            * 1-D (Series / ``(T,)`` array): treated as a portfolio return
              series; the *whole* weight vector is scaled by the sign of the
              trailing portfolio return.
            * a ``StrategyContext``-like object exposing ``.returns``.

        Returns
        -------
        numpy.ndarray
            Gated weights, validated via :func:`enforce_exposure_contract`.
        """
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w)):
            raise ValueError("weights must be a non-empty finite 1-D vector")
        ret = self._coerce_returns(returns)

        signals = self._trailing_signal(ret, n_assets=w.size)
        multiplier = signals if self.allow_short else np.maximum(signals, 0.0)
        gated = w * multiplier

        input_gross = float(np.abs(w).sum())
        max_gross = max(1.0, input_gross) + 1e-9
        return enforce_exposure_contract(
            gated, max_gross=max_gross, name=type(self).__name__
        )

    # ------------------------------------------------------------------ #
    def _trailing_signal(self, ret: np.ndarray, n_assets: int) -> np.ndarray:
        """Compute the per-asset (or broadcast) sign-of-trailing-return signal.

        The trailing window ends at ``t-1`` (the most recent fully observed bar
        excluding the contemporaneous one): we drop the final row, then take the
        last ``lookback`` rows and compound their simple returns.
        """
        arr = np.asarray(ret, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ValueError("returns must be 1-D or 2-D")
        if arr.shape[1] == 0 or not np.all(np.isfinite(arr)):
            raise ValueError("returns must contain finite observations")
        if np.any(arr < -1.0):
            raise ValueError("simple returns cannot be less than -100%")

        # Causal: exclude the contemporaneous (last) observation if we have more
        # than one row, so the signal is known strictly before bar t.
        usable = arr[:-1]
        window = usable[-self.lookback:]
        if window.shape[0] == 0:
            trailing = np.zeros(arr.shape[1], dtype=np.float64)
        else:
            trailing = np.prod(1.0 + window, axis=0) - 1.0
        signs = np.sign(trailing)  # in {-1, 0, +1}

        if signs.size == 1 and n_assets > 1:
            # Portfolio-level (1-D) return series -> broadcast to all assets.
            signs = np.full(n_assets, signs[0], dtype=np.float64)
        elif signs.size != n_assets:
            if signs.size == 1:
                signs = np.full(n_assets, signs[0], dtype=np.float64)
            else:
                raise ValueError(
                    f"returns has {signs.size} asset columns but weights has "
                    f"{n_assets}"
                )
        return signs.astype(np.float64)

    @staticmethod
    def _coerce_returns(
        returns: Union[pd.DataFrame, pd.Series, np.ndarray, Any],
    ) -> Union[np.ndarray]:
        """Normalise ``returns`` (incl. context objects) to a numpy array."""
        if isinstance(returns, pd.DataFrame):
            return returns.to_numpy(dtype=np.float64)
        if isinstance(returns, pd.Series):
            return returns.to_numpy(dtype=np.float64)
        if isinstance(returns, np.ndarray):
            return returns.astype(np.float64)
        if isinstance(returns, (list, tuple)):
            return np.asarray(returns, dtype=np.float64)

        ctx_returns = getattr(returns, "returns", None)
        if ctx_returns is not None:
            return TSMomentum._coerce_returns(ctx_returns)
        raise TypeError(
            "TSMomentum.apply requires returns (DataFrame/Series/array) or a "
            "context object exposing .returns"
        )
