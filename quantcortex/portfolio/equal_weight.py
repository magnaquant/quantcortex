"""Equal-weight (1/N) portfolio optimizer.

The 1/N portfolio is a famously hard-to-beat benchmark (DeMiguel, Garlappi &
Uppal, 2009).  It allocates an identical fraction of capital to every asset in
the investable set and therefore needs no return or covariance estimate at all,
which makes it the natural sanity-check optimizer for the weight contract.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from quantcortex.portfolio.base import PortfolioMode, PortfolioOptimizer

__all__ = ["EqualWeight"]


def _infer_n_assets(returns) -> int:
    """Best-effort extraction of the asset count from a variety of inputs.

    DataFrames (columns = assets) and Series (ONE asset's return history, so
    ``n = 1``) are the canonical inputs.  A 1-D ndarray is ambiguous - one
    asset's history vs. one observation per asset - and keeps the historical
    interpretation of ``shape[0]`` elements = assets; prefer passing a
    DataFrame/Series (or an explicit ``n_assets``) to avoid the ambiguity.
    """
    if returns is None:
        raise ValueError("EqualWeight needs either `returns` or `n_assets`.")
    if isinstance(returns, (int, np.integer)):
        return int(returns)
    if isinstance(returns, pd.Series):
        # A Series is a single asset's return history, not one asset per row.
        return 1
    shape = getattr(returns, "shape", None)
    if shape is not None:
        # DataFrame / 2-D ndarray -> columns are assets; 1-D ndarray -> elements
        # (ambiguous; see docstring).
        return int(shape[1]) if len(shape) > 1 else int(shape[0])
    return len(returns)


class EqualWeight(PortfolioOptimizer):
    """Allocate ``1/N`` to each asset (long-only by construction)."""

    def __init__(self, **kwargs) -> None:
        # Equal weight is only meaningful long-only; force the mode.
        kwargs.setdefault("mode", PortfolioMode.LONG_ONLY)
        if PortfolioMode.coerce(kwargs["mode"]) is not PortfolioMode.LONG_ONLY:
            raise ValueError("EqualWeight only supports long_only mode.")
        super().__init__(**kwargs)

    def _compute_weights(
        self,
        returns=None,
        *,
        n_assets: Optional[int] = None,
        **_: Union[int, float],
    ) -> np.ndarray:
        n = int(n_assets) if n_assets is not None else _infer_n_assets(returns)
        if n <= 0:
            raise ValueError(f"n_assets must be positive, got {n}")
        return np.full(n, 1.0 / n, dtype=np.float64)
