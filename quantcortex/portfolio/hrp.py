"""Hierarchical Risk Parity (HRP) portfolio optimizer.

This module implements the Hierarchical Risk Parity allocation method of
López de Prado (2016), *"Building Diversified Portfolios that Outperform
Out-of-Sample"* (Journal of Portfolio Management, 42(4), 59-69), from scratch
on top of :mod:`scipy.cluster.hierarchy`.

HRP sidesteps the instability of Markowitz mean-variance optimization (which
requires inverting an often ill-conditioned covariance matrix) by replacing the
single global optimization with three robust, sequential steps:

1. **Tree clustering** - assets are clustered hierarchically using a distance
   derived from their correlation matrix.
2. **Quasi-diagonalization** - the rows/columns of the covariance matrix are
   reordered so that the largest covariances lie along the diagonal, placing
   similar assets next to one another.
3. **Recursive bisection** - capital is split top-down through the reordered
   tree, allocating between the two halves inversely to their cluster variance.

The result is a long-only, fully-invested portfolio that requires neither
matrix inversion nor a return forecast. Its out-of-sample behavior remains an
empirical question for the dataset and validation design in use.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from quantcortex.portfolio.base import (
    PortfolioMode,
    PortfolioOptimizer,
    prepare_return_panel,
)

__all__ = ["HierarchicalRiskParity"]


class HierarchicalRiskParity(PortfolioOptimizer):
    """Long-only Hierarchical Risk Parity allocator.

    Parameters
    ----------
    mode:
        Allocation regime.  HRP is intrinsically a long-only, fully-invested
        method; the optimizer therefore always produces non-negative weights
        summing to 1.0 and only :data:`PortfolioMode.LONG_ONLY` is supported.
    linkage_method:
        Linkage criterion passed to :func:`scipy.cluster.hierarchy.linkage`
        (e.g. ``"single"``, ``"complete"``, ``"average"``, ``"ward"``).  The
        original paper uses single linkage; it is the default here.
    **kw:
        Forwarded to :class:`~quantcortex.portfolio.base.PortfolioOptimizer` (``tolerance``,
        ``weight_bounds``).

    Notes
    -----
    ``N == 1`` returns the sole asset. Degenerate covariance inputs are
    rejected. ``N == 2`` is handled exactly as a single
    recursive-bisection step (López de Prado's allocation is well-defined
    there): capital is split inversely to the two cluster variances.
    """

    def __init__(
        self,
        mode: Union[PortfolioMode, str] = PortfolioMode.LONG_ONLY,
        linkage_method: str = "single",
        **kw,
    ) -> None:
        super().__init__(mode, **kw)
        if self.mode is not PortfolioMode.LONG_ONLY:
            raise ValueError(
                "HierarchicalRiskParity only supports PortfolioMode.LONG_ONLY"
            )
        self.linkage_method = str(linkage_method)
        supported = {
            "single",
            "complete",
            "average",
            "weighted",
            "centroid",
            "median",
            "ward",
        }
        if self.linkage_method not in supported:
            raise ValueError(f"unsupported linkage_method {self.linkage_method!r}")

    # ------------------------------------------------------------------ #
    # Static / helper methods implementing the López de Prado algorithm.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _corr_dist(corr: pd.DataFrame) -> pd.DataFrame:
        """Correlation distance ``d_ij = sqrt(0.5 * (1 - rho_ij))``.

        This maps a correlation of ``+1`` to distance ``0`` and ``-1`` to
        distance ``1``, yielding a proper metric on the correlation matrix.
        """
        dist = ((1.0 - corr) / 2.0).clip(lower=0.0) ** 0.5
        # Guarantee an exact-zero, symmetric diagonal for ``squareform``.
        arr = dist.to_numpy(dtype=np.float64, copy=True)
        np.fill_diagonal(arr, 0.0)
        return pd.DataFrame(arr, index=dist.index, columns=dist.columns)

    @staticmethod
    def _get_quasi_diag(link: np.ndarray) -> list[int]:
        """Recover the leaf order from a SciPy linkage matrix.

        Implements López de Prado's ``getQuasiDiag``: starting from the final
        merge, recursively replace every cluster id ``>= N`` with the two items
        it was formed from, until only original-asset (leaf) indices remain.
        The resulting order places correlated assets adjacent to one another.
        """
        link = link.astype(int)
        # Number of original observations = clusters formed + 1.
        num_items = link[-1, 3]
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)  # make space
            clusters = sort_ix[sort_ix >= num_items]  # find clusters to expand
            i = clusters.index
            j = clusters.values - num_items
            sort_ix[i] = link[j, 0]  # item 1
            df0 = pd.Series(link[j, 1], index=i + 1)  # item 2
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()  # re-sort
            sort_ix.index = range(sort_ix.shape[0])  # re-index
        return sort_ix.tolist()

    @staticmethod
    def _get_ivp(cov: np.ndarray) -> np.ndarray:
        """Inverse-variance portfolio weights for a (sub-)covariance matrix."""
        ivp = 1.0 / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    @classmethod
    def _get_cluster_var(cls, cov: pd.DataFrame, items: list) -> float:
        """Variance of a cluster under its inverse-variance weighting.

        The cluster is collapsed to a single synthetic asset whose variance is
        ``w' Sigma w`` with ``w`` the inverse-variance weights of its members.
        """
        cov_slice = cov.loc[items, items].values
        w = cls._get_ivp(cov_slice).reshape(-1, 1)
        cluster_var = float((w.T @ cov_slice @ w)[0, 0])
        return cluster_var

    @classmethod
    def _get_rec_bipart(cls, cov: pd.DataFrame, sort_ix: list) -> pd.Series:
        """Recursive bisection allocation over the quasi-diagonal ordering.

        Capital starts fully allocated to the whole (ordered) universe.  At each
        step every current cluster is split into two contiguous halves; the
        weight of each half is scaled by ``1 - var_half / (var_left +
        var_right)`` so the lower-variance half receives more capital.  The
        process repeats until every cluster is a single asset.
        """
        w = pd.Series(1.0, index=sort_ix, dtype=float)
        clusters = [sort_ix]  # initialize with one cluster containing all items
        while len(clusters) > 0:
            # Bisect each cluster into two halves, dropping length-1 clusters.
            clusters = [
                half
                for cluster in clusters
                for half in (cluster[: len(cluster) // 2], cluster[len(cluster) // 2 :])
                if len(cluster) > 1
            ]
            # Process pairs (left half, right half).
            for i in range(0, len(clusters), 2):
                left = clusters[i]
                right = clusters[i + 1]
                var_left = cls._get_cluster_var(cov, left)
                var_right = cls._get_cluster_var(cov, right)
                denom = var_left + var_right
                alpha = 1.0 - var_left / denom if denom > 0 else 0.5
                w[left] *= alpha
                w[right] *= 1.0 - alpha
        return w

    # ------------------------------------------------------------------ #
    # Optimizer entry point.
    # ------------------------------------------------------------------ #
    def _compute_weights(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        """Compute HRP weights for the asset universe in ``returns``.

        Parameters
        ----------
        returns:
            ``(T x N)`` DataFrame of per-asset returns (columns = assets).

        Returns
        -------
        numpy.ndarray
            Long-only weights of shape ``(N,)`` summing to 1.0, in the original
            column order of ``returns``.
        """
        if not isinstance(returns, pd.DataFrame):
            returns = pd.DataFrame(np.asarray(returns, dtype=float))
        returns = prepare_return_panel(returns, name="HRP returns")

        n = returns.shape[1]
        columns = list(returns.columns)

        # --- Tiny universes. ---
        if n == 0:
            raise ValueError("HRP requires at least one asset")
        if n == 1:
            return self._project_configured_bounds(
                np.array([1.0], dtype=np.float64)
            )
        if n == 2:
            # LdP's recursive bisection is well-defined for two assets: a
            # single bisection allocating inversely to the cluster variances,
            # alpha = 1 - var([0]) / (var([0]) + var([1])) - i.e. weights
            # proportional to 1/variance.  Equal weight is kept only for the
            # degenerate-covariance case.
            cov2 = returns.cov()
            diag2 = np.diag(cov2.values)
            if not np.all(np.isfinite(cov2.values)) or not np.all(diag2 > 0.0):
                raise ValueError("HRP requires positive variance for every asset")
            var_left = self._get_cluster_var(cov2, [columns[0]])
            var_right = self._get_cluster_var(cov2, [columns[1]])
            denom = var_left + var_right
            if denom <= 0.0 or not np.isfinite(denom):
                raise ValueError("HRP cluster variances must be finite and positive")
            alpha = 1.0 - var_left / denom
            return self._project_configured_bounds(
                np.array([alpha, 1.0 - alpha], dtype=np.float64)
            )

        cov = returns.cov()
        corr = returns.corr()
        if (
            not np.all(np.isfinite(cov.values))
            or not np.all(np.isfinite(corr.values))
            or np.any(np.diag(cov.values) <= 0.0)
        ):
            raise ValueError("HRP requires finite positive asset variances")

        corr_arr = corr.to_numpy(dtype=np.float64, copy=True)
        np.fill_diagonal(corr_arr, 1.0)
        corr = pd.DataFrame(corr_arr, index=corr.index, columns=corr.columns)

        # (a) correlation distance.
        dist = self._corr_dist(corr)

        # (b) hierarchical clustering on the condensed distance matrix.
        condensed = squareform(dist.to_numpy(dtype=np.float64), checks=False)
        link = linkage(condensed, method=self.linkage_method)

        # (c) quasi-diagonalization: recover leaf order, mapped to labels.
        sort_ix_pos = self._get_quasi_diag(link)
        sort_ix = [columns[i] for i in sort_ix_pos]

        # (d) recursive bisection.
        hrp = self._get_rec_bipart(cov, sort_ix)

        # Restore original column ordering and normalize defensively.
        weights = hrp.reindex(columns).values.astype(np.float64)
        weights = np.clip(weights, 0.0, None)
        total = weights.sum()
        if total <= 0.0 or not np.isfinite(total):
            raise RuntimeError("HRP produced invalid recursive-bisection weights")
        return self._project_configured_bounds(weights / total)
