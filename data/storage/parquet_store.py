"""Partitioned Parquet storage backed by PyArrow.

A thin, round-trippable store for tabular datasets (OHLCV, features,
fundamentals).  Datasets live under ``root/<dataset>`` as partitioned Parquet
and can be read back with optional column projection and predicate pushdown.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq

__all__ = ["ParquetStore"]

PathLike = Union[str, Path]


class ParquetStore:
    """Filesystem-backed partitioned Parquet store.

    Parameters
    ----------
    root:
        Root directory under which datasets are stored (created if absent).
    """

    def __init__(self, root: PathLike) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dataset_path(self, dataset: str) -> Path:
        return self.root / dataset

    def write(
        self,
        df: pd.DataFrame,
        dataset: str,
        partition_cols: Optional[List[str]] = None,
        mode: str = "overwrite",
    ) -> Path:
        """Write ``df`` to ``dataset``.

        Parameters
        ----------
        df:
            DataFrame to persist.
        dataset:
            Dataset name (a subdirectory of ``root``).
        partition_cols:
            Columns to hive-partition by.
        mode:
            ``"overwrite"`` replaces the whole dataset directory; ``"append"``
            adds new Parquet files alongside the existing ones.
        """
        if mode not in ("overwrite", "append"):
            raise ValueError(f"unknown mode {mode!r}; use 'overwrite' or 'append'")

        path = self._dataset_path(dataset)
        if mode == "overwrite" and path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

        # Reset a meaningful index into a column so it survives the round trip.
        if df.index.name is not None or not isinstance(df.index, pd.RangeIndex):
            table_df = df.reset_index()
        else:
            table_df = df
        table = pa.Table.from_pandas(table_df, preserve_index=False)

        existing = "overwrite_or_ignore" if mode == "append" else "overwrite_or_ignore"
        pq.write_to_dataset(
            table,
            root_path=str(path),
            partition_cols=partition_cols or None,
            existing_data_behavior=existing,
        )
        return path

    def read(
        self,
        dataset: str,
        columns: Optional[List[str]] = None,
        filters=None,
    ) -> pd.DataFrame:
        """Read ``dataset`` back into a DataFrame.

        Parameters
        ----------
        columns:
            Optional column projection.
        filters:
            Optional predicate, either a PyArrow :class:`Expression` or the
            disjunctive-normal-form list-of-tuples accepted by
            :func:`pyarrow.parquet.read_table`.
        """
        path = self._dataset_path(dataset)
        if not path.exists():
            raise FileNotFoundError(f"dataset {dataset!r} not found under {self.root}")

        dataset_obj = pa_ds.dataset(str(path), format="parquet", partitioning="hive")

        expr = None
        if filters is not None:
            if isinstance(filters, pa_ds.Expression):
                expr = filters
            else:
                expr = pq.filters_to_expression(filters)

        table = dataset_obj.to_table(columns=columns, filter=expr)
        return table.to_pandas()

    def list_datasets(self) -> List[str]:
        """Return the names of all datasets under ``root``."""
        if not self.root.exists():
            return []
        return sorted(
            p.name for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")
        )

    def exists(self, dataset: str) -> bool:
        """Return ``True`` if ``dataset`` exists."""
        path = self._dataset_path(dataset)
        return path.exists() and path.is_dir()
