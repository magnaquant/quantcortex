"""Partitioned Parquet storage backed by PyArrow.

A thin, round-trippable store for tabular datasets (OHLCV, features,
fundamentals).  Datasets live under ``root/<dataset>`` as partitioned Parquet
and can be read back with optional column projection and predicate pushdown.
"""

from __future__ import annotations

import base64
import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq

__all__ = ["ParquetStore"]

PathLike = Union[str, Path]
_DATASET_MARKER = ".quantcortex-dataset"


class ParquetStore:
    """Filesystem-backed partitioned Parquet store.

    Parameters
    ----------
    root:
        Root directory under which datasets are stored (created if absent).
    """

    def __init__(self, root: PathLike) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _dataset_path(self, dataset: str) -> Path:
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("dataset must be a non-empty relative path")
        relative = Path(dataset)
        if relative.is_absolute():
            raise ValueError("dataset must be relative to the store root")
        candidate = (self.root / relative).resolve()
        if candidate == self.root or not candidate.is_relative_to(self.root):
            raise ValueError("dataset path escapes the store root")
        return candidate

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

        Notes
        -----
        A meaningful index (named, or any non-``RangeIndex``) is materialized
        as an ordinary column before writing.  :meth:`read` does *not* restore
        it as the index: it comes back as a plain column, and callers must
        ``set_index`` themselves if they want the index back.
        """
        if mode not in ("overwrite", "append"):
            raise ValueError(f"unknown mode {mode!r}; use 'overwrite' or 'append'")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty:
            raise ValueError("df must contain at least one row")

        partition_cols = list(partition_cols or [])
        if len(partition_cols) != len(set(partition_cols)) or any(
            not isinstance(column, str) or not column for column in partition_cols
        ):
            raise ValueError("partition_cols must contain unique non-empty names")
        missing_partitions = [
            column for column in partition_cols if column not in df.columns
        ]
        if missing_partitions:
            raise ValueError(
                f"partition columns are missing from df: {missing_partitions}"
            )

        path = self._dataset_path(dataset)

        # Reset a meaningful index into a column so it survives the round trip.
        if df.index.name is not None or not isinstance(df.index, pd.RangeIndex):
            table_df = df.reset_index()
        else:
            table_df = df
        table = pa.Table.from_pandas(table_df, preserve_index=False)

        if mode == "append" and path.exists():
            marker = path / _DATASET_MARKER
            if marker.exists():
                metadata = json.loads(marker.read_text(encoding="ascii"))
                existing_schema = pa.ipc.read_schema(
                    pa.BufferReader(base64.b64decode(metadata["schema"]))
                )
                existing_partitions = metadata["partition_cols"]
            else:
                existing_schema = pa_ds.dataset(
                    str(path), format="parquet", partitioning="hive"
                ).schema
                existing_partitions = partition_cols
            if not table.schema.equals(existing_schema, check_metadata=False):
                raise ValueError(
                    "append schema does not match the existing dataset schema"
                )
            if partition_cols != existing_partitions:
                raise ValueError(
                    "append partition_cols do not match the existing dataset"
                )

        if mode == "overwrite":
            path.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(prefix=f".{path.name}-staging-", dir=path.parent)
            )
            backup: Optional[Path] = None
            try:
                self._write_table(table, staging, partition_cols)
                if path.exists():
                    backup = path.with_name(f".{path.name}-backup-{uuid.uuid4().hex}")
                    path.rename(backup)
                try:
                    staging.rename(path)
                except Exception:
                    if backup is not None and backup.exists() and not path.exists():
                        backup.rename(path)
                    raise
                if backup is not None:
                    shutil.rmtree(backup)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        else:
            path.mkdir(parents=True, exist_ok=True)
            self._write_table(table, path, partition_cols)
        return path

    @staticmethod
    def _write_table(
        table: pa.Table, path: Path, partition_cols: List[str]
    ) -> None:
        pq.write_to_dataset(
            table,
            root_path=str(path),
            partition_cols=partition_cols or None,
            basename_template=f"part-{uuid.uuid4().hex}-{{i}}.parquet",
            existing_data_behavior="overwrite_or_ignore",
        )
        metadata = {
            "partition_cols": partition_cols,
            "schema": base64.b64encode(table.schema.serialize().to_pybytes()).decode(
                "ascii"
            ),
        }
        (path / _DATASET_MARKER).write_text(
            json.dumps(metadata, sort_keys=True), encoding="ascii"
        )

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

        Notes
        -----
        An index materialized as a column by :meth:`write` is returned as a
        plain column (with a default ``RangeIndex``); it is *not* restored as
        the DataFrame index.  Call ``set_index`` on the result if needed.
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
            str(marker.parent.relative_to(self.root))
            for marker in self.root.rglob(_DATASET_MARKER)
            if not any(
                part.startswith(".")
                for part in marker.parent.relative_to(self.root).parts
            )
        )

    def exists(self, dataset: str) -> bool:
        """Return ``True`` if ``dataset`` exists."""
        path = self._dataset_path(dataset)
        return path.exists() and path.is_dir()
