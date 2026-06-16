"""Canonical target-tape validation for backtest conformance tests.

The target tape is a long-form, engine-neutral representation of close-of-bar
portfolio decisions. It deliberately contains no prices or returns, so the same
decision stream can be evaluated by multiple engines against a separately
specified market-data tape.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

TARGET_TAPE_COLUMNS = (
    "decision_timestamp",
    "symbol",
    "target_weight",
)
TARGET_TAPE_SCHEMA_VERSION = 1


def _validated_max_gross(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("max_gross must be a finite positive number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("max_gross must be a finite positive number") from exc
    if not np.isfinite(parsed):
        raise TypeError("max_gross must be a finite positive number")
    if parsed <= 0.0:
        raise ValueError("max_gross must be positive")
    return parsed


def validate_target_tape(
    tape: pd.DataFrame,
    *,
    max_gross: float = 1.0,
    expected_symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Validate and normalize a canonical long-form target tape.

    Returned timestamps are UTC-normalized and timezone-naive, matching the
    repository's internal convention. Every decision must explicitly contain
    the same complete symbol set.
    """
    if not isinstance(tape, pd.DataFrame):
        raise TypeError("tape must be a pandas DataFrame")
    if tape.empty:
        raise ValueError("target tape must not be empty")
    if tuple(tape.columns) != TARGET_TAPE_COLUMNS:
        raise ValueError(
            "target tape columns must be exactly " + ", ".join(TARGET_TAPE_COLUMNS)
        )
    max_gross = _validated_max_gross(max_gross)

    normalized = tape.copy()
    try:
        timestamps = pd.to_datetime(
            normalized["decision_timestamp"],
            utc=True,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("decision_timestamp contains an invalid timestamp") from exc
    normalized["decision_timestamp"] = timestamps.dt.tz_convert("UTC").dt.tz_localize(
        None
    )
    if normalized["decision_timestamp"].isna().any():
        raise ValueError("decision_timestamp must not be missing")

    symbols = normalized["symbol"]
    if symbols.isna().any() or not symbols.map(lambda value: isinstance(value, str)).all():
        raise ValueError("symbol values must be non-empty strings")
    normalized["symbol"] = symbols.str.strip()
    if (normalized["symbol"] == "").any():
        raise ValueError("symbol values must be non-empty strings")

    normalized["target_weight"] = pd.to_numeric(
        normalized["target_weight"],
        errors="coerce",
    )
    weights = normalized["target_weight"].to_numpy(dtype=float)
    if not np.all(np.isfinite(weights)):
        raise ValueError("target_weight values must be finite")
    if (weights < -1e-12).any():
        raise ValueError("target_weight values must be long-only")
    normalized.loc[normalized["target_weight"].abs() < 1e-15, "target_weight"] = 0.0

    duplicate = normalized.duplicated(
        subset=["decision_timestamp", "symbol"],
        keep=False,
    )
    if duplicate.any():
        raise ValueError("target tape contains duplicate timestamp-symbol rows")

    if expected_symbols is None:
        required_symbols = tuple(sorted(normalized["symbol"].unique()))
    else:
        if isinstance(expected_symbols, (str, bytes)):
            raise TypeError("expected_symbols must be a sequence of symbol strings")
        required_symbols = tuple(expected_symbols)
        if not required_symbols or any(
            not isinstance(symbol, str) or not symbol.strip()
            for symbol in required_symbols
        ):
            raise ValueError("expected_symbols must contain non-empty strings")
        required_symbols = tuple(sorted(symbol.strip() for symbol in required_symbols))
        if len(set(required_symbols)) != len(required_symbols):
            raise ValueError("expected_symbols must be unique after normalization")

    required_set = set(required_symbols)
    for timestamp, group in normalized.groupby("decision_timestamp", sort=False):
        observed = set(group["symbol"])
        if observed != required_set:
            missing = sorted(required_set - observed)
            extra = sorted(observed - required_set)
            raise ValueError(
                f"decision {timestamp!s} has incomplete symbols; "
                f"missing={missing}, extra={extra}"
            )
        gross = float(group["target_weight"].abs().sum())
        if gross > max_gross + 1e-12:
            raise ValueError(
                f"decision {timestamp!s} gross exposure {gross:.12g} "
                f"exceeds {max_gross:.12g}"
            )

    return normalized.sort_values(
        ["decision_timestamp", "symbol"],
        kind="stable",
    ).reset_index(drop=True)


def target_tape_to_weights(
    tape: pd.DataFrame,
    *,
    max_gross: float = 1.0,
    expected_symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Convert a validated long-form target tape to an engine weight frame."""
    normalized = validate_target_tape(
        tape,
        max_gross=max_gross,
        expected_symbols=expected_symbols,
    )
    weights = normalized.pivot(
        index="decision_timestamp",
        columns="symbol",
        values="target_weight",
    )
    weights.columns.name = None
    weights.index.name = None
    return weights.sort_index().sort_index(axis=1)


def weights_to_target_tape(
    weights: pd.DataFrame,
    *,
    max_gross: float = 1.0,
) -> pd.DataFrame:
    """Convert a complete target-weight frame to canonical long form."""
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("weights must be a pandas DataFrame")
    if weights.empty:
        raise ValueError("weights must not be empty")
    if not isinstance(weights.index, pd.DatetimeIndex):
        raise TypeError("weights must use a DatetimeIndex")
    if weights.index.hasnans or weights.index.has_duplicates:
        raise ValueError("weights index must contain unique valid timestamps")
    if weights.columns.has_duplicates or weights.shape[1] == 0:
        raise ValueError("weights must have unique, non-empty symbol columns")
    if any(not isinstance(symbol, str) or not symbol.strip() for symbol in weights.columns):
        raise ValueError("weights columns must be non-empty strings")

    normalized = weights.copy()
    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_convert("UTC").tz_localize(None)
    normalized.index.name = "decision_timestamp"
    normalized.columns.name = "symbol"
    tape = normalized.reset_index().melt(
        id_vars="decision_timestamp",
        var_name="symbol",
        value_name="target_weight",
    )
    return validate_target_tape(
        tape.loc[:, TARGET_TAPE_COLUMNS],
        max_gross=max_gross,
        expected_symbols=list(weights.columns),
    )


def target_tape_to_payload(
    tape: pd.DataFrame,
    *,
    max_gross: float = 1.0,
    expected_symbols: Sequence[str] | None = None,
) -> dict[str, object]:
    """Serialize a target tape to the versioned JSON-compatible envelope."""
    normalized = validate_target_tape(
        tape,
        max_gross=max_gross,
        expected_symbols=expected_symbols,
    )
    parsed_max_gross = _validated_max_gross(max_gross)
    records = []
    for row in normalized.itertuples(index=False):
        timestamp = pd.Timestamp(row.decision_timestamp).tz_localize("UTC")
        records.append(
            {
                "decision_timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                "symbol": str(row.symbol),
                "target_weight": float(row.target_weight),
            }
        )
    return {
        "schema_version": TARGET_TAPE_SCHEMA_VERSION,
        "symbols": sorted(normalized["symbol"].unique()),
        "max_gross": parsed_max_gross,
        "records": records,
    }


def target_tape_from_payload(payload: Mapping[str, object]) -> pd.DataFrame:
    """Deserialize and validate a canonical target-tape JSON envelope."""
    if not isinstance(payload, Mapping):
        raise TypeError("target-tape payload must be a mapping")
    required_keys = {"schema_version", "symbols", "max_gross", "records"}
    if set(payload) != required_keys:
        raise ValueError(
            "target-tape payload keys must be exactly "
            + ", ".join(sorted(required_keys))
        )
    if payload["schema_version"] != TARGET_TAPE_SCHEMA_VERSION:
        raise ValueError("unsupported target-tape schema version")
    symbols = payload["symbols"]
    if not isinstance(symbols, list):
        raise TypeError("target-tape symbols must be a list")
    records = payload["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("target-tape records must be a non-empty list")
    if any(not isinstance(record, Mapping) for record in records):
        raise TypeError("every target-tape record must be a mapping")
    if any(set(record) != set(TARGET_TAPE_COLUMNS) for record in records):
        raise ValueError(
            "target-tape record keys must be exactly "
            + ", ".join(TARGET_TAPE_COLUMNS)
        )
    tape = pd.DataFrame(records, columns=TARGET_TAPE_COLUMNS)
    return validate_target_tape(
        tape,
        max_gross=_validated_max_gross(payload["max_gross"]),
        expected_symbols=symbols,
    )


__all__ = [
    "TARGET_TAPE_COLUMNS",
    "TARGET_TAPE_SCHEMA_VERSION",
    "target_tape_from_payload",
    "target_tape_to_payload",
    "target_tape_to_weights",
    "validate_target_tape",
    "weights_to_target_tape",
]
