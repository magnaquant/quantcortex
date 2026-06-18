#!/usr/bin/env python3
"""Fetch the frozen expansion panels into ignored local storage."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

YFINANCE_TERMS = "https://ranaroussi.github.io/yfinance/"
YAHOO_TERMS = "https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_tracked(repo_root: Path, path: Path) -> bool:
    relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"could not determine Git status for {relative}")
    return result.returncode == 0


def _adjusted_close(download: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if not isinstance(download, pd.DataFrame) or download.empty:
        raise ValueError("provider returned no observations")
    if isinstance(download.columns, pd.MultiIndex):
        if "Adj Close" not in download.columns.get_level_values(0):
            raise ValueError("provider response has no adjusted-close field")
        adjusted = download["Adj Close"].copy()
    else:
        if symbols != [str(download.columns.name)] or "Adj Close" not in download:
            raise ValueError("provider response has an unexpected column layout")
        adjusted = download[["Adj Close"]].rename(columns={"Adj Close": symbols[0]})
    adjusted.columns = [str(column) for column in adjusted.columns]
    missing = [symbol for symbol in symbols if symbol not in adjusted.columns]
    extra = [symbol for symbol in adjusted.columns if symbol not in symbols]
    if missing or extra:
        raise ValueError(f"provider symbol mismatch; missing={missing}, extra={extra}")
    adjusted = adjusted.loc[:, symbols].copy()
    adjusted.index = pd.to_datetime(adjusted.index, utc=True).tz_localize(None)
    adjusted.index.name = "date"
    if adjusted.index.hasnans or adjusted.index.has_duplicates:
        raise ValueError("provider returned invalid or duplicate dates")
    return adjusted.sort_index().apply(pd.to_numeric, errors="coerce")


def _validate_panel(
    adjusted: pd.DataFrame,
    *,
    evaluation_start: str,
    evaluation_end: str,
    minimum_pre_evaluation_sessions: int,
    minimum_mature_training_months: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    missing_by_symbol = {
        symbol: int(count)
        for symbol, count in adjusted.isna().sum().items()
    }
    complete = adjusted.dropna(how="any")
    values = complete.to_numpy(dtype=float)
    if complete.empty or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("complete panel must contain finite positive prices")
    before = complete.index < pd.Timestamp(evaluation_start)
    pre_evaluation_sessions = int(before.sum())
    if pre_evaluation_sessions < minimum_pre_evaluation_sessions:
        raise ValueError(
            f"panel has {pre_evaluation_sessions} pre-evaluation sessions; "
            f"requires {minimum_pre_evaluation_sessions}"
        )
    evaluation = complete.loc[evaluation_start:evaluation_end]
    if evaluation.empty:
        raise ValueError("panel has no evaluation observations")
    expected_months = pd.period_range(
        pd.Timestamp(evaluation_start).to_period("M"),
        pd.Timestamp(evaluation_end).to_period("M"),
        freq="M",
    )
    observed_months = evaluation.index.to_period("M").unique()
    missing_months = expected_months.difference(observed_months)
    if len(missing_months):
        raise ValueError(
            "panel is missing evaluation months: "
            + ", ".join(str(month) for month in missing_months)
        )
    mature_months = complete.index[before].to_period("M").nunique()
    if mature_months < minimum_mature_training_months:
        raise ValueError(
            f"panel has {mature_months} pre-evaluation months; "
            f"requires {minimum_mature_training_months}"
        )
    diagnostics = {
        "provider_rows": int(len(adjusted)),
        "complete_rows": int(len(complete)),
        "dropped_incomplete_rows": int(len(adjusted) - len(complete)),
        "missing_by_symbol": missing_by_symbol,
        "pre_evaluation_sessions": pre_evaluation_sessions,
        "pre_evaluation_months": int(mature_months),
        "evaluation_sessions": int(len(evaluation)),
        "first_date": complete.index[0].date().isoformat(),
        "last_date": complete.index[-1].date().isoformat(),
    }
    return complete, diagnostics


def _write_panel(path: Path, panel: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    panel.to_csv(temporary, index=True, float_format="%.17g")
    os.replace(temporary, path)


def fetch_panels(
    protocol_path: Path,
    output_dir: Path,
    *,
    retrieved_at: str,
) -> list[dict[str, object]]:
    """Fetch every frozen panel and return local provenance records."""
    import yfinance as yf

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != (
        "repository_frozen_prospective_not_externally_registered"
    ):
        raise ValueError("expansion protocol is not frozen")
    data = protocol["data"]
    request = data["provider_request"]
    cash_symbol = data["cash_proxy"]
    repo_root = protocol_path.resolve().parents[2]
    resolved_output = output_dir.expanduser().resolve()
    try:
        resolved_output.relative_to((repo_root / "local_data").resolve())
    except ValueError as exc:
        raise ValueError("output_dir must remain under ignored local_data/") from exc

    records = []
    for panel_name, risky_symbols in protocol["panels"].items():
        symbols = [*risky_symbols, cash_symbol]
        downloaded = yf.download(
            tickers=symbols,
            start=request["request_start"],
            end=request["request_end_exclusive"],
            auto_adjust=request["auto_adjust"],
            actions=request["actions"],
            repair=request["repair"],
            threads=request["threads"],
            progress=False,
            group_by="column",
            multi_level_index=True,
        )
        adjusted = _adjusted_close(downloaded, symbols)
        panel, diagnostics = _validate_panel(
            adjusted,
            evaluation_start=data["evaluation_start"],
            evaluation_end=data["evaluation_end"],
            minimum_pre_evaluation_sessions=data[
                "minimum_pre_evaluation_sessions"
            ],
            minimum_mature_training_months=data[
                "minimum_mature_training_months"
            ],
        )
        output_path = resolved_output / f"{panel_name}.csv"
        metadata_path = resolved_output / f"{panel_name}.metadata.json"
        if _is_tracked(repo_root, output_path) or _is_tracked(repo_root, metadata_path):
            raise ValueError("raw expansion data paths must not be tracked")
        _write_panel(output_path, panel)
        record = {
            "schema_version": 1,
            "panel": panel_name,
            "symbols": symbols,
            "provider": "Yahoo Finance via yfinance",
            "provider_terms_independently_verified": False,
            "retrieved_at": retrieved_at,
            "request": request,
            "protocol_path": protocol_path.relative_to(repo_root).as_posix(),
            "protocol_sha256": _protocol_digest(protocol_path),
            "yfinance_version": importlib.metadata.version("yfinance"),
            "terms_urls": [YFINANCE_TERMS, YAHOO_TERMS],
            "raw_data_committed": False,
            "local_file": output_path.relative_to(repo_root).as_posix(),
            "input_sha256": _sha256(output_path),
            **diagnostics,
        }
        metadata_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("paper/expansion/protocol.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("local_data/expansion"),
    )
    parser.add_argument(
        "--retrieved-at",
        default=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00",
            "Z",
        ),
    )
    args = parser.parse_args()
    records = fetch_panels(
        args.protocol.resolve(),
        args.output_dir,
        retrieved_at=args.retrieved_at,
    )
    for record in records:
        print(
            f"{record['panel']}: {record['complete_rows']} rows, "
            f"SHA-256 {record['input_sha256']}"
        )
    print("Raw observations remain local and untracked.")
    print("Provider terms and publication rights are not certified by this script.")


if __name__ == "__main__":
    main()
