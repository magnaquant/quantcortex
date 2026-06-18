from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from quantcortex.research.expansion import (
    FROZEN_PROTOCOL_COMMIT,
    FROZEN_PROTOCOL_SHA256,
)
from scripts.run_expansion_experiments import _source_tree_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPANSION_ROOT = REPO_ROOT / "paper" / "expansion"
RESULT_ROOT = EXPANSION_ROOT / "results"

EXPECTED_ROWS = {
    "contract_effects.csv": 120,
    "engine_conformance.csv": 16,
    "family_summary.csv": 48,
    "learned_fit_diagnostics.csv": 10,
    "rank_reversals.csv": 48,
    "seed_variant_metrics.csv": 96,
}
EXPECTED_ARTIFACTS = {
    *(f"results/{name}" for name in EXPECTED_ROWS),
    "results/data_provenance.json",
    "results/generated_values.tex",
    "results/target_tape_hashes.json",
    *(
        f"figures/{stem}.{suffix}"
        for stem in (
            "baseline_performance",
            "contract_effects_return",
            "contract_effects_sharpe",
            "engine_conformance",
            "learned_seed_sensitivity",
        )
        for suffix in ("pdf", "png")
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aware_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None
    return parsed


def test_expansion_manifest_binds_clean_source_inputs_and_artifacts():
    manifest = json.loads((RESULT_ROOT / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    generated_at = _aware_timestamp(manifest["generated_at"])
    protocol = manifest["protocol"]
    assert protocol == {
        "freeze_commit": FROZEN_PROTOCOL_COMMIT,
        "path": "paper/expansion/protocol.json",
        "sha256": FROZEN_PROTOCOL_SHA256,
        "status": "repository_frozen_prospective_not_externally_registered",
    }
    assert _sha256(EXPANSION_ROOT / "protocol.json") == FROZEN_PROTOCOL_SHA256

    git = manifest["git"]
    source_commit = git["source_commit"]
    assert len(source_commit) == 40
    int(source_commit, 16)
    assert git["tracked_worktree_clean_at_start"] is True
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=REPO_ROOT,
        check=True,
    )

    source_tree = manifest["source_tree"]
    assert source_tree["file_count"] == len(source_tree["files"])
    assert source_tree == _source_tree_manifest(REPO_ROOT)
    subprocess.run(
        ["git", "diff", "--quiet", source_commit, "--", *source_tree["files"]],
        cwd=REPO_ROOT,
        check=True,
    )
    for required in (
        "paper/preregistration.md",
        "scripts/fetch_expansion_data.py",
        "scripts/release_expansion_artifacts.sh",
        "scripts/run_expansion_experiments.py",
        "schemas/canonical_target_tape.schema.json",
    ):
        assert required in source_tree["files"]

    environment = manifest["environment"]
    assert environment["threadpoolctl"]
    assert isinstance(environment["threadpools"], list)

    artifacts = manifest["artifacts"]
    assert set(artifacts) == EXPECTED_ARTIFACTS
    for relative_path, expected_digest in artifacts.items():
        artifact = EXPANSION_ROOT / relative_path
        assert artifact.is_file(), relative_path
        assert _sha256(artifact) == expected_digest, relative_path
        if artifact.suffix == ".pdf":
            assert artifact.read_bytes().startswith(b"%PDF-")
        if artifact.suffix == ".png":
            assert artifact.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    data = manifest["data"]
    assert json.loads(
        (RESULT_ROOT / "data_provenance.json").read_text(encoding="utf-8")
    ) == data
    assert len(data) == 2
    for panel in data:
        retrieved_at = _aware_timestamp(panel["retrieved_at"])
        assert retrieved_at <= generated_at
        assert panel["protocol_sha256"] == FROZEN_PROTOCOL_SHA256
        assert panel["provider_terms_independently_verified"] is False
        assert panel["raw_data_committed"] is False
        assert panel["dropped_incomplete_rows"] == 0
        assert panel["complete_rows"] == 3018
        assert panel["pre_evaluation_sessions"] == 1007
        assert panel["evaluation_sessions"] == 2011
        assert set(panel["missing_by_symbol"].values()) == {0}
        assert len(panel["input_sha256"]) == 64
        int(panel["input_sha256"], 16)

    assert manifest["counts"] == {
        "contract_effect_rows": 120,
        "panels": 2,
        "seed_variant_rows": 96,
        "strategy_families": 4,
        "target_tapes": 16,
    }


def test_expansion_tables_and_target_hashes_are_complete_and_finite():
    for name, expected_rows in EXPECTED_ROWS.items():
        frame = pd.read_csv(RESULT_ROOT / name)
        assert len(frame) == expected_rows, name
        assert not frame.isna().any(axis=None), name
        numeric = frame.select_dtypes(include=[np.number])
        assert np.isfinite(numeric.to_numpy()).all(), name

    summary = pd.read_csv(RESULT_ROOT / "family_summary.csv")
    assert set(summary["panel"]) == {"us_sector_etfs", "country_equity_etfs"}
    assert set(summary["strategy"]) == {
        "cross_sectional_momentum",
        "learned_gbrt",
        "short_term_reversal",
        "ts_momentum",
    }
    assert set(summary["variant"]) == {
        "baseline",
        "costed_comparator",
        "same_close",
        "vectorized",
        "zero_cash",
        "zero_cost",
    }

    effects = pd.read_csv(RESULT_ROOT / "contract_effects.csv")
    assert set(effects["block_length"]) == {5, 21, 63}
    assert set(effects["replications"]) == {5000}
    assert (effects["annualized_mean_ci_95_lower"] <= effects["annualized_mean_ci_95_upper"]).all()
    assert (effects["sharpe_ci_95_lower"] <= effects["sharpe_ci_95_upper"]).all()

    targets = json.loads(
        (RESULT_ROOT / "target_tape_hashes.json").read_text(encoding="utf-8")
    )
    assert len(targets) == 16
    assert len({row["sha256"] for row in targets}) == 16
    for row in targets:
        assert row["decision_count"] == 96
        assert row["symbols"] == sorted(row["symbols"])
        assert row["record_count"] == row["decision_count"] * len(row["symbols"])
        assert len(row["sha256"]) == 64
        int(row["sha256"], 16)

    generated = (RESULT_ROOT / "generated_values.tex").read_text(encoding="ascii")
    for expected in (
        r"\newcommand{\ExpansionPanelCount}{2}",
        r"\newcommand{\ExpansionFamilyCount}{4}",
        r"\newcommand{\ExpansionFamilyPanelCount}{8}",
        r"\newcommand{\ExpansionBootstrapReplications}{5,000}",
        rf"\newcommand{{\ExpansionProtocolDigest}}{{{FROZEN_PROTOCOL_SHA256}}}",
        r"\newcommand{\ExpansionSectorInputDigest}{c5947c4ca6ad6d21ad834c8f344dcdd07acc59ba411e3dcb2202a2413642b2f9}",
        r"\newcommand{\ExpansionCountryInputDigest}{870fa926b5080378c65bd629b9959bd73b16391dded5720c894c0f35558132a9}",
        r"\newcommand{\ExpansionBaselineRows}",
        r"\newcommand{\ExpansionEffectRows}",
    ):
        assert expected in generated
