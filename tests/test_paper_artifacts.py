from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from scripts.run_paper_experiments import source_tree_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_ROOT = REPO_ROOT / "paper"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_paper_artifacts_match_manifest_and_generator():
    manifest = json.loads(
        (PAPER_ROOT / "results" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 3

    source = manifest["source"]
    assert source["raw_input_committed"] is False
    assert source["permission_basis"]
    assert len(source["input_sha256"]) == 64
    int(source["input_sha256"], 16)

    generator = manifest["generator"]
    script = REPO_ROOT / generator["path"]
    assert script.is_file()
    assert _sha256(script) == generator["script_sha256"]
    assert len(generator["git"]["base_commit"]) == 40
    int(generator["git"]["base_commit"], 16)
    assert isinstance(generator["git"]["worktree_clean"], bool)
    source_tree = generator["source_tree"]
    assert source_tree == source_tree_manifest(
        REPO_ROOT,
        list(source_tree["files"]),
    )

    design = manifest["design"]
    assert design["strategy_parameters"] == {
        "ir_lookback": 126,
        "max_position_weight": 0.6,
        "mom_gap": 21,
        "mom_lookback": 126,
        "regime_backend": "gmm",
        "regime_covariance_type": "full",
        "regime_feature_vol_lookback": 20,
        "regime_n_iter": 100,
        "regime_n_states": 3,
        "regime_reg_covar": 1e-05,
        "regime_seed": 42,
        "target_vix": 20.0,
        "top_n_groups": 2,
        "vix_cap": 1.0,
        "vix_floor": 0.3,
        "vix_proxy_lookback": 21,
    }
    assert design["bootstrap"]["sensitivity_block_lengths_sessions"] == [5, 21, 63]
    assert design["bootstrap"]["ablation_intervals"].startswith("joint across")
    assert design["input_validation"] == {
        "complete_rows_required": True,
        "max_forward_fill_sessions": 0,
    }
    assert design["warmup"] == {
        "available_sessions": 503,
        "required_sessions": 274,
    }
    assert design["signal_fallback"].startswith(
        "hold cash when mature selected-group residual scores"
    )
    assert design["return_decomposition"]["method"] == (
        "exact daily arithmetic identity"
    )
    assert design["bootstrap"]["interval"] == (
        "two-sided unstudentized percentile interval"
    )
    assert design["primary_engine"] == "event_driven"
    assert "pseudo-shares" in design["primary_engine_semantics"]

    generated_values = (
        PAPER_ROOT / "results" / "generated_values.tex"
    ).read_text(encoding="ascii")
    assert source["input_sha256"] in generated_values
    assert generator["source_tree"]["sha256"] in generated_values
    assert "\\newcommand{\\PaperWarmupSessions}{503}" in generated_values
    assert "\\newcommand{\\PaperRequiredWarmupSessions}{274}" in generated_values
    assert "\\newcommand{\\PaperBootstrapReplications}{5,000}" in generated_values
    assert "\\newcommand{\\PaperGrossActiveFullLower}" in generated_values
    assert "\\newcommand{\\PaperGrossTradedNotional}" in generated_values
    assert "\\newcommand{\\PaperDecompositionRows}" in generated_values

    artifacts = manifest["artifacts"]
    assert artifacts
    assert "results/ablation_uncertainty.csv" in artifacts
    assert "results/return_decomposition.csv" in artifacts
    assert "results/protocol_switches.csv" in artifacts
    assert "figures/bootstrap_robustness.pdf" in artifacts
    assert "figures/return_attribution_and_protocol_switches.pdf" in artifacts
    for relative_path, expected_digest in artifacts.items():
        artifact = PAPER_ROOT / relative_path
        assert artifact.is_file(), relative_path
        assert _sha256(artifact) == expected_digest, relative_path

    with (PAPER_ROOT / "results" / "protocol_switches.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        protocols = {row["protocol"]: row for row in csv.DictReader(handle)}
    assert protocols["audited"]["diagnostic_class"] == "reference"
    assert protocols["zero_return_cash"]["diagnostic_class"] == (
        "economic_counterfactual"
    )
    assert protocols["zero_modeled_costs"]["diagnostic_class"] == (
        "economic_counterfactual"
    )
    assert protocols["invalid_same_close"]["diagnostic_class"] == (
        "causally_invalid"
    )
    assert protocols["invalid_same_close"]["causally_valid"] == "False"

    with (PAPER_ROOT / "results" / "ablation_uncertainty.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        ablation_intervals = list(csv.DictReader(handle))
    assert {row["variant"] for row in ablation_intervals} == {
        "full",
        "no_regime",
        "no_vol_scaler",
        "signal_only",
    }
    assert all(float(row["ci_95_upper"]) < 0.0 for row in ablation_intervals)
    full_interval = next(
        row for row in ablation_intervals if row["variant"] == "full"
    )
    with (PAPER_ROOT / "results" / "bootstrap_sensitivity.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        gross_primary = next(
            row
            for row in csv.DictReader(handle)
            if row["comparison"]
            == "strategy_gross_minus_exposure_matched_equal_weight"
            and row["block_length"] == "21"
        )
    for field in ("annualized_mean", "ci_95_lower", "ci_95_upper"):
        assert full_interval[field] == gross_primary[field]


def test_paper_source_and_reviewed_pdf_are_published():
    main = (PAPER_ROOT / "main.tex").read_text(encoding="utf-8")

    assert "\\usepackage[preprint]{neurips_2026}" in main
    assert "\\usepackage{orcidlink}" in main
    assert "\\input{results/generated_values}" in main
    assert "Kevin Lee\\,\\orcidlink{0009-0004-0388-9260}" in main
    assert "University of California, Los Angeles" in main
    assert "\\url{https://orcid.org/0009-0004-0388-9260}" not in main
    assert "\\input{checklist}" in main
    assert "\\PaperInputDigest" in main
    assert "{bootstrap_robustness.pdf}" in main

    pdf = PAPER_ROOT / "quantcortex_audit_neurips2026.pdf"
    assert pdf.is_file()
    assert pdf.stat().st_size > 100_000
    assert pdf.read_bytes().startswith(b"%PDF-")
    digest, file_name = (
        PAPER_ROOT / "quantcortex_audit_neurips2026.sha256"
    ).read_text(encoding="ascii").split()
    assert file_name == pdf.name
    assert digest == _sha256(pdf)

    source_manifest = (
        PAPER_ROOT / "quantcortex_audit_neurips2026.sources.sha256"
    )
    source_entries = {}
    for line in source_manifest.read_text(encoding="ascii").splitlines():
        expected_digest, relative_path = line.split(maxsplit=1)
        source_entries[relative_path] = expected_digest
    assert {
        "main.tex",
        "checklist.tex",
        "references.bib",
        "neurips_2026.sty",
        "results/generated_values.tex",
        "figures/accounting_summary.pdf",
        "figures/audit_protocol.pdf",
        "figures/bootstrap_robustness.pdf",
        "figures/engine_comparison.pdf",
        "figures/return_attribution_and_protocol_switches.pdf",
        "figures/sensitivity_and_ablation.pdf",
    } == set(source_entries)
    for relative_path, expected_digest in source_entries.items():
        assert _sha256(PAPER_ROOT / relative_path) == expected_digest

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "paper/quantcortex_audit_neurips2026.pdf" in readme
    assert "paper/figures/return_attribution_and_protocol_switches.png" in readme
    assert "paper/figures/sensitivity_and_ablation.png" in readme
    assert "docs/architecture.md" in readme


def test_paper_body_respects_neurips_nine_page_limit():
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        import pytest

        pytest.skip("pdftotext is required to inspect PDF page boundaries")

    pdf = PAPER_ROOT / "quantcortex_audit_neurips2026.pdf"

    def page_text(first: int, last: int) -> str:
        return subprocess.run(
            [pdftotext, "-f", str(first), "-l", str(last), "-layout", pdf, "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    first_nine = page_text(1, 9)
    page_ten = page_text(10, 10)
    references_pattern = re.compile(r"^\s*References\s*$", re.MULTILINE)
    if references_pattern.search(first_nine):
        return

    substantive_lines = [
        line.strip()
        for line in page_ten.replace("\f", "").splitlines()
        if line.strip() and not line.strip().isdigit()
    ]
    assert substantive_lines
    assert substantive_lines[0] == "References"


def test_paper_citations_are_defined_used_and_unique():
    main = (PAPER_ROOT / "main.tex").read_text(encoding="utf-8")
    bibliography = (PAPER_ROOT / "references.bib").read_text(encoding="utf-8")

    cited_keys = {
        key.strip()
        for group in re.findall(r"\\cite\w*\{([^}]*)\}", main)
        for key in group.split(",")
    }
    bibliography_keys = set(re.findall(r"@\w+\{([^,]+),", bibliography))

    assert cited_keys == bibliography_keys
    assert "{L{\\'o}pez de Prado}, Marcos" in bibliography
    assert "{d'Alch{\\'e}-Buc}, Florence" in bibliography
    assert "The Statistics of {Sharpe} Ratios" in bibliography
    assert "The Deflated {Sharpe} Ratio" in bibliography

    dois = [
        value.lower()
        for value in re.findall(r"doi\s*=\s*\{([^}]+)\}", bibliography)
    ]
    eprints = re.findall(r"eprint\s*=\s*\{([^}]+)\}", bibliography)
    assert len(dois) == len(set(dois))
    assert len(eprints) == len(set(eprints))

    for eprint in eprints:
        assert f"https://arxiv.org/abs/{eprint}" in bibliography
