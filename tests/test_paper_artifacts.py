from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from scripts.run_paper_experiments import _json_sha256, source_tree_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_ROOT = REPO_ROOT / "paper"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_paper_artifacts_match_manifest_and_generator():
    manifest = json.loads(
        (PAPER_ROOT / "results" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 5

    source = manifest["source"]
    assert source["raw_input_committed"] is False
    assert source["permission_basis"]
    assert len(source["input_sha256"]) == 64
    int(source["input_sha256"], 16)

    generator = manifest["generator"]
    script = REPO_ROOT / generator["path"]
    assert script.is_file()
    assert _sha256(script) == generator["script_sha256"]
    source_commit = generator["git"]["source_commit"]
    assert len(source_commit) == 40
    int(source_commit, 16)
    assert generator["git"]["worktree_clean_at_start"] is True
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            source_commit,
            "--",
            "quantcortex",
            "scripts/release_paper_artifacts.sh",
            "scripts/run_paper_experiments.py",
            "schemas",
            "pyproject.toml",
            "poetry.lock",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    dependency_lock = generator["dependency_lock"]
    assert dependency_lock["path"] == "poetry.lock"
    assert dependency_lock["sha256"] == _sha256(REPO_ROOT / "poetry.lock")
    assert generator["threadpools"]
    source_tree = generator["source_tree"]
    assert "scripts/release_paper_artifacts.sh" in source_tree["files"]
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
    assert manifest["configuration_sha256"] == _json_sha256(design)
    assert design["universe"]["group_display_names"] == {
        "growth": "growth equities",
        "real_assets": "gold and nominal duration",
        "defensive": "broad and dividend equities",
    }
    assert design["comparators"]["realized_exposure_attribution_control"].endswith(
        "ex-post and gross of comparator costs"
    )
    assert "next-bar execution" in design["comparators"][
        "target_exposure_costed_comparator"
    ]

    contract_metadata = manifest["evaluation_contract"]
    contract_path = PAPER_ROOT / contract_metadata["path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract_metadata["schema_version"] == 1
    assert contract_metadata["canonical_sha256"] == _json_sha256(contract)
    assert contract["comparators"]["realized_exposure_attribution_control"][
        "implementable"
    ] is False
    assert contract["comparators"]["target_exposure_costed_comparator"][
        "costed"
    ] is True
    assert contract["overlays"]["may_reduce_risky_exposure"] is True
    assert "block automatic retry" in contract["order_state"][
        "uncertain_submission"
    ]

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
    assert "\\newcommand{\\PaperCostedComparatorCAGR}" in generated_values
    assert "\\newcommand{\\PaperCostedActiveSharpeLower}" in generated_values
    assert "\\newcommand{\\PaperDecompositionRows}" in generated_values

    artifacts = manifest["artifacts"]
    assert artifacts
    assert "results/ablation_uncertainty.csv" in artifacts
    assert "results/comparator_diagnostics.csv" in artifacts
    assert "results/evaluation_contract.json" in artifacts
    assert "results/target_tape_hashes.json" in artifacts
    assert "results/return_decomposition.csv" in artifacts
    assert "results/sharpe_uncertainty.csv" in artifacts
    assert "results/protocol_switches.csv" in artifacts
    assert "figures/bootstrap_robustness.pdf" in artifacts
    assert "figures/return_attribution_and_protocol_switches.pdf" in artifacts
    for relative_path, expected_digest in artifacts.items():
        artifact = PAPER_ROOT / relative_path
        assert artifact.is_file(), relative_path
        assert _sha256(artifact) == expected_digest, relative_path

    target_tape_path = PAPER_ROOT / manifest["decision_streams"]["path"]
    target_tapes = json.loads(target_tape_path.read_text(encoding="utf-8"))
    assert target_tapes == manifest["decision_streams"]["variants"]
    assert set(target_tapes) == {"full", "no_regime", "no_vol_scaler", "signal_only"}
    for metadata in target_tapes.values():
        assert metadata["symbols"] == sorted(metadata["symbols"])
        assert metadata["record_count"] == (
            metadata["decision_count"] * len(metadata["symbols"])
        )
        assert len(metadata["canonical_payload_sha256"]) == 64
        int(metadata["canonical_payload_sha256"], 16)

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

    with (PAPER_ROOT / "results" / "sharpe_uncertainty.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        sharpe_rows = list(csv.DictReader(handle))
    primary_costed = next(
        row
        for row in sharpe_rows
        if row["series"]
        == "strategy_net_minus_target_exposure_costed_comparator"
        and row["block_length"] == "21"
    )
    assert float(primary_costed["sample_sharpe"]) < 0.0
    assert float(primary_costed["ci_95_upper"]) < 0.0


def test_paper_source_and_reviewed_pdf_are_published():
    main = (PAPER_ROOT / "main.tex").read_text(encoding="utf-8")
    release_script = (REPO_ROOT / "scripts" / "release_paper_artifacts.sh").read_text(
        encoding="utf-8"
    )

    assert "\\usepackage[preprint]{neurips_2026}" in main
    assert "\\usepackage{orcidlink}" in main
    assert "\\input{results/generated_values}" in main
    assert "\\input{expansion/results/generated_values}" in main
    assert "Executable Evaluation Contracts" in main
    assert "Kevin Lee\\,\\orcidlink{0009-0004-0388-9260}" in main
    assert "University of California, Los Angeles" in main
    assert "\\url{https://orcid.org/0009-0004-0388-9260}" not in main
    assert "\\input{checklist}" in main
    assert "\\PaperInputDigest" in main
    assert "{bootstrap_robustness.pdf}" in main
    assert "{contract_effects_return.pdf}" in main
    assert "{baseline_performance.pdf}" in main
    assert "target-exposure comparator" in main
    assert "actual_input_digest" in release_script
    assert "expected_input_digest" in release_script
    assert "ls-files --error-unmatch" in release_script
    assert "reviewed_generated_at" in release_script
    assert "reviewed_source_commit" in release_script
    assert "release_source_paths" in release_script
    assert "scripts/release_paper_artifacts.sh" in release_script
    assert "QUANTCORTEX_GENERATED_AT is required for changed release source" in (
        release_script
    )
    assert '--data-provider "${provider}"' in release_script

    anonymous_source = (PAPER_ROOT / "anonymous.tex").read_text(encoding="ascii")
    assert "\\def\\quantcortexanonymous{1}" in anonymous_source

    for stem in ("quantcortex_audit_neurips2026", "quantcortex_audit_anonymous"):
        pdf = PAPER_ROOT / f"{stem}.pdf"
        assert pdf.is_file()
        assert pdf.stat().st_size > 100_000
        assert pdf.read_bytes().startswith(b"%PDF-")
        digest, file_name = (PAPER_ROOT / f"{stem}.sha256").read_text(
            encoding="ascii"
        ).split()
        assert file_name == pdf.name
        assert digest == _sha256(pdf)

    pdftotext = shutil.which("pdftotext")
    if pdftotext is not None:
        anonymous_text = subprocess.run(
            [
                pdftotext,
                str(PAPER_ROOT / "quantcortex_audit_anonymous.pdf"),
                "-",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for identifying_text in (
            "Kevin Lee",
            "kevinlee69720",
            "University of California",
            "micro1",
            "github.com/magnaquant",
            "Submitted to",
        ):
            assert identifying_text not in anonymous_text

    build_manifest = json.loads(
        (PAPER_ROOT / "build_manifest.json").read_text(encoding="ascii")
    )
    assert build_manifest["tectonic_version"] == "0.16.9"
    assert build_manifest["tectonic_bundle"] == {
        "name": "default_bundle_v33.tar",
        "sha256": "6ffe055852f8faf66c0acbe1a7fb27f87b869a90bad1204f3bf4d9683f597c7c",
    }
    assert build_manifest["source_commit"] == json.loads(
        (PAPER_ROOT / "results" / "manifest.json").read_text(encoding="utf-8")
    )["generator"]["git"]["source_commit"]
    assert build_manifest["pdf"]["sha256"] == _sha256(
        PAPER_ROOT / build_manifest["pdf"]["path"]
    )
    assert build_manifest["anonymous_pdf"]["sha256"] == _sha256(
        PAPER_ROOT / build_manifest["anonymous_pdf"]["path"]
    )

    source_manifest = (
        PAPER_ROOT / "quantcortex_audit_neurips2026.sources.sha256"
    )
    source_entries = {}
    for line in source_manifest.read_text(encoding="ascii").splitlines():
        expected_digest, relative_path = line.split(maxsplit=1)
        source_entries[relative_path] = expected_digest
    assert {
        "main.tex",
        "anonymous.tex",
        "checklist.tex",
        "references.bib",
        "neurips_2026.sty",
        "preregistration.md",
        "results/generated_values.tex",
        "results/manifest.json",
        "expansion/protocol.json",
        "expansion/results/generated_values.tex",
        "expansion/results/manifest.json",
        "figures/accounting_summary.pdf",
        "figures/audit_protocol.pdf",
        "figures/bootstrap_robustness.pdf",
        "figures/engine_comparison.pdf",
        "figures/return_attribution_and_protocol_switches.pdf",
        "figures/sensitivity_and_ablation.pdf",
        "expansion/figures/baseline_performance.pdf",
        "expansion/figures/contract_effects_return.pdf",
        "expansion/figures/contract_effects_sharpe.pdf",
        "expansion/figures/engine_conformance.pdf",
        "expansion/figures/learned_seed_sensitivity.pdf",
    } == set(source_entries)
    for relative_path, expected_digest in source_entries.items():
        assert _sha256(PAPER_ROOT / relative_path) == expected_digest

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "paper/quantcortex_audit_neurips2026.pdf" in readme
    assert "paper/quantcortex_audit_anonymous.pdf" in readme
    assert "paper/figures/return_attribution_and_protocol_switches.png" in readme
    assert "paper/figures/sensitivity_and_ablation.png" in readme
    assert "docs/architecture.md" in readme


def test_paper_body_respects_neurips_nine_page_limit():
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        import pytest

        pytest.skip("pdftotext is required to inspect PDF page boundaries")

    def page_text(pdf: Path, first: int, last: int) -> str:
        return subprocess.run(
            [
                pdftotext,
                "-f",
                str(first),
                "-l",
                str(last),
                "-layout",
                pdf,
                "-",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    for pdf_name in (
        "quantcortex_audit_neurips2026.pdf",
        "quantcortex_audit_anonymous.pdf",
    ):
        pdf = PAPER_ROOT / pdf_name
        first_nine = page_text(pdf, 1, 9)
        page_ten = page_text(pdf, 10, 10)
        references_pattern = re.compile(
            r"^\s*(?:\d+\s+)?References\s*$",
            re.MULTILINE,
        )
        if references_pattern.search(first_nine):
            continue

        substantive_lines = [
            re.sub(r"^\d+\s+", "", line.strip())
            for line in page_ten.replace("\f", "").splitlines()
            if line.strip() and not line.strip().isdigit()
        ]
        assert substantive_lines, pdf_name
        assert substantive_lines[0] == "References", pdf_name


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
