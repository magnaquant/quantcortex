from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_ROOT = REPO_ROOT / "paper"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_paper_artifacts_match_manifest_and_generator():
    manifest = json.loads(
        (PAPER_ROOT / "results" / "manifest.json").read_text(encoding="utf-8")
    )

    source = manifest["source"]
    assert source["raw_input_committed"] is False
    assert source["permission_basis"]
    assert len(source["input_sha256"]) == 64
    int(source["input_sha256"], 16)

    generator = manifest["generator"]
    script = REPO_ROOT / generator["path"]
    assert script.is_file()
    assert _sha256(script) == generator["script_sha256"]
    assert len(generator["git_commit"]) == 40
    int(generator["git_commit"], 16)

    artifacts = manifest["artifacts"]
    assert artifacts
    for relative_path, expected_digest in artifacts.items():
        artifact = PAPER_ROOT / relative_path
        assert artifact.is_file(), relative_path
        assert _sha256(artifact) == expected_digest, relative_path


def test_paper_source_and_reviewed_pdf_are_published():
    main = (PAPER_ROOT / "main.tex").read_text(encoding="utf-8")
    manifest = json.loads(
        (PAPER_ROOT / "results" / "manifest.json").read_text(encoding="utf-8")
    )
    input_digest = manifest["source"]["input_sha256"]

    assert "\\usepackage[preprint]{neurips_2026}" in main
    assert "\\usepackage{orcidlink}" in main
    assert "Kevin Lee\\,\\orcidlink{0009-0004-0388-9260}" in main
    assert "\\url{https://orcid.org/0009-0004-0388-9260}" not in main
    assert "\\input{checklist}" in main
    assert input_digest in main

    pdf = PAPER_ROOT / "quantcortex_audit_neurips2026.pdf"
    assert pdf.is_file()
    assert pdf.stat().st_size > 100_000
    assert pdf.read_bytes().startswith(b"%PDF-")
    digest, file_name = (
        PAPER_ROOT / "quantcortex_audit_neurips2026.sha256"
    ).read_text(encoding="ascii").split()
    assert file_name == pdf.name
    assert digest == _sha256(pdf)

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "paper/quantcortex_audit_neurips2026.pdf" in readme
    assert "paper/figures/sensitivity_and_ablation.png" in readme
