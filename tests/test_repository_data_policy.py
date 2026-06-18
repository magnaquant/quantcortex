from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_MARKET_DATA = [
    "data/sample/rotation_prices.csv",
    "quantcortex/data/sample/rotation_prices.csv",
]
PUBLISHED_CHARTS = {
    "allocation_and_exposure.png",
    "drawdown.png",
    "equity_vs_benchmarks.png",
    "monthly_returns.png",
    "performance_attribution.png",
    "report_overview.png",
    "return_distribution.png",
    "rolling_risk.png",
    "rolling_sharpe.png",
    "turnover_and_costs.png",
}
DOCUMENTATION_FILES = [
    "README.md",
    "PERFORMANCE.md",
    "CONTRIBUTING.md",
    "AGENTS.md",
    "CLAUDE.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "local_data/README.md",
    "docs/architecture.md",
    "docs/production-readiness.md",
    "paper/README.md",
]


def test_redistributed_market_data_is_absent():
    present = [path for path in FORBIDDEN_MARKET_DATA if (REPO_ROOT / path).exists()]
    assert not present, f"redistributed market-data snapshots reappeared: {present}"

    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert [path for path in tracked if path.startswith("local_data/")] == [
        "local_data/README.md"
    ]

    raw_columns = {"date", "QQQ", "VGT", "GLD", "TLT", "SPY", "VIG", "SHV"}
    leaked = []
    for relative_path in tracked:
        if not relative_path.endswith(".csv"):
            continue
        with (REPO_ROOT / relative_path).open(encoding="utf-8", newline="") as handle:
            header = set(next(csv.reader(handle), []))
        if raw_columns <= header:
            leaked.append(relative_path)
    assert not leaked, f"tracked raw paper price matrices detected: {leaked}"


def test_published_performance_charts_match_manifest_and_readme():
    image_dir = REPO_ROOT / "docs" / "img"
    manifest_path = image_dir / "performance_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 4

    source = manifest["source"]
    assert source["permission_basis"]
    assert source["raw_input_committed"] is False
    input_digest = source["input_sha256"]
    assert len(input_digest) == 64
    int(input_digest, 16)

    artifacts = manifest["artifacts"]
    assert set(artifacts) == PUBLISHED_CHARTS
    assert {path.name for path in image_dir.glob("*.png")} == PUBLISHED_CHARTS
    for name, expected_digest in artifacts.items():
        artifact = image_dir / name
        actual_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert actual_digest == expected_digest, name

    source_tree = manifest["generator"]["source_tree"]
    assert source_tree["file_count"] == len(source_tree["files"])
    aggregate_digest = hashlib.sha256()
    for relative, expected_digest in source_tree["files"].items():
        source_path = REPO_ROOT / relative
        actual_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        assert actual_digest == expected_digest, relative
        aggregate_digest.update(relative.encode("utf-8"))
        aggregate_digest.update(b"\0")
        aggregate_digest.update(bytes.fromhex(expected_digest))
    assert aggregate_digest.hexdigest() == source_tree["sha256"]

    git_metadata = manifest["generator"]["git"]
    source_commit = git_metadata["source_commit"]
    assert len(source_commit) == 40
    int(source_commit, 16)
    assert git_metadata["worktree_clean_at_start"] is True

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert input_digest in readme
    assert "docs/img/performance_manifest.json" in readme
    for name in PUBLISHED_CHARTS:
        assert f"docs/img/{name}" in readme


def test_research_notebooks_have_no_committed_execution_state():
    for path in sorted((REPO_ROOT / "research").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            assert cell.get("execution_count") is None, f"{path}:{index} is executed"
            assert not cell.get("outputs"), f"{path}:{index} has committed outputs"
            assert "execution" not in cell.get("metadata", {}), (
                f"{path}:{index} has execution timestamps"
            )


def test_published_documentation_headline_metrics_match_paper_results():
    with (REPO_ROOT / "paper" / "results" / "accounting.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        accounting = {
            row["series"]: row for row in csv.DictReader(handle)
        }
    with (REPO_ROOT / "paper" / "results" / "ablation.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        ablation = {row["variant"]: row for row in csv.DictReader(handle)}
    with (
        REPO_ROOT / "paper" / "results" / "bootstrap_sensitivity.csv"
    ).open(encoding="utf-8", newline="") as handle:
        sensitivity = list(csv.DictReader(handle))
    with (
        REPO_ROOT / "paper" / "results" / "return_decomposition.csv"
    ).open(encoding="utf-8", newline="") as handle:
        decomposition = list(csv.DictReader(handle))

    net = accounting["strategy_net_shv"]
    gross = accounting["strategy_gross_shv"]
    matched = accounting["exposure_matched_equal_weight"]
    full = ablation["full"]
    gross_primary = next(
        row
        for row in sensitivity
        if row["comparison"]
        == "strategy_gross_minus_exposure_matched_equal_weight"
        and row["block_length"] == "21"
    )
    primary_components = {
        row["component"]: row
        for row in decomposition
        if row["block_length"] == "21"
    }

    expected_strings = {
        f"{float(net['nominal_cagr']):+.2%}",
        f"{float(gross['nominal_cagr']):+.2%}",
        f"{float(net['cash_excess_sharpe']):+.2f}",
        f"{float(net['max_drawdown']):+.2%}",
        f"{float(full['annualized_one_way_turnover']):.2f}x",
        f"{float(full['annualized_gross_traded_notional']):.2f}x",
        f"{float(full['mean_gross_exposure']):.2%}",
        f"{float(full['fully_cash_fraction']):.2%}",
        f"{float(matched['cash_excess_sharpe']):+.2f}",
        f"{float(gross_primary['annualized_mean']):.2%}",
        (
            f"[{float(gross_primary['ci_95_lower']):.2%}, "
            f"{float(gross_primary['ci_95_upper']):.2%}]"
        ),
        f"{float(primary_components['net_excess_over_cash']['annualized_mean']):+.2%}",
        f"{float(primary_components['active_risky_allocation']['annualized_mean']):+.2%}",
        f"{float(primary_components['dynamic_exposure_timing']['annualized_mean']):+.2%}",
        f"{float(primary_components['passive_risky_exposure']['annualized_mean']):+.2%}",
        f"{float(primary_components['implementation_cost']['annualized_mean']):+.2%}",
    }
    for relative_path in ("README.md", "PERFORMANCE.md"):
        document = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        normalized = " ".join(document.split())
        missing = sorted(value for value in expected_strings if value not in normalized)
        assert not missing, f"{relative_path} headline evidence drifted: {missing}"

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "paper/figures/return_attribution_and_protocol_switches.png" in readme


def test_documentation_local_links_resolve():
    missing = []
    for relative_path in DOCUMENTATION_FILES:
        document = REPO_ROOT / relative_path
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"!?\[[^]]*\]\(([^)]+)\)", text):
            target = target.strip().split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{relative_path}: {target}")
    assert not missing, f"broken local documentation links: {missing}"
