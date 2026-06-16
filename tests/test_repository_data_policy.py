from __future__ import annotations

import hashlib
import json
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


def test_redistributed_market_data_is_absent():
    present = [path for path in FORBIDDEN_MARKET_DATA if (REPO_ROOT / path).exists()]
    assert not present, f"redistributed market-data snapshots reappeared: {present}"


def test_published_performance_charts_match_manifest_and_readme():
    image_dir = REPO_ROOT / "docs" / "img"
    manifest_path = image_dir / "performance_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

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
