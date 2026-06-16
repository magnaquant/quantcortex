from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_ARTIFACTS = [
    "data/sample/rotation_prices.csv",
    "quantcortex/data/sample/rotation_prices.csv",
    "docs/img/equity_vs_benchmarks.png",
    "docs/img/drawdown.png",
    "docs/img/rolling_sharpe.png",
]


def test_redistributed_market_data_and_results_are_absent():
    present = [path for path in FORBIDDEN_ARTIFACTS if (REPO_ROOT / path).exists()]
    assert not present, f"redistributed market-data artifacts reappeared: {present}"


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
