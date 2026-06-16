"""Verify that executed research notebooks actually rendered their figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_FIGURES = {
    "01_data_quality.ipynb": 1,
    "02_factor_research.ipynb": 1,
    "03_portfolio_construction.ipynb": 1,
    "04_backtest_analysis.ipynb": 1,
}


def rendered_figure_count(path: Path) -> int:
    """Count rich outputs containing an image MIME type."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return sum(
        any(str(mime).startswith("image/") for mime in output.get("data", {}))
        for cell in notebook.get("cells", [])
        for output in cell.get("outputs", [])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    for filename, minimum in EXPECTED_FIGURES.items():
        path = args.output_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"executed notebook is missing: {path}")
        count = rendered_figure_count(path)
        if count < minimum:
            raise RuntimeError(
                f"{filename} rendered {count} figures; expected at least {minimum}"
            )
        print(f"{filename}: {count} rendered figure(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
