"""Build deterministic, test-only market-data fixtures for notebook CI."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "NVDA",
    "JPM",
    "XOM",
    "PG",
    "KO",
    "GOOGL",
    "JNJ",
    "UNH",
    "HD",
    "QQQ",
    "VGT",
    "GLD",
    "TLT",
    "SPY",
    "VIG",
]


def build(output_dir: Path) -> tuple[Path, Path]:
    """Write a wide price matrix and one valid OHLCV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20240615)
    # The notebooks request data from 2018 onward. This leaves roughly 590
    # sessions after that cut, enough for the longest 526-session strategy
    # initialization while keeping the CI smoke fast.
    dates = pd.bdate_range("2017-01-03", periods=850, name="date")

    market = rng.normal(0.00025, 0.008, size=(len(dates), 1))
    loadings = np.linspace(0.7, 1.3, len(SYMBOLS)).reshape(1, -1)
    idiosyncratic = rng.normal(0.0, 0.006, size=(len(dates), len(SYMBOLS)))
    returns = market * loadings + idiosyncratic
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=SYMBOLS,
    )

    prices_path = output_dir / "prices.csv"
    prices.to_csv(prices_path, index_label="date")

    close = prices["AAPL"]
    open_ = close * np.exp(rng.normal(0.0, 0.002, size=len(close)))
    intraday = np.abs(rng.normal(0.003, 0.001, size=len(close)))
    high = np.maximum(open_, close) * (1.0 + intraday)
    low = np.minimum(open_, close) * (1.0 - intraday)
    ohlcv = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(500_000, 5_000_000, size=len(close)),
        },
        index=dates,
    )
    ohlcv_path = output_dir / "aapl_ohlcv.csv"
    ohlcv.to_csv(ohlcv_path, index_label="date")
    return prices_path, ohlcv_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    prices_path, ohlcv_path = build(args.output_dir)
    print(prices_path)
    print(ohlcv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
