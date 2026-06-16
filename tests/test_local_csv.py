from __future__ import annotations

import pandas as pd
import pytest

from quantcortex.data.local_csv import (
    LocalDataError,
    load_ohlcv_csv,
    load_price_matrix,
    sha256_file,
)


def test_load_price_matrix_filters_and_orders_symbols(tmp_path):
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "AAA": [10.0, None, 12.0],
            "BBB": [20.0, 21.0, 22.0],
        }
    ).to_csv(path, index=False)

    result = load_price_matrix(
        path,
        symbols=["BBB", "AAA"],
        start="2024-01-02",
        end="2024-01-03",
    )

    assert list(result.columns) == ["BBB", "AAA"]
    assert result.index.name == "date"
    assert result.loc["2024-01-02", "AAA"] == 10.0


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"AAA": [1.0]}), "date"),
        (pd.DataFrame({"date": [None], "AAA": [1.0]}), "empty date"),
        (
            pd.DataFrame(
                {"date": ["2024-01-01", "2024-01-01"], "AAA": [1.0, 2.0]}
            ),
            "duplicate dates",
        ),
        (pd.DataFrame({"date": ["2024-01-01"], "AAA": [0.0]}), "positive"),
    ],
)
def test_load_price_matrix_rejects_invalid_data(tmp_path, frame, message):
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(LocalDataError, match=message):
        load_price_matrix(path)


def test_load_price_matrix_rejects_missing_symbol(tmp_path):
    path = tmp_path / "prices.csv"
    pd.DataFrame({"date": ["2024-01-01"], "AAA": [1.0]}).to_csv(
        path, index=False
    )

    with pytest.raises(LocalDataError, match="BBB"):
        load_price_matrix(path, symbols=["AAA", "BBB"])


def test_load_price_matrix_rejects_duplicate_csv_headers(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("date,AAA,AAA\n2024-01-01,1.0,2.0\n", encoding="utf-8")

    with pytest.raises(LocalDataError, match="duplicate CSV columns"):
        load_price_matrix(path)


def test_load_ohlcv_csv_validates_canonical_shape(tmp_path):
    path = tmp_path / "ohlcv.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "open": [10.0, 11.0],
            "high": [12.0, 13.0],
            "low": [9.0, 10.0],
            "close": [11.0, 12.0],
            "adj_close": [11.0, 12.0],
            "volume": [100.0, 120.0],
        }
    ).to_csv(path, index=False)

    result = load_ohlcv_csv(path, start="2024-01-02")

    assert list(result.columns) == [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert result.index.tolist() == [pd.Timestamp("2024-01-02")]
    assert len(sha256_file(path)) == 64
