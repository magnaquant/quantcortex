import numpy as np
import pandas as pd
import pytest

from quantcortex.alpha.factors.classical._cross_section import (
    rolling_cov,
    to_returns,
    validate_prices,
)
from quantcortex.alpha.factors.classical._fundamentals import (
    panel_axes,
    pit_panel,
    validate_fundamentals,
)
from quantcortex.alpha.factors.classical.low_vol import LowVolFactor
from quantcortex.alpha.factors.classical.momentum import MomentumFactor
from quantcortex.alpha.factors.classical.quality import QualityFactor
from quantcortex.alpha.factors.classical.value import ValueFactor
from quantcortex.alpha.factors.ml.neural_factor import NeuralFactor
from quantcortex.alpha.feature_engineering.alpha158 import Alpha158
from quantcortex.alpha.validation.alphalens_report import (
    _newey_west_tstat,
    compute_information_coefficient,
    quantile_returns,
)
from quantcortex.data.providers.base import canonical_fundamental_field
from quantcortex.data.providers.fmp_provider import FMPProvider


def _quarterly_fundamentals() -> pd.DataFrame:
    periods = pd.to_datetime(
        ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31"]
    )
    announcements = periods + pd.Timedelta(days=30)
    values = {
        "net_income": [1.0, 2.0, 3.0, 4.0, 5.0],
        "book_value": [100.0, 110.0, 120.0, 130.0, 140.0],
        "gross_profit": [40.0, 44.0, 48.0, 52.0, 56.0],
        "revenue": [100.0, 110.0, 120.0, 130.0, 140.0],
        "operating_cashflow": [0.5, 1.5, 2.5, 3.5, 4.5],
        "total_assets": [200.0, 210.0, 220.0, 230.0, 240.0],
        "ebitda": [2.0, 3.0, 4.0, 5.0, 6.0],
        "enterprise_value": [500.0, 510.0, 520.0, 530.0, 540.0],
        "shares_outstanding": [10.0, 10.0, 10.0, 10.0, 10.0],
    }
    records = []
    for field, field_values in values.items():
        for period, announcement, value in zip(
            periods, announcements, field_values, strict=True
        ):
            records.append(
                {
                    "symbol": "AAA",
                    "period_end": period,
                    "announcement_date": announcement,
                    "field": field,
                    "value": value,
                }
            )
    return pd.DataFrame(records)


def test_market_series_type_must_be_explicit():
    market = pd.Series(
        [0.5, 0.6, 0.7], index=pd.date_range("2024-01-01", periods=3)
    )

    with pytest.raises(ValueError, match="explicitly"):
        to_returns(market)
    returns = to_returns(market, market_is_returns=False)
    assert returns.iloc[-1] == pytest.approx(0.7 / 0.6 - 1.0)


def test_price_validation_rejects_invalid_panels():
    index = pd.date_range("2024-01-01", periods=2)
    with pytest.raises(ValueError, match="strictly positive"):
        validate_prices(pd.DataFrame({"AAA": [1.0, 0.0]}, index=index))
    with pytest.raises(ValueError, match="infinite"):
        validate_prices(pd.DataFrame({"AAA": [1.0, np.inf]}, index=index))
    with pytest.raises(ValueError, match="whitespace"):
        validate_prices(
            pd.DataFrame([[1.0, 2.0]], index=index[:1], columns=["AAA", " AAA "])
        )


def test_rolling_covariance_matches_population_covariance():
    index = pd.date_range("2024-01-01", periods=4)
    x = pd.Series([1.0, 2.0, 4.0, 8.0], index=index)
    y = pd.Series([2.0, 1.0, 3.0, 7.0], index=index)

    result = rolling_cov(x, y, window=3)

    assert result.iloc[-1] == pytest.approx(np.cov(x.iloc[-3:], y.iloc[-3:], ddof=0)[0, 1])


def test_classical_factor_windows_are_strict_and_causal():
    with pytest.raises(ValueError, match="integer"):
        MomentumFactor(lookback=5.5, gap=1)
    with pytest.raises(ValueError, match="integer"):
        LowVolFactor(window=True)

    dates = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame({"AAA": [100.0, 101.0, 102.0, 103.0, 206.0, 207.0]}, index=dates)
    momentum = MomentumFactor(lookback=3, gap=1).compute(prices)
    assert momentum.loc[dates[3], "AAA"] == pytest.approx(102.0 / 100.0 - 1.0)

    volatility = LowVolFactor(window=2).realized_volatility(prices)
    assert volatility.loc[dates[4], "AAA"] < volatility.loc[dates[5], "AAA"]


def test_fundamental_vintages_use_ttm_and_do_not_replace_latest_period():
    fundamentals = _quarterly_fundamentals()
    fundamentals.loc[len(fundamentals)] = {
        "symbol": "AAA",
        "period_end": pd.Timestamp("2023-12-31"),
        "announcement_date": pd.Timestamp("2024-05-15"),
        "field": "net_income",
        "value": 40.0,
    }
    frame = validate_fundamentals(fundamentals)
    index = pd.DatetimeIndex(
        ["2024-04-30", "2024-05-01", "2024-05-15", "2024-05-16"]
    )
    columns = pd.Index(["AAA"])

    latest = pit_panel(frame, "net_income", index, columns, mode="latest")
    ttm = pit_panel(frame, "net_income", index, columns, mode="ttm")

    assert latest.loc["2024-05-15", "AAA"] == 5.0
    assert ttm.loc["2024-04-30", "AAA"] == 10.0
    assert ttm.loc["2024-05-01", "AAA"] == 14.0
    assert ttm.loc["2024-05-15", "AAA"] == 14.0
    assert ttm.loc["2024-05-16", "AAA"] == 50.0


def test_quality_uses_ttm_flows_and_average_balance():
    fundamentals = _quarterly_fundamentals()
    index = pd.DatetimeIndex(["2024-05-01"])

    roe = QualityFactor().roe(fundamentals, index=index)

    assert roe.loc["2024-05-01", "AAA"] == pytest.approx(14.0 / 120.0)


def test_value_requires_compatible_market_cap_basis():
    fundamentals = _quarterly_fundamentals()
    prices = pd.DataFrame(
        {"AAA": [100.0]}, index=pd.DatetimeIndex(["2024-05-01"])
    )

    with pytest.raises(ValueError, match="market_caps"):
        ValueFactor().earnings_yield(fundamentals, prices)

    market_caps = pd.DataFrame({"AAA": [1_000.0]}, index=prices.index)
    result = ValueFactor().earnings_yield(
        fundamentals, prices, market_caps=market_caps
    )
    assert result.loc["2024-05-01", "AAA"] == pytest.approx(14.0 / 1_000.0)


def test_value_multiple_and_yield_have_consistent_orientation():
    fundamentals = _quarterly_fundamentals()
    prices = pd.DataFrame(
        {"AAA": [100.0]}, index=pd.DatetimeIndex(["2024-05-01"])
    )

    factor = ValueFactor()
    multiple = factor.ev_to_ebitda(fundamentals, prices)
    yield_ = factor.ebitda_yield(fundamentals, prices)

    assert multiple.loc["2024-05-01", "AAA"] == pytest.approx(540.0 / 18.0)
    assert yield_.loc["2024-05-01", "AAA"] == pytest.approx(18.0 / 540.0)


def test_period_average_shares_are_not_point_in_time_shares_outstanding():
    assert canonical_fundamental_field("Ordinary Shares Number") == "shares_outstanding"
    assert canonical_fundamental_field("Diluted Average Shares") == (
        "diluted_weighted_average_shares"
    )


def test_provider_fields_are_canonical_and_fmp_includes_cash_flow(monkeypatch):
    assert canonical_fundamental_field("Net Income Loss") == "net_income"
    assert canonical_fundamental_field("operatingCashFlow") == "operating_cashflow"

    provider = FMPProvider(api_key="unused")
    paths = []

    def fake_get(path, **params):
        paths.append(path)
        if path.startswith("cash-flow-statement"):
            return [
                {
                    "date": "2024-03-31",
                    "filingDate": "2024-04-30",
                    "operatingCashFlow": 123.0,
                }
            ]
        return []

    monkeypatch.setattr(provider, "_get", fake_get)
    result = provider.fetch_fundamentals(["AAA"], fields=["operating_cashflow"])

    assert any(path.startswith("cash-flow-statement") for path in paths)
    assert result.iloc[0]["field"] == "operating_cashflow"
    assert result.iloc[0]["value"] == 123.0


def test_panel_axes_rejects_duplicate_target_dates():
    frame = validate_fundamentals(_quarterly_fundamentals())
    default_index, _ = panel_axes(frame)
    announcements = pd.DatetimeIndex(sorted(frame["announcement_date"].unique()))
    assert (default_index == announcements + pd.Timedelta(1, unit="ns")).all()

    duplicate_index = pd.DatetimeIndex(["2024-04-30", "2024-04-30"])

    with pytest.raises(ValueError, match="unique"):
        panel_axes(frame, duplicate_index)


def test_alpha158_rejects_invalid_market_bars_and_duplicate_windows():
    with pytest.raises(ValueError, match="unique"):
        Alpha158(windows=(5, 5))

    bars = pd.DataFrame(
        {
            "open": [100.0],
            "high": [99.0],
            "low": [98.0],
            "close": [100.0],
            "volume": [1_000.0],
        },
        index=pd.DatetimeIndex(["2024-01-02"]),
    )
    with pytest.raises(ValueError, match="high"):
        Alpha158(windows=(2,)).compute(bars)


def test_alpha158_count_features_require_a_full_return_window():
    index = pd.bdate_range("2024-01-01", periods=3)
    bars = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1_000.0, 1_100.0, 1_200.0],
        },
        index=index,
    )

    features = Alpha158(windows=(2,)).compute(bars)

    assert pd.isna(features.loc[index[1], "CNTP2"])
    assert features.loc[index[2], "CNTP2"] == pytest.approx(1.0)


def test_quantile_spread_uses_same_date_top_and_bottom_returns():
    index = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    factor = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
        index=index,
        columns=list("ABCD"),
    )
    forward = pd.DataFrame(
        [[0.0, 0.0, 0.1, 0.1], [1.0, 1.0, np.nan, np.nan]],
        index=index,
        columns=list("ABCD"),
    )

    result = quantile_returns(factor, forward, quantiles=2)

    assert result.loc["long_short", "mean_return"] == pytest.approx(0.1)


def test_factor_validation_rejects_unsorted_dates():
    index = pd.DatetimeIndex(["2024-01-03", "2024-01-02"])
    factor = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=index, columns=["A", "B"])
    forward = pd.DataFrame(
        [[0.1, 0.0], [0.0, 0.1]], index=index, columns=["A", "B"]
    )

    with pytest.raises(ValueError, match="sorted"):
        compute_information_coefficient(factor, forward)


def test_newey_west_tstat_accounts_for_serial_dependence():
    values = pd.Series(np.repeat([0.01, 0.02, 0.03, 0.04], 20))
    naive = values.mean() / (values.std(ddof=1) / np.sqrt(len(values)))

    assert _newey_west_tstat(values) < naive


@pytest.mark.filterwarnings("ignore:Stochastic Optimizer:sklearn.exceptions.ConvergenceWarning")
def test_neural_factor_rejects_invalid_backend_predictions():
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0.0, 0.1, 0.2, 0.3])
    factor = NeuralFactor(
        hidden=(2,), epochs=2, batch_size=2, backend="sklearn", random_state=1
    ).fit(X, y)

    class BadModel:
        def predict(self, values):
            return np.full(len(values), np.nan)

    factor.model_ = BadModel()
    with pytest.raises(RuntimeError, match="non-finite"):
        factor.predict(X)


def test_torch_neural_factor_preserves_caller_rng_state():
    torch = pytest.importorskip("torch")
    features = pd.DataFrame(
        {"x": np.linspace(-1.0, 1.0, 12), "z": np.linspace(1.0, 2.0, 12)}
    )
    target = pd.Series(np.linspace(-0.1, 0.1, 12), index=features.index)
    torch.manual_seed(12345)
    state = torch.random.get_rng_state().clone()

    NeuralFactor(
        hidden=(4,), epochs=1, batch_size=4, backend="torch", random_state=7
    ).fit(features, target)

    assert torch.equal(torch.random.get_rng_state(), state)
