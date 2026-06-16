import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from quantcortex.alpha.feature_engineering.macro_features import MacroFeatures
from quantcortex.data.processors.adjustments import AdjustmentError, AdjustmentValidator
from quantcortex.data.processors.calendar import (
    TradingCalendar,
    first_session_each_week,
    last_session_each_month,
)
from quantcortex.data.providers.alpaca_provider import AlpacaProvider
from quantcortex.data.providers.base import DataProvider
from quantcortex.data.providers.ccxt_provider import CCXTProvider
from quantcortex.data.storage.redis_cache import RedisCache, RedisCacheError
from quantcortex.data.storage.timescale_store import TimescaleStore
from quantcortex.data.universe.base import PITMembership
from quantcortex.data.universe.nasdaq100_universe import Nasdaq100Universe
from quantcortex.data.universe.sp500_universe import SP500Universe
from quantcortex.data.universe.sp500_wikipedia import build_pit_membership


def test_pit_membership_uses_half_open_intervals():
    membership = PITMembership(
        pd.DataFrame(
            {
                "symbol": ["OLD", "NEW"],
                "start_date": ["2020-01-01", "2021-06-01"],
                "end_date": ["2021-06-01", None],
            }
        )
    )

    assert membership.members_asof("2021-05-31") == ["OLD"]
    assert membership.members_asof("2021-06-01") == ["NEW"]
    assert membership.is_member("new", "2021-06-01")
    assert membership.coverage_start == pd.Timestamp("2020-01-01")
    with pytest.raises(ValueError, match="valid timestamp"):
        membership.members_asof(None)


def test_pit_membership_rejects_implicit_numeric_data_and_normalizes_tickers():
    with pytest.raises(ValueError, match="symbols must be strings"):
        PITMembership(
            pd.DataFrame(
                {
                    "symbol": [123],
                    "start_date": ["2020-01-01"],
                    "end_date": [None],
                }
            )
        )
    with pytest.raises(ValueError, match="numeric epochs"):
        PITMembership(
            pd.DataFrame(
                {
                    "symbol": ["AAA"],
                    "start_date": [1_700_000_000],
                    "end_date": [None],
                }
            )
        )

    membership = PITMembership(
        pd.DataFrame(
            {
                "symbol": ["brk.b"],
                "start_date": ["2020-01-01"],
                "end_date": [None],
            }
        )
    )
    assert membership.members_asof("2020-01-01") == ["BRK-B"]


def test_pit_membership_rejects_overlapping_intervals():
    with pytest.raises(ValueError, match="overlapping intervals"):
        PITMembership(
            pd.DataFrame(
                {
                    "symbol": ["AAA", "AAA"],
                    "start_date": ["2020-01-01", "2020-06-01"],
                    "end_date": ["2021-01-01", None],
                }
            )
        )


def test_wikipedia_reconstruction_preserves_prior_readded_tenure():
    current = pd.DataFrame(
        {
            "Symbol": ["AAA", "CCC"],
            "Date added": ["2022-01-03", "2019-01-01"],
        }
    )
    changes = pd.DataFrame(
        {
            ("Date", "Date"): ["2020-01-02", "2021-01-04", "2022-01-03"],
            ("Added", "Ticker"): ["AAA", "BBB", "AAA"],
            ("Removed", "Ticker"): ["LEGACY", "AAA", "BBB"],
        }
    )

    membership = build_pit_membership(current, changes)

    assert membership.members_asof("2020-06-01") == ["AAA", "CCC"]
    assert membership.members_asof("2021-06-01") == ["BBB", "CCC"]
    assert membership.members_asof("2022-01-03") == ["AAA", "CCC"]
    aaa = membership.frame[membership.frame["symbol"] == "AAA"]
    assert len(aaa) == 2


def test_wikipedia_reconstruction_fails_closed_before_change_log_coverage():
    current = pd.DataFrame(
        {
            "Symbol": ["AAA", "BBB"],
            "Date added": ["1990-01-01", "2020-01-02"],
        }
    )
    changes = pd.DataFrame(
        {
            ("Date", "Date"): ["2020-01-02"],
            ("Added", "Ticker"): ["BBB"],
            ("Removed", "Ticker"): ["CCC"],
        }
    )

    membership = build_pit_membership(current, changes)

    with pytest.raises(ValueError, match="predates membership coverage"):
        membership.members_asof("2019-12-31")
    assert membership.members_asof("2020-01-02") == ["AAA", "BBB"]


def test_wikipedia_reconstruction_requires_real_change_log_coverage():
    current = pd.DataFrame({"Symbol": ["AAA"], "Date added": ["2020-01-02"]})
    with pytest.raises(ValueError, match="non-empty"):
        build_pit_membership(current, pd.DataFrame())

    changes = pd.DataFrame(
        {
            ("Date", "Date"): ["2020-01-02"],
            ("Added", "Ticker"): ["AAA"],
            ("Removed", "Ticker"): [None],
        }
    )
    with pytest.raises(ValueError, match="cannot precede"):
        build_pit_membership(current, changes, floor=pd.Timestamp("2019-01-01"))


def test_named_index_universes_do_not_use_demo_subsets_implicitly():
    with pytest.raises(ValueError, match="requires membership_csv"):
        SP500Universe().membership()
    with pytest.raises(ValueError, match="requires membership_csv"):
        Nasdaq100Universe().membership()

    with pytest.warns(RuntimeWarning, match="survivorship-biased"):
        assert SP500Universe(allow_static_demo=True).constituents("2020-01-01")


def test_calendar_juneteenth_starts_in_2022(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "pandas_market_calendars", None)
    calendar = TradingCalendar("NYSE")

    assert calendar.is_trading_day("2021-06-18")
    assert not calendar.is_trading_day("2022-06-20")


def test_calendar_timezone_aware_holiday_preserves_local_date(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "pandas_market_calendars", None)
    calendar = TradingCalendar("NYSE")

    assert not calendar.is_trading_day("2024-07-04T12:00:00-04:00")


def test_rebalance_schedules_use_observed_sessions_not_calendar_labels():
    sessions = pd.DatetimeIndex(
        [
            "2024-01-02",  # Monday was New Year's Day.
            "2024-01-03",
            "2024-01-05",
            "2024-01-08",
            "2024-01-31",
            "2024-02-01",
            "2024-02-29",
        ]
    )

    assert first_session_each_week(sessions).tolist()[:2] == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-08"),
    ]
    assert last_session_each_month(sessions).tolist() == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-29"),
    ]


def test_rebalance_schedules_reject_unsorted_or_duplicate_sessions():
    with pytest.raises(ValueError, match="sorted"):
        first_session_each_week(
            pd.DatetimeIndex(["2024-01-03", "2024-01-02"])
        )
    with pytest.raises(ValueError, match="duplicate"):
        last_session_each_month(
            pd.DatetimeIndex(["2024-01-02", "2024-01-02"])
        )


def test_calendar_rejects_unknown_exchange_without_backend(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "pandas_market_calendars", None)
    with pytest.raises(ValueError, match="does not support"):
        TradingCalendar("LSE")


def test_timescale_write_uses_primary_key_upsert(monkeypatch):
    class Connection:
        def __init__(self):
            self.statement = None
            self.records = None

        def execute(self, statement, records):
            self.statement = statement
            self.records = records

    class Begin:
        def __init__(self, connection):
            self.connection = connection

        def __enter__(self):
            return self.connection

        def __exit__(self, *args):
            return None

    connection = Connection()
    engine = SimpleNamespace(begin=lambda: Begin(connection))
    monkeypatch.setitem(sys.modules, "sqlalchemy", SimpleNamespace(text=lambda x: x))
    store = TimescaleStore("postgresql://unused")
    store._engine = engine
    bars = pd.DataFrame(
        {"close": [100.0], "volume": [1_000.0]},
        index=pd.DatetimeIndex(["2024-01-02"], name="time"),
    )

    assert store.write_ohlcv(bars, "AAA", table="market.ohlcv") == 1
    assert "INSERT INTO market.ohlcv" in connection.statement
    assert "ON CONFLICT (symbol, time) DO UPDATE SET" in connection.statement
    assert connection.records[0]["symbol"] == "AAA"


def test_timescale_write_rejects_duplicate_timestamps():
    bars = pd.DataFrame(
        {"time": ["2024-01-02", "2024-01-02"], "close": [100.0, 101.0]}
    )
    with pytest.raises(ValueError, match="duplicate timestamps"):
        TimescaleStore("postgresql://unused").write_ohlcv(bars, "AAA")


def test_timescale_write_rejects_incomplete_or_impossible_bars():
    index = pd.DatetimeIndex(["2024-01-02"], name="time")
    with pytest.raises(ValueError, match="present"):
        TimescaleStore("postgresql://unused").write_ohlcv(
            pd.DataFrame({"close": [None]}, index=index), "AAA"
        )
    with pytest.raises(ValueError, match="high must be at least open"):
        TimescaleStore("postgresql://unused").write_ohlcv(
            pd.DataFrame(
                {"open": [101.0], "high": [100.0], "low": [99.0], "close": [100.0]},
                index=index,
            ),
            "AAA",
        )


def test_timescale_read_validates_query_bounds_before_connecting():
    store = TimescaleStore("postgresql://unused")
    with pytest.raises(ValueError, match="non-empty"):
        store.read_ohlcv(" ")
    with pytest.raises(ValueError, match="before"):
        store.read_ohlcv("AAA", start="2024-02-01", end="2024-01-01")


def test_adjustments_reject_duplicate_split_events():
    index = pd.date_range("2024-01-01", periods=3)
    bars = pd.DataFrame({"close": [100.0, 50.0, 51.0]}, index=index)
    splits = pd.Series([2.0, 2.0], index=[index[1], index[1]])

    with pytest.raises(AdjustmentError, match="duplicate event dates"):
        AdjustmentValidator().apply_adjustments(bars, splits=splits)


def test_adjustments_reject_impossible_dividend():
    index = pd.date_range("2024-01-01", periods=3)
    bars = pd.DataFrame({"close": [10.0, 9.0, 9.5]}, index=index)
    dividends = pd.Series([10.0], index=[index[1]])

    with pytest.raises(AdjustmentError, match="not smaller than prior close"):
        AdjustmentValidator().apply_adjustments(bars, dividends=dividends)


def test_adjustment_validation_requires_identical_timestamp_coverage():
    raw = pd.Series(
        [100.0, 50.0, 51.0], index=pd.date_range("2024-01-01", periods=3)
    )
    adjusted = pd.Series([50.0, 51.0], index=raw.index[1:])

    with pytest.raises(AdjustmentError, match="identical timestamp coverage"):
        AdjustmentValidator().validate_adjustment(raw, adjusted)


def test_shared_ohlcv_normalizer_checks_partial_high_low_rows():
    index = pd.DatetimeIndex(["2024-01-02"])

    with pytest.raises(ValueError, match="high is below"):
        DataProvider._standardize_ohlcv(
            pd.DataFrame({"close": [101.0], "high": [100.0]}, index=index)
        )
    with pytest.raises(ValueError, match="low is above"):
        DataProvider._standardize_ohlcv(
            pd.DataFrame({"open": [99.0], "low": [100.0]}, index=index)
        )


def test_unadjusted_detection_rejects_malformed_price_panels():
    validator = AdjustmentValidator()
    duplicate_dates = pd.DataFrame(
        {"AAA": [100.0, 50.0]},
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-01"]),
    )

    with pytest.raises(ValueError, match="unique"):
        validator.detect_unadjusted(duplicate_dates)
    with pytest.raises(ValueError, match="strictly positive"):
        validator.detect_unadjusted(
            pd.DataFrame(
                {"AAA": [100.0, 0.0]}, index=pd.date_range("2024-01-01", periods=2)
            )
        )


def test_macro_publication_lag_is_nonnegative_and_alias_aware():
    with pytest.raises(ValueError, match="non-negative integers"):
        MacroFeatures(publication_lags={"PMI": -1})

    macro = pd.DataFrame(
        {"ISM": [50.0, 51.0], "DGS10": [4.0, 4.1]},
        index=pd.to_datetime(["2024-01-01", "2024-02-01"]),
    )
    prepared = MacroFeatures(publication_lags={"PMI": 5})._prepare(macro)

    assert pd.isna(prepared.loc["2024-01-05", "ISM"])
    assert prepared.loc["2024-01-08", "ISM"] == 50.0


def test_macro_features_reject_malformed_or_ambiguous_observations():
    index = pd.DatetimeIndex(["2024-01-01"])

    with pytest.raises(ValueError, match="non-numeric"):
        MacroFeatures().compute(pd.DataFrame({"DGS10": ["bad"]}, index=index))
    with pytest.raises(ValueError, match="ambiguous"):
        MacroFeatures().compute(
            pd.DataFrame([[4.0, 4.1]], columns=["DGS10", "dgs10"], index=index)
        )
    with pytest.raises(ValueError, match="positive integers"):
        MacroFeatures(vix_change_window=True)


def test_redis_cache_uses_validated_thread_safe_local_mode():
    cache = RedisCache(url=None, default_ttl=0, namespace="test")

    assert cache.using_fallback
    cache.set("key", {"value": 1})
    assert cache.get("key") == {"value": 1}
    assert cache.exists("key")
    cache.delete("key")
    assert not cache.exists("key")

    with pytest.raises(ValueError, match="non-empty"):
        cache.get(" ")
    with pytest.raises(ValueError, match="non-negative"):
        cache.set("key", 1, ttl=-1)


def test_redis_runtime_failure_does_not_silently_change_semantics():
    class FailingClient:
        def get(self, _key):
            raise ConnectionError("down")

    cache = RedisCache(url=None)
    cache._client = FailingClient()
    cache._using_fallback = False

    with pytest.raises(RedisCacheError, match="GET failed"):
        cache.get("key")


def test_ccxt_pagination_fails_when_exchange_cursor_does_not_advance():
    class Client:
        def fetch_ohlcv(self, *args, **kwargs):
            return [[1, 1.0, 1.0, 1.0, 1.0, 1.0]] * 1_000

    with pytest.raises(RuntimeError, match="did not advance"):
        CCXTProvider()._fetch_symbol(Client(), "BTC/USD", "1d", 0, None)


def test_alpaca_provider_uses_current_sdk_and_splits_multi_symbol_bars(monkeypatch):
    index = pd.MultiIndex.from_product(
        [["AAPL", "MSFT"], pd.to_datetime(["2024-01-02", "2024-01-03"])],
        names=["symbol", "timestamp"],
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 200.0, 202.0],
            "high": [102.0, 103.0, 204.0, 205.0],
            "low": [99.0, 100.0, 198.0, 201.0],
            "close": [101.0, 102.0, 202.0, 204.0],
            "volume": [1_000.0, 1_100.0, 2_000.0, 2_100.0],
        },
        index=index,
    )

    class Client:
        instance = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            type(self).instance = self

        def get_stock_bars(self, request):
            self.request = request
            return SimpleNamespace(df=frame)

    class Request(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class Unit:
        Day = "day"
        Hour = "hour"
        Week = "week"
        Month = "month"
        Minute = "minute"

    class Adjustment:
        ALL = "all"

    sdk = {
        "Adjustment": Adjustment,
        "StockHistoricalDataClient": Client,
        "StockBarsRequest": Request,
        "TimeFrame": lambda amount, unit: (amount, unit),
        "TimeFrameUnit": Unit,
    }
    provider = AlpacaProvider("key", "secret", data_url="https://data.example")
    monkeypatch.setattr(provider, "_load_sdk", lambda: sdk)

    result = provider.fetch_ohlcv(
        ["AAPL", "MSFT"],
        start="2024-01-02",
        end="2024-01-03",
        timeframe="1d",
    )

    assert Client.instance.kwargs["url_override"] == "https://data.example"
    assert Client.instance.request.symbol_or_symbols == ["AAPL", "MSFT"]
    assert Client.instance.request.timeframe == (1, "day")
    assert Client.instance.request.adjustment == "all"
    assert result["AAPL"]["adj_close"].tolist() == [101.0, 102.0]
    assert result["MSFT"]["volume"].tolist() == [2_000.0, 2_100.0]


def test_alpaca_provider_does_not_reuse_trading_endpoint_environment(monkeypatch):
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.delenv("ALPACA_DATA_URL", raising=False)

    assert AlpacaProvider("key", "secret").data_url is None
