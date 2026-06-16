"""Regression tests for research and accounting inputs that must fail closed."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantcortex.alpha.factors.ml.gbdt_factor import GBDTFactor
from quantcortex.alpha.factors.ml.neural_factor import NeuralFactor
from quantcortex.alpha.factors.nlp.finbert_sentiment import FinBERTSentiment
from quantcortex.alpha.factors.nlp.news_scorer import NewsScorer
from quantcortex.backtest.costs.transaction_costs import TransactionCostModel
from quantcortex.backtest.engines.event_driven import EventDrivenBacktest
from quantcortex.backtest.engines.vectorized import BacktestResult, VectorizedBacktest
from quantcortex.backtest.execution_models.market_impact import AlmgrenChriss
from quantcortex.backtest.execution_models.vwap_fill import VWAPFill
from quantcortex.backtest.metrics.tearsheet import Tearsheet
from quantcortex.backtest.validation.deflated_sharpe import compute_dsr
from quantcortex.backtest.validation.multiple_testing import bonferroni
from quantcortex.data.local_csv import load_price_matrix
from quantcortex.data.processors.pit_enforcer import PITEnforcer, PITViolationError
from quantcortex.data.providers.base import DataProvider
from quantcortex.data.providers.fmp_provider import FMPProvider
from quantcortex.data.providers.polygon_provider import PolygonProvider
from quantcortex.data.storage.parquet_store import ParquetStore
from quantcortex.data.universe.sp500_wikipedia import fetch_sp500_tables
from quantcortex.portfolio.base import (
    PortfolioMode,
    WeightContractViolationError,
    enforce_weight_contract,
    normalize_long_only,
    normalize_market_neutral,
    project_bounded_sum,
)
from quantcortex.portfolio.black_litterman import BlackLitterman
from quantcortex.portfolio.drl_allocator import DRLAllocator, _portfolio_variance
from quantcortex.portfolio.equal_weight import EqualWeight
from quantcortex.portfolio.hrp import HierarchicalRiskParity
from quantcortex.portfolio.mean_variance import MeanVariance
from quantcortex.portfolio.minimum_variance import MinimumVariance
from quantcortex.portfolio.risk_parity import RiskParity
from quantcortex.risk.circuit_breaker import CircuitBreaker, compute_drawdown
from quantcortex.risk.factor_exposure import FactorExposureLimiter
from quantcortex.risk.kelly import KellyCriterion
from quantcortex.risk.var_cvar import VaRCVaR
from quantcortex.risk.vol_targeting import VolTargeting
from quantcortex.strategies.base_strategy import Strategy, StrategyContext
from quantcortex.strategies.drl_portfolio import DRLPortfolioStrategy
from quantcortex.strategies.macro_timing import MacroTimingStrategy
from quantcortex.strategies.momentum_ml import MomentumMLStrategy
from quantcortex.strategies.multi_asset_rotation import MultiAssetRotation
from quantcortex.strategies.sentiment_nlp import SentimentNLPStrategy
from quantcortex.timing.hmm_regime import BEAR, BULL, SIDEWAYS, HMMRegime
from quantcortex.timing.kama import KAMA
from quantcortex.timing.tsmom import TSMomentum
from quantcortex.timing.vix_scaler import VIXScaler


def test_long_only_contract_rejects_a_short_leg_even_when_sum_is_one():
    with pytest.raises(WeightContractViolationError, match="short positions"):
        enforce_weight_contract(
            np.array([0.8, 0.3, -0.1]), mode=PortfolioMode.LONG_ONLY
        )


def test_score_normalization_does_not_invent_a_portfolio_from_no_signal():
    with pytest.raises(WeightContractViolationError, match="positive score"):
        normalize_long_only(np.array([-2.0, -1.0, 0.0]))
    with pytest.raises(WeightContractViolationError, match="dispersion"):
        normalize_market_neutral(np.array([1.0, 1.0, 1.0]))


@pytest.mark.parametrize("column", ["feature_date", "announcement_date"])
def test_pit_enforcer_rejects_missing_dates(column):
    frame = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "feature_date": ["2024-04-01"],
            "announcement_date": ["2024-03-01"],
        }
    )
    frame.loc[0, column] = None
    with pytest.raises(PITViolationError, match="missing or invalid dates"):
        PITEnforcer().enforce(frame)


def test_pit_merge_rejects_missing_announcement_date():
    features = pd.DataFrame({"symbol": ["A"], "feature_date": ["2024-01-01"]})
    fundamentals = pd.DataFrame(
        {"symbol": ["A"], "announcement_date": [pd.NaT], "eps": [1.0]}
    )
    with pytest.raises(PITViolationError, match="missing or invalid dates"):
        PITEnforcer().point_in_time_merge(features, fundamentals)


def test_pit_merge_rejects_ambiguous_tidy_rows_and_preserves_feature_order():
    enforcer = PITEnforcer()
    features = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "feature_date": ["2024-04-01", "2024-02-01"],
            "row": ["late", "early"],
        }
    )
    ambiguous = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "announcement_date": ["2024-01-15", "2024-01-15"],
            "field": ["revenue", "net_income"],
            "value": [100.0, 10.0],
        }
    )
    with pytest.raises(PITViolationError, match="reshape tidy fields"):
        enforcer.point_in_time_merge(features, ambiguous)

    wide = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "announcement_date": ["2024-01-15", "2024-03-15"],
            "eps": [1.0, 2.0],
        }
    )
    merged = enforcer.point_in_time_merge(features, wide)
    assert merged["row"].tolist() == ["late", "early"]
    assert merged["eps"].tolist() == [2.0, 1.0]


def test_transaction_cost_model_rejects_nonfinite_liquidity_inputs():
    model = TransactionCostModel()
    with pytest.raises(ValueError, match="adv must contain finite"):
        model.apply_costs(
            np.array([0.0]),
            np.array([1.0]),
            adv=np.array([np.nan]),
            capital=1_000.0,
        )
    with pytest.raises(ValueError, match="capital must be finite"):
        model.apply_costs(np.array([0.0]), np.array([1.0]), capital=np.nan)


def test_drawdown_includes_starting_nav():
    returns = pd.Series([-0.5, 0.1])
    assert Tearsheet(returns).compute()["max_drawdown"] == pytest.approx(-0.5)

    result = BacktestResult(
        returns=returns,
        equity_curve=(1.0 + returns).cumprod(),
        weights=pd.DataFrame(index=returns.index),
        gross_returns=returns,
        costs=pd.Series(0.0, index=returns.index),
        turnover=pd.Series(0.0, index=returns.index),
    )
    assert result.summary()["max_drawdown"] == pytest.approx(-0.5)


def test_tearsheet_accepts_time_varying_risk_free_returns():
    index = pd.bdate_range("2024-01-01", periods=3)
    returns = pd.Series([0.01, 0.02, 0.03], index=index)
    risk_free = pd.Series([0.005, 0.01, 0.015], index=index)
    metrics = Tearsheet(returns, risk_free=risk_free).compute()

    excess = returns - risk_free
    expected_sharpe = excess.mean() / excess.std(ddof=1) * np.sqrt(252.0)
    assert metrics["sharpe"] == pytest.approx(expected_sharpe)

    with pytest.raises(ValueError, match="every return"):
        Tearsheet(returns, risk_free=risk_free.iloc[:-1])


def test_event_engine_executes_close_signal_on_next_bar():
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=dates)
    weights = pd.DataFrame({"A": [1.0]}, index=[dates[0]])
    costs = TransactionCostModel(commission=0.0, slippage=0.0, tax=0.0)

    result = EventDrivenBacktest(costs, capital=1_000.0).run(weights, prices)

    assert result.returns.to_list() == pytest.approx([0.0, 0.0, 0.1])
    assert result.equity_curve.to_list() == pytest.approx([1_000.0, 1_000.0, 1_100.0])


def test_event_engine_cost_series_reconciles_gross_and_net_returns():
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 100.0, 110.0]}, index=dates)
    weights = pd.DataFrame({"A": [1.0]}, index=[dates[0]])
    costs = TransactionCostModel(commission=0.001, slippage=0.0, tax=0.0)

    result = EventDrivenBacktest(costs, capital=1_000.0).run(weights, prices)

    pd.testing.assert_series_equal(
        result.gross_returns,
        result.returns + result.costs,
        check_names=False,
    )
    expected_cost = 0.001 / 1.001
    assert result.costs.iloc[1] == pytest.approx(expected_cost)
    assert result.turnover.iloc[1] == pytest.approx(1.0 / 1.001)
    assert result.traded_notional.iloc[1] == pytest.approx(1.0 / 1.001)
    assert result.weights.iloc[1, 0] == pytest.approx(1.0)
    assert result.cash_weights.iloc[1] == pytest.approx(0.0, abs=1e-12)


def test_event_engine_cost_drag_reconciles_to_pretrade_notional_after_a_move():
    dates = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame(
        {
            "A": [100.0, 100.0, 110.0, 110.0],
            "B": [100.0, 100.0, 100.0, 100.0],
        },
        index=dates,
    )
    weights = pd.DataFrame(
        {"A": [1.0, 0.0], "B": [0.0, 1.0]},
        index=dates[:2],
    )
    rate = 0.001
    result = EventDrivenBacktest(
        TransactionCostModel(commission=rate, slippage=0.0, tax=0.0),
        capital=1_000.0,
    ).run(weights, prices)

    expected_drag = (
        rate
        * result.traded_notional.iloc[2]
        * (1.0 + result.gross_returns.iloc[2])
    )
    assert result.costs.iloc[2] == pytest.approx(expected_drag)


def test_event_engine_rejects_implicit_financing_from_adverse_fills():
    class AdverseFill:
        def fill(self, symbol, target_qty, bar, **kwargs):
            return float(bar["close"]) * (1.10 if target_qty > 0 else 0.90)

    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=dates)
    weights = pd.DataFrame({"A": [1.0]}, index=[dates[0]])
    costs = TransactionCostModel(commission=0.0, slippage=0.0, tax=0.0)

    with pytest.raises(ValueError, match="unmodeled financing"):
        EventDrivenBacktest(
            costs, execution_model=AdverseFill(), capital=1_000.0
        ).run(weights, prices)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TransactionCostModel(commission=True),
        lambda: VectorizedBacktest(TransactionCostModel(), capital=True),
        lambda: EventDrivenBacktest(TransactionCostModel(), max_gross=True),
        lambda: VWAPFill(participation=True),
        lambda: AlmgrenChriss(eta=True),
        lambda: KellyCriterion(fraction=True),
        lambda: VolTargeting(target_vol=True),
        lambda: FactorExposureLimiter(max_exposure=True),
        lambda: CircuitBreaker(max_drawdown=True),
        lambda: VaRCVaR(alpha=True),
        lambda: VIXScaler(target_vix=True),
        lambda: MeanVariance(risk_aversion=True),
        lambda: RiskParity(max_iter=True),
        lambda: BlackLitterman(tau=True),
        lambda: DRLAllocator(transaction_cost=True),
    ],
)
def test_money_path_configuration_rejects_booleans(factory):
    with pytest.raises(TypeError, match="boolean|integer"):
        factory()


def test_weight_and_return_contracts_reject_boolean_data():
    with pytest.raises(WeightContractViolationError, match="boolean"):
        enforce_weight_contract([True], mode=PortfolioMode.LONG_ONLY)
    with pytest.raises(ValueError, match="boolean"):
        MeanVariance().optimize(pd.DataFrame({"A": [True, False]}))
    with pytest.raises(TypeError, match="boolean"):
        EqualWeight().optimize(None, n_assets=True)


def test_loss_metrics_do_not_report_negative_losses():
    metrics = Tearsheet(pd.Series([0.01, 0.02, 0.03])).compute()
    assert metrics["var_95"] == 0.0
    assert metrics["cvar_95"] == 0.0


def test_statistical_validation_rejects_invalid_controls():
    returns = pd.Series([0.01, -0.01, 0.02, -0.005, 0.003])
    with pytest.raises(ValueError, match="positive integer"):
        compute_dsr(returns, n_trials=0)
    with pytest.raises(ValueError, match="positive integer"):
        compute_dsr(returns, n_trials=1.5)
    with pytest.raises(ValueError, match="alpha"):
        bonferroni([0.01, 0.02], alpha=0.0)


def test_providers_do_not_backdate_missing_filing_dates():
    fmp_row = {"date": "2024-03-31", "revenue": 100.0}
    fmp = FMPProvider()._melt_row("AAA", fmp_row, None, {"date"})
    assert fmp.empty

    financials = SimpleNamespace(income_statement={"revenue": {"value": 100.0}})
    polygon_report = SimpleNamespace(
        end_date="2024-03-31", filing_date=None, financials=financials
    )
    polygon = PolygonProvider()._melt_financial("AAA", polygon_report, None)
    assert polygon.empty


def test_standard_library_http_helpers_reject_non_https_urls():
    fmp = FMPProvider(api_key="not-a-real-key")
    fmp._BASE_URL = "file:///tmp"  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="HTTPS"):
        fmp._get("prices")
    with pytest.raises(ValueError, match="HTTPS"):
        fetch_sp500_tables("file:///etc/passwd")


def test_ohlcv_standardization_rejects_duplicate_normalized_columns():
    frame = pd.DataFrame(
        {"close": [100.0], "Adj Close": [100.0], "adjclose": [100.0]},
        index=["2024-01-01"],
    )
    with pytest.raises(ValueError, match="unique"):
        DataProvider._standardize_ohlcv(frame)


def test_local_price_loader_bounds_forward_fill(tmp_path):
    path = tmp_path / "prices.csv"
    dates = pd.bdate_range("2024-01-01", periods=8)
    pd.DataFrame(
        {
            "date": dates,
            "A": [100.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 108.0],
            "B": np.arange(200.0, 208.0),
        }
    ).to_csv(path, index=False)

    loaded = load_price_matrix(path)
    assert dates[6] not in loaded.index
    assert dates[5] in loaded.index


def test_parquet_store_rejects_dataset_path_traversal(tmp_path):
    store = ParquetStore(tmp_path / "store")
    with pytest.raises(ValueError, match="escapes"):
        store.exists("../outside")


def test_parquet_store_rejects_incompatible_append_without_mutating_data(tmp_path):
    store = ParquetStore(tmp_path / "store")
    original = pd.DataFrame({"symbol": ["A"], "value": [1.0]})
    store.write(original, "prices")

    with pytest.raises(ValueError, match="schema"):
        store.write(
            pd.DataFrame({"symbol": ["B"], "value": ["bad"]}),
            "prices",
            mode="append",
        )

    pd.testing.assert_frame_equal(store.read("prices"), original)


def test_parquet_store_lists_nested_dataset_roots_not_partitions(tmp_path):
    store = ParquetStore(tmp_path / "store")
    store.write(
        pd.DataFrame({"year": [2024], "value": [1.0]}),
        "market/prices",
        partition_cols=["year"],
    )

    assert store.list_datasets() == ["market/prices"]


def test_risk_overlays_reject_missing_risk_inputs():
    with pytest.raises(ValueError, match="equity_curve"):
        compute_drawdown([100.0, np.nan])
    with pytest.raises(ValueError, match="current_drawdown"):
        CircuitBreaker().apply(np.array([1.0]), current_drawdown=np.nan)
    with pytest.raises(ValueError, match="realized_vol"):
        VolTargeting().apply(np.array([1.0]), realized_vol=np.nan)


def test_vol_and_kelly_caps_apply_to_final_gross_exposure():
    vol_scaled = VolTargeting(max_leverage=1.0).apply(
        np.array([1.0, 1.0]), realized_vol=0.0
    )
    assert np.abs(vol_scaled).sum() == pytest.approx(1.0)

    kelly_scaled = KellyCriterion(max_leverage=1.0).apply(
        np.array([1.0, 1.0]),
        expected_returns=np.array([0.1, 0.1]),
        cov=np.eye(2) * 0.01,
    )
    assert np.abs(kelly_scaled).sum() == pytest.approx(1.0)


def test_factor_limiter_rejects_nan_and_records_post_adjustment_exposure():
    limiter = FactorExposureLimiter(max_exposure=0.2)
    with pytest.raises(ValueError, match="finite"):
        limiter.apply(np.array([1.0]), pd.DataFrame({"beta": [np.nan]}))

    adjusted = limiter.apply(
        np.array([1.0, 0.0]), pd.DataFrame({"beta": [1.0, 0.0]})
    )
    assert adjusted[0] == pytest.approx(0.2)
    assert limiter.last_exposures.loc["beta"] == pytest.approx(0.2)


def test_factor_limiter_does_not_create_new_positions_by_default():
    loadings = pd.DataFrame({"beta": [1.0, -1.0]})
    weights = np.array([1.0, 0.0])

    preserved = FactorExposureLimiter(max_exposure=0.2).apply(weights, loadings)
    unconstrained = FactorExposureLimiter(
        max_exposure=0.2,
        preserve_signs=False,
    ).apply(weights, loadings)

    assert preserved[0] >= 0.0
    assert preserved[1] == pytest.approx(0.0)
    assert unconstrained[1] > 0.0


def test_var_estimators_floor_gain_quantiles_at_zero_loss():
    risk = VaRCVaR(alpha=0.95)
    returns = np.full(20, 0.01)
    assert risk.historical_var(returns) == 0.0
    assert risk.historical_cvar(returns) == 0.0
    assert risk.parametric_var(returns) == 0.0
    assert risk.parametric_cvar(returns) == 0.0
    assert risk.cornish_fisher_var(returns) == 0.0
    with pytest.raises(ValueError, match="infinite"):
        risk.historical_var([0.01, np.inf])


def test_strategy_generation_emits_explicit_flat_rebalance():
    class FlatStrategy(Strategy):
        def select(self, ctx):
            return pd.Series(dtype=float)

    dates = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0, 103.0]}, index=dates)
    panel = FlatStrategy(EqualWeight()).generate_weights(prices, [dates[-1]])
    assert panel.index.to_list() == [dates[-1]]
    assert panel.loc[dates[-1], "A"] == 0.0


def test_strategy_context_rejects_data_after_as_of():
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0]}, index=dates)
    returns = prices.pct_change(fill_method=None)

    with pytest.raises(ValueError, match="after as_of"):
        StrategyContext(dates[1], prices, returns)


def test_default_allocator_rejects_missing_selected_return_history():
    class MissingAssetStrategy(Strategy):
        def select(self, ctx):
            return pd.Series({"B": 1.0})

    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0]}, index=dates)
    returns = prices.pct_change(fill_method=None).dropna()
    ctx = StrategyContext(dates[-1], prices, returns)

    with pytest.raises(ValueError, match="absent from strategy data"):
        MissingAssetStrategy(EqualWeight()).rebalance(ctx)


def test_strategy_generation_sorts_input_and_rejects_duplicate_decisions():
    class FullStrategy(Strategy):
        def select(self, ctx):
            return pd.Series({"A": 1.0})

    dates = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame(
        {"A": [103.0, 100.0, 102.0, 101.0]},
        index=dates[[3, 0, 2, 1]],
    )
    strategy = FullStrategy(EqualWeight())

    panel = strategy.generate_weights(prices, [dates[-1]])
    assert panel.loc[dates[-1], "A"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unique"):
        strategy.generate_weights(prices, [dates[-1], dates[-1]])


def test_strategy_risk_parameters_reject_booleans():
    class FullStrategy(Strategy):
        def select(self, ctx):
            return pd.Series({"A": 1.0})

    with pytest.raises(TypeError, match="max_gross"):
        FullStrategy(EqualWeight(), max_gross=True)
    with pytest.raises(TypeError, match="half_life"):
        NewsScorer(half_life=True)
    with pytest.raises(TypeError, match="target_vix"):
        MultiAssetRotation(target_vix=True)
    with pytest.raises(TypeError, match="max_position_weight"):
        MultiAssetRotation(max_position_weight=True)
    with pytest.raises(ValueError, match="smaller than"):
        SentimentNLPStrategy(base_lookback=20, base_gap=20)


def test_multi_asset_rotation_caps_concentrated_score_allocations():
    strategy = MultiAssetRotation(regime=False, vix_scale=False)

    allocated = strategy.allocate(pd.Series({"AAA": 9.0, "BBB": 1.0}), None)
    weights = strategy._position_limit_overlay(allocated, None)

    assert weights == pytest.approx([0.6, 0.4])


def test_multi_asset_rotation_position_cap_leaves_cash_when_needed():
    strategy = MultiAssetRotation(
        max_position_weight=0.6, regime=False, vix_scale=False
    )

    weights = strategy._position_limit_overlay(np.array([1.0]), None)

    assert weights == pytest.approx([0.6])

    with pytest.raises(ValueError, match="negative"):
        strategy._position_limit_overlay(np.array([-0.1, 1.0]), None)


def test_rotation_does_not_invest_when_all_fallback_momentum_is_negative():
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 99.0, 98.0]}, index=dates)

    scores = MultiAssetRotation(regime=False, vix_scale=False)._fallback_momentum(
        prices
    )

    assert scores.empty


def test_rotation_holds_cash_when_selected_residual_scores_are_nonpositive():
    dates = pd.bdate_range("2024-01-01", periods=8)
    base = np.arange(len(dates), dtype=float)
    prices = pd.DataFrame(
        {
            "QQQ": 100.0 + base,
            "VGT": 100.0 + 1.2 * base,
            "GLD": 100.0 + 0.8 * base,
            "TLT": 100.0 + 0.6 * base,
            "SPY": 100.0 + 0.9 * base,
            "VIG": 100.0 + 0.7 * base,
        },
        index=dates,
    )
    returns = prices.pct_change(fill_method=None).dropna()
    strategy = MultiAssetRotation(
        ir_lookback=3,
        mom_lookback=2,
        mom_gap=0,
        regime=False,
        vix_scale=False,
    )
    strategy._residual_momentum = lambda members, benchmark: pd.Series(
        -1.0,
        index=members.columns,
    )
    strategy._fallback_momentum = lambda frame: (_ for _ in ()).throw(
        AssertionError("whole-universe fallback must not replace a mature signal")
    )

    scores = strategy.select(StrategyContext(dates[-1], prices, returns))

    assert scores.empty


def test_momentum_ml_gap_feature_uses_a_full_lookback_before_the_gap():
    dates = pd.bdate_range("2023-01-01", periods=80)
    prices = pd.DataFrame({"A": np.arange(100.0, 180.0)}, index=dates)
    strategy = MomentumMLStrategy(gap=21)
    features = strategy._price_features(prices)
    expected = prices.iloc[-22, 0] / prices.iloc[-43, 0] - 1.0
    assert features.loc[(dates[-1], "A"), "mom_21"] == pytest.approx(expected)
    assert expected != 0.0


def test_momentum_ml_requires_ohlcv_for_every_price_symbol():
    dates = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame(
        {"A": [100.0, 101.0, 102.0, 103.0], "B": [90.0, 91.0, 92.0, 93.0]},
        index=dates,
    )
    ohlcv_a = pd.DataFrame(
        {
            "open": prices["A"],
            "high": prices["A"] + 1.0,
            "low": prices["A"] - 1.0,
            "close": prices["A"],
            "volume": 1_000.0,
        },
        index=dates,
    )

    with pytest.raises(ValueError, match="every price symbol"):
        MomentumMLStrategy()._alpha158_features(prices, {"A": ohlcv_a})


def test_residual_momentum_uses_separate_estimation_and_formation_windows():
    rng = np.random.default_rng(12)
    n = 180
    dates = pd.bdate_range("2023-01-01", periods=n)
    benchmark_returns = rng.normal(0.0002, 0.01, n)
    asset_returns = benchmark_returns.copy()
    # With lookback=50 and gap=5, the formation return positions are 125:175.
    asset_returns[125:175] += 0.002
    benchmark_prices = pd.Series(
        100.0 * np.cumprod(1.0 + benchmark_returns), index=dates
    )
    member_prices = pd.DataFrame(
        {
            "A": 100.0 * np.cumprod(1.0 + asset_returns),
            "B": benchmark_prices.to_numpy(),
        },
        index=dates,
    )

    scores = MultiAssetRotation(
        mom_lookback=50, mom_gap=5, regime=False, vix_scale=False
    )._residual_momentum(member_prices, benchmark_prices)

    assert scores is not None
    assert scores["A"] > 0.05
    assert abs(scores["B"]) < 1e-10


def test_residual_momentum_requires_full_estimation_and_formation_windows():
    dates = pd.bdate_range("2024-01-01", periods=30)
    benchmark = pd.Series(np.linspace(100.0, 110.0, len(dates)), index=dates)
    members = pd.DataFrame({"A": benchmark * 1.01}, index=dates)
    strategy = MultiAssetRotation(
        mom_lookback=20, mom_gap=2, regime=False, vix_scale=False
    )

    assert strategy._residual_momentum(members, benchmark) is None


def test_gbdt_walk_forward_predicts_live_tail_without_a_future_label(monkeypatch):
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    dates = pd.bdate_range("2024-01-01", periods=8)
    symbols = [f"S{i}" for i in range(6)]
    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    rng = np.random.default_rng(4)
    features = pd.DataFrame(
        {"x1": rng.normal(size=len(index)), "x2": rng.normal(size=len(index))},
        index=index,
    )
    labels = pd.Series(rng.normal(size=len(index)), index=index)
    labels.loc[(dates[-1], slice(None))] = np.nan

    model = GBDTFactor(
        model="sklearn", max_iter=5, min_samples_leaf=2, max_leaf_nodes=5
    )
    predictions = model.fit_predict_cross_sectional(
        features, labels, train_window=3, min_train_obs=12
    )

    assert predictions.loc[dates[-1]].notna().all()


def test_gbdt_purge_preserves_the_configured_training_window(monkeypatch):
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    dates = pd.bdate_range("2024-01-01", periods=8)
    symbols = [f"S{i}" for i in range(4)]
    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    rng = np.random.default_rng(9)
    features = pd.DataFrame({"x": rng.normal(size=len(index))}, index=index)
    labels = pd.Series(rng.normal(size=len(index)), index=index)
    model = GBDTFactor(
        model="sklearn", max_iter=3, min_samples_leaf=2, max_leaf_nodes=3
    )

    predictions = model.fit_predict_cross_sectional(
        features,
        labels,
        train_window=3,
        purge=2,
        min_train_obs=12,
    )

    assert predictions.iloc[:5].isna().all(axis=None)
    assert predictions.iloc[5].notna().all()


def test_gbdt_rejects_infinite_features():
    model = GBDTFactor(model="sklearn", max_iter=2)
    with pytest.raises(ValueError, match="infinite"):
        model.fit(pd.DataFrame({"x": [1.0, np.inf]}), pd.Series([0.0, 1.0]))


def test_kama_treats_price_equal_to_average_as_in_trend():
    prices = pd.Series(np.full(20, 100.0))
    signal = KAMA(er_window=5).trend_signal(prices)
    assert signal.iloc[5:].eq(1).all()


def test_tsmom_uses_compounded_return_sign_and_excludes_current_bar():
    overlay = TSMomentum(lookback=2)
    weights = np.array([1.0])
    # 90% then -60% compounds to -24%, although the arithmetic sum is +30%.
    result = overlay.apply(weights, returns=np.array([0.9, -0.6, 0.5]))
    assert result[0] == 0.0
    # With only the contemporaneous bar there is no prior signal to trade on.
    assert overlay.apply(weights, returns=np.array([0.5]))[0] == 0.0


def test_kelly_vector_rejects_unbounded_zero_variance_edge():
    model = KellyCriterion()
    with pytest.raises(ValueError, match="unbounded"):
        model.kelly_vector(
            np.array([0.01, 0.02]),
            np.array([[1.0, 0.0], [0.0, 0.0]]),
        )


def test_timing_overlays_reject_malformed_shapes_and_parameters():
    with pytest.raises(ValueError, match="1-D"):
        VIXScaler().apply(np.array([[1.0]]), 20.0)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        KAMA().apply(np.array([1.0]), np.ones((2, 2, 2)))
    with pytest.raises(TypeError, match="boolean"):
        TSMomentum(allow_short="false")
    with pytest.raises(TypeError, match="booleans"):
        KAMA(er_window=True)


def test_drl_risk_penalty_uses_portfolio_return_variance():
    returns = np.array(
        [
            [0.10, -0.10],
            [-0.10, 0.10],
            [0.10, -0.10],
            [-0.10, 0.10],
        ]
    )

    concentrated = _portfolio_variance(returns, np.array([1.0, 0.0]))
    hedged = _portfolio_variance(returns, np.array([0.5, 0.5]))

    assert concentrated > 0.0
    assert hedged == pytest.approx(0.0)


def test_trained_drl_policy_rejects_asset_schema_changes():
    class Model:
        def predict(self, obs, deterministic):
            return np.array([0.0, 0.0]), None

    allocator = DRLAllocator(window=2)
    allocator.model = Model()
    allocator._n_assets = 2
    allocator._asset_names = ("AAA", "BBB")
    returns = pd.DataFrame(
        [[0.01, 0.02], [0.02, 0.01]], columns=["BBB", "AAA"]
    )

    with pytest.raises(ValueError, match="asset schema"):
        allocator.optimize(returns)


def test_trained_drl_policy_requires_and_observes_previous_weights():
    class Model:
        observation = None

        def predict(self, obs, deterministic):
            self.observation = np.asarray(obs)
            return np.array([0.0, 0.0]), None

    allocator = DRLAllocator(window=2)
    allocator.model = Model()
    allocator._n_assets = 2
    allocator._asset_names = ("AAA", "BBB")
    returns = pd.DataFrame(
        [[0.01, 0.02], [0.02, 0.01]], columns=["AAA", "BBB"]
    )

    with pytest.raises(ValueError, match="previous_weights"):
        allocator.optimize(returns)

    weights = allocator.optimize(returns, previous_weights=[0.25, 0.50])

    assert weights.tolist() == pytest.approx([0.5, 0.5])
    assert allocator.model.observation.shape == (6,)
    assert allocator.model.observation[-2:].tolist() == pytest.approx([0.25, 0.50])


def test_drl_strategy_requires_labeled_complete_current_holdings():
    class Model:
        def predict(self, obs, deterministic):
            return np.array([0.0, 0.0]), None

    dates = pd.bdate_range("2024-01-01", periods=4)
    returns = pd.DataFrame(
        {"AAA": [0.01, 0.00, 0.02, 0.01], "BBB": [0.00, 0.01, 0.01, 0.02]},
        index=dates,
    )
    prices = (1.0 + returns).cumprod() * 100.0
    allocator = DRLAllocator(window=2)
    allocator.model = Model()
    allocator._n_assets = 2
    allocator._asset_names = ("AAA", "BBB")
    strategy = DRLPortfolioStrategy(optimizer=allocator, train_window=4)
    strategy._trained = True
    strategy._last_train = dates[-1]
    scores = pd.Series({"AAA": 1.0, "BBB": 1.0})

    unlabeled = StrategyContext(
        dates[-1], prices, returns, extra={"current_weights": [0.5, 0.5]}
    )
    with pytest.raises(ValueError, match="labeled numeric mapping"):
        strategy.allocate(scores, unlabeled)

    incomplete = StrategyContext(
        dates[-1],
        prices,
        returns,
        extra={"current_weights": {"AAA": 0.4, "BBB": 0.4, "CCC": 0.2}},
    )
    with pytest.raises(ValueError, match="outside its asset schema"):
        strategy.allocate(scores, incomplete)


def test_sentiment_strategy_rejects_ambiguous_precomputed_panel_order():
    dates = pd.bdate_range("2024-01-01", periods=3)
    returns = pd.DataFrame({"AAA": [0.0, 0.01, 0.02]}, index=dates)
    prices = (1.0 + returns).cumprod() * 100.0
    sentiment = pd.DataFrame(
        {"AAA": [0.1, 0.2]}, index=pd.DatetimeIndex([dates[1], dates[0]])
    )
    ctx = StrategyContext(
        dates[-1], prices, returns, extra={"sentiment": sentiment}
    )

    with pytest.raises(ValueError, match="sorted"):
        SentimentNLPStrategy()._sentiment_zscore(ctx, prices.columns)


def test_sentiment_strategy_uses_an_empty_selection_for_no_positive_signal(
    monkeypatch,
):
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame(
        {"AAA": [100.0, 99.0, 98.0], "BBB": [100.0, 99.5, 99.0]},
        index=dates,
    )
    ctx = StrategyContext(
        dates[-1],
        prices,
        prices.pct_change(fill_method=None).dropna(),
    )
    strategy = SentimentNLPStrategy()
    monkeypatch.setattr(
        strategy,
        "_momentum_zscore",
        lambda frame: pd.Series({"AAA": -1.0, "BBB": 0.0}),
    )
    monkeypatch.setattr(strategy, "_sentiment_zscore", lambda ctx, symbols: None)

    result = strategy.rebalance(ctx)

    assert result.symbols == []
    assert result.target_weights.size == 0


def test_untrained_drl_requires_explicit_heuristic_policy():
    returns = pd.DataFrame(
        [[0.01, 0.02], [0.02, 0.01]], columns=["AAA", "BBB"]
    )

    with pytest.raises(RuntimeError, match="no trained PPO model"):
        DRLAllocator(window=2).optimize(returns)
    weights = DRLAllocator(window=2, untrained_policy="heuristic").optimize(returns)
    assert weights.sum() == pytest.approx(1.0)


def test_multi_asset_regime_gate_is_flat_without_enough_history():
    dates = pd.bdate_range("2024-01-01", periods=20)
    returns = pd.DataFrame({"QQQ": np.full(20, 0.001)}, index=dates)
    prices = (1.0 + returns).cumprod() * 100.0
    ctx = StrategyContext(dates[-1], prices, returns)
    strategy = MultiAssetRotation(regime=True, vix_scale=False)

    gated = strategy._regime_overlay(np.array([1.0]), ctx)

    assert gated.tolist() == [0.0]


def test_multi_asset_regime_model_failure_stops_the_run(monkeypatch):
    dates = pd.bdate_range("2023-01-01", periods=100)
    returns = pd.DataFrame(
        {"QQQ": np.sin(np.arange(100)) * 0.01}, index=dates
    )
    prices = (1.0 + returns).cumprod() * 100.0
    ctx = StrategyContext(dates[-1], prices, returns)
    strategy = MultiAssetRotation(regime=True, vix_scale=False)

    def fail(_features):
        raise RuntimeError("regime fit failed")

    monkeypatch.setattr(strategy._hmm, "fit", fail)
    with pytest.raises(RuntimeError, match="regime fit failed"):
        strategy._regime_overlay(np.array([1.0]), ctx)


def test_strategy_regime_backends_do_not_depend_on_optional_packages():
    assert MultiAssetRotation()._hmm.backend == "gmm"
    assert MacroTimingStrategy()._hmm.backend == "gmm"


def test_regime_fit_failure_does_not_switch_model_class(monkeypatch):
    class FailingModel:
        def fit(self, _features):
            raise ValueError("bad fit")

    model = HMMRegime(backend="hmm")
    monkeypatch.setattr(model, "_build_model", lambda: (FailingModel(), "hmm"))
    features = pd.DataFrame(
        {
            "returns": [0.0, 0.01, -0.01],
            "realized_vol": [0.1, 0.1, 0.2],
            "vix": [15.0, 16.0, 25.0],
        }
    )

    with pytest.raises(RuntimeError, match="hmm backend failed"):
        model.fit(features)
    assert model.backend_ is None


def test_regime_model_rejects_ambiguous_clock_and_integer_parameters():
    features = pd.DataFrame(
        {
            "returns": [0.0, 0.01],
            "realized_vol": [0.1, 0.2],
            "vix": [15.0, 20.0],
        },
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-01"]),
    )

    with pytest.raises(ValueError, match="sorted"):
        HMMRegime().fit(features)
    with pytest.raises(ValueError, match="n_iter"):
        HMMRegime(n_iter=1.5)
    with pytest.raises(ValueError, match="seed"):
        HMMRegime(seed=True)
    with pytest.raises(ValueError, match="reg_covar"):
        HMMRegime(reg_covar=-1.0)
    with pytest.raises(ValueError, match="reg_covar"):
        HMMRegime(reg_covar="invalid")


def test_regime_state_labels_rank_observed_return_means():
    states = np.array([0, 0, 1, 1, 2, 2])
    returns = np.array([0.00, 0.01, 0.08, 0.10, -0.08, -0.06])

    labels = HMMRegime(n_states=3)._label_states(states, returns)

    assert labels == {0: SIDEWAYS, 1: BULL, 2: BEAR}


def test_regime_state_labels_leave_unobserved_states_neutral():
    states = np.array([0, 0, 2, 2])
    returns = np.array([-0.05, -0.03, 0.04, 0.06])

    labels = HMMRegime(n_states=3)._label_states(states, returns)

    assert labels == {0: BEAR, 1: SIDEWAYS, 2: BULL}


def test_regime_state_labels_leave_tied_state_means_neutral():
    states = np.array([0, 0, 1, 1, 2, 2])
    returns = np.array([-0.01, 0.01, -0.02, 0.02, -0.03, 0.03])

    labels = HMMRegime(n_states=3)._label_states(states, returns)

    assert labels == {0: SIDEWAYS, 1: SIDEWAYS, 2: SIDEWAYS}


def test_rotation_strategy_rejects_invalid_volatility_lookbacks():
    with pytest.raises(ValueError, match="regime_feature_vol_lookback"):
        MultiAssetRotation(regime_feature_vol_lookback=1)
    with pytest.raises(ValueError, match="vix_proxy_lookback"):
        MultiAssetRotation(vix_proxy_lookback=1)


def test_regime_features_honor_configured_volatility_lookback():
    returns = pd.Series([0.00, 0.02, 0.06])

    features = HMMRegime._features_from_returns(
        returns,
        realized_vol_lookback=2,
    )

    assert features.loc[2, "realized_vol"] == pytest.approx(
        returns.iloc[-2:].std()
    )
    with pytest.raises(ValueError, match="realized_vol_lookback"):
        HMMRegime._features_from_returns(returns, realized_vol_lookback=1)


def test_rotation_strategy_honors_two_session_proxy_lookback():
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"QQQ": [100.0, 101.0, 104.03]}, index=dates)
    returns = prices.pct_change(fill_method=None).dropna()
    ctx = StrategyContext(dates[-1], prices, returns)

    proxy = MultiAssetRotation(vix_proxy_lookback=2)._realized_vol_proxy(ctx)

    expected = returns["QQQ"].std(ddof=0) * np.sqrt(252.0) * 100.0
    assert proxy == pytest.approx(expected)


def test_macro_strategy_treats_missing_regime_history_as_bear():
    dates = pd.bdate_range("2024-01-01", periods=20)
    returns = pd.DataFrame(
        {"SPY": np.full(20, 0.01), "XLP": np.full(20, 0.001)}, index=dates
    )
    prices = (1.0 + returns).cumprod() * 100.0
    ctx = StrategyContext(dates[-1], prices, returns)

    scores = MacroTimingStrategy(regime=True).select(ctx)

    assert scores.index.tolist() == ["XLP"]


def test_macro_strategy_slices_explicit_features_at_decision_time():
    dates = pd.bdate_range("2024-01-01", periods=3)
    returns = pd.DataFrame({"SPY": [0.0, 0.01]}, index=dates[:2])
    prices = (1.0 + returns).cumprod() * 100.0
    macro = pd.DataFrame(
        {
            "returns": [0.0, 0.01, 9.0],
            "realized_vol": [0.1, 0.1, 9.0],
            "vix": [15.0, 16.0, 99.0],
        },
        index=dates,
    )
    ctx = StrategyContext(dates[1], prices, returns, extra={"macro": macro})

    features = MacroTimingStrategy()._regime_features(ctx)

    assert features.index.max() == dates[1]
    assert 9.0 not in features.to_numpy()


def test_hmm_context_features_are_sliced_at_decision_time():
    dates = pd.bdate_range("2024-01-01", periods=3)
    returns = pd.DataFrame({"SPY": [0.0, 0.01]}, index=dates[:2])
    prices = (1.0 + returns).cumprod() * 100.0
    regime_features = pd.DataFrame(
        {
            "returns": [0.0, 0.01, 9.0],
            "realized_vol": [0.1, 0.1, 9.0],
            "vix": [15.0, 16.0, 99.0],
        },
        index=dates,
    )
    ctx = StrategyContext(
        dates[1], prices, returns, extra={"regime_features": regime_features}
    )

    features = HMMRegime()._coerce_features(ctx)

    assert features.index.max() == dates[1]
    assert 9.0 not in features.to_numpy()


def test_news_scorer_rejects_negative_lookback():
    news = pd.DataFrame(
        {"date": ["2024-01-01"], "symbol": ["AAA"], "headline": ["profit"]}
    )
    with pytest.raises(ValueError, match="non-negative"):
        NewsScorer().aggregate_daily(news, lookback_days=-1)


def test_ml_models_reject_misaligned_targets():
    X = pd.DataFrame({"x": [1.0, 2.0]}, index=["a", "b"])
    y = pd.Series([0.1, 0.2], index=["b", "a"])

    with pytest.raises(ValueError, match="index must exactly match"):
        GBDTFactor(model="sklearn", max_iter=2).fit(X, y)
    with pytest.raises(ValueError, match="index must exactly match"):
        NeuralFactor(epochs=1).fit(X, y)


def test_neural_factor_does_not_silently_switch_after_torch_runtime_error(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    model = NeuralFactor(backend="torch", epochs=1)
    monkeypatch.setattr(
        model,
        "_fit_torch",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("training failed")),
    )
    X = pd.DataFrame({"x": [1.0, 2.0]})
    y = pd.Series([0.1, 0.2])

    with pytest.raises(RuntimeError, match="training failed"):
        model.fit(X, y)
    assert model.backend_ is None


def test_finbert_runtime_error_does_not_silently_change_backend(monkeypatch):
    scorer = FinBERTSentiment(backend="auto")
    monkeypatch.setattr(
        scorer,
        "_score_transformers",
        lambda _texts: (_ for _ in ()).throw(RuntimeError("bad inference")),
    )

    with pytest.raises(RuntimeError, match="bad inference"):
        scorer.score(["profits rise"])
    assert scorer.backend_ is None


def test_news_scorer_normalizes_timezone_and_deduplicates_headlines():
    news = pd.DataFrame(
        {
            "date": ["2024-01-02T20:00:00-05:00"] * 2,
            "symbol": ["AAA", "AAA"],
            "headline": ["profits rise", "profits rise"],
        }
    )
    panel = NewsScorer().aggregate_daily(news, as_of="2024-01-03T02:00:00Z")

    assert panel.index.tolist() == [pd.Timestamp("2024-01-03")]
    assert panel.loc[pd.Timestamp("2024-01-03"), "AAA"] > 0.0


def test_news_scorer_excludes_later_headline_on_same_day():
    news = pd.DataFrame(
        {
            "date": ["2024-01-03T10:00:00Z", "2024-01-03T16:00:00Z"],
            "symbol": ["AAA", "AAA"],
            "headline": ["profits rise", "loss warning"],
        }
    )
    panel = NewsScorer().aggregate_daily(news, as_of="2024-01-03T12:00:00Z")

    assert panel.loc[pd.Timestamp("2024-01-03"), "AAA"] > 0.0


def test_covariance_optimizer_drops_incomplete_rows_instead_of_filling():
    returns = pd.DataFrame(
        {
            "A": [0.01, np.nan, 0.03],
            "B": [0.02, 0.04, 0.01],
        }
    )

    clean = MeanVariance._clean(returns)

    assert clean.index.tolist() == [0, 2]
    assert clean.loc[2, "A"] == pytest.approx(0.03)


def test_covariance_optimizer_rejects_no_complete_sample():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, np.nan, np.nan],
            "B": [np.nan, np.nan, 0.01, 0.02],
        }
    )
    with pytest.raises(ValueError, match="complete observations"):
        MeanVariance().optimize(returns)


@pytest.mark.parametrize(
    "optimizer",
    [MeanVariance(), MinimumVariance(), RiskParity()],
)
def test_covariance_optimizers_reject_nonnumeric_dead_assets(optimizer):
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03],
            "BROKEN": ["bad", "data", "column"],
        }
    )

    with pytest.raises(ValueError, match="non-numeric"):
        optimizer.optimize(returns)


def test_bounded_sum_projection_enforces_feasibility_without_renormalization():
    projected = project_bounded_sum(
        np.array([0.9, 0.1]), target_sum=1.0, lower=0.2, upper=0.8
    )
    assert projected == pytest.approx([0.8, 0.2])

    with pytest.raises(WeightContractViolationError, match="infeasible"):
        project_bounded_sum(
            np.ones(3), target_sum=1.0, lower=0.0, upper=0.3
        )


@pytest.mark.parametrize(
    "optimizer",
    [
        MeanVariance(weight_bounds=(0.1, 0.7)),
        MinimumVariance(weight_bounds=(0.1, 0.7)),
        RiskParity(weight_bounds=(0.1, 0.7)),
        HierarchicalRiskParity(weight_bounds=(0.1, 0.7)),
        BlackLitterman(weight_bounds=(0.1, 0.7)),
        DRLAllocator(
            window=5,
            weight_bounds=(0.1, 0.7),
            untrained_policy="heuristic",
        ),
    ],
)
def test_portfolio_optimizers_honor_feasible_custom_bounds(optimizer):
    rng = np.random.default_rng(123)
    returns = pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(120, 3)), columns=["A", "B", "C"]
    )

    weights = optimizer.optimize(returns)

    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights >= 0.1 - 1e-8)
    assert np.all(weights <= 0.7 + 1e-8)


def test_black_litterman_without_views_reproduces_supplied_market_prior():
    rng = np.random.default_rng(321)
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(120, 3)), columns=["A", "B", "C"]
    )
    market = pd.Series([0.6, 0.3, 0.1], index=returns.columns)

    weights = BlackLitterman().optimize(returns, market_weights=market)

    assert weights == pytest.approx(market.to_numpy())


def test_minimum_variance_rejects_undefined_market_neutral_problem():
    with pytest.raises(ValueError, match="additional exposure or return constraint"):
        MinimumVariance(mode=PortfolioMode.MARKET_NEUTRAL)


def test_black_litterman_rejects_invalid_view_confidence():
    returns = pd.DataFrame(
        {"A": [0.01, 0.02, 0.00], "B": [0.00, 0.01, 0.02]}
    )
    with pytest.raises(ValueError, match="strictly between"):
        BlackLitterman().optimize(
            returns,
            views=np.array([[1.0, -1.0]]),
            q=np.array([0.01]),
            view_confidences=np.array([1.0]),
        )


def test_black_litterman_rejects_unknown_or_empty_view_assets():
    returns = pd.DataFrame(
        {"A": [0.01, 0.02, 0.00], "B": [0.00, 0.01, 0.02]}
    )
    model = BlackLitterman()

    with pytest.raises(ValueError, match="unknown assets"):
        model.optimize(
            returns,
            views=pd.DataFrame([[1.0]], columns=["TYPO"]),
            q=np.array([0.01]),
        )
    with pytest.raises(ValueError, match="at least one asset"):
        model.optimize(
            returns,
            views=np.zeros((1, 2)),
            q=np.array([0.01]),
        )


def test_hrp_rejects_zero_variance_assets():
    returns = pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.0],
            "B": [0.01, -0.01, 0.02],
        }
    )
    with pytest.raises(ValueError, match="positive variance"):
        HierarchicalRiskParity().optimize(returns)
