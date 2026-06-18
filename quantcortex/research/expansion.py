"""Frozen strategy and evaluation primitives for the prospective expansion."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantcortex.backtest.conformance import (
    target_tape_from_payload,
    target_tape_to_payload,
    target_tape_to_weights,
    weights_to_target_tape,
)
from quantcortex.backtest.costs.transaction_costs import TransactionCostModel
from quantcortex.backtest.engines.event_driven import EventDrivenBacktest
from quantcortex.backtest.engines.vectorized import BacktestResult, VectorizedBacktest

PERIODS_PER_YEAR = 252
FROZEN_PROTOCOL_COMMIT = "4018f4063f46889f41d6981db5a71079e1dbd713"
FROZEN_PROTOCOL_SHA256 = (
    "e49e41a12a19fa5404a573ba5e21eb8a2888e616985f8c610d9652866923315c"
)


@dataclass(frozen=True)
class LearnedTargetResult:
    """Walk-forward learned targets and fit diagnostics for one random seed."""

    weights: pd.DataFrame
    training_rows: pd.Series
    training_months: pd.Series


def validate_price_panel(
    prices: pd.DataFrame,
    *,
    risky_symbols: Sequence[str],
    cash_symbol: str,
) -> pd.DataFrame:
    """Validate a complete, positive daily adjusted-close panel."""
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty DataFrame")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must use a DatetimeIndex")
    if prices.index.hasnans or prices.index.has_duplicates:
        raise ValueError("prices index must contain unique valid timestamps")
    if prices.columns.has_duplicates:
        raise ValueError("prices columns must be unique")
    symbols = _validated_symbols(risky_symbols)
    if not isinstance(cash_symbol, str) or not cash_symbol.strip():
        raise ValueError("cash_symbol must be a non-empty string")
    cash_symbol = cash_symbol.strip()
    if cash_symbol in symbols:
        raise ValueError("cash_symbol must not be a risky symbol")
    required = [*symbols, cash_symbol]
    missing = [symbol for symbol in required if symbol not in prices.columns]
    if missing:
        raise ValueError(f"price panel is missing symbols: {missing}")

    normalized = prices.loc[:, required].copy().sort_index()
    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_convert("UTC").tz_localize(None)
    normalized = normalized.apply(pd.to_numeric, errors="coerce").astype("float64")
    values = normalized.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("price panel must be complete and finite")
    if np.any(values <= 0.0):
        raise ValueError("price panel must be strictly positive")
    return normalized


def monthly_decision_dates(
    index: pd.DatetimeIndex,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """Return the first observed panel session in each calendar month."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("index must be sorted with unique valid timestamps")
    selected = index
    if start is not None:
        selected = selected[selected >= pd.Timestamp(start)]
    if end is not None:
        selected = selected[selected <= pd.Timestamp(end)]
    if selected.empty:
        raise ValueError("decision window contains no sessions")
    periods = selected.to_period("M")
    first = ~periods.duplicated(keep="first")
    return pd.DatetimeIndex(selected[first])


def time_series_momentum_targets(
    prices: pd.DataFrame,
    *,
    symbols: Sequence[str],
    decisions: pd.DatetimeIndex,
    lookback: int = 252,
) -> pd.DataFrame:
    """Equal-weight assets with positive trailing total return."""
    symbols = _validated_symbols(symbols)
    _validate_lookback(lookback)
    panel = _risky_panel(prices, symbols)
    rows = []
    for decision in decisions:
        position = _decision_position(panel.index, decision, lookback)
        signal = panel.iloc[position] / panel.iloc[position - lookback] - 1.0
        selected = sorted(symbol for symbol in symbols if signal[symbol] > 0.0)
        row = pd.Series(0.0, index=symbols, dtype=float)
        if selected:
            row.loc[selected] = 1.0 / len(selected)
        rows.append(row)
    return _canonical_targets(rows, decisions, symbols)


def cross_sectional_momentum_targets(
    prices: pd.DataFrame,
    *,
    symbols: Sequence[str],
    decisions: pd.DatetimeIndex,
    lookback_start: int = 252,
    skip_recent: int = 21,
    selection_count: int = 3,
) -> pd.DataFrame:
    """Select the strongest long-horizon returns with a recent-month gap."""
    symbols = _validated_symbols(symbols)
    _validate_lookback(lookback_start)
    _validate_selection_count(selection_count, len(symbols))
    if not isinstance(skip_recent, int) or not 0 < skip_recent < lookback_start:
        raise ValueError("skip_recent must be in (0, lookback_start)")
    panel = _risky_panel(prices, symbols)
    rows = []
    for decision in decisions:
        position = _decision_position(panel.index, decision, lookback_start)
        signal = (
            panel.iloc[position - skip_recent]
            / panel.iloc[position - lookback_start]
            - 1.0
        )
        selected = _ordered_symbols(signal, descending=True)[:selection_count]
        row = pd.Series(0.0, index=symbols, dtype=float)
        row.loc[selected] = 1.0 / selection_count
        rows.append(row)
    return _canonical_targets(rows, decisions, symbols)


def short_term_reversal_targets(
    prices: pd.DataFrame,
    *,
    symbols: Sequence[str],
    decisions: pd.DatetimeIndex,
    lookback: int = 5,
    selection_count: int = 3,
) -> pd.DataFrame:
    """Select up to three assets with the lowest negative short-term return."""
    symbols = _validated_symbols(symbols)
    _validate_lookback(lookback)
    _validate_selection_count(selection_count, len(symbols))
    panel = _risky_panel(prices, symbols)
    rows = []
    for decision in decisions:
        position = _decision_position(panel.index, decision, lookback)
        trailing = panel.iloc[position] / panel.iloc[position - lookback] - 1.0
        eligible = trailing[trailing < 0.0]
        selected = _ordered_symbols(eligible, descending=False)[:selection_count]
        row = pd.Series(0.0, index=symbols, dtype=float)
        row.loc[selected] = 1.0 / selection_count
        rows.append(row)
    return _canonical_targets(rows, decisions, symbols)


def learned_gbrt_targets(
    prices: pd.DataFrame,
    *,
    symbols: Sequence[str],
    all_decisions: pd.DatetimeIndex,
    evaluation_decisions: pd.DatetimeIndex,
    config: Mapping[str, object],
    seed: int,
) -> LearnedTargetResult:
    """Fit the frozen pooled walk-forward GBRT and emit causal monthly targets."""
    from sklearn.ensemble import GradientBoostingRegressor
    from threadpoolctl import threadpool_limits

    symbols = _validated_symbols(symbols)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    panel = _risky_panel(prices, symbols)
    features, labels = _learned_feature_table(
        panel,
        symbols=symbols,
        decisions=all_decisions,
        return_windows=_integer_sequence(config, "feature_return_windows"),
        volatility_windows=_integer_sequence(
            config,
            "feature_volatility_windows",
        ),
        label_horizon=_positive_int(config, "label_horizon_sessions"),
    )
    maximum_months = _positive_int(config, "training_decision_months")
    minimum_months = _positive_int(config, "minimum_training_decision_months")
    selection_count = _positive_int(config, "selection_count")
    _validate_selection_count(selection_count, len(symbols))
    if minimum_months > maximum_months:
        raise ValueError("minimum training months exceed the rolling window")

    evaluation_set = set(pd.DatetimeIndex(evaluation_decisions))
    rows = []
    row_index = []
    training_rows: dict[pd.Timestamp, int] = {}
    training_months: dict[pd.Timestamp, int] = {}
    for decision in all_decisions:
        decision = pd.Timestamp(decision)
        if decision not in evaluation_set:
            continue
        current = features.xs(decision, level="decision_timestamp")
        candidates = labels.loc[labels["label_end"] <= decision]
        candidate_dates = pd.DatetimeIndex(
            candidates.index.get_level_values("decision_timestamp").unique()
        ).sort_values()
        selected_dates = candidate_dates[-maximum_months:]
        if len(selected_dates) < minimum_months:
            raise ValueError(
                f"learned model has only {len(selected_dates)} mature months "
                f"at {decision.date().isoformat()}"
            )
        train_index = candidates.index[
            candidates.index.get_level_values("decision_timestamp").isin(
                selected_dates
            )
        ]
        train_x = features.loc[train_index]
        train_y = candidates.loc[train_index, "label"]
        if train_x.isna().any(axis=None) or train_y.isna().any():
            raise ValueError("learned training data must be complete")
        model = GradientBoostingRegressor(
            n_estimators=_positive_int(config, "n_estimators"),
            learning_rate=_positive_float(config, "learning_rate"),
            max_depth=_positive_int(config, "max_depth"),
            min_samples_leaf=_positive_int(config, "min_samples_leaf"),
            subsample=_unit_interval(config, "subsample"),
            random_state=seed,
            loss="squared_error",
        )
        with threadpool_limits(limits=1):
            model.fit(train_x.to_numpy(dtype=float), train_y.to_numpy(dtype=float))
        predictions = pd.Series(
            model.predict(current.to_numpy(dtype=float)),
            index=current.index,
            dtype=float,
        )
        positive = predictions[predictions > 0.0]
        selected = _ordered_symbols(positive, descending=True)[:selection_count]
        row = pd.Series(0.0, index=symbols, dtype=float)
        row.loc[selected] = 1.0 / selection_count
        rows.append(row)
        row_index.append(decision)
        training_rows[decision] = int(len(train_x))
        training_months[decision] = int(len(selected_dates))

    if not rows:
        raise ValueError("learned model produced no evaluation decisions")
    weights = _canonical_targets(
        rows,
        pd.DatetimeIndex(row_index),
        symbols,
    )
    return LearnedTargetResult(
        weights=weights,
        training_rows=pd.Series(training_rows, name="training_rows"),
        training_months=pd.Series(training_months, name="training_months"),
    )


def run_engine(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    cash_returns: pd.Series,
    *,
    cost_rate: float,
    engine: str,
) -> BacktestResult:
    """Run one frozen cost/cash/engine configuration."""
    if not np.isfinite(cost_rate) or not 0.0 <= cost_rate < 1.0:
        raise ValueError("cost_rate must be finite and in [0, 1)")
    cost_model = TransactionCostModel(commission=0.0, slippage=cost_rate)
    if engine == "event_driven":
        runner = EventDrivenBacktest(cost_model, capital=1.0)
    elif engine == "vectorized":
        runner = VectorizedBacktest(cost_model, capital=1.0)
    else:
        raise ValueError("engine must be 'event_driven' or 'vectorized'")
    return runner.run(weights, prices, cash_returns=cash_returns)


def invalid_same_close_result(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    cash_returns: pd.Series,
    *,
    cost_rate: float,
) -> BacktestResult:
    """Deliberately apply close-derived targets to the return ending there."""
    if not isinstance(weights.index, pd.DatetimeIndex):
        raise TypeError("weights must use a DatetimeIndex")
    decisions = weights.sort_index().reindex(columns=prices.columns, fill_value=0.0)
    positions = prices.index.searchsorted(decisions.index, side="left")
    keep = positions < len(prices)
    decisions = decisions.iloc[keep].copy()
    execution_positions = np.maximum(positions[keep] - 1, 0)
    decisions.index = pd.DatetimeIndex(
        [
            prices.index[position] - pd.Timedelta(microseconds=1)
            for position in execution_positions
        ]
    )
    decisions = decisions[~decisions.index.duplicated(keep="last")]
    return run_engine(
        decisions,
        prices,
        cash_returns,
        cost_rate=cost_rate,
        engine="event_driven",
    )


def exposure_matched_comparator_targets(weights: pd.DataFrame) -> pd.DataFrame:
    """Allocate each target's risky gross exposure equally across its panel."""
    if not isinstance(weights, pd.DataFrame) or weights.empty:
        raise ValueError("weights must be a non-empty DataFrame")
    if weights.shape[1] == 0 or weights.columns.has_duplicates:
        raise ValueError("weights must have unique, non-empty columns")
    normalized = weights.apply(pd.to_numeric, errors="coerce").astype("float64")
    values = normalized.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("target weights must be finite")
    if np.any(values < -1e-12):
        raise ValueError("target weights must be long-only")
    normalized = normalized.clip(lower=0.0)
    exposure = normalized.sum(axis=1)
    if (exposure > 1.0 + 1e-12).any():
        raise ValueError("target exposure must remain in [0, 1]")
    comparator = pd.DataFrame(
        np.repeat((exposure / weights.shape[1]).to_numpy()[:, None], weights.shape[1], axis=1),
        index=weights.index,
        columns=weights.columns,
    )
    return _canonical_targets(
        [row for _, row in comparator.iterrows()],
        comparator.index,
        list(comparator.columns),
    )


def performance_metrics(
    result: BacktestResult,
    cash_returns: pd.Series,
    evaluation_index: pd.DatetimeIndex,
) -> dict[str, float | int]:
    """Compute the frozen daily performance estimands."""
    returns = result.returns.reindex(evaluation_index)
    cash = cash_returns.reindex(evaluation_index)
    if returns.isna().any() or cash.isna().any():
        raise ValueError("returns and cash must cover the evaluation window")
    return_values = returns.to_numpy(dtype=float)
    cash_values = cash.to_numpy(dtype=float)
    if not np.all(np.isfinite(return_values)) or not np.all(np.isfinite(cash_values)):
        raise ValueError("returns and cash must be finite")
    if np.any(return_values <= -1.0):
        raise ValueError("returns must remain above -100 percent")
    excess = returns - cash
    years = len(returns) / PERIODS_PER_YEAR
    growth = (1.0 + returns).cumprod()
    total_return = float(growth.iloc[-1] - 1.0)
    cagr = float(growth.iloc[-1] ** (1.0 / years) - 1.0)
    drawdown = growth / growth.cummax().clip(lower=1.0) - 1.0
    standard_deviation = float(excess.std(ddof=1))
    sharpe = (
        float(excess.mean() / standard_deviation * np.sqrt(PERIODS_PER_YEAR))
        if standard_deviation > 0.0
        else float("nan")
    )
    active_exposure = result.weights.shift(1).reindex(evaluation_index).fillna(0.0)
    return {
        "observations": int(len(returns)),
        "annualized_arithmetic_return": float(returns.mean() * PERIODS_PER_YEAR),
        "cash_excess_sharpe": sharpe,
        "cagr": cagr,
        "total_return": total_return,
        "max_drawdown": float(drawdown.min()),
        "annualized_volatility": float(returns.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)),
        "annualized_one_way_turnover": float(
            result.turnover.reindex(evaluation_index).mean() * PERIODS_PER_YEAR
        ),
        "annualized_gross_traded_notional": float(
            result.traded_notional.reindex(evaluation_index).mean()
            * PERIODS_PER_YEAR
        ),
        "arithmetic_cost_drag": float(
            result.costs.reindex(evaluation_index).sum()
        ),
        "mean_risky_exposure": float(active_exposure.abs().sum(axis=1).mean()),
    }


def circular_block_indices(
    observations: int,
    *,
    block_length: int,
    replications: int,
    seed: int,
) -> np.ndarray:
    """Return reproducible circular-block row indices."""
    if isinstance(observations, bool) or not isinstance(observations, int):
        raise TypeError("observations must be an integer")
    if observations < 2:
        raise ValueError("at least two observations are required")
    if isinstance(block_length, bool) or not isinstance(block_length, int):
        raise TypeError("block_length must be an integer")
    if not 0 < block_length <= observations:
        raise ValueError("block_length must be in [1, observations]")
    if isinstance(replications, bool) or not isinstance(replications, int):
        raise TypeError("replications must be an integer")
    if replications <= 0:
        raise ValueError("replications must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(observations / block_length))
    starts = rng.integers(0, observations, size=(replications, block_count))
    offsets = np.arange(block_length)
    return ((starts[:, :, None] + offsets) % observations).reshape(
        replications,
        -1,
    )[:, :observations]


def bootstrap_metric_difference(
    lhs_returns: Sequence[pd.Series],
    rhs_returns: Sequence[pd.Series],
    cash_returns: pd.Series,
    *,
    block_length: int,
    replications: int,
    seed: int,
    chunk_size: int = 100,
) -> dict[str, float | int]:
    """Joint block intervals for family-mean return and Sharpe differences."""
    if len(lhs_returns) == 0 or len(lhs_returns) != len(rhs_returns):
        raise ValueError("lhs_returns and rhs_returns must have equal non-zero size")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    aligned = pd.concat(
        {
            **{f"lhs_{i}": series for i, series in enumerate(lhs_returns)},
            **{f"rhs_{i}": series for i, series in enumerate(rhs_returns)},
            "cash": cash_returns,
        },
        axis=1,
        join="inner",
    )
    if aligned.isna().any(axis=None) or len(aligned) < 2:
        raise ValueError("bootstrap inputs must be complete with at least two rows")
    values = aligned.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("bootstrap inputs must be finite")
    family_size = len(lhs_returns)
    lhs = values[:, :family_size].T
    rhs = values[:, family_size : 2 * family_size].T
    cash = values[:, -1]
    indices = circular_block_indices(
        len(aligned),
        block_length=block_length,
        replications=replications,
        seed=seed,
    )
    annualized = np.empty(replications, dtype=float)
    sharpe = np.empty(replications, dtype=float)
    for start in range(0, replications, chunk_size):
        stop = min(replications, start + chunk_size)
        sample = indices[start:stop]
        lhs_sample = lhs[:, sample]
        rhs_sample = rhs[:, sample]
        cash_sample = cash[sample]
        annualized[start:stop] = (
            (lhs_sample.mean(axis=2) - rhs_sample.mean(axis=2)).mean(axis=0)
            * PERIODS_PER_YEAR
        )
        lhs_excess = lhs_sample - cash_sample[None, :, :]
        rhs_excess = rhs_sample - cash_sample[None, :, :]
        lhs_sharpe = _sample_sharpe(lhs_excess)
        rhs_sharpe = _sample_sharpe(rhs_excess)
        sharpe[start:stop] = (lhs_sharpe - rhs_sharpe).mean(axis=0)
    if not np.all(np.isfinite(sharpe)):
        raise ValueError("bootstrap Sharpe difference is undefined")
    annualized_point = float(
        (lhs.mean(axis=1) - rhs.mean(axis=1)).mean() * PERIODS_PER_YEAR
    )
    lhs_point_sharpe = _sample_sharpe((lhs - cash[None, :])[:, None, :])[:, 0]
    rhs_point_sharpe = _sample_sharpe((rhs - cash[None, :])[:, None, :])[:, 0]
    if not np.all(np.isfinite(lhs_point_sharpe)) or not np.all(
        np.isfinite(rhs_point_sharpe)
    ):
        raise ValueError("point Sharpe difference is undefined")
    sharpe_point = float((lhs_point_sharpe - rhs_point_sharpe).mean())
    annualized_interval = np.quantile(annualized, [0.025, 0.975])
    sharpe_interval = np.quantile(sharpe, [0.025, 0.975])
    return {
        "observations": int(len(aligned)),
        "family_size": int(family_size),
        "block_length": int(block_length),
        "replications": int(replications),
        "seed": int(seed),
        "annualized_mean_difference": annualized_point,
        "annualized_mean_ci_95_lower": float(annualized_interval[0]),
        "annualized_mean_ci_95_upper": float(annualized_interval[1]),
        "sharpe_difference": sharpe_point,
        "sharpe_ci_95_lower": float(sharpe_interval[0]),
        "sharpe_ci_95_upper": float(sharpe_interval[1]),
    }


def _learned_feature_table(
    prices: pd.DataFrame,
    *,
    symbols: Sequence[str],
    decisions: pd.DatetimeIndex,
    return_windows: Sequence[int],
    volatility_windows: Sequence[int],
    label_horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    maximum_window = max(*return_windows, *volatility_windows, 252)
    log_prices = np.log(prices)
    daily_log_returns = log_prices.diff()
    feature_rows = []
    label_rows = []
    index_rows = []
    for decision in decisions:
        position = prices.index.get_indexer([decision])[0]
        if position < maximum_window:
            continue
        raw: dict[str, pd.Series] = {}
        for window in return_windows:
            raw[f"log_return_{window}"] = (
                log_prices.iloc[position] - log_prices.iloc[position - window]
            )
        for window in volatility_windows:
            raw[f"log_volatility_{window}"] = daily_log_returns.iloc[
                position - window + 1 : position + 1
            ].std(ddof=1)
        trailing_high = prices.iloc[position - 251 : position + 1].max(axis=0)
        raw["trailing_high_distance_252"] = (
            prices.iloc[position] / trailing_high - 1.0
        )
        raw["rank_log_return_21"] = _normalized_ordinal_rank(
            raw["log_return_21"]
        )
        raw["rank_log_return_252"] = _normalized_ordinal_rank(
            raw["log_return_252"]
        )
        feature_frame = pd.DataFrame(raw, index=symbols)
        if feature_frame.isna().any(axis=None):
            raise ValueError("learned feature construction produced missing values")
        label_end_position = position + label_horizon
        has_label = label_end_position < len(prices)
        if has_label:
            label = log_prices.iloc[label_end_position] - log_prices.iloc[position]
            label_end = prices.index[label_end_position]
        for symbol in symbols:
            index_rows.append((pd.Timestamp(decision), symbol))
            feature_rows.append(feature_frame.loc[symbol].to_dict())
            if has_label:
                label_rows.append(
                    {
                        "decision_timestamp": pd.Timestamp(decision),
                        "symbol": symbol,
                        "label": float(label[symbol]),
                        "label_end": pd.Timestamp(label_end),
                    }
                )
    if not feature_rows or not label_rows:
        raise ValueError("insufficient history for learned feature construction")
    feature_index = pd.MultiIndex.from_tuples(
        index_rows,
        names=["decision_timestamp", "symbol"],
    )
    features = pd.DataFrame(feature_rows, index=feature_index).sort_index()
    labels = (
        pd.DataFrame(label_rows)
        .set_index(["decision_timestamp", "symbol"])
        .sort_index()
    )
    return features, labels


def _canonical_targets(
    rows: Sequence[pd.Series],
    decisions: pd.DatetimeIndex,
    symbols: Sequence[str],
) -> pd.DataFrame:
    if len(rows) != len(decisions):
        raise ValueError("target rows and decisions must have equal length")
    weights = pd.DataFrame(rows, index=decisions, columns=symbols, dtype=float)
    tape = weights_to_target_tape(weights, max_gross=1.0)
    payload = target_tape_to_payload(
        tape,
        max_gross=1.0,
        expected_symbols=list(symbols),
    )
    decoded = json.loads(json.dumps(payload, allow_nan=False, sort_keys=True))
    restored = target_tape_to_weights(
        target_tape_from_payload(decoded),
        max_gross=1.0,
        expected_symbols=list(symbols),
    )
    restored.index.name = None
    return restored


def _risky_panel(prices: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    missing = [symbol for symbol in symbols if symbol not in prices.columns]
    if missing:
        raise ValueError(f"prices are missing risky symbols: {missing}")
    panel = prices.loc[:, list(symbols)].copy()
    if panel.isna().any(axis=None) or not np.all(
        np.isfinite(panel.to_numpy(dtype=float))
    ):
        raise ValueError("risky price panel must be complete and finite")
    if (panel <= 0.0).any(axis=None):
        raise ValueError("risky prices must be strictly positive")
    return panel


def _validated_symbols(symbols: Sequence[str]) -> list[str]:
    if isinstance(symbols, (str, bytes)):
        raise TypeError("symbols must be a sequence")
    parsed = list(symbols)
    if not parsed or any(not isinstance(symbol, str) or not symbol.strip() for symbol in parsed):
        raise ValueError("symbols must contain non-empty strings")
    parsed = [symbol.strip() for symbol in parsed]
    if len(parsed) != len(set(parsed)):
        raise ValueError("symbols must be unique")
    return parsed


def _decision_position(
    index: pd.DatetimeIndex,
    decision: pd.Timestamp,
    required_history: int,
) -> int:
    position = int(index.get_indexer([decision])[0])
    if position < 0:
        raise ValueError(f"decision {decision!s} is not a price row")
    if position < required_history:
        raise ValueError(f"decision {decision!s} lacks required history")
    return position


def _ordered_symbols(values: pd.Series, *, descending: bool) -> list[str]:
    if values.empty:
        return []
    if not np.all(np.isfinite(values.to_numpy(dtype=float))):
        raise ValueError("ranking values must be finite")
    direction = -1.0 if descending else 1.0
    return sorted(values.index, key=lambda symbol: (direction * values[symbol], symbol))


def _normalized_ordinal_rank(values: pd.Series) -> pd.Series:
    ordered = _ordered_symbols(values, descending=False)
    denominator = max(1, len(ordered) - 1)
    return pd.Series(
        {symbol: position / denominator for position, symbol in enumerate(ordered)},
        index=values.index,
        dtype=float,
    )


def _sample_sharpe(excess: np.ndarray) -> np.ndarray:
    standard_deviation = excess.std(axis=2, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return excess.mean(axis=2) / standard_deviation * np.sqrt(PERIODS_PER_YEAR)


def _validate_lookback(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("lookback must be a positive integer")


def _validate_selection_count(value: int, symbol_count: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= symbol_count:
        raise ValueError("selection_count must be in [1, symbol_count]")


def _positive_int(config: Mapping[str, object], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _positive_float(config: Mapping[str, object], key: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be positive")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{key} must be positive")
    return parsed


def _unit_interval(config: Mapping[str, object], key: str) -> float:
    value = _positive_float(config, key)
    if value > 1.0:
        raise ValueError(f"{key} must be in (0, 1]")
    return value


def _integer_sequence(config: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty integer list")
    parsed = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in parsed):
        raise ValueError(f"{key} must contain positive integers")
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{key} must not contain duplicates")
    return parsed


__all__ = [
    "FROZEN_PROTOCOL_COMMIT",
    "FROZEN_PROTOCOL_SHA256",
    "LearnedTargetResult",
    "PERIODS_PER_YEAR",
    "bootstrap_metric_difference",
    "circular_block_indices",
    "cross_sectional_momentum_targets",
    "exposure_matched_comparator_targets",
    "invalid_same_close_result",
    "learned_gbrt_targets",
    "monthly_decision_dates",
    "performance_metrics",
    "run_engine",
    "short_term_reversal_targets",
    "time_series_momentum_targets",
    "validate_price_panel",
]
