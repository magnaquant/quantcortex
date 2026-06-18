# Prospective Evaluation Protocol

Status: repository-frozen prospective protocol. Commit
`4018f4063f46889f41d6981db5a71079e1dbd713` and protocol SHA-256
`e49e41a12a19fa5404a573ba5e21eb8a2888e616985f8c610d9652866923315c`
are the pre-retrieval public record. This is not an external registry entry. The
2018-2025 six-ETF case in the paper was inspected before this freeze and is not
confirmatory evidence for the expansion.

## Research Question

How much do explicit timing, cash, cost, comparator, and engine contracts change
reported performance across heterogeneous target-weight strategies and real
data panels?

## Frozen Panels

Both panels use daily adjusted closes from 2014-01-02 through 2025-12-31 and
SHV as the residual-cash proxy. Evaluation begins 2018-01-02 and ends
2025-12-31. Raw matrices remain untracked; each run records provider metadata,
retrieval time, input SHA-256, and row coverage.

1. `us_sector_etfs`: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY.
2. `country_equity_etfs`: EWA, EWC, EWG, EWH, EWJ, EWL, EWP, EWQ, EWS, EWU.

The accepted matrix is the complete-row intersection of the declared symbols
and SHV. There is no forward fill or symbol substitution. A panel is excluded
only if it has fewer than 252 complete pre-evaluation sessions or a missing
evaluation month. The learned model additionally requires 24 mature monthly
training dates before evaluation. Any exclusion is reported as a protocol
deviation; a new universe is not substituted after outcomes are inspected.
Provider terms and publication rights are reported per panel without legal
inference.

The frozen retrieval adapter is yfinance with `auto_adjust=False`,
`actions=False`, `repair=False`, and `threads=False`; the adjusted-close field
is selected explicitly. The request ends at 2026-01-01 because the provider's
end date is exclusive. This choice does not assert permission to redistribute
the observations.

## Frozen Strategies

All decisions occur on the first observed session of each calendar month after
every feature is mature. Signals use that session's close and execute on the
first strictly later panel row.

- `ts_momentum`: 252-session total return. Each positive-signal asset receives
  `1 / N_positive`; otherwise capital remains in SHV.
- `cross_sectional_momentum`: return from session `t-252` through `t-21`.
  The top three assets receive one-third each, irrespective of sign.
- `short_term_reversal`: negative five-session return. Up to the three assets
  with the lowest negative returns receive one-third each; unused exposure
  remains in SHV.
- `learned_gbrt`: a walk-forward gradient-boosted regression model predicts
  21-session forward log return. Features are 5-, 21-, 63-, 126-, and
  252-session log returns; sample standard deviations of daily log returns over
  21 and 63 sessions; `price / trailing_252_session_high - 1`; and normalized
  ordinal cross-sectional ranks of 21- and 252-session returns. The lowest rank
  is zero and the highest is one. Training examples are prior monthly decisions
  whose labels end on or before the current decision. The pooled training rows
  are asset-month pairs; symbol identity is not a feature. The rolling training
  set is capped at 60 decision months and must contain at least 24. The estimator
  is scikit-learn `GradientBoostingRegressor` with 100 estimators, learning rate
  0.03, depth 2, minimum leaf size 10, and subsample 0.8. Seeds are 11, 29, 47,
  71, and 97. Each seed is reported; the family estimate is the arithmetic mean
  of seed-level metrics. Up to three positive predictions receive one-third
  each.

All score and rank ties are broken by ascending symbol. Features are not
standardized or winsorized. No hyperparameter, universe, threshold, seed, or
window may change in response to observed performance.

## Execution And Comparators

Primary accounting uses the event-driven engine, initial NAV 1.0, next-bar
close execution, SHV residual-cash returns, and 13 basis points per unit of
one-way gross traded notional. Targets are long-only with gross exposure at
most one. The vectorized engine is a model-convention sensitivity diagnostic on
the identical canonical target tape; equality with pseudo-share accounting is
not expected.

For each strategy, a causal costed comparator follows the strategy's target
risky exposure, allocates it equally across the panel, rebalances on the same
dates, holds residual SHV, and pays the same cost rate. A buy-and-hold equal
weight panel portfolio is descriptive only.

The one-switch diagnostics are:

1. next-bar close versus deliberately invalid same-close assignment;
2. SHV residual cash versus zero cash return;
3. 13 basis-point cost versus zero cost;
4. raw net return versus the causal exposure-matched comparator; and
5. event-driven versus vectorized accounting.

The invalid same-close result is a diagnostic, not a candidate strategy.

## Outcomes And Uncertainty

Primary outcomes are the paired change in annualized arithmetic return and the
paired change in conventional sample Sharpe for each switch, strategy family,
and panel. Annualized arithmetic return is 252 times the daily mean. Sharpe is
the mean daily strategy return minus the SHV return, divided by its sample
standard deviation and multiplied by the square root of 252. Secondary outcomes
are net CAGR, maximum drawdown, turnover, cost drag, active return versus the
costed comparator, engine return divergence, strategy-rank reversals, and
learned-model seed dispersion.

Joint circular-block bootstrap intervals use 5,000 draws, primary block length
21 sessions, sensitivity lengths 5 and 63, and seed 20260618. Return components
are resampled jointly. Exact accounting identities must hold in every original
series and sampled draw. Intervals are percentile intervals and are not called
posterior probabilities or multiple-testing-adjusted evidence.

## Researcher Degrees Of Freedom

1. Archive all configurations, seeds, failures, exclusions, and deviations.
2. Preserve the current negative case without retuning it.
3. Label analyses outside this protocol exploratory.
4. Do not call any period inspected before this commit out of sample.
5. Publish the machine-readable protocol, canonical target-tape hashes,
   aggregate result tables, forest plots, engine-conformance matrix,
   rank-reversal summary, environment lock, and content hashes.
6. Do not publish raw provider matrices unless explicit redistribution rights
   are documented.

The machine-readable source of truth is `paper/expansion/protocol.json`.

Post-freeze clarification: the JSON key
`cost_per_one_way_gross_notional` is a naming error retained to preserve the
pre-retrieval protocol hash. The implementation applies its 0.0013 value to
gross two-sided traded notional, `sum(abs(delta_weight))`, not to one-way
turnover. This clarification changes no calculation or frozen parameter.
