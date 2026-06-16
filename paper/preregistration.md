# Prospective Evaluation Protocol

Status: draft only. This document has not been registered, and the published
2018-2025 case is retrospective. A future study must freeze and timestamp a
completed version before inspecting confirmatory outcomes.

## Research Question

How much do explicit timing, cash, cost, comparator, and engine contracts change
reported performance across heterogeneous target-weight strategies and real
data panels?

## Confirmatory Scope

- Strategy archetypes: time-series momentum, cross-sectional ranking,
  mean-reversion, and at least one substantive learned model.
- Data: at least two real panels that pass
  `docs/data-source-due-diligence.md`. Crypto is excluded from the first study to
  avoid mixing calendar and microstructure changes with the contract effects.
- Feature maturity: no decision is evaluated before every declared feature and
  training window is mature.
- Learned models: architecture, features, training window, hyperparameters,
  stopping rule, and seed set are frozen. Every declared seed is reported.
- Engines: the canonical target tape is frozen before any engine comparison.

Panel names, date windows, universes, and precise strategy configurations must
be inserted here before registration. Placeholders make this draft incomplete.

## Primary Outcomes

For each strategy-panel pair, estimate paired changes in annualized arithmetic
return and conventional sample Sharpe caused by one declared contract switch at
a time. Primary switches are execution timing, residual cash return,
transaction costs, comparator exposure, and engine semantics. Report rank
reversals across strategies as a separate outcome.

The main uncertainty analysis uses joint circular-block resampling with block
lengths fixed before outcome inspection. Exact accounting identities are checked
within every draw. Sharpe intervals resample the statistic directly; no iid
normal standard error is used.

## Controls Against Researcher Degrees of Freedom

1. Archive all configurations, seeds, failures, and exclusions.
2. Define the primary metric, comparator, block lengths, and missing-data policy
   before running the confirmatory windows.
3. Preserve the current negative case without retuning it.
4. Label exploratory analyses and keep them out of confirmatory claims.
5. Do not call a previously inspected period out of sample.
6. Record every protocol deviation with its timestamp and rationale.

## Planned Evidence

Publish the machine-readable evaluation contract, canonical target tapes,
conformance fixtures, aggregate result tables, forest plots of paired effects,
engine-conformance matrices, rank-reversal summaries, environment locks, and
content hashes. Raw data availability and reviewer access are reported per
panel; no unavailable data are implied to be open.
