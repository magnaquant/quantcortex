# Performance Evaluation

This repository publishes one owner-authorized reference run as derived chart
artifacts. The raw market data remains uncommitted. Ordinary generated reports
and executed notebook outputs remain excluded unless publication is explicitly
approved and accompanied by complete provenance and artifact hashes.

## Published Reference Run

The README charts were generated on June 16, 2026 with yfinance 1.4.1 adjusted
closes for QQQ, VGT, GLD, TLT, SPY, VIG, and SHV. The source spans January 4,
2016 to December 31, 2025; evaluation spans January 2, 2018 to December 31,
2025 after 503 warm-up sessions. SHV is the residual-cash return proxy. The
owner authorizes publication of the derived charts, but the project does not
independently establish that the provider's terms permit publication. When the
mature selected-group residual-momentum signal has no positive member, the
strategy holds cash.

| Metric | Value |
|---|---:|
| Net nominal CAGR | +1.40% |
| Gross nominal CAGR before modeled costs | +3.17% |
| Annualized volatility | 5.86% |
| Net cash-excess Sharpe | -0.15 |
| Gross cash-excess Sharpe before modeled costs | +0.14 |
| Maximum drawdown | -9.83% |
| Annualized one-way turnover | 10.30x |
| Annualized gross traded notional | 13.25x |
| Arithmetic sum of modeled transaction-cost return drag | 13.75% |
| Mean active gross exposure | 30.14% |
| Fully-cash session fraction | 47.94% |
| Cash proxy CAGR | +2.50% |
| Exposure-matched equal-initial-weight basket cash-excess Sharpe, gross | +0.80 |

Strategy returns are net of the platform default: 3 bps commission plus 10 bps
flat slippage per dollar traded. The paper experiment encodes the same 13 bps
as one all-in symmetric charge; the two representations are numerically
identical under this linear model. Benchmarks are gross and the ADV cap is
inactive. One-way turnover is useful for portfolio-change reporting; the 13 bps
charge is applied to gross two-sided traded notional measured against pre-trade
NAV. The 13.75% aggregate is instead the arithmetic sum of daily return drag
against prior-close NAV, so the two quantities differ slightly on rebalance
days with nonzero returns. Cash-aware accounting changes the interpretation: the strategy has
positive nominal growth but negative return in excess of SHV after modeled
costs, and it trails a passive benchmark matched to the same daily risky
exposure. The DSR of 0.022 uses an assumed 10 trials and a single-series
variance estimate. Because the true historical trial count is unknown, it is
not a validated multiple-testing correction. Exact source and artifact hashes
are in `docs/img/performance_manifest.json`.

### Exact return attribution

The paper experiment decomposes each daily net return in excess of SHV into an
exposure-matched active-allocation effect, exposure timing around the
full-sample mean, passive risky exposure, and modeled cost. The identity holds
on every date; annualized arithmetic means therefore add exactly before
rounding. The active-allocation effect combines group selection, within-risky
weighting, and the event engine's between-rebalance share drift;
it is not a pure security-selection effect.

| Component | Annualized mean | 21-session bootstrap interval |
|---|---:|---:|
| Active risky allocation | -3.38% | [-5.85%, -0.90%] |
| Dynamic exposure timing | +0.30% | [-2.57%, +3.26%] |
| Passive risky exposure | +3.90% | [+0.82%, +6.85%] |
| Modeled implementation cost | -1.72% | [-2.08%, -1.38%] |
| Net excess over SHV | -0.90% | [-4.77%, +3.14%] |

The constant-exposure series is an ex-post diagnostic, not a tradable
benchmark, and the intervals are unstudentized percentile intervals
conditional on this historical path. `paper/results/return_decomposition.csv`
contains 5-, 21-, and 63-session results.

### Protocol diagnostics

One-assumption diagnostics hold the target weights and input matrix fixed. The
audited SHV-excess Sharpe is -0.15. Setting modeled costs to zero raises it to
+0.14; deliberately applying a close-derived target to the return ending at
that same close raises it to +0.03. The same-close result is look-ahead and is
never an executable path. Assigning zero return to residual cash lowers the
consistently measured SHV-excess Sharpe to -0.42. See
`paper/results/protocol_switches.csv`.

## Generate a Report

Use a licensed or otherwise permitted wide adjusted-close CSV:

```bash
PYTHONPATH=. python scripts/generate_report.py \
  --prices-csv local_data/published_rotation_prices.csv \
  --cash-proxy-symbol SHV \
  --start 2018 --end 2025 --n-trials 10 \
  --manifest-out reports/performance_manifest.json \
  --data-provider "$DATA_PROVIDER" \
  --permission-basis "$DATA_PERMISSION_BASIS" \
  --retrieved-at "$DATA_RETRIEVED_AT" \
  --adjustment-method "$DATA_ADJUSTMENT_METHOD"
```

The required columns are documented in
[local_data/README.md](local_data/README.md). The command
writes charts to ignored `reports/img/` and prints markdown tables containing
the local file path, SHA-256 digest, and observed date window. By default it
loads two years before `--start` to warm the signals, carries that strategy
state into the requested evaluation window, and excludes the pre-roll returns
from reported metrics. The source must contain at least 274 pre-evaluation
sessions. Use `--warmup-years 0` only when a deliberately cold-started report
is appropriate; that override is disclosed in the generated settings.
The provenance options record owner-supplied facts and assertions; they do not
constitute independent verification that publication or redistribution is
permitted. Missing fields are labeled incomplete in the report. The requested
manifest records the input, source tree, settings, and artifact hashes.

The fixed paper experiment is stricter than the general report: it accepts no
missing price rows, performs no forward fill, and requires at least 274 signal
warm-up sessions before the evaluation window.

For an explicitly requested live download:

```bash
PYTHONPATH=. python scripts/generate_report.py --live-yfinance
PYTHONPATH=. python scripts/validate_performance.py --live-yfinance --pit
```

Review Yahoo's terms and the
[yfinance legal disclaimer](https://ranaroussi.github.io/yfinance/) before use.
Live historical data may be revised, so preserve your own permitted input if
exact reproduction matters.

## Generated Artifacts

The command writes `reports/report.md` plus these plots under `reports/img/`:

- `report_overview.png`: equity, drawdown, allocation, turnover, and costs in a
  compact review image.
- `equity_vs_benchmarks.png`: strategy net of modeled costs versus gross SPY
  and an equal-initial-weight buy-and-hold basket.
- `performance_attribution.png`: strategy before and after modeled costs,
  the exposure-matched passive basket, and the cash proxy on one capital clock.
- `drawdown.png`: the strategy underwater curve.
- `rolling_sharpe.png`: trailing 126-session cash-excess Sharpe.
- `rolling_risk.png`: trailing 126-session annualized volatility and beta to
  SPY.
- `allocation_and_exposure.png`: realized asset weights, invested gross
  exposure, and cash under the share-based event engine.
- `turnover_and_costs.png`: executed one-way turnover, gross traded notional,
  and cumulative modeled transaction-cost return drag.
- `monthly_returns.png`: monthly net-return heatmap.
- `return_distribution.png`: log-count daily net-return histogram, historical
  tail markers, and a normal Q-Q diagnostic.

The Markdown report links every plot and includes performance metrics,
evaluation settings, data provenance, and the monthly-return table. Outputs
remain ignored by default. Publish them only with explicit owner approval,
complete adjacent provenance, an input digest, and hashes for every artifact.

The current command must not fabricate diagnostics it cannot support. Add
walk-forward or live-start boundaries only when the run records those regimes;
capacity and slippage curves only with spread, volume, and order-size inputs;
factor attribution only with validated factor returns/exposures; and fill
quality only from authenticated order and execution records.

## Reporting Requirements

- Report the data provider, license or permission basis, retrieval date, date
  window, symbols, adjustment method, and input-file digest.
- Set `--n-trials` to the actual number of configurations evaluated. The
  default `10` is a convenience, not a factual statement about a research run.
  Correlated trials also require a defensible effective trial count or Sharpe
  variance estimate; a raw configuration count alone does not make DSR valid.
  Pass the measured cross-trial variance of per-observation Sharpes with
  `--sr-variance`. Without it, the scripts disclose that they use the metric's
  simplifying single-series variance estimate.
- Compare against relevant buy-and-hold, equal-weight, cash, and
  exposure-matched benchmarks.
- Label whether benchmarks are gross or cost-adjusted and state the risk-free
  or cash-return series used for Sharpe. The reference benchmarks are gross.
- Report costs, turnover, and maximum drawdown. Report the Deflated Sharpe
  Ratio only when its trial-count and Sharpe-variance assumptions are
  defensible; otherwise omit it or label it exploratory with those assumptions.
- Keep failed variants in the trial count; do not tune until a target is met.

## Known Limitations

The default transaction-cost model uses flat commission and slippage rates.
Size-, spread-, and volatility-aware impact is available in
`quantcortex/backtest/execution_models/market_impact.py` but is not wired into
the reference report. The report supplies no ADV series, so its configured
volume cap is also inactive. The reference report uses the event-driven engine,
which holds adjusted-close pseudo-shares and sizes targets against post-cost
NAV. These are total-return accounting units, not nominal broker shares. The
vectorized engine remains available for approximations and sweeps but holds
target weights constant between explicit rebalances without charging for the
implied re-pegging trades.
Residual cash must be supplied explicitly when it earns a nonzero return; the
engines reject missing cash-proxy bars rather than filling them silently.
Single-name tests remain survivorship-biased unless the price feed includes
delisted securities and the universe is point-in-time.
`SP500Universe.from_wikipedia()` supplies approximate, coverage-limited
historical membership, not delisted price history. yfinance fundamentals use a
documented 45-day filing-date proxy rather than exact announcement timestamps;
date-only records become available strictly after that timestamp by default.

Close-derived target weights execute at the next available bar's close in both
backtest engines. Weekly schedules use the first observed session of each week,
so a Monday exchange holiday moves the decision to Tuesday rather than skipping
the week. Monthly validation decisions use the last observed session of each
month. Report any separate signal warm-up period; using the evaluation window
itself for model warm-up can materially bias comparisons.

No metric threshold in this repository is evidence that a strategy is
profitable or deployable. Interpret every result against its data vintage,
comparator, costs, exposure, uncertainty, and recorded research trial history.
