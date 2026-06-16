# Performance Evaluation

This repository publishes one owner-authorized reference run as derived chart
artifacts. The raw market data remains uncommitted. Ordinary generated reports
and executed notebook outputs remain excluded unless publication is explicitly
approved and accompanied by complete provenance and artifact hashes.

## Published Reference Run

The README charts were generated on June 15, 2026 with yfinance 1.4.1 adjusted
closes for QQQ, VGT, GLD, TLT, SPY, and VIG. The source spans January 4, 2016 to
December 31, 2025; evaluation spans January 2, 2018 to December 31, 2025 after
503 warm-up sessions. The owner confirms permission to publish the derived
charts; that permission is not independently verified by the software.

| Metric | Value |
|---|---:|
| CAGR | -0.07% |
| Annualized volatility | 6.53% |
| Sharpe | 0.02 |
| Maximum drawdown | -14.22% |
| Annualized one-way turnover | 10.84x |
| Sum of modeled cost fractions | 15.06% |
| SPY Sharpe, gross | 0.78 |
| Equal-weight six-ETF Sharpe, gross | 0.96 |

Strategy returns are net of 3 bps commission and 10 bps flat slippage per
trade; benchmarks are gross and the ADV cap is inactive. The DSR of 0.065 uses
an assumed 10 trials and a single-series variance estimate. Because the true
historical trial count is unknown, it is not a validated multiple-testing
correction. Exact source and artifact hashes are in
`docs/img/performance_manifest.json`.

## Generate a Report

Use a licensed or otherwise permitted wide adjusted-close CSV:

```bash
PYTHONPATH=. python scripts/generate_report.py \
  --prices-csv local_data/rotation_prices.csv \
  --start 2018 --end 2025 --n-trials 10 \
  --data-provider "$DATA_PROVIDER" \
  --permission-basis "$DATA_PERMISSION_BASIS" \
  --retrieved-at "$DATA_RETRIEVED_AT" \
  --adjustment-method "$DATA_ADJUSTMENT_METHOD"
```

The required columns are documented in `local_data/README.md`. The command
writes charts to ignored `reports/img/` and prints markdown tables containing
the local file path, SHA-256 digest, and observed date window. By default it
loads two years before `--start` to warm the signals, carries that strategy
state into the requested evaluation window, and excludes the pre-roll returns
from reported metrics. The source must contain at least 274 pre-evaluation
sessions. Use `--warmup-years 0` only when a deliberately cold-started report
is appropriate; that override is disclosed in the generated settings.
The provenance options record owner-supplied facts and assertions; they do not
constitute independent verification that publication or redistribution is
permitted. Missing fields are labeled incomplete in the report.

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
  and equal-weight buy-and-hold benchmarks.
- `drawdown.png`: the strategy underwater curve.
- `rolling_sharpe.png`: trailing 126-session Sharpe.
- `rolling_risk.png`: trailing 126-session annualized volatility and beta to
  SPY.
- `allocation_and_exposure.png`: post-trade asset weights, invested gross
  exposure, and cash.
- `turnover_and_costs.png`: executed one-way turnover and the cumulative sum of
  modeled per-period cost fractions.
- `monthly_returns.png`: monthly net-return heatmap.
- `return_distribution.png`: daily net-return histogram, historical tail
  markers, and a normal Q-Q diagnostic.

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
- Compare against relevant buy-and-hold and equal-weight benchmarks.
- Label whether benchmarks are gross or cost-adjusted. The reference report's
  buy-and-hold benchmarks are gross.
- Report costs, turnover, maximum drawdown, and the Deflated Sharpe Ratio.
- Keep failed variants in the trial count; do not tune until a target is met.

## Known Limitations

The default transaction-cost model uses flat commission and slippage rates.
Size-, spread-, and volatility-aware impact is available in
`quantcortex/backtest/execution_models/market_impact.py` but is not wired into
the reference report. The report supplies no ADV series, so its configured
volume cap is also inactive. The vectorized engine holds target weights constant
between explicit rebalances without charging for the implied re-pegging trades;
use the event-driven engine when position drift and fill mechanics matter.
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

Design targets in the README are aspirational. They are not evidence that a
strategy is profitable, deployable, or expected to meet the target on unseen
data.
