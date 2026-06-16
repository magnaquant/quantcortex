# Performance Evaluation

This repository does not publish a fixed performance result. Market data,
executed notebook outputs, and generated charts are intentionally excluded.
Evaluate the reference strategies on data you are permitted to use and retain
the source metadata needed to reproduce the run.

## Generate a Report

Use a licensed or otherwise permitted wide adjusted-close CSV:

```bash
PYTHONPATH=. python scripts/generate_report.py \
  --prices-csv local_data/rotation_prices.csv \
  --start 2018 --end 2025 --n-trials 10
```

The required columns are documented in `local_data/README.md`. The command
writes charts to ignored `reports/img/` and prints markdown tables containing
the local file path, SHA-256 digest, and observed date window. By default it
loads two years before `--start` to warm the signals, carries that strategy
state into the requested evaluation window, and excludes the pre-roll returns
from reported metrics. The source must contain at least 274 pre-evaluation
sessions. Use `--warmup-years 0` only when a deliberately cold-started report
is appropriate; that override is disclosed in the generated settings.

For an explicitly requested live download:

```bash
PYTHONPATH=. python scripts/generate_report.py --live-yfinance
PYTHONPATH=. python scripts/validate_performance.py --live-yfinance --pit
```

Review Yahoo's terms and the
[yfinance legal disclaimer](https://ranaroussi.github.io/yfinance/) before use.
Live historical data may be revised, so preserve your own permitted input if
exact reproduction matters.

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
