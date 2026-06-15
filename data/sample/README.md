# Sample data

`rotation_prices.csv` is a fixed snapshot of split/dividend-adjusted daily closes
for the six multi-asset-rotation ETFs (QQQ, VGT, GLD, TLT, SPY, VIG),
2018-01-02 to 2025-12-30, indexed by date.

**Why it is bundled.** `scripts/generate_report.py` and the README "Results"
section read this snapshot by default so the published charts, tables, and Sharpe
figures are exactly reproducible. The backtest is deterministic, so a re-run on
this file reproduces the numbers bit for bit. Without a fixed snapshot the
results would drift: `yfinance` re-adjusts historical closes over time as
dividends accrue, so every live fetch returns slightly different data.

**Provenance / licensing.** Derived from Yahoo Finance via `yfinance` for
reproducible examples and tests only. It is a small derived dataset, not a
redistribution of a licensed feed; do not treat it as an authoritative or
survivorship-safe source. Regenerate or extend it with:

```bash
python -c "from data.providers.yfinance_provider import YFinanceProvider; \
YFinanceProvider().get_prices(['QQQ','VGT','GLD','TLT','SPY','VIG'], start='2018-01-01', end='2025-12-31') \
.rename_axis('date').to_csv('data/sample/rotation_prices.csv', float_format='%.6f')"
```

Or pass `--live` to `scripts/generate_report.py` to bypass the snapshot and fetch
fresh data.
