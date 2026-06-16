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

**Provenance / licensing.** This is *real* market data (not synthetic), derived
from Yahoo Finance via `yfinance` for reproducible examples and tests. Note that
`yfinance` documents Yahoo Finance data as intended for **personal use**;
whether committing this Yahoo-derived snapshot to a given repo (public,
redistributed, or commercial) is acceptable is a repo-owner / legal decision
this file cannot settle, so observe Yahoo's terms for your use. To avoid relying
on the committed copy, regenerate it locally (below) or pass `--live` to
`scripts/generate_report.py`, which fetches fresh and writes nothing. It is not
an authoritative or survivorship-safe source. Regenerate or extend the committed
snapshot with:

```bash
python -c "from quantcortex.data.providers.yfinance_provider import YFinanceProvider; \
YFinanceProvider().get_prices(['QQQ','VGT','GLD','TLT','SPY','VIG'], start='2018-01-01', end='2025-12-31') \
.rename_axis('date').to_csv('quantcortex/data/sample/rotation_prices.csv', float_format='%.6f')"
```

Or pass `--live` to `scripts/generate_report.py` to bypass the snapshot and fetch
fresh data.
