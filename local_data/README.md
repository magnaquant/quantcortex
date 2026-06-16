# Local Market Data

This directory is for owner-supplied market data. Everything except this file
is gitignored. Do not commit provider downloads unless their terms explicitly
permit redistribution.

Record the provider, permission or license basis, retrieval timestamp, symbol
mapping, corporate-action adjustment method, timezone, and any cleaning steps
outside the CSV.

## Wide Adjusted-Close CSV

`scripts/generate_report.py` and all research notebooks accept a date-by-symbol
matrix:

```csv
date,QQQ,VGT,GLD,TLT,SPY,VIG
2024-01-02,100.0,101.0,102.0,103.0,104.0,105.0
```

Dates must be unique and parseable. Symbol columns must be numeric and strictly
positive. Missing observations are forward-filled from earlier rows for at
most five sessions; rows that remain incomplete are dropped. Notebook
universes require their full set of symbol columns.

Set `QUANTCORTEX_PRICES_CSV` to the absolute path or pass
`--prices-csv local_data/<file>.csv` to the report generator.
For a performance report, include at least the requested pre-evaluation warm-up
history (two calendar years by default); the loader cannot reconstruct signal
history that is absent from the file.

## Single-Symbol OHLCV CSV

Notebook 02 also requires actual OHLCV data when using local files:

```csv
date,open,high,low,close,adj_close,volume
2024-01-02,100.0,102.0,99.0,101.0,101.0,1500000
```

All six fields are required. Prices must be positive, volume non-negative, and
high/low internally consistent. Set `QUANTCORTEX_OHLCV_CSV` to this file.
