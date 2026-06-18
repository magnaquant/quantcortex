# Local Market Data

This directory is for owner-supplied market data. Everything except this file
is gitignored. Do not commit provider downloads unless their terms explicitly
permit redistribution.

Record the provider, permission or license basis, retrieval timestamp, symbol
mapping, corporate-action adjustment method, timezone, and any cleaning steps
outside the CSV.

## Wide Adjusted-Close CSV

`scripts/generate_report.py`, `scripts/run_paper_experiments.py`,
`scripts/run_expansion_experiments.py`, and all research notebooks accept a
date-by-symbol matrix:

```csv
date,QQQ,VGT,GLD,TLT,SPY,VIG,SHV
2024-01-02,100.0,101.0,102.0,103.0,104.0,105.0,106.0
```

Dates must be unique and parseable. Symbol columns must be numeric and strictly
positive. Reports and notebooks forward-fill missing observations from earlier
rows for at most five sessions, then drop incomplete rows. The fixed paper
experiment instead requires complete rows and performs no forward fill.
Notebook universes require their full set of symbol columns.

Set `QUANTCORTEX_PRICES_CSV` to the absolute path for notebooks, or pass
`--prices-csv local_data/<file>.csv` to a script. Notebook runs must choose
exactly one adjusted-close source: this variable or
`QUANTCORTEX_LIVE_YFINANCE=1`. `QUANTCORTEX_OHLCV_CSV` is a supplemental local
input for the Alpha158 notebook, not a second adjusted-close source.
Include the configured cash-proxy column, such as `SHV`, whenever residual
cash should earn a nonzero return. The fixed paper experiment requires all
seven columns shown above.
For a performance report, include at least the requested pre-evaluation warm-up
history (two calendar years by default); the loader cannot reconstruct signal
history that is absent from the file.

For a report intended for external review, also pass `--data-provider`,
`--permission-basis`, `--retrieved-at`, and `--adjustment-method`. These fields
record the owner's provenance assertions but do not independently establish
that redistribution is permitted. Pass `--manifest-out` as well so the input,
source tree, settings, and generated artifacts are hash-bound.

The expansion uses two separate complete matrices plus metadata sidecars under
`local_data/expansion/`:

- `us_sector_etfs.csv`: `date`, XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY,
  and SHV.
- `country_equity_etfs.csv`: `date`, EWA, EWC, EWG, EWH, EWJ, EWL, EWP, EWQ,
  EWS, EWU, and SHV.

Each `.metadata.json` records the exact request, provider version, retrieval
timestamp, protocol digest, row coverage, missingness, and CSV SHA-256. The
frozen expansion requires complete rows from 2014-01-02 through 2025-12-31,
performs no forward fill, and rejects a missing evaluation month or hash
mismatch. `scripts/fetch_expansion_data.py` can create these files through the
explicit provider adapter; using it does not establish permission to publish or
redistribute the observations.

## Single-Symbol OHLCV CSV

Notebook 02 also requires actual OHLCV data when using local files:

```csv
date,open,high,low,close,adj_close,volume
2024-01-02,100.0,102.0,99.0,101.0,101.0,1500000
```

All six fields are required. Prices must be positive, volume non-negative, and
high/low internally consistent. Set `QUANTCORTEX_OHLCV_CSV` to this file.
