# Measured Performance

Reproduce with:

```bash
python scripts/validate_performance.py            # 2018-2025, real adjusted prices
python scripts/validate_performance.py 2010 2020  # any window
```

The harness fetches real split/dividend-adjusted prices via `yfinance`, runs the
reference strategies through the mandatory-cost backtest engine (3 bps
commission + 10 bps slippage, 10% ADV cap), and reports measured metrics next to
naive buy-and-hold benchmarks. Numbers below were measured on the **2018-01-02
to 2025-12-30** window (2010 trading days); they depend on the `yfinance` data
vintage and will move as data is revised.

## Results (2018-2025)

| Strategy / benchmark | Sharpe | CAGR | Sortino | Calmar | Max DD | DSR | Design target |
|---|---|---|---|---|---|---|---|
| SPY buy & hold | 0.78 | +14.3% | - | - | - | - | benchmark |
| Equal-weight 6-ETF buy & hold | 1.00 | - | - | - | - | - | benchmark |
| **multi_asset_rotation** | **0.05** | +0.1% | 0.07 | 0.01 | -15.4% | 0.08 | Sharpe > 1.10 - **not met** |
| **momentum_ml** (survivorship-biased) | **0.64** | +13.0% | 0.90 | 0.41 | -31.5% | 0.59 | Sharpe > 0.9 - **not met** |

## Honest interpretation

- **The README's Sharpe targets are aspirational design goals; the reference
  implementations do not meet them on this window.** This is reported as-is
  rather than tuned away. Searching hyper-parameters until a single backtest
  clears 1.10 is exactly the overfitting the platform's Deflated Sharpe Ratio
  and BHY multiple-testing tooling exist to flag.
- **multi_asset_rotation underperforms a naive equal-weight buy-and-hold** (0.05
  vs 1.00). This is expected, not a bug: it is a *defensive* rotation that holds
  only 2 of 3 asset-class groups and an HMM regime gate that sits in cash ~36% of
  weeks. In a bull-dominated 2018-2025 sample, time spent flat is pure return
  drag; an ablation shows the core selection alone scores ~0.40 and the regime
  gate removes most of that. The strategy is logically correct (selection sign,
  causal residual momentum, contract-valid weights all verified) but offers no
  edge over buy-and-hold in this regime.
- **momentum_ml shows real positive alpha** (13% CAGR, DSR 0.59) but its read is
  **survivorship-biased** - the universe is today's large-caps, which excludes
  firms that were delisted or merged over the window, inflating the result.
- **A clean evaluation needs a licensed point-in-time feed** with
  delisted-name coverage and historical index constituents. The point-in-time
  membership is already available via `SP500Universe.from_wikipedia()`, and
  `python scripts/survivorship_demo.py` quantifies the gap concretely: of the
  501 S&P 500 names as of 2018-06-01, 122 are gone today and **55 are no longer
  priceable** on `yfinance` (acquired/delisted) - exactly the rows a
  survivor-only single-name backtest omits. `yfinance` adjusted closes are
  adequate for the all-ETF rotation (none were delisted) but a clean single-name
  read still needs delisted-name *price* history from a licensed vendor.

## What this validates

The harness confirms the end-to-end machinery is sound and honest: real data
flows through selection -> allocation -> timing -> risk -> mandatory-cost
backtest -> tearsheet/DSR, benchmarks compute correctly, and the reported
numbers are trustworthy (if unflattering). It does **not** claim the strategies
are profitable as shipped; they are correct, well-tested baselines.
