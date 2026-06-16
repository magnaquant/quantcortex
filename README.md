# quantcortex

> **Modular quantitative research and paper-execution platform**
> Data -> Alpha -> Portfolio -> Timing -> Risk -> Backtest -> Execution

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

**quantcortex** is a modular quantitative research platform built around a
strict **weight-centric interface contract** (inspired by FinRL-X). Every layer
- from alpha signal to broker adapter - speaks the same language: a normalized
weight vector `w_t in R^n`.

This eliminates the most common gap in quant stacks: strategies that backtest cleanly but behave differently in paper and live trading because the architecture changes between environments.

```
w_t = R_t( T_t( A_t( S_t( X<=t ) ) ) )
       ^       ^       ^       ^
     Risk   Timing  Alloc  Selection
```

**Design targets are not performance claims.** The reference strategies retain
aspirational Sharpe targets of 1.10 (multi-asset rotation) and 0.9 (momentum
ML), but this repository publishes no fixed backtest result. Evaluate them on
data you are permitted to use, record the true number of trials, and compare
against appropriate benchmarks. See [PERFORMANCE.md](PERFORMANCE.md).

---

## Getting Started

The scientific core, test suite, broker mocks, and labeled paper-trading dry run
work offline. Research notebooks and performance reports require an explicit
real-data source; the repository does not bundle market data or silently replace
failed downloads with generated prices. Optional providers, ML libraries,
broker SDKs, and storage clients remain lazy imports.

### Install

```bash
git clone https://github.com/magnaquant/quantcortex.git
cd quantcortex
python3.11 -m venv .venv && source .venv/bin/activate

# Core (required) - enough to run the full test suite
pip install numpy pandas scipy scikit-learn matplotlib pyarrow pytest

# Optional accelerators / integrations (Poetry extras):
poetry install -E all          # or, with pip:  pip install '.[all]'
#   ml        -> xgboost, lightgbm, catboost       (GBDT cross-sectional alpha)
#   nlp       -> transformers, torch               (FinBERT sentiment)
#   rl        -> stable-baselines3, gymnasium       (PPO DRL allocator)
#   regime    -> hmmlearn                           (HMM regime overlay)
#   providers -> yfinance, polygon-api-client, fredapi  (market / macro data)
#   brokers   -> alpaca-trade-api, ib_insync, ccxt  (live execution)
#   storage   -> redis, sqlalchemy, psycopg2-binary (feature cache + TimescaleDB)
```

> **macOS note:** LightGBM/XGBoost need the OpenMP runtime (`brew install
> libomp`). Without it quantcortex transparently falls back to the
> scikit-learn GBDT backend - nothing breaks.

### Run the tests

```bash
pytest tests/ -v   # weight contract, transaction costs, look-ahead, risk overlay, order state machine
```

### Run the research notebooks

```bash
# Owner-supplied real data; see local_data/README.md for schemas.
export QUANTCORTEX_PRICES_CSV="$PWD/local_data/prices.csv"
export QUANTCORTEX_OHLCV_CSV="$PWD/local_data/aapl_ohlcv.csv"  # notebook 02
jupyter lab research/

# Or explicitly opt into live yfinance instead of local CSVs.
unset QUANTCORTEX_PRICES_CSV QUANTCORTEX_OHLCV_CSV
export QUANTCORTEX_LIVE_YFINANCE=1
jupyter lab research/
```

Set exactly one price source. Live yfinance use is subject to the provider's
[terms and legal disclaimer](https://ranaroussi.github.io/yfinance/); confirm
that your use is permitted. Notebook outputs are intentionally not committed.

### Validation & operations scripts

```bash
python scripts/validate_performance.py --live-yfinance
python scripts/validate_performance.py --live-yfinance --pit
python scripts/generate_report.py --prices-csv local_data/rotation_prices.csv
python scripts/generate_report.py --live-yfinance
python scripts/survivorship_demo.py --live-yfinance # quantify survivorship bias
python scripts/verify_brokers.py              # broker adapters vs faithful SDK mocks (no account)
python scripts/paper_trade_cycle.py --offline # labeled synthetic dry-run; no broker calls
```

`validate_performance.py` explicitly fetches live yfinance data; `--pit` uses a
fixed start-date cohort from historical index membership. `generate_report.py`
accepts either an owner-supplied wide CSV or explicit live yfinance, writes three
charts under ignored `reports/img/`, and prints source metadata plus markdown
tables. `survivorship_demo.py` requires the same explicit live-data opt-in and
shows the current pricing gap for past index members. `verify_brokers.py` exercises the
Alpaca/IB/CCXT adapters end-to-end against SDK-shaped mocks (request build +
response parsing); it does not verify a live SDK/authenticated connection.
`paper_trade_cycle.py` runs the full execution path (use
`--live-yfinance --submit` with `ALPACA_*` set to place paper orders).

> **On UI:** quantcortex is a library + notebooks + exported charts, like its
> peers (qlib, zipline, vectorbt). There is no bundled web app by design - a
> heavy SPA would be maintenance overhead for a research platform. Results are
> surfaced through notebooks and locally generated reports; an
> optional Streamlit dashboard could be layered on if interactive exploration is
> wanted, but it is intentionally out of the core.

### Paper trading (Phase 4)

Copy `.env.example` to `.env`, add your Alpaca / Interactive Brokers credentials,
then drive one rebalance cycle through `research/05_live_trading_bridge.ipynb`
against your paper account. Or bring up the full stack (app + Redis +
TimescaleDB) with `docker compose up`.

---

## Performance Reporting

No market-data snapshot, executed notebook output, generated chart, or fixed
performance number is published in this repository. This avoids redistributing
provider data and prevents a changing data vintage from looking immutable.

For a licensed local dataset, run:

```bash
PYTHONPATH=. python scripts/generate_report.py \
  --prices-csv local_data/rotation_prices.csv \
  --n-trials 10  # replace 10 with the actual configurations tested
```

The report records the file path, SHA-256 digest, observed date window, cost
assumptions, DSR trial count, and whether liquidity constraints are active.
Generated charts remain under ignored `reports/`. See
[PERFORMANCE.md](PERFORMANCE.md) for interpretation requirements and known
limitations.

---

## Architecture

The platform is organized as seven composable layers. Each layer produces or consumes the same weight vector interface, so any component can be swapped without touching downstream code.

| Layer | Role | Key modules |
|-------|------|-------------|
| **Data** | Point-in-time clean market + alternative data | `providers/`, `pit_enforcer.py`, `lookahead_detector.py` |
| **Alpha** | Factor research, ML signals, NLP sentiment | `factors/`, `alpha158.py`, `feature_engineering/` |
| **Portfolio** | Weight optimization (MV, HRP, RL) | `equal_weight.py`, `hrp.py`, `drl_allocator.py` |
| **Timing** | Regime detection, momentum overlays | `hmm_regime.py`, `tsmom.py`, `vix_scaler.py` |
| **Risk** | Drawdown limits, VaR/CVaR, Kelly sizing | `circuit_breaker.py`, `var_cvar.py`, `vol_targeting.py` |
| **Backtest** | Walk-forward validation, pitfall detection | `walk_forward.py`, `deflated_sharpe.py`, `lookahead_audit.py` |
| **Execution** | Live broker routing, order/position mgmt | `brokers/`, `order_manager.py`, `pre_trade_risk.py` |

### Weight Contract

Every **portfolio optimizer** output satisfies the strict contract (enforced at
runtime by `enforce_weight_contract`):

```python
# output: np.ndarray, shape (n_assets,)
# dtype:  float64
# sum:    1.0  (long-only) or 0.0 (market-neutral)
# range:  each weight in [-1.0, 1.0]
# violation raises: WeightContractViolationError
```

Timing and risk **overlays** legitimately scale gross exposure down (a fully
de-risked book is flat, a half-scaled long-only book sums to 0.5 with the
remainder in cash), so the *post-overlay* strategy output satisfies the relaxed
**exposure contract** (`enforce_exposure_contract`): finite, 1-D float64, each
weight in `[-1.0, 1.0]`, and gross (`sum |w|`) no greater than the input. In
other words, `sum == 1.0` holds at the allocation layer; `sum <= 1.0` holds
after timing and risk scaling.

---

## Repository Structure

All importable code lives under the single top-level `quantcortex` package, so
it installs and imports without colliding with any other project's modules:
`from quantcortex.portfolio.base import enforce_weight_contract`.

```
quantcortex/                     # repo root
├── quantcortex/                 # the importable package (import quantcortex.*)
│   ├── data/
│   │   ├── providers/          # base.py ABC + yfinance, Polygon, Alpaca, CCXT, FRED, FMP
│   │   ├── processors/         # calendar.py, adjustments.py, pit_enforcer.py, lookahead_detector.py
│   │   ├── storage/            # parquet_store.py, timescale_store.py, redis_cache.py
│   │   ├── universe/           # base ABC, sp500/nasdaq100 + sp500_wikipedia.py (PIT)
│   │   └── local_csv.py        # validated owner-supplied CSV loaders
│   │
│   ├── alpha/
│   │   ├── factors/
│   │   │   ├── classical/      # momentum, value, quality, low-vol (+ _cross_section helpers)
│   │   │   ├── ml/             # GBDT (XGBoost/LightGBM/CatBoost), neural
│   │   │   └── nlp/            # finbert_sentiment.py, news_scorer.py
│   │   ├── validation/         # alphalens_report.py, factor_decay.py
│   │   └── feature_engineering/ # alpha158.py, macro_features.py
│   │
│   ├── portfolio/
│   │   ├── base.py             # Abstract ABC with weight contract enforcement
│   │   ├── equal_weight.py
│   │   ├── mean_variance.py
│   │   ├── minimum_variance.py
│   │   ├── risk_parity.py
│   │   ├── hrp.py              # Hierarchical Risk Parity (López de Prado)
│   │   ├── black_litterman.py
│   │   └── drl_allocator.py    # PPO-based RL allocator
│   │
│   ├── timing/
│   │   ├── hmm_regime.py       # Hidden Markov Model regime detection
│   │   ├── vix_scaler.py       # VIX-based vol scaling
│   │   ├── tsmom.py            # Time-series momentum
│   │   └── kama.py             # Kaufman Adaptive Moving Average
│   │
│   ├── risk/
│   │   ├── circuit_breaker.py  # Hard stop on drawdown threshold
│   │   ├── var_cvar.py         # Historical & parametric VaR/CVaR
│   │   ├── vol_targeting.py    # Annualized vol targeting
│   │   ├── factor_exposure.py  # Barra-style factor exposure limits
│   │   └── kelly.py            # Fractional Kelly sizing
│   │
│   ├── backtest/
│   │   ├── engines/
│   │   │   ├── vectorized.py   # Fast NumPy/pandas vectorized engine
│   │   │   ├── event_driven.py # Tick-level event loop
│   │   │   └── walk_forward.py # Expanding/rolling WFO with embargo
│   │   ├── execution_models/
│   │   │   ├── ideal_fill.py
│   │   │   ├── vwap_fill.py
│   │   │   └── market_impact.py  # Almgren-Chriss market impact
│   │   ├── costs/
│   │   │   └── transaction_costs.py  # 3bps commission + 10bps slippage
│   │   ├── validation/
│   │   │   ├── deflated_sharpe.py    # Bailey & López de Prado DSR
│   │   │   ├── multiple_testing.py   # BHY correction
│   │   │   ├── lookahead_audit.py    # Automated look-ahead bias detection
│   │   │   └── survivorship_check.py
│   │   └── metrics/
│   │       └── tearsheet.py    # Full pyfolio-compatible tearsheet
│   │
│   ├── execution/
│   │   ├── brokers/
│   │   │   ├── base.py
│   │   │   ├── alpaca_broker.py
│   │   │   ├── ib_broker.py        # Interactive Brokers via ib_insync
│   │   │   └── ccxt_broker.py      # 100+ crypto exchanges
│   │   ├── order_manager.py
│   │   ├── position_manager.py
│   │   ├── state_persistence.py    # Redis-backed state across restarts
│   │   └── pre_trade_risk.py       # Pre-flight weight contract check
│   │
│   └── strategies/
│       ├── base_strategy.py
│       ├── momentum_ml.py          # GBDT cross-sectional momentum
│       ├── macro_timing.py         # Macro regime + asset rotation
│       ├── drl_portfolio.py        # PPO end-to-end RL portfolio
│       ├── sentiment_nlp.py        # FinBERT earnings sentiment overlay
│       └── multi_asset_rotation.py # Growth/Real Assets/Defensive rotation
│
├── research/                       # Jupyter notebooks (add repo root to sys.path)
│   ├── 01_data_quality.ipynb
│   ├── 02_factor_research.ipynb
│   ├── 03_portfolio_construction.ipynb
│   ├── 04_backtest_analysis.ipynb
│   └── 05_live_trading_bridge.ipynb
│
├── scripts/
│   ├── validate_performance.py  # explicit live-data validation (--pit: PIT universe)
│   ├── generate_report.py       # charts + markdown from an explicit data source
│   ├── paper_trade_cycle.py     # one rebalance cycle (offline / Alpaca paper)
│   ├── survivorship_demo.py     # quantify S&P 500 survivorship bias (PIT)
│   └── verify_brokers.py        # broker adapters vs faithful SDK mocks
│
├── tests/
│   ├── conftest.py             # shared synthetic-data fixtures
│   ├── test_lookahead_detector.py
│   ├── test_transaction_costs.py
│   ├── test_weight_interface.py
│   ├── test_risk_overlay.py
│   ├── test_order_manager.py
│   └── test_regression_guards.py  # core-dep regression guards (audit fixes)
│
├── local_data/README.md         # ignored local-data schemas and provenance rules
├── reports/                     # ignored generated charts and report output
├── docs/history-rewrite-plan.md # optional purge procedure; not executed
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── .env.example
├── .dockerignore               # excludes secrets, local data, and reports
├── .gitignore
├── PERFORMANCE.md              # evaluation method and reporting requirements
└── LICENSE
```

---

## Key Design Principles

### 1. Point-in-Time (PIT) Discipline
Financial report data should use **announcement dates**, not period-end dates.
`pit_enforcer.py` rejects records that violate that contract when the enforcer
is applied.

### 2. Walk-Forward Validation with Embargo
The walk-forward engine supports expanding or rolling windows plus a purge and
embargo gap. Research code must select and use it when models are refit or
configurations are evaluated over time.

### 3. Deflated Sharpe Ratio (DSR)
The report generator includes DSR (Bailey & López de Prado, 2014) to account
for multiple testing and non-normal return distributions:

```
DSR = Phi[ (SR* - SR0)*sqrt(T-1) / sqrt(1 - gamma3*SR* + (gamma4-1)/4*SR*^2) ]
```

Where `SR*` = observed max Sharpe, `SR0` = expected max under the null, `gamma3` = skewness, `gamma4` = excess kurtosis.

### 4. Backtesting Pitfall Controls
1. **Look-ahead bias** - `lookahead_audit.py` and PIT checks detect leakage when invoked.
2. **Overfitting** - DSR and BHY multiple-testing utilities quantify trial risk.
3. **Survivorship bias** - point-in-time universe utilities reconstruct historical membership; delisted-security prices still require an appropriate feed.
4. **Data adjustment errors** - processors validate split/dividend-adjusted inputs.
5. **Multiple testing bias** - BHY correction is available for factor and strategy tests.
6. **Transaction cost neglect** - every backtest engine requires a cost model.
7. **Liquidity assumptions** - a 10% ADV cap applies only when ADV data is supplied.

### 5. Transaction Cost Model
```python
commission  = 0.0003   # 3 bps
slippage    = 0.0010   # 10 bps
volume_cap  = 0.10     # max 10% of ADV when an ADV series is supplied
```

---

## ML / AI Stack

| Technique | Use case | Module |
|-----------|----------|--------|
| XGBoost / LightGBM / CatBoost | Cross-sectional alpha (GBDT dominates tabular financial data) | `quantcortex/alpha/factors/ml/` |
| PPO (Stable-Baselines3) | End-to-end RL portfolio allocation | `quantcortex/portfolio/drl_allocator.py` |
| Hidden Markov Model | Regime detection (bull/bear/sideways) | `quantcortex/timing/hmm_regime.py` |
| FinBERT | Earnings call & news sentiment scoring | `quantcortex/alpha/factors/nlp/` |
| Hierarchical Clustering (HRP) | Robust portfolio construction without inverting covariance | `quantcortex/portfolio/hrp.py` |

---

## Strategies

### Multi-Asset Rotation (`quantcortex/strategies/multi_asset_rotation.py`)
- **Universe:** Growth (QQQ, VGT), Real Assets (GLD, TLT), Defensive (SPY, VIG)
- **Rebalance:** Weekly
- **Selection:** Information Ratio relative to QQQ
- **Allocation:** Residual momentum within selected asset groups
- **Risk gate:** HMM regime + VIX scaling
- **Design target:** Sharpe > 1.10; this is aspirational, not a published result.

### Momentum ML (`quantcortex/strategies/momentum_ml.py`)
- GBDT cross-sectional momentum with alpha158 features
- Walk-forward refit every quarter
- **Design target:** Sharpe > 0.9; this is aspirational, not a published result.

### DRL Portfolio (`quantcortex/strategies/drl_portfolio.py`)
- PPO agent trained on rolling 3-year windows
- Action space: continuous weight vector over universe
- Reward: risk-adjusted return minus transaction costs

---

## Development Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1** | Data layer + PIT enforcement + universe construction | Complete |
| **Phase 2** | Alpha factor library + walk-forward validation harness | Complete |
| **Phase 3** | Portfolio construction + backtest engines + DSR reporting | Complete |
| **Phase 4** | Live execution layer (Alpaca paper -> IB live) | Adapters are behaviorally covered by SDK-shaped mocks (`scripts/verify_brokers.py`, 15/15), and `scripts/paper_trade_cycle.py` runs the full local cycle. Live SDK compatibility and the account-authenticated paper handshake remain unverified. |
| **Phase 5** | DRL allocator + FinBERT sentiment overlay | Complete |

---

## Framework Rationale

| Framework | Role in quantcortex | Not used for |
|-----------|---------------------|--------------|
| **vectorbt** | Fast parameter sweeps in research notebooks | Live trading |
| **qlib** | ML alpha factor benchmarks | Broker connectivity |
| **Lean/QuantConnect** | Reference event-driven engine comparison | Primary architecture |
| **FinRL-X** | Weight-contract interface pattern | Direct dependency |

---

## References

- Bailey, D. & López de Prado, M. (2014). *The Deflated Sharpe Ratio.* Journal of Portfolio Management.
- Yang, H. et al. (2026). *FinRL-X: An AI-Native Modular Infrastructure for Quantitative Trading.* [arXiv:2603.21330](https://arxiv.org/abs/2603.21330).
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Qian, E. (2005). *Risk Parity Portfolios.* PanAgora Asset Management.

---

*MIT licensed.*
