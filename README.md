# quantcortex

> **State-of-the-art modular quant trading platform**
> Data → Alpha → Portfolio → Timing → Risk → Backtest → Execution

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Private](https://img.shields.io/badge/visibility-private-red.svg)]()

---

## Overview

**quantcortex** is a research-grade, production-consistent quant trading platform built around a strict **weight-centric interface contract** (inspired by FinRL-X). Every layer — from alpha signal to live broker — speaks the same language: a normalized weight vector `w_t ∈ ℝⁿ`.

This eliminates the most common gap in quant stacks: strategies that backtest cleanly but behave differently in paper and live trading because the architecture changes between environments.

```
w_t = R_t( T_t( A_t( S_t( X≤t ) ) ) )
       ↑       ↑       ↑       ↑
     Risk   Timing  Alloc  Selection
```

**Target performance (2018–2025 backtest):**
- Multi-asset rotation: Sharpe > 1.10
- Momentum ML: Sharpe > 0.9
- Max drawdown < 15% with vol-targeting risk overlay

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

All portfolio and strategy components must satisfy this contract (enforced at runtime):

```python
# output: np.ndarray, shape (n_assets,)
# dtype:  float64
# sum:    1.0  (long-only) or 0.0 (market-neutral)
# range:  each weight ∈ [-1.0, 1.0]
# violation raises: WeightContractViolationError
```

---

## Repository Structure

```
quantcortex/
├── data/
│   ├── providers/          # yfinance, Polygon, Alpaca, CCXT, FRED, FMP
│   ├── processors/         # calendar.py, adjustments.py, pit_enforcer.py
│   ├── storage/            # parquet_store.py, timescale_store.py, redis_cache.py
│   └── universe/           # sp500_universe.py, nasdaq100_universe.py
│
├── alpha/
│   ├── factors/
│   │   ├── classical/      # momentum, value, quality, low-vol
│   │   ├── ml/             # GBDT (XGBoost/LightGBM/CatBoost), neural
│   │   └── nlp/            # FinBERT sentiment, earnings call scoring
│   ├── validation/         # alphalens_report.py, factor_decay.py
│   └── feature_engineering/ # alpha158.py, macro_features.py
│
├── portfolio/
│   ├── base.py             # Abstract ABC with weight contract enforcement
│   ├── equal_weight.py
│   ├── mean_variance.py
│   ├── minimum_variance.py
│   ├── risk_parity.py
│   ├── hrp.py              # Hierarchical Risk Parity (López de Prado)
│   ├── black_litterman.py
│   └── drl_allocator.py    # PPO-based RL allocator
│
├── timing/
│   ├── hmm_regime.py       # Hidden Markov Model regime detection
│   ├── vix_scaler.py       # VIX-based vol scaling
│   ├── tsmom.py            # Time-series momentum
│   └── kama.py             # Kaufman Adaptive Moving Average
│
├── risk/
│   ├── circuit_breaker.py  # Hard stop on drawdown threshold
│   ├── var_cvar.py         # Historical & parametric VaR/CVaR
│   ├── vol_targeting.py    # Annualized vol targeting
│   ├── factor_exposure.py  # Barra-style factor exposure limits
│   └── kelly.py            # Fractional Kelly sizing
│
├── backtest/
│   ├── engines/
│   │   ├── vectorized.py   # Fast NumPy/pandas vectorized engine
│   │   ├── event_driven.py # Tick-level event loop
│   │   └── walk_forward.py # Expanding/rolling WFO with embargo
│   ├── execution_models/
│   │   ├── ideal_fill.py
│   │   ├── vwap_fill.py
│   │   └── market_impact.py  # Almgren-Chriss market impact
│   ├── costs/
│   │   └── transaction_costs.py  # 3bps commission + 10bps slippage
│   ├── validation/
│   │   ├── deflated_sharpe.py    # Bailey & López de Prado DSR
│   │   ├── multiple_testing.py   # BHY correction
│   │   ├── lookahead_audit.py    # Automated look-ahead bias detection
│   │   └── survivorship_check.py
│   └── metrics/
│       └── tearsheet.py    # Full pyfolio-compatible tearsheet
│
├── execution/
│   ├── brokers/
│   │   ├── base.py
│   │   ├── alpaca_broker.py
│   │   ├── ib_broker.py        # Interactive Brokers via ib_insync
│   │   └── ccxt_broker.py      # 100+ crypto exchanges
│   ├── order_manager.py
│   ├── position_manager.py
│   ├── state_persistence.py    # Redis-backed state across restarts
│   └── pre_trade_risk.py       # Pre-flight weight contract check
│
├── strategies/
│   ├── base_strategy.py
│   ├── momentum_ml.py          # GBDT cross-sectional momentum
│   ├── macro_timing.py         # Macro regime + asset rotation
│   ├── drl_portfolio.py        # PPO end-to-end RL portfolio
│   ├── sentiment_nlp.py        # FinBERT earnings sentiment overlay
│   └── multi_asset_rotation.py # Growth/Real Assets/Defensive rotation
│
├── research/
│   ├── 01_data_quality.ipynb
│   ├── 02_factor_research.ipynb
│   ├── 03_portfolio_construction.ipynb
│   ├── 04_backtest_analysis.ipynb
│   └── 05_live_trading_bridge.ipynb
│
├── tests/
│   ├── test_lookahead_detector.py
│   ├── test_transaction_costs.py
│   ├── test_weight_interface.py
│   ├── test_risk_overlay.py
│   └── test_order_manager.py
│
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## Key Design Principles

### 1. Point-in-Time (PIT) Discipline
Financial report data uses **announcement dates**, not period-end dates. `pit_enforcer.py` raises at ingestion time if any feature would introduce forward-looking information.

### 2. Walk-Forward Validation with Embargo
All strategy evaluation uses expanding or rolling walk-forward optimization. An embargo gap between train and test windows purges samples whose label windows overlap, preventing subtle leakage.

### 3. Deflated Sharpe Ratio (DSR)
All strategy results are reported with DSR (Bailey & López de Prado, 2014) to account for multiple testing and non-normal return distributions:

```
DSR = Φ[ (SR* − SR₀)·√(T−1) / √(1 − γ₃·SR* + (γ₄−1)/4·SR*²) ]
```

Where `SR*` = observed max Sharpe, `SR₀` = expected max under the null, `γ₃` = skewness, `γ₄` = excess kurtosis.

### 4. Seven Backtesting Pitfall Categories (enforced programmatically)
1. **Look-ahead bias** — `lookahead_audit.py` detects future data leakage
2. **Overfitting** — DSR + BHY multiple-testing correction
3. **Survivorship bias** — `survivorship_check.py` validates universe construction
4. **Data adjustment errors** — split/dividend-adjusted price validation
5. **Multiple testing bias** — BHY correction on all factor IC tests
6. **Transaction cost neglect** — costs mandatory in all backtest engines
7. **Liquidity assumptions** — volume limit: ≤ 10% of 20-day ADV per symbol

### 5. Transaction Cost Model
```python
commission  = 0.0003   # 3 bps
slippage    = 0.0010   # 10 bps
volume_cap  = 0.10     # max 10% of 20-day ADV
```

---

## ML / AI Stack

| Technique | Use case | Module |
|-----------|----------|--------|
| XGBoost / LightGBM / CatBoost | Cross-sectional alpha (GBDT dominates tabular financial data) | `alpha/factors/ml/` |
| PPO (Stable-Baselines3) | End-to-end RL portfolio allocation | `portfolio/drl_allocator.py` |
| Hidden Markov Model | Regime detection (bull/bear/sideways) | `timing/hmm_regime.py` |
| FinBERT | Earnings call & news sentiment scoring | `alpha/factors/nlp/` |
| Hierarchical Clustering (HRP) | Robust portfolio construction without inverting covariance | `portfolio/hrp.py` |

---

## Strategies

### Multi-Asset Rotation (`strategies/multi_asset_rotation.py`)
- **Universe:** Growth (QQQ, VGT), Real Assets (GLD, TLT), Defensive (SPY, VIG)
- **Rebalance:** Weekly
- **Selection:** Information Ratio relative to QQQ
- **Allocation:** Residual momentum within selected asset groups
- **Risk gate:** HMM regime + VIX scaling
- **Target:** Sharpe > 1.10 (2018–2025)

### Momentum ML (`strategies/momentum_ml.py`)
- GBDT cross-sectional momentum with alpha158 features
- Walk-forward refit every quarter
- Target: Sharpe > 0.9

### DRL Portfolio (`strategies/drl_portfolio.py`)
- PPO agent trained on rolling 3-year windows
- Action space: continuous weight vector over universe
- Reward: risk-adjusted return minus transaction costs

---

## Development Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1** | Data layer + PIT enforcement + universe construction | ✅ Complete |
| **Phase 2** | Alpha factor library + walk-forward validation harness | ✅ Complete |
| **Phase 3** | Portfolio construction + backtest engines + DSR reporting | ✅ Complete |
| **Phase 4** | Live execution layer (Alpaca paper → IB live) | ✅ Code complete, tested offline — live broker round-trip pending account credentials |
| **Phase 5** | DRL allocator + FinBERT sentiment overlay | ✅ Complete |

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
- Liu, X. et al. (2024). *FinRL-X: A Unified Framework for Financial Reinforcement Learning.* arXiv:2603.21330.
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Qian, E. (2005). *Risk Parity Portfolios.* PanAgora Asset Management.

---

*Private repository — magnaquant*
