# Systematic Trading Platform

An end-to-end systematic trading system built from first principles.
The goal is engineering correctness: reproducible data, event-driven backtesting,
and a single strategy code path that runs unchanged in backtest and live trading.

The platform has two layers, fused into one codebase:

- **Execution engine** — event-driven backtester, live Binance WebSocket feed, testnet order routing
- **Allocation research library** — portfolio construction algorithms (MVO, risk parity, Black-Litterman, CVaR) that plug directly into the engine as a multi-asset backtestable allocation strategy

---

## Architecture

```
RESEARCH                  CONSTRUCTION              EXECUTION
────────                  ────────────              ─────────
risk/covariance.py   ──►  alloc/*.py          ──►   strategy/rebalance.py
  sample_cov               equal_weight               generate_weights()
  ledoit_wolf              gmv_weights                     │
  factor_cov               erc_weights                     ▼
  pca_cov                  mvp_weights         backtest/multiasset.run()
                           bl.posterior_mean              │
                                                          ▼
                                                   backtest/metrics.py
                                                   compute_metrics()

SINGLE-ASSET PATH (legacy, unchanged)
──────────────────────────────────────
data/loader.py ──► strategy/ma_crossover.py ──► backtest/event_driven.run()
                                                  ║  (same _Portfolio, _Broker)
                                                  ▼
                                             live/engine.py ──► feed/binance.py
                                                                 execution/binance_broker.py
```

> **North Star:** The same `_Portfolio` and `_Broker` objects run unchanged
> in backtest, live paper feed, and testnet execution.
> The same `Allocator` interface connects any portfolio construction method
> to the multi-asset engine — no glue code.

---

## Build Phases

### Core Engine (Phases 0–6)

| Phase | Goal | Status |
|-------|------|--------|
| 0 | Reproducible OHLCV download and parquet cache | Done |
| 1 | Vectorized backtest of a MA crossover with fees and IS/OOS split | Done |
| 2 | Event-driven backtester — structural look-ahead prevention | Done |
| 3 | Live paper feed — Binance WebSocket, backtest/live parity | Done |
| 4 | Testnet execution — real HMAC-signed orders, actual fills | Done |
| 5 | Risk layer — vol targeting, circuit-breaker, slippage | Done |
| 6 | Equal-weight multi-asset engine + concurrent live runner | Done |

### Allocation Research Layer (Phases A–D)

| Phase | Goal | Status |
|-------|------|--------|
| A | Extract allocation algorithms into testable `alloc/` + `risk/` packages | Done |
| B | Generalise the event-driven engine to continuous multi-asset weights | Done |
| C | Rebalancing strategy adapter — plugs any allocator into the engine | Done |
| D | Walk-forward evaluation: compare allocators OOS, net of costs | Planned |

---

## Project Structure

```
fullproject/
├── config.yaml              # all parameters — edit here, not in code
├── main.py                  # backtest entry point (Phases 0–6)
│
├── data/
│   ├── loader.py            # get_bars()   — single-asset OHLCV + cache
│   │                        # get_panel()  — multi-asset close panel
│   └── cache/               # parquet files (gitignored)
│
├── alloc/                   # ← portfolio construction algorithms (Phase A)
│   ├── protocol.py          #   Allocator + CovEstimator structural types
│   ├── base.py              #   equal_weight, gmv_weights, tangency_weights,
│   │                        #   risk_contributions, port_vol, port_return
│   ├── mean_variance.py     #   constrained QP frontier (cvxpy)
│   ├── risk_parity.py       #   Equal Risk Contribution (log-barrier)
│   └── black_litterman.py   #   reverse optimise + BL posterior mean
│
├── risk/
│   ├── covariance.py        #   sample_cov, ledoit_wolf, factor_cov, pca_cov
│   │                        #   is_psd, nearest_psd, condition_number
│   └── var_cvar.py          #   VaR / ES (parametric, historical, MC)
│                            #   min_cvar_weights (Rockafellar-Uryasev LP)
│
├── strategy/
│   ├── ma_crossover.py      #   generate_signals() — single-asset MA crossover
│   └── rebalance.py         #   generate_weights() — causal sliding-window
│                            #   adapter: allocator → (dates × N) weight matrix
│
├── backtest/
│   ├── vectorized.py        #   run() — array-at-once, Phase 1 ground truth
│   ├── event_driven.py      #   run() — scalar event loop; _Portfolio, _Broker
│   ├── multiasset.py        #   run() — vector event loop (Phase B)
│   │                        #   INVARIANT: N=1, binary weights ≡ event_driven
│   ├── metrics.py           #   compute_metrics() — return, vol, Sharpe, MDD, turnover
│   └── portfolio.py         #   run_portfolio() — equal-weight multi-asset (Phase 6)
│
├── feed/
│   └── binance.py           #   BinanceFeed — REST bootstrap + async WebSocket
│
├── live/
│   ├── engine.py            #   LiveEngine — online event loop (Phase 3)
│   └── runner.py            #   entry point: paper / testnet / --portfolio
│
├── execution/
│   └── binance_broker.py    #   TestnetBroker — real orders (Phase 4)
│
└── tests/
    ├── test_loader.py
    ├── test_vectorized.py
    ├── test_event_driven.py       # parity: vectorized vs event-driven (8 cases)
    ├── test_live_engine.py        # parity: backtest vs live (7 cases)
    ├── test_risk.py               # vol targeting, circuit-breaker (Phase 5)
    ├── test_portfolio.py          # equal-weight portfolio (Phase 6)
    ├── test_metrics.py
    ├── test_alloc.py              # alloc/ + risk/ unit tests (Phase A)
    ├── test_multiasset_parity.py  # N=1 reduction invariant (Phase B)
    ├── test_no_lookahead.py       # causal window discipline (Phase C)
    ├── test_binance_feed.py
    └── test_testnet_broker.py
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Full backtest pipeline
python main.py

# All tests
pytest tests/ -v

# Live paper session (Ctrl+C to stop)
python live/runner.py
```

First run downloads SPY and/or BTCUSDT bars from Yahoo Finance / Binance REST
and caches them to `data/cache/`. Every subsequent run reads from cache —
same data, no network call.

---

## Key Concepts

**Look-ahead prevention** — the event-driven engine makes look-ahead structurally
impossible. Bar T's close price does not exist in the system while the strategy
decides. The `generate_weights()` adapter enforces the same guarantee for
allocation strategies: the window at bar T covers strictly bars 0 through T-1.

**Backtest/live parity** — the North Star. `_Portfolio` and `_Broker` are
imported unchanged into the live engine. Seven parity tests verify to 1e-9.

**Reproducibility** — same inputs, same outputs. Every backtest traces to a
frozen parquet file. The alloc/risk functions are pure (no I/O, no randomness).

**Transaction cost realism** — the multi-asset engine charges fees on turnover
`Σ|Δwᵢ|`, not just scalar `|Δposition|`. Walk-forward results are always
reported net of costs — a strategy that beats equal-weight gross but not net
is explicitly flagged as such.

**Single allocation contract** — any function matching `(mu, Sigma) → w`
is a valid `Allocator`. Swap ERC for Ledoit-Wolf GMV for Black-Litterman
by changing one argument to `generate_weights()`. The engine never changes.

---

## Tech Stack

- **Data**: `yfinance`, cached as `parquet` via `pyarrow`
- **Numerics**: `pandas`, `numpy`, `scipy`
- **Optimisation**: `cvxpy` (MVO, ERC, min-CVaR LP)
- **Covariance shrinkage**: `scikit-learn` (`LedoitWolf`)
- **Live feed**: `websockets`, Binance REST + WebSocket API
- **Testing**: `pytest` — 130 tests, 0 failures
