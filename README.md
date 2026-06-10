# Systematic Trading System

An end-to-end systematic trading system built as a quant-developer learning project.
The goal is to prove engineering correctness — reproducible data, event-driven backtesting,
and a single strategy code path that runs unchanged in both backtest and live trading.

## Architecture

```
         +----------------------------------------------------------+
         |                        ENGINE                             |
         |              (the event loop / clock)                     |
         +----------------------------------------------------------+
              |            |            |            |
              v            v            v            v
        +---------+  +----------+  +-----------+  +-----------+
        |  DATA   |->| STRATEGY |->| PORTFOLIO |->| EXECUTION |
        |  feed   |  | (signal) |  |  & risk   |  |  (fills)  |
        +---------+  +----------+  +-----------+  +-----------+
              ^                                          |
              |            BACKTEST: historical file     |
              +----------  LIVE:     market data feed <--+
                          (same event types either way)
```

## Build Phases

| Phase | Goal | Status |
|-------|------|--------|
| 0 | Data layer — reproducible OHLCV download and parquet cache | Done |
| 1 | Vectorized backtest with fees and walk-forward split | Done |
| 2 | Event-driven backtester (architectural core) | Done |
| 3 | Live paper data feed — backtest/live parity | Done |
| 4 | Testnet execution — real market orders via Binance Spot Testnet | Done |
| 5 | Risk, accounting, and correctness hardening | Done |
| 6 | Portfolio engine — multi-asset equal-weight runner | Done |

## Strategies

Switch strategies by changing one line in `config.yaml`:

```yaml
strategy: ma_crossover   # ← change this
```

| Key | Description |
|-----|-------------|
| `ma_crossover` | Long when fast SMA > slow SMA (trend-following) |
| `ema_crossover` | Same but uses exponential MAs — reacts faster to recent prices |
| `rsi` | Long when RSI is oversold, flat when overbought (mean-reversion) |
| `bollinger_bands` | Long at lower Bollinger Band, flat at upper band (mean-reversion) |
| `momentum` | Long when price is higher than N bars ago (rate-of-change) |
| `macd` | Long when MACD line crosses above its signal line |
| `mean_reversion` | Long when z-score falls below a threshold, flat when it reverts |

Each strategy's parameters live under `strategies:` in `config.yaml` — no code changes needed.

## Project Structure

```
fullproject/
├── config.yaml              # single source of truth — strategy selection + all params
├── main.py                  # entry point (Phases 0-2, BTC validation, Phase 6)
├── requirements.txt
├── data/
│   ├── loader.py            # get_bars() — the only public data interface
│   └── cache/               # parquet files (gitignored)
├── strategy/
│   ├── __init__.py          # load_strategy(cfg) — config-driven strategy factory
│   ├── ma_crossover.py      # SMA crossover
│   ├── ema_crossover.py     # EMA crossover
│   ├── rsi.py               # RSI threshold with hysteresis
│   ├── bollinger_bands.py   # Bollinger Band mean-reversion
│   ├── momentum.py          # Rate-of-change momentum
│   ├── macd.py              # MACD line vs. signal line
│   └── mean_reversion.py    # Z-score mean-reversion
├── backtest/
│   ├── vectorized.py        # run() — vectorized backtest engine (Phase 1)
│   ├── event_driven.py      # run() — event-driven backtest engine (Phase 2)
│   ├── portfolio.py         # run_portfolio() — multi-asset equal-weight runner (Phase 6)
│   └── metrics.py           # compute_metrics() — return, vol, Sharpe, drawdown, turnover
├── feed/
│   └── binance.py           # BinanceFeed — REST bootstrap + WebSocket stream (Phase 3)
├── live/
│   ├── engine.py            # LiveEngine — online event pipeline (Phase 3)
│   └── runner.py            # entry point: python live/runner.py [--testnet]
├── execution/
│   └── binance_broker.py    # TestnetBroker — real orders via Binance Spot Testnet (Phase 4)
├── tests/
│   ├── test_loader.py
│   ├── test_vectorized.py
│   ├── test_event_driven.py
│   ├── test_live_engine.py
│   ├── test_testnet_broker.py
│   ├── test_metrics.py
│   ├── test_portfolio.py
│   ├── test_risk.py
│   └── test_binance_feed.py
└── notes/
    ├── phase0_data_layer.txt
    ├── phase1_vectorized_backtest.txt
    ├── phase2_event_driven.txt
    ├── phase3_live_feed.txt
    └── phase4_testnet.txt
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

First run downloads SPY daily bars from 2010–2024 and caches them to `data/cache/`.
Every subsequent run reads from cache — same data, no network call.

```bash
pytest tests/ -v
```

## Tech Stack

- **Data**: `yfinance` (Yahoo Finance), cached as `parquet` via `pyarrow`
- **Analysis**: `pandas`, `numpy`
- **Config**: `pyyaml`
- **Testing**: `pytest`
- **Language**: Python

## Key Concepts

**Look-ahead bias** — using information that wouldn't have existed at decision time.
The event-driven engine (Phase 2) makes this structurally impossible: signals computed
at bar T determine the position entered at bar T+1.

**Backtest/live parity** — the North Star. One strategy code path, two data sources.
The same `_Portfolio` and `_Broker` objects run in both the event-driven backtester
and the live engine.

**Reproducibility** — same inputs, same outputs. Every backtest traces to a
frozen, versioned dataset.

**Transaction cost realism** — fees, spread, and slippage are first-class inputs,
not footnotes. They change conclusions, not just magnitudes.
