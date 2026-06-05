# Systematic Trading System

An end-to-end systematic trading system built as a quant-developer learning project.
The goal is to prove engineering correctness — reproducible data, event-driven backtesting,
and a single strategy code path that runs unchanged in both backtest and live trading.

This project is bound to eveolve, testing multiple algorithms and implementations.

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
| 1 | Vectorized backtest of a simple strategy with fees | Done |
| 2 | Event-driven backtester (architectural core) | Done |
| 3 | Live paper data feed — backtest/live parity | Done |
| 4 | Testnet execution — real market orders via Binance Spot Testnet | Done |
| 5 | Risk, accounting, and correctness hardening | Next |
| 6 | Portfolio polish | Planned |

## Project Structure

```
fullproject/
├── config.yaml              # symbol, date range, backtest params — edit here, not in code
├── main.py                  # entry point (runs Phase 0 + Phase 1)
├── requirements.txt
├── data/
│   ├── loader.py            # get_bars() — the only public data interface
│   └── cache/               # parquet files (gitignored)
├── strategy/
│   └── ma_crossover.py      # generate_signals() — 50/200-day MA crossover
├── backtest/
│   ├── vectorized.py        # run() — vectorized backtest engine (Phase 1)
│   ├── event_driven.py      # run() — event-driven backtest engine (Phase 2)
│   └── metrics.py           # compute_metrics() — return, vol, Sharpe, drawdown, turnover
├── feed/
│   └── binance.py           # BinanceFeed — REST bootstrap + WebSocket stream (Phase 3)
├── live/
│   ├── engine.py            # LiveEngine — online event pipeline (Phase 3)
│   └── runner.py            # entry point: python live/runner.py [--testnet]
├── execution/
│   └── binance_broker.py    # TestnetBroker — real orders via Binance Spot Testnet (Phase 4)
├── tests/
│   ├── test_loader.py       # data layer tests
│   ├── test_vectorized.py   # backtest engine tests (including look-ahead check)
│   ├── test_event_driven.py # event-driven engine tests (including parity check)
│   ├── test_live_engine.py  # live engine tests (including backtest/live parity)
│   ├── test_testnet_broker.py # testnet broker tests (mocked HTTP)
│   └── test_metrics.py      # metrics unit tests
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
- **Performance metrics**: `empyrical` (Phase 1+)
- **Portfolio analytics**: `pyfolio` (Phase 1+)
- **Testing**: `pytest` + `hypothesis` (property-based, Phase 5)
- **Language**: Python, with a performance-critical component planned in C

## Key Concepts

**Look-ahead bias** — using information that wouldn't have existed at decision time.
The event-driven engine (Phase 2) makes this structurally impossible.

**Backtest/live parity** — the North Star. One strategy, two data sources.

**Reproducibility** — same inputs, same outputs. Every backtest traces to a
frozen, versioned dataset.

**Transaction cost realism** — fees, spread, and slippage are first-class inputs,
not footnotes. They change conclusions, not just magnitudes.
