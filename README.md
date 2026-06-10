# Systematic Trading System

An end-to-end systematic trading system built as a quant-developer learning project.
The goal is to prove engineering correctness — reproducible data, event-driven backtesting,
and a single strategy code path that runs unchanged in both backtest and live trading.

## Strategies

Switch strategies by changing one line in `config.yaml`:

 `ma_crossover` : Long when fast SMA > slow SMA (trend-following) 
 `ema_crossover` : Same but uses exponential MAs — reacts faster to recent prices 
 `rsi` : Long when RSI is oversold, flat when overbought (mean-reversion) 
 `bollinger_bands` : Long at lower Bollinger Band, flat at upper band (mean-reversion)
 `momentum` : Long when price is higher than N bars ago (rate-of-change) 
 `macd` : Long when MACD line crosses above its signal line 
 `mean_reversion` : Long when z-score falls below a threshold, flat when it reverts 

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
    └── binance_broker.py    # TestnetBroker — real orders via Binance Spot Testnet (Phase 4)
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
