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
