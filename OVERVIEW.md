# Systematic Trading System — Interviewer Overview

A self-contained portfolio project demonstrating a production-grade systematic
trading system built from first principles in Python.

---

## What this is

An end-to-end MA crossover trading system where the same strategy code runs
unchanged in backtest, paper trading, and live testnet execution. The system
design — not the alpha — is the artifact.

**North Star invariant:** the strategy never knows whether it is running on
historical parquet data or a live WebSocket feed. The event pipeline is
identical in both cases.

---

## Run it in 30 seconds

```bash
# Full backtest pipeline (Phases 0-2 + BTC + portfolio demo)
python main.py

# Live paper session (press Ctrl+C to stop)
python live/runner.py

# Multi-asset live session (BTCUSDT + ETHUSDT concurrently)
python live/runner.py --portfolio

# All 133 tests
python -m pytest tests/ -v
```

---

## Architecture in one diagram

```
DATA FEED          STRATEGY          PORTFOLIO          BROKER
(bars)        →   (signal)      →   (order)        →   (fill)

MarketEvent   →   SignalEvent   →   OrderEvent     →   FillEvent
     │                                                      │
  BACKTEST: parquet file (data/loader.py)                   │
  LIVE:     Binance WebSocket (feed/binance.py) ────────────┘
```

The four event types are defined once (`backtest/event_driven.py`) and
flow unchanged through both environments. `_Portfolio` and `_Broker` are
imported literally the same objects in the live engine.

---

## The six phases

| Phase | What was built | Key file |
|-------|---------------|----------|
| 0 | Reproducible OHLCV download + parquet cache | `data/loader.py` |
| 1 | Vectorized backtest with fees and IS/OOS split | `backtest/vectorized.py` |
| 2 | Event-driven backtester — structural look-ahead prevention | `backtest/event_driven.py` |
| 3 | Binance WebSocket live feed — same portfolio/broker objects | `live/engine.py` |
| 4 | Real testnet execution — HMAC-signed orders, actual fills | `execution/binance_broker.py` |
| 5 | Risk layer — vol targeting, circuit-breaker, slippage | `backtest/event_driven.py` |
| 6 | Multi-asset portfolio engine + concurrent live runner | `backtest/portfolio.py` |

---

## Key engineering decisions worth discussing

**1. Look-ahead prevention by structure, not discipline**

The vectorized engine relies on `signals.shift(1)` — a one-line discipline
check. The event-driven engine makes look-ahead structurally impossible:
bar T's close price literally does not exist in the system when the strategy
decides what to do. The live system is identical — you can't act on a candle
that hasn't closed yet.

The parity test proves they agree to `1e-9` on every column.

**2. The bootstrap alignment problem (Phase 3)**

A subtle off-by-one: to match the backtest's first-bar initialisation, the
live engine needs to observe *two* consecutive signals from bootstrap history,
not one. This requires `slow+1` bootstrap bars (not `slow`). With `slow` bars,
the first live bar charges a spurious fee whenever the signal changed on the
last bootstrap bar. Seven parity test scenarios verify this to `1e-9`.

**3. Vol targeting with no look-ahead (Phase 5)**

The vectorized engine computes `market_return.shift(1).rolling(N).std()` —
the extra shift ensures at bar t we only see returns through t-1.
The event-driven engine fills its deque in `on_fill()`, which is called
*after* the bar's return is known — so `on_signal()` at bar t+1 sees
exactly the same window. Both produce identical scaled positions.

**4. Portfolio equity is not the average of equity curves (Phase 6)**

A common mistake: average per-asset equity curves. The correct calculation
compounds the portfolio *return* — the equal-weighted mean of per-asset
net returns each period. The difference is small short-term but significant
over time (Jensen's inequality). A test explicitly verifies this.

**5. Single source of truth for parameters**

Every numeric parameter (MA windows, fee bps, vol target, risk thresholds,
trading pairs) lives in `config.yaml`. No Python file contains hardcoded
strategy constants. The annualisation factor map (`PERIODS_PER_YEAR`) is
defined once in `backtest/metrics.py` and imported by `live/engine.py` and
`live/runner.py`.

---

## Numbers to know

```
Test suite     : 133 tests, 0 failures
Engine parity  : vectorized vs event-driven, 0.00e+00 max diff
Phases         : 6 complete
Lines of code  : ~1,200 (excluding tests)
Test coverage  : ~1,100 lines of tests across 9 test files
```

---

## What this does NOT do

- No alpha. The 50/200 MA crossover is a textbook rule; the strategy
  underperforms buy-and-hold on most periods. That is expected and intentional.
- No portfolio optimisation (no covariance matrix, no mean-variance weights).
  Phase 6 uses equal weighting to keep the math auditable.
- No order book or limit orders — market orders only, with a simulated
  slippage model.
- No live P&L tracking beyond the session window.

---

## Files to read first

1. `GUIDE.md` — complete mathematical and architectural reference
2. `backtest/event_driven.py` — the core: events, `_Portfolio`, `_Broker`
3. `live/engine.py` — the North Star made literal
4. `tests/test_live_engine.py` — the bootstrap alignment parity tests
5. `backtest/portfolio.py` — Phase 6 portfolio engine
