# Data Flow Guide

This document traces how data moves through the platform, folder by folder.
Three independent execution paths share common building blocks; understanding
where they diverge (and why) is the fastest way to navigate the codebase.

---

## The three paths at a glance

```
PATH 1 — Single-asset backtest (MA crossover)
─────────────────────────────────────────────
data/loader        strategy/ma_crossover     backtest/vectorized
get_bars()    ──►  generate_signals()   ──►  run()
                                              │
                                              ▼
                                         backtest/event_driven
                                         run()
                                              │
                                              ▼
                                         backtest/metrics
                                         compute_metrics()

PATH 2 — Multi-asset research backtest (allocation strategies)
──────────────────────────────────────────────────────────────
data/loader        risk/covariance       alloc/              strategy/rebalance
get_panel()   ──►  ledoit_wolf()    ──►  erc_weights()  ──►  generate_weights()
                   sample_cov()          gmv_weights()             │
                   factor_cov()          mvp_weights()             ▼
                                         bl.posterior_mean    backtest/multiasset
                                                              run()
                                                                   │
                                                                   ▼
                                                              backtest/metrics
                                                              compute_metrics()

PATH 3 — Live trading (WebSocket + testnet)
───────────────────────────────────────────
feed/binance          live/engine           execution/binance_broker
bootstrap()  ──►  LiveEngine.on_bar()  ──►  TestnetBroker.on_order()
stream()           (uses _Portfolio,         (real HMAC-signed order)
                    _Broker from                     │
                    backtest/event_driven)            ▼
                        │                      FillEvent (real fill)
                        ▼
                   backtest/metrics
                   compute_metrics()
```

---

## Folder responsibilities

### `data/`

**Produces:** clean OHLCV DataFrames with no NaN.
**Consumes:** Yahoo Finance via `yfinance` (network or parquet cache).
**Isolation rule:** nothing outside this folder imports `yfinance`.

```
data/loader.py
├── get_bars(symbol, start, end)  →  DataFrame(date × [open,high,low,close,volume])
│     Cache key: data/cache/{symbol}_{start}_{end}.parquet
│     First call: downloads + saves.  All subsequent calls: reads from disk.
│
└── get_panel(symbols, start, end)  →  DataFrame(date × symbols)
      Calls get_bars() once per symbol, joins on common dates, drops any NaN row.
      Used by: strategy/rebalance.py, tests/test_no_lookahead.py
```

The cache design means two things: every backtest is reproducible (the parquet
is frozen), and the rest of the system can call `get_bars()` freely without
worrying about network availability.

---

### `risk/`

**Produces:** covariance matrices `(N, N)`, scalar risk statistics.
**Consumes:** returns arrays `(T, N)` — no prices, no dates, no I/O.
**Isolation rule:** pure functions; no imports from any other project folder.

```
risk/covariance.py
├── sample_cov(returns)           →  (N,N) ndarray   — np.cov * 252
├── ledoit_wolf(returns)          →  (N,N) ndarray   — sklearn shrinkage * 252
├── factor_cov(returns, factors)  →  (N,N) ndarray   — BFB^T + D (annualised)
├── pca_cov(returns, k)           →  (N,N) ndarray   — top-k eigenvectors only
├── is_psd(Sigma)                 →  bool             — all eigenvalues >= -tol
├── nearest_psd(Sigma)            →  (N,N) ndarray   — clip negative eigenvalues
└── condition_number(Sigma)       →  float            — λ_max / λ_min

risk/var_cvar.py
├── var_parametric(mu, sigma, c)  →  float   — Gaussian quantile
├── var_historical(losses, c)     →  float   — empirical quantile
├── var_mc(mu, sigma, c)          →  float   — simulation
├── es_parametric(mu, sigma, c)   →  float   — Gaussian tail mean
├── es_historical(losses, c)      →  float   — tail sample mean
└── min_cvar_weights(returns, c)  →  (N,)    — Rockafellar-Uryasev LP (cvxpy)
```

All estimators in `covariance.py` conform to the `CovEstimator` protocol
(`risk/` → `alloc/protocol.py`): they accept a returns array and return a
matrix. This lets `strategy/rebalance.py` swap estimators with one argument.

---

### `alloc/`

**Produces:** weight vectors `(N,)` summing to 1.
**Consumes:** covariance matrices and optionally expected-return vectors.
**Isolation rule:** pure functions; imports only `numpy` and `cvxpy`.

```
alloc/protocol.py
├── Allocator     — Protocol: (mu, Sigma, **params) → (N,) ndarray
└── CovEstimator  — Protocol: (returns) → (N,N) ndarray

alloc/base.py
├── equal_weight(n)                     →  (N,)  — 1/N
├── gmv_weights(Sigma)                  →  (N,)  — closed-form unconstrained GMV
├── tangency_weights(mu, Sigma, rf)     →  (N,)  — max-Sharpe
├── risk_contributions(w, Sigma)        →  (N,)  — Euler RC_i = w_i(Σw)_i / σ_p
├── port_vol(w, Sigma)                  →  float
├── port_return(w, mu)                  →  float
├── log_returns(prices)                 →  DataFrame
└── sample_moments(returns)             →  (mu, Sigma)

alloc/mean_variance.py
└── mvp_weights(mu, Sigma, target, long_only, max_w)  →  (N,) | None
      Constrained QP via cvxpy: min w^TΣw s.t. μ^Tw >= target

alloc/risk_parity.py
└── erc_weights(mu, Sigma)  →  (N,) | None
      Maillard-Roncalli-Teïletche log-barrier: equal RC_i for all i
      Validation: max|RC_i - RC_j| < 1e-4

alloc/black_litterman.py
├── reverse_optimize(Sigma, w_mkt, lambda)  →  (N,)  — implied returns Π
└── posterior_mean(Sigma, Pi, P, q, Omega)  →  (N,)  — BL posterior μ̄
      No-view limit: Omega → ∞  ⟹  μ̄ → Π  (tested)
```

`gmv_weights` and `erc_weights` ignore `mu` — they need only `Sigma`.
This is intentional: return estimation error dominates the benefit of
using μ̂, so covariance-only methods are more robust out-of-sample.

---

### `strategy/`

**Produces:** signals / weight matrices with strict causal guarantee.
**Consumes:** price data from `data/`, covariance estimators from `risk/`, allocators from `alloc/`.

```
strategy/ma_crossover.py
└── generate_signals(bars, fast, slow)  →  pd.Series
      signal[t] = 1.0 if SMA(fast)[t] > SMA(slow)[t], else 0.0
      signal[t] = NaN for t < slow - 1  (warmup)
      NOTE: no 1-bar lag applied here — the engine does that.

strategy/rebalance.py
└── generate_weights(panel, allocator, cov_estimator, lookback, rebalance_every)
        →  pd.DataFrame (dates × symbols)
      At bar i: window = log_returns.iloc[i-lookback : i]  (strictly up to T-1)
      Calls allocator(None, cov_estimator(window)) every rebalance_every bars.
      Forward-fills between rebalance dates; fills pre-warmup with 0.
      CAUSAL GUARANTEE: corrupting bar i does not change weight at bar i.
```

The rebalancing adapter is the bridge between the research layer and the
execution layer. It knows about both but isolates them: the engine never
imports from `alloc/` or `risk/`; the allocators never import from `strategy/`.

---

### `backtest/`

**Produces:** result DataFrames (equity curves, returns, fees) and metric dicts.
**Consumes:** `data/` (bars/panels) and `strategy/` (signals/weights).
**Isolation rule:** does not import from `alloc/` or `risk/` directly.

```
backtest/vectorized.py
└── run(bars, signals, fee_bps)  →  DataFrame
      Scalar path, array-at-once. 1-bar lag via signals.shift(1).
      Columns: position, market_return, gross_return, fee, net_return, equity.
      Serves as ground truth for event_driven parity tests.

backtest/event_driven.py
├── _Portfolio  — holds scalar position; emits OrderEvent from SignalEvent
│     Phase 5: vol targeting (on_signal scales target) + circuit-breaker
├── _Broker     — fills at close; fee = |delta| × fee_rate
└── run(bars, signals, fee_bps, ...)  →  DataFrame
      Same columns as vectorized.run().
      INVARIANT: vectorized.run() == event_driven.run() to 1e-9 on all columns.
      Shared with live/engine.py — _Portfolio and _Broker imported directly.

backtest/multiasset.py
├── _Portfolio  — holds weight vector (N,); gross_return = w^T r  (dot product)
├── _Broker     — fee = turnover × fee_rate  where turnover = Σ|Δwᵢ|
└── run(panel, weights, fee_bps, ...)  →  DataFrame
      Columns: gross_return, fee, slippage, net_return, equity, turnover.
      INVARIANT: when N=1 and weights ∈ {0,1}, equals event_driven.run() to 1e-9.
      Tested in test_multiasset_parity.py across 8 scenarios.

backtest/metrics.py
└── compute_metrics(returns, positions)  →  dict
      total_return, ann_return, ann_vol, sharpe, max_drawdown, ann_turnover.
      periods_per_year param: 252 (daily), 8760 (1h), etc.

backtest/portfolio.py
└── run_portfolio(assets, fee_bps, ...)  →  PortfolioResult
      assets: dict[symbol → (bars, signals)]
      Runs event_driven.run() independently per asset, then:
      r_portfolio[t] = mean(r_net_i[t])   (equal weight 1/N)
      equity[t]      = cumprod(1 + r_portfolio)
      NOTE: this is NOT the average of equity curves.
```

The two portfolio engines solve different problems:
- `backtest/portfolio.py` — equal-weight, each asset has its own MA crossover signal
- `backtest/multiasset.py` — continuous weights from an allocation algorithm, single joint rebalancing strategy

---

### `feed/`

**Produces:** bar dicts `{timestamp, open, high, low, close, volume}`.
**Consumes:** Binance REST API (bootstrap) and WebSocket (stream).
**Isolation rule:** only called from `live/` — never from backtest paths.

```
feed/binance.py
└── BinanceFeed(symbol, interval, warmup_bars)
    ├── bootstrap()              →  list[dict]   — sync REST call, last N closed candles
    │     GET /api/v3/klines?symbol=BTCUSDT&interval=1h&limit=N
    │     Takes ~0.5s. Must be called before asyncio.run().
    └── stream()                 →  async generator[dict]
          wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}
          Filters to k.x == True (closed candles only).
          Reconnects with exponential backoff on disconnect.

Bar dict schema (identical from both endpoints):
    { timestamp: pd.Timestamp,  open: float,  high: float,
      low: float,  close: float,  volume: float }
```

The bar dict schema is the same format as a single row from `get_bars()`.
This is intentional: `LiveEngine` consumes it the same way the backtest
engine consumes `bars.iterrows()`.

---

### `live/`

**Produces:** row dicts (same schema as backtest result rows).
**Consumes:** `feed/binance.py` (bars), `backtest/event_driven.py` (_Portfolio, _Broker).
**North Star:** `_Portfolio` and `_Broker` are imported literally unchanged from `backtest/event_driven.py`.

```
live/engine.py
└── LiveEngine(portfolio, broker, fast, slow)
    ├── initialize(bootstrap_bars)    — warms up MA deque; sets portfolio.position
    │     Requires slow+1 bars to observe two consecutive signals.
    │     (With only slow bars, a spurious fee fires on the first live bar
    │      whenever the signal changed on the last bootstrap bar.)
    ├── on_bar(bar)  →  dict | None   — processes one closed candle
    │     window.append(close)
    │     current_signal = fast_ma > slow_ma  (from deque)
    │     ┌──────────────────────────────────────────────────────┐
    │     │  MarketEvent → SignalEvent(pending) → OrderEvent     │
    │     │            → _Portfolio.on_signal()                  │
    │     │            → _Broker.on_order()     → FillEvent      │
    │     │            → _Portfolio.on_fill()   → row dict       │
    │     └──────────────────────────────────────────────────────┘
    │     pending_signal = current_signal  (advance 1-bar lag)
    ├── get_results()  →  DataFrame
    └── get_metrics(interval)  →  dict

live/runner.py   — entry point
    ├── python live/runner.py            (single-asset, paper)
    ├── python live/runner.py --testnet  (single-asset, real testnet orders)
    └── python live/runner.py --portfolio (N assets concurrent, asyncio.gather)
```

---

### `execution/`

**Produces:** `FillEvent` with real fill price and real fee from exchange.
**Consumes:** `OrderEvent` from `live/engine.py`'s `_Portfolio`.
**Replaces:** `_Broker` from `backtest/event_driven.py` at the execution seam.

```
execution/binance_broker.py
└── TestnetBroker(api_key, api_secret, symbol, order_qty)
    └── on_order(OrderEvent)  →  FillEvent
          Submits HMAC-SHA256 signed market order to Binance Spot Testnet.
          Fee extracted from actual fills array (real BNB/USDT commission).
          Everything above this seam (_Portfolio, LiveEngine, events) is unchanged.
```

---

## What imports what

This graph shows allowed import directions. Arrows point from importer to imported.
A missing arrow means the import does not exist and must not be added.

```
config.yaml
    │
    ▼
main.py / live/runner.py
    │
    ├──► data/loader.py  ──────────────────────────────────────────(no upstream imports)
    │
    ├──► risk/covariance.py ────────────────────────────────────── numpy, sklearn
    ├──► risk/var_cvar.py  ─────────────────────────────────────── numpy, scipy, cvxpy
    │
    ├──► alloc/base.py ─────────────────────────────────────────── numpy
    ├──► alloc/mean_variance.py ────────────────────────────────── numpy, cvxpy
    ├──► alloc/risk_parity.py ──────────────────────────────────── numpy, cvxpy
    ├──► alloc/black_litterman.py ──────────────────────────────── numpy
    │
    ├──► strategy/ma_crossover.py ──────────────────────────────── pandas, numpy
    ├──► strategy/rebalance.py ─────────────────────────────────── alloc/protocol, numpy
    │
    ├──► backtest/vectorized.py ────────────────────────────────── pandas, numpy
    ├──► backtest/event_driven.py ──────────────────────────────── pandas
    ├──► backtest/multiasset.py ────────────────────────────────── pandas, numpy
    ├──► backtest/metrics.py ───────────────────────────────────── pandas, numpy
    ├──► backtest/portfolio.py ─────────────────────────────────── backtest/event_driven
    │
    ├──► feed/binance.py ───────────────────────────────────────── websockets, urllib
    ├──► live/engine.py ────────────────────────────────────────── backtest/event_driven
    ├──► live/runner.py ────────────────────────────────────────── live/engine, feed/binance
    │                                                               execution/binance_broker (--testnet)
    │
    └──► execution/binance_broker.py ───────────────────────────── backtest/event_driven (events only)
```

**Key isolation rules:**
- `alloc/` and `risk/` do not import from `data/`, `strategy/`, or `backtest/`
- `backtest/` does not import from `alloc/` or `risk/`
- `feed/` is never imported by `backtest/`
- `execution/` is never imported by `backtest/`

These rules mean you can unit-test any folder in complete isolation, and you
can swap implementations (e.g., different allocator, different exchange) by
changing exactly one call site.

---

## End-to-end example: multi-asset ERC backtest

Tracing a call to `backtest/multiasset.run()` driven by `strategy/rebalance.py`:

```python
# 1. Load prices
panel = get_panel(["AAPL","MSFT","GOOGL","AMZN","META"], "2020-01-01", "2024-01-01")
#    data/loader.py calls get_bars() per symbol → joins on common dates
#    returns DataFrame(dates × 5 symbols)

# 2. Build weight matrix
weights = generate_weights(
    panel,
    allocator     = erc_weights,     # alloc/risk_parity.py
    cov_estimator = ledoit_wolf,     # risk/covariance.py
    lookback      = 60,
    rebalance_every = 21,
)
#    strategy/rebalance.py loop:
#      at each rebalance bar i:
#        window  = log_returns(panel).iloc[i-60 : i].dropna()   ← strictly T-1
#        Sigma   = ledoit_wolf(window)         ← (5,5) shrunk covariance
#        w       = erc_weights(None, Sigma)    ← (5,) equal-risk weights
#        weights[i] = w
#      ffill between rebalance dates; 0 during warmup
#    returns DataFrame(dates × 5 symbols)

# 3. Run multi-asset backtest
result = multiasset.run(panel, weights, fee_bps=10.0)
#    backtest/multiasset.py:
#      lagged = weights.shift(1)      ← position at T decided by weights at T-1
#      asset_returns = panel.pct_change()
#      for each bar T:
#        signal_evt.target_weights = lagged[T]        ← (5,) vector
#        order_evt.deltas  = target - positions        ← (5,) changes
#        fill_evt.fee      = Σ|deltas| × fee_rate      ← turnover-based cost
#        gross_return      = positions @ asset_returns  ← dot product w^T r
#        equity            *= 1 + net_return
#    returns DataFrame(dates, columns=[gross_return, fee, net_return, equity, turnover])

# 4. Compute metrics
m = compute_metrics(result["net_return"], result["equity"])
#    backtest/metrics.py: sharpe, max_drawdown, ann_turnover, ...
```

---

## The parity chain

Every generalisation in this codebase must reduce to the simpler case it extends.
These are tested and must stay green:

```
vectorized.run(bars, signals)
    ‖  (to 1e-9, 8 scenarios)
event_driven.run(bars, signals)
    ‖  (to 1e-9, 7 scenarios, after bootstrap alignment)
LiveEngine.on_bar() driven by same bars

multiasset.run(panel_1col, weights_binary)
    ‖  (to 1e-9, 8 scenarios)
event_driven.run(bars, signals)

run_portfolio({"A": (bars, signals)})
    ‖  (to 1e-9)
event_driven.run(bars, signals)
```

If any of these break, there is an accounting bug. The tests in
`test_event_driven.py`, `test_live_engine.py`, `test_multiasset_parity.py`,
and `test_portfolio.py` are the first place to look.
