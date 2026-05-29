# Building an End-to-End Systematic Trading System
### A project guide & learning roadmap (crypto, quant-developer track)

This is the document you keep open while you build. It is **not** a list of code to copy — it's the map: the architecture, the order to build things in, the concepts that matter, and the specific moments where your understanding will click.

**Who this is for:** you — strong math/coding fundamentals (Python, Java, C, Bash), real crypto trading experience, targeting a **quant developer** role in a few months.

**What this project proves to an employer:** that you can build correct, reproducible trading infrastructure — that you understand event-driven systems, look-ahead bias, transaction-cost realism, and the messy gap between a backtest and live trading. Quant dev hiring cares far more about *engineering correctness and judgment* than about a secret money-making signal.

---

## 0. The North Star (read this before writing any code)

> **The same strategy code must run, unchanged, in both backtest and live trading.**

This single commitment shapes every architectural decision below. Real systems do not have a "backtest version" and a separate "live version" of the logic — that's how subtle differences creep in and make your backtest a lie. Instead, the strategy is a black box that **receives events and emits orders**, blind to whether those events come from a historical file or a live websocket.

If your finished project has this property, you have demonstrated the core thing a quant-dev interviewer is looking for. Everything else serves this goal.

A second principle, equally important:

> **Build the thinnest vertical slice that touches every layer, get it working end-to-end, then thicken each layer.**

Do not perfect the data module in isolation for three weeks. Get a dumb strategy "trading" in simulation in the first sitting, then improve. A walking skeleton beats a beautiful unfinished wing.

---

## 1. The mental model — how the system is shaped

A systematic trading system is a pipeline of well-separated layers, wired together by an engine that passes **events** between them:

```
         ┌─────────────────────────────────────────────────────────┐
         │                        ENGINE                            │
         │              (the event loop / clock)                    │
         └─────────────────────────────────────────────────────────┘
              │            │            │            │
              ▼            ▼            ▼            ▼
        ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐
        │  DATA   │─►│ STRATEGY │─►│ PORTFOLIO │─►│ EXECUTION │
        │  feed   │  │ (signal) │  │  & risk   │  │  (fills)  │
        └─────────┘  └──────────┘  └───────────┘  └───────────┘
              ▲                                          │
              │            BACKTEST: historical file     │
              └──────────  LIVE: exchange websocket  ◄────┘
                          (same event types either way)
```

The flow of one tick:

1. **Data** emits a `MarketEvent` (a new bar or order-book update).
2. **Strategy** consumes it, updates its internal state, and may emit an `OrderEvent` (target position / signal).
3. **Portfolio/risk** consumes the order, applies sizing and limits, and decides the actual order to send.
4. **Execution** turns that into a `FillEvent` — simulated against historical prices in backtest, or a real API call live.
5. **Portfolio** consumes the fill, updates positions, cash, and PnL.

The crucial design fact: **the strategy only knows about events.** It never imports the data loader or the exchange client. Swap the data source from "file" to "websocket" and the strategy code does not change. That's the North Star, made concrete.

---

## 2. The phased build plan

Each phase ends with a **working system** and teaches one big idea. Resist the urge to skip ahead.

### Phase 0 — Foundations & data, one symbol
**Goal:** reproducibly get clean historical data onto disk.
**Build:** a data loader for one exchange and one symbol (use `ccxt` for a unified crypto API). Download OHLCV bars, cache them to `parquet`, and expose them through one clean function/interface. Add config + logging from the start.
**Done when:** you can re-run a command and get the identical dataset, fast, from cache.
**Learning point:** *reproducibility is a feature.* If you can't regenerate a dataset deterministically, you can't trust or debug anything built on it.

### Phase 1 — Vectorized backtest of a dumb strategy
**Goal:** prove your data and metrics are correct on the simplest possible strategy.
**Build:** a moving-average crossover (or similar trivial rule), backtested over the whole array at once, **with fees and walk-forward discipline**. Output the metrics you already know: return, vol, Sharpe, max drawdown, turnover.
**Done when:** the numbers are sane and you understand every one of them.
**Learning point:** *this is the textbook pipeline you already built* — now treat it as a baseline to beat and a correctness check, not the destination.

### Phase 2 — Rewrite as an EVENT-DRIVEN backtester ⭐
**Goal:** the architectural heart of the project.
**Build:** replace the vectorized loop with an event loop. Define your event types (`MarketEvent`, `OrderEvent`, `FillEvent`). Feed historical bars through one at a time, in order. The strategy reacts to each event with no ability to see the future.
**Done when:** the same dumb strategy from Phase 1 produces the same results — but now through the event loop.
**Learning point ⭐:** *this is the single biggest leap.* Event-driven is slower than vectorized, but it makes look-ahead bias structurally impossible and gives you the exact interface you'll reuse live. This is what separates "I followed a tutorial" from "I understand trading systems."

### Phase 3 — Plug in a live (paper) data feed
**Goal:** make backtest–live parity real.
**Build:** a websocket client that emits the **same event types** as your historical feed. Now your strategy runs against live data with zero code change.
**Done when:** you flip a config flag between "historical" and "live" and the same strategy runs in both.
**Learning point:** *the payoff of Phase 2.* This is the moment the North Star becomes tangible — and the thing to demo in an interview.

### Phase 4 — Execution against a testnet
**Goal:** close the loop with real (simulated-money) orders.
**Build:** route `OrderEvent`s to an exchange **testnet** (Binance, Bybit, etc. all offer one). Handle order acknowledgements, fills, partial fills, and reconcile your internal positions against the exchange's reported positions.
**Done when:** your system places, fills, and tracks paper trades, and your internal position always matches the exchange's.
**Learning point:** *the sim/live gap is never zero.* Live introduces latency, partial fills, rejected orders, disconnects, and rate limits — messy reality your backtest didn't model. Handling it gracefully is a feature, not an afterthought.

### Phase 5 — Risk, accounting, and correctness hardening
**Goal:** make the boring-but-fatal layer bulletproof.
**Build:** position limits, max leverage, a kill-switch. Then write a serious **test suite** focused on the accounting: positions, cash, PnL, fees, funding, partial fills. Add property-based tests (e.g. "cash + position value is conserved across any sequence of fills").
**Done when:** you trust your PnL number absolutely.
**Learning point:** *accounting bugs are silent and fatal.* A sign error in fee handling or position tracking makes a losing strategy look like a winner — and you'd never see it from the equity curve alone. Test this layer harder than any other.

### Phase 6 — Polish for the portfolio
**Goal:** make it legible to someone evaluating you.
**Build:** a clean README explaining the architecture (include the diagram), a small results dashboard (equity curve, drawdown, metrics), structured logging, and a short write-up of design decisions and known limitations.
**Done when:** a stranger can clone it, read the README, and understand the architecture in ten minutes.
**Learning point:** *being able to explain the design is half the value.* Interviewers will ask "why did you build it this way?" — the write-up is your rehearsal.

---

## 3. Key ideas (the conceptual anchors)

These are the concepts the whole project is built to teach. Understand them deeply enough to explain them on a whiteboard.

**Event-driven vs vectorized.** Vectorized = operate on the entire price history at once (fast, great for research, easy to accidentally peek at the future). Event-driven = process one event at a time in chronological order (slower, but mirrors reality and prevents look-ahead). Real execution is inherently event-driven; your backtester being event-driven is what makes it trustworthy.

**Look-ahead bias.** Using information you couldn't have had at decision time. It hides everywhere: deciding a trade on a bar's close then "filling" at that same close, resampling that leaks future data into a current bar, using a symbol list that excludes things that later got delisted (survivorship), or using revised/restated data instead of what was actually published then. **Point-in-time correctness** is the discipline of only ever using what was knowable at the moment.

**Backtest–live parity.** The North Star. One strategy code path, two data sources. The architectural commitment that everything else serves.

**Transaction-cost & slippage realism.** Costs aren't a footnote — they change *conclusions*, not just magnitudes. (Recall how accounting for turnover turned a "best return" strategy into ruin.) Model maker/taker fees, the bid-ask spread, and slippage. For larger size, market impact becomes nonlinear.

**Accounting correctness.** The unglamorous core: positions, cash, realized/unrealized PnL, fees, funding, partial fills. The layer most likely to have silent, catastrophic bugs.

**Determinism & reproducibility.** Same inputs → same outputs. Seed every random process, version your data, log your config. A backtest you can't reproduce is a backtest you can't trust.

**Separation of concerns.** The strategy doesn't know where data comes from. Execution doesn't know what the strategy thinks. Each layer talks only through events. This is what lets you swap, test, and reason about each piece independently.

---

## 4. Turning learning points — the moments understanding clicks

Watch for these. They're the inflection points where a developer's mental model genuinely shifts.

1. **"The strategy shouldn't know its data source."** The moment you stop writing two versions of everything and design around events. This reframes the entire architecture.

2. **"Event-driven is the bias-prevention mechanism, not just a style choice."** Realizing that processing one event at a time, never seeing the future, makes look-ahead bias *structurally impossible* rather than something you have to remember to avoid.

3. **"Look-ahead bias is everywhere and subtle."** The first time you find one hiding in your own code — a fill at a price you couldn't have gotten, a resample that leaked — and realize how easily a backtest lies.

4. **"The equity curve can be a complete fiction."** Discovering (ideally via a test, not via losing money) that a sign error in fee or position accounting made a losing strategy look profitable. This is why you test the accounting layer hardest.

5. **"Costs change the answer, not just the number."** Internalizing that frictions are first-class inputs that flip rankings — the lesson from watching turnover×cost destroy a high-return strategy.

6. **"Live is a different animal."** The first disconnect, partial fill, or rejected order that your clean backtest never had to handle. Robustness and error-handling become features you're proud of, not chores.

7. **"This is literally the shape of a real firm's stack."** Recognizing that data → alpha → portfolio → execution → engine maps directly onto how a real shop is organized — and being able to say which layer maps to which team in an interview.

---

## 5. Crypto-specific considerations (your edge — use it)

Your trading experience is a genuine differentiator. Most CS grads applying have never touched an order book. Get these right and it shows:

- **Maker/taker fees** differ and matter a lot for high-turnover strategies. Model them correctly, not as a single flat number.
- **Funding rates** on perpetual futures are a real, periodic cash flow (paid/received every few hours) — they belong in your PnL accounting. Getting this right is exactly the "someone who's actually traded" signal.
- **Order-book mechanics** — spread, depth, and how your fills depend on available liquidity. A fast order-book reconstruction is a great place to show off.
- **Websockets** are the live data backbone — handle reconnects, sequence gaps, and heartbeats.
- **Testnets** (Binance, Bybit, and others) let you run Phase 4 with no real money.
- **Data quirks** — 24/7 markets (no clean daily close), exchange downtime, symbol renames, wildly varying liquidity across pairs.

Build the data layer around **the exchange you've actually traded on** — your intuition for its quirks is worth more than a generic implementation.

---

## 6. Tech stack (suggested)

- **Python** for the bulk — research, backtest, orchestration. The lingua franca.
- **`ccxt`** for unified exchange access (historical + live + order routing across most crypto venues).
- **`parquet` / `pyarrow`** (or DuckDB) for fast, columnar local storage.
- **`pandas` / `numpy`** for analysis and the vectorized baseline.
- **`pytest`** + a property-based testing library for the accounting tests.
- **A performance-critical component in C or Java** ⭐ — e.g. a low-latency order-book reconstructor or a fast event dispatcher. Quant dev is one of the few fields where your C background is a direct asset; calling this out makes you stand out.

---

## 7. How not to fool yourself (the anti-patterns)

- **Don't optimize for a beautiful backtest.** An honest 0.8 Sharpe with realistic costs beats a fantasy 10 Sharpe. Interviewers can smell overfitting.
- **Don't tune your strategy on the same data you report on.** Out-of-sample discipline, always.
- **Don't trust an equity curve you haven't tested the accounting behind.**
- **Don't chase exotic alpha.** This is a *developer* portfolio — the strategy can be simple and well-known. The *system* is the artifact.
- **Don't try to build all layers at once.** Walking skeleton first.

---

## 8. Suggested first session

1. Pick the exchange you've traded on.
2. Set up the repo skeleton, config, and logging.
3. Write the Phase 0 data loader: download one symbol's OHLCV, cache to parquet, serve through one function.
4. Verify reproducibility (re-run → identical data, from cache).

That's a complete, satisfying first sitting — and the foundation everything else sits on.

---

### One-line summary
Build the thinnest end-to-end slice first; make the strategy blind to its data source so the same code runs in backtest and live; treat costs and accounting as first-class; and remember that for a quant-dev role, **the system is the artifact, not the alpha.**
