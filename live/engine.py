"""
live/engine.py — Online execution engine for live paper trading.

Drives the same four-event pipeline used in backtest/event_driven.py:

    MarketEvent → SignalEvent → OrderEvent → FillEvent

The key difference from the backtest engine: instead of replaying a
pre-computed signals array, the LiveEngine maintains a rolling window of
the last `slow` closes and recomputes the MA crossover signal online,
one bar at a time.

The _Portfolio and _Broker components are imported directly from
backtest/event_driven.py — unchanged. This is the North Star principle
made literal: the same execution components run in both backtest and live.

Phase 5 addition: pass bootstrap_fn=feed.bootstrap at construction time.
When a WebSocket gap is detected, the engine calls bootstrap_fn() to
re-fetch the last slow+1 bars from the REST API and re-initialises the
MA window, preventing stale-state trading after a disconnect.

PUBLIC INTERFACE
────────────────
    engine = LiveEngine(portfolio, broker, fast, slow, bootstrap_fn=None)
    engine.initialize(bootstrap_bars)   # warm up MA from historical data
    engine.on_bar(bar) → dict | None    # process one live bar
    engine.get_results() → pd.DataFrame
    engine.get_metrics() → dict | None
"""

from __future__ import annotations

import logging
import math
from collections import deque

import pandas as pd

from backtest.event_driven import (
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
    _Broker,
    _Portfolio,
)
from backtest.metrics import PERIODS_PER_YEAR, compute_metrics

logger = logging.getLogger(__name__)


class LiveEngine:
    """
    Online event engine.

    Maintains a rolling deque of the last `slow` closing prices to compute
    the MA crossover signal after each bar. Applies the same 1-bar signal
    lag used in the backtest engines: the signal computed from bar t's close
    determines the position held on bar t+1.
    """

    def __init__(
        self,
        portfolio:    _Portfolio,
        broker:       _Broker,
        fast:         int,
        slow:         int,
        bootstrap_fn=None,
    ) -> None:
        """
        Parameters
        ----------
        portfolio, broker : shared execution components from event_driven.py.
        fast, slow        : MA window sizes.
        bootstrap_fn      : optional callable () → list[dict].  When provided,
                            a gap in the live stream triggers a call to
                            bootstrap_fn() followed by self.initialize() to
                            restore a correct MA window state.
        """
        self._portfolio    = portfolio
        self._broker       = broker
        self._fast         = fast
        self._slow         = slow
        self._bootstrap_fn = bootstrap_fn

        self._window: deque[float] = deque(maxlen=slow)
        self._prev_close:     float | None = None
        self._pending_signal: float | None = None  # signal from the previous bar
        self._rows: list[dict] = []

        # Gap detection: track last seen timestamp and derive expected interval
        # from the first two consecutive bars.
        self._prev_ts:       pd.Timestamp | None = None
        self._bar_interval:  pd.Timedelta  | None = None

    # ── Initialization ─────────────────────────────────────────────────────────

    def initialize(self, bootstrap_bars: list[dict]) -> None:
        """
        Prime the rolling MA window using the historical bootstrap bars.

        These bars are in the past — no trades are recorded.
        After this call, the engine is ready to process live candles.

        ALIGNMENT WITH THE BACKTEST ENGINE
        ───────────────────────────────────
        The event-driven backtest silently initializes two pieces of state
        from its first-bar:
            portfolio.position = signal[t-1]   (the "previous" signal)
            prev_close         = close[t]
        and records rows from bar t+1 onward.

        To match this, initialize() tracks TWO consecutive signals from the
        bootstrap — it needs at least slow+1 bars to do so:
            prev_signal    = signal after processing bar[slow-1]
            pending_signal = signal after processing bar[slow]
        Then portfolio.position = prev_signal mirrors the backtest.

        With fewer than slow+1 bars there is no prev_signal, so
        portfolio.position defaults to pending_signal. The first row may
        then carry a spurious fee if the signal changed on the last bar.
        """
        prev_signal: float | None = None

        for bar in bootstrap_bars:
            self._window.append(float(bar["close"]))
            if len(self._window) == self._slow:
                # Each time the window first becomes full, rotate signals:
                # prev gets the old pending, pending gets the freshly computed one.
                prev_signal          = self._pending_signal
                self._pending_signal = self._compute_signal()

        if len(self._window) < self._slow:
            logger.warning(
                "Bootstrap only provided %d/%d bars. Engine will complete "
                "warmup from the live stream before trading.",
                len(self._window), self._slow,
            )
            return

        # portfolio.position = prev_signal so first-row fee matches the backtest.
        self._portfolio.position = prev_signal if prev_signal is not None else self._pending_signal
        self._prev_close         = float(bootstrap_bars[-1]["close"])
        self._prev_ts            = bootstrap_bars[-1]["timestamp"]

        logger.info(
            "Engine ready. MA%d=%.2f  MA%d=%.2f  prev_pos=%.0f  pending=%.0f",
            self._fast, self._fast_ma(),
            self._slow, self._slow_ma(),
            self._portfolio.position, self._pending_signal,
        )

    # ── Per-bar processing ─────────────────────────────────────────────────────

    def on_bar(self, bar: dict) -> dict | None:
        """
        Process one closed candle from the live feed.

        Returns the result row dict if a full event cycle was completed,
        or None during the warmup period.

        THE 1-BAR LAG
        ─────────────
        Signal computed at close of bar T → position held on bar T+1.
        Stored as self._pending_signal between calls.

        GAP RE-BOOTSTRAP
        ─────────────────
        If a gap is detected and bootstrap_fn was provided, the MA window is
        re-initialised from fresh REST data before processing the current bar.
        self._prev_ts is updated at the end of the call (via finally) so that
        after a re-bootstrap it correctly reflects the current bar, not the
        stale last-bootstrap bar.
        """
        close: float        = float(bar["close"])
        ts:    pd.Timestamp = bar["timestamp"]

        # Gap detection: warn if a disconnect caused missed candles.
        if self._prev_ts is not None:
            if self._bar_interval is None:
                self._bar_interval = ts - self._prev_ts
            elif ts - self._prev_ts > self._bar_interval * 1.5:
                missed = round((ts - self._prev_ts) / self._bar_interval) - 1
                logger.warning(
                    "GAP: %d bar(s) missed between %s and %s.",
                    missed, self._prev_ts, ts,
                )
                if self._bootstrap_fn is not None:
                    logger.info("Re-bootstrapping MA window after gap...")
                    fresh_bars = self._bootstrap_fn()
                    self.initialize(fresh_bars)
                    logger.info("Re-bootstrap complete.")
                else:
                    logger.warning(
                        "MA window is stale — no bootstrap_fn provided."
                    )

        try:
            self._window.append(close)

            # Still in warmup — not enough history to compute both MAs.
            if len(self._window) < self._slow:
                logger.debug("Warmup: %d/%d bars.", len(self._window), self._slow)
                self._prev_close = close
                return None

            current_signal = self._compute_signal()

            # First bar after reaching the warmup threshold.
            if self._pending_signal is None or self._prev_close is None:
                self._pending_signal     = current_signal
                self._prev_close         = close
                self._portfolio.position = current_signal
                logger.info(
                    "%s | Warmup complete. Initial position=%.0f", ts, current_signal,
                )
                return None

            # ── Event pipeline ─────────────────────────────────────────────────
            market_return = close / self._prev_close - 1.0

            market_evt = MarketEvent(timestamp=ts, close=close, market_return=market_return)
            signal_evt = SignalEvent(timestamp=ts, target_position=self._pending_signal)
            order_evt  = self._portfolio.on_signal(signal_evt)
            fill_evt   = self._broker.on_order(order_evt)
            row        = self._portfolio.on_fill(fill_evt, market_return)
            row["timestamp"] = ts

            self._rows.append(row)
            _log_bar(ts, close, market_return, row, self._fast_ma(), self._slow_ma())

            # Advance state — store current bar's signal for next bar.
            self._pending_signal = current_signal
            self._prev_close     = close

            return row

        finally:
            # Always update _prev_ts AFTER gap detection and any re-bootstrap,
            # so the next call sees this bar's timestamp as the reference point.
            self._prev_ts = ts

    # ── Results ────────────────────────────────────────────────────────────────

    def get_results(self) -> pd.DataFrame:
        """Return all processed bars as a DataFrame indexed by timestamp."""
        if not self._rows:
            return pd.DataFrame()
        return pd.DataFrame(self._rows).set_index("timestamp")

    def get_metrics(self, interval: str = "1h") -> dict | None:
        """
        Compute performance metrics from the session so far.

        Uses the correct annualization factor for the candle interval:
            1h → 8760 periods/year,  1d → 252,  etc.
        Returns None if fewer than 2 bars have been processed.
        """
        df = self.get_results()
        if len(df) < 2:
            return None
        periods = PERIODS_PER_YEAR.get(interval, 252)
        return compute_metrics(df["net_return"], df["position"], periods_per_year=periods)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _fast_ma(self) -> float:
        closes = list(self._window)
        return math.fsum(closes[-self._fast:]) / self._fast

    def _slow_ma(self) -> float:
        return math.fsum(self._window) / len(self._window)

    def _compute_signal(self) -> float:
        return 1.0 if self._fast_ma() > self._slow_ma() else 0.0


# ── Logging helper ─────────────────────────────────────────────────────────────

def _log_bar(
    ts: pd.Timestamp,
    close: float,
    market_return: float,
    row: dict,
    fast_ma: float,
    slow_ma: float,
) -> None:
    side      = "LONG" if row["position"] == 1.0 else "FLAT"
    trade_tag = f"  [TRADE  fee={row['fee']:.4%}]" if row["fee"] > 0 else ""
    logger.info(
        "%s | close=%10.2f | ret=%+7.3f%% | MA50=%10.2f  MA200=%10.2f | %-4s | equity=%.6f%s",
        ts.strftime("%Y-%m-%d %H:%M UTC"),
        close,
        market_return * 100,
        fast_ma,
        slow_ma,
        side,
        row["equity"],
        trade_tag,
    )
