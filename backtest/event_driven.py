"""
backtest/event_driven.py — Event-driven backtester.

Processes one bar at a time through a four-event pipeline:

    MarketEvent → SignalEvent → OrderEvent → FillEvent

Unlike the vectorized engine, this loop structurally cannot look ahead:
each bar is processed with only the information available at that point in
time. The strategy cannot accidentally reference tomorrow's close because
tomorrow's bar has not been emitted yet.

The critical test of this engine: run() must produce the SAME output as
backtest.vectorized.run() on the same inputs (within floating-point tolerance).
If the numbers differ, there is an accounting bug — finding it is the lesson.

PUBLIC INTERFACE
────────────────
    run(bars, signals, fee_bps) → pd.DataFrame

The signature is identical to backtest.vectorized.run() so both engines
can be swapped without any changes at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# ── Events ─────────────────────────────────────────────────────────────────────

@dataclass
class MarketEvent:
    """A new price bar has arrived from the data feed."""
    timestamp: pd.Timestamp
    close: float
    market_return: float  # close-to-close; NaN on the very first bar


@dataclass
class SignalEvent:
    """The strategy's desired target position for this bar."""
    timestamp: pd.Timestamp
    target_position: float  # 1.0 = long, 0.0 = flat


@dataclass
class OrderEvent:
    """Instruction to change position by `delta` units."""
    timestamp: pd.Timestamp
    delta: float  # positive = buy, negative = sell, 0 = hold


@dataclass
class FillEvent:
    """Confirmed execution of an order at the bar's closing price."""
    timestamp: pd.Timestamp
    delta: float  # position change that was executed
    fee: float    # transaction cost paid (fee_bps applied to |delta|)


# ── Handlers ───────────────────────────────────────────────────────────────────

class _Portfolio:
    """
    Tracks position and equity; converts SignalEvents into OrderEvents.

    The portfolio is responsible for knowing its current position and
    deciding how much to trade to reach the target. In Phase 2 the logic
    is trivial (always trade to the exact target), but this is the
    component that will grow into risk management and position sizing in
    later phases.
    """

    def __init__(self) -> None:
        self.position: float = 0.0
        self.equity: float = 1.0

    def on_signal(self, event: SignalEvent) -> OrderEvent:
        return OrderEvent(
            timestamp=event.timestamp,
            delta=event.target_position - self.position,
        )

    def on_fill(self, event: FillEvent, market_return: float) -> dict:
        """
        Apply the fill, compute returns, and return the row dict for this bar.

        The position changes to the target, then earns the bar's market return.
        Fee is subtracted from the return. Equity compounds.
        """
        self.position += event.delta
        gross_return = self.position * market_return
        net_return = gross_return - event.fee
        self.equity *= 1.0 + net_return
        return {
            "position":      self.position,
            "market_return": market_return,
            "gross_return":  gross_return,
            "fee":           event.fee,
            "net_return":    net_return,
            "equity":        self.equity,
        }


class _Broker:
    """
    Simulated execution engine.

    Fills orders immediately at the current bar's closing price (no slippage,
    no partial fills). In Phase 4 this will be replaced by a real exchange
    connector; the event types stay the same.
    """

    def __init__(self, fee_bps: float) -> None:
        self._fee_rate = fee_bps / 10_000.0

    def on_order(self, event: OrderEvent) -> FillEvent:
        return FillEvent(
            timestamp=event.timestamp,
            delta=event.delta,
            fee=abs(event.delta) * self._fee_rate,
        )


# ── Public run() ───────────────────────────────────────────────────────────────

def run(
    bars: pd.DataFrame,
    signals: pd.Series,
    fee_bps: float = 10.0,
) -> pd.DataFrame:
    """
    Run an event-driven backtest and return a result DataFrame.

    Parameters
    ----------
    bars : pd.DataFrame
        OHLCV data with a DatetimeIndex. Must have a 'close' column.
    signals : pd.Series
        Raw position signal from the strategy (1.0 = long, 0.0 = flat,
        NaN = warmup). Must be aligned to bars (same index).
    fee_bps : float
        One-way transaction cost in basis points. Applied whenever the
        position changes.

    Returns
    -------
    pd.DataFrame with columns: position, market_return, gross_return,
        fee, net_return, equity — identical schema to vectorized.run().

    1-BAR SIGNAL LAG
    ────────────────
    The signal from bar T determines the position held on bar T+1.
    This is applied via shift(1) — the same arithmetic as the vectorized engine.

    What the event structure does guarantee: portfolio and broker logic cannot
    accidentally access a future bar's price, because future bars haven't been
    emitted yet. But the signal is still precomputed over the full history before
    the loop begins. The truly online version — where signals are computed one
    bar at a time with no future data at all — is live/engine.py (LiveEngine).
    """
    # Apply the 1-bar lag: lagged_signals[T] = signals[T-1].
    # The position held on bar T was decided by the signal from bar T-1.
    lagged_signals = signals.shift(1)

    # Drop warmup rows where the lagged signal is still NaN.
    mask           = lagged_signals.notna()
    lagged_signals = lagged_signals[mask].astype(float)
    bars_clean     = bars.loc[mask]

    portfolio = _Portfolio()
    broker = _Broker(fee_bps)

    rows: list[dict] = []
    prev_close: float | None = None

    for ts, bar in bars_clean.iterrows():
        close = float(bar["close"])

        if prev_close is None:
            # First bar: set the initial portfolio position without charging a fee.
            # No prior position to diff against — mirrors the vectorized engine
            # dropping its first row (NaN from pct_change and diff).
            portfolio.position = float(lagged_signals.loc[ts])
            prev_close = close
            continue

        # ── MarketEvent ───────────────────────────────────────────────────────
        market_return = close / prev_close - 1.0
        market_evt = MarketEvent(timestamp=ts, close=close, market_return=market_return)

        # ── SignalEvent ───────────────────────────────────────────────────────
        signal_evt = SignalEvent(timestamp=ts, target_position=float(lagged_signals.loc[ts]))

        # ── OrderEvent ────────────────────────────────────────────────────────
        order_evt = portfolio.on_signal(signal_evt)

        # ── FillEvent ─────────────────────────────────────────────────────────
        fill_evt = broker.on_order(order_evt)

        # ── Portfolio update ──────────────────────────────────────────────────
        row = portfolio.on_fill(fill_evt, market_evt.market_return)
        rows.append(row)

        prev_close = close

    return pd.DataFrame(rows, index=bars_clean.index[1:])
