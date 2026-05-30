"""
backtest/vectorized.py — Vectorized (array-at-once) backtester.

Operates on the full price history simultaneously using pandas/numpy operations.
This is Phase 1's approach: fast to write and run, useful as a correctness
baseline, but structurally unable to prevent look-ahead bias on its own.

Phase 2 replaces this with an event-driven loop that makes look-ahead impossible
by design. The key test of Phase 2: it must reproduce these numbers exactly.

PUBLIC INTERFACE
────────────────
    run(bars, signals, fee_bps) → pd.DataFrame
"""

import pandas as pd


def run(
    bars: pd.DataFrame,
    signals: pd.Series,
    fee_bps: float = 10.0,
) -> pd.DataFrame:
    """
    Run a vectorized backtest and return a result DataFrame.

    Parameters
    ----------
    bars : pd.DataFrame
        OHLCV data with a DatetimeIndex. Must have a 'close' column.
    signals : pd.Series
        Raw position signal from the strategy (1.0 = long, 0.0 = flat, NaN = warmup).
        Must be aligned to bars (same index).
    fee_bps : float
        One-way transaction cost in basis points.
        10 bps = 0.10% per trade side.
        Applied whenever the position changes (i.e. on each trade).

    Returns
    -------
    pd.DataFrame with columns:
        position       — actual position held on this bar (after 1-bar lag)
        market_return  — raw close-to-close return (buy-and-hold reference)
        gross_return   — strategy return before fees (position × market_return)
        fee            — cost paid on this bar (0 unless a trade occurred)
        net_return     — strategy return after fees
        equity         — cumulative equity curve starting at 1.0

    Warmup rows (where the signal is NaN) are dropped before returning.

    THE ANTI-LOOK-AHEAD MECHANISM
    ──────────────────────────────
    The single most important line in this file:

        position = signals.shift(1)

    Today's signal is computed from today's closing prices. We cannot act
    on information from today's close BEFORE today's close happens. So we
    shift the signal forward by one bar: a signal generated at close of day T
    enters our position at the open of day T+1 (approximated as close of T+1).

    Without this shift, the backtest assumes you can trade at the exact closing
    price at the moment you decide to trade — which is impossible. This is
    a classic look-ahead bias. One missing .shift(1) can turn a losing strategy
    into a phantom winner.

    FEE MODEL
    ─────────
    fee_bps applies to each side of a trade. When position changes from 0→1,
    we are "buying in" and pay fee_bps on the notional. When 1→0, we are
    "selling out" and pay again. In this simplified model, we treat each bar's
    position change as triggering a proportional cost:

        trade_size[t] = |position[t] - position[t-1]|   (0 or 1 here)
        fee[t]        = trade_size[t] × (fee_bps / 10_000)

    Real systems add slippage (market impact) and the bid-ask spread on top
    of the explicit fee — see the project guide for Phase 5 hardening.
    """
    fee_rate = fee_bps / 10_000.0

    # Shift signal by 1 bar to avoid look-ahead.
    position = signals.shift(1).rename("position")

    # Drop warmup rows where position is NaN.
    mask = position.notna()
    position = position[mask].astype(float)
    bars_clean = bars.loc[mask]

    # Close-to-close daily return.
    market_return = bars_clean["close"].pct_change().rename("market_return")

    # Strategy gross return: what fraction of the market return we captured.
    gross_return = (position * market_return).rename("gross_return")

    # Fee: paid on every position change, proportional to position size moved.
    trade = position.diff().abs()
    fee = (trade * fee_rate).rename("fee")

    # Net return after costs.
    net_return = (gross_return - fee).rename("net_return")

    result = pd.DataFrame({
        "position":      position,
        "market_return": market_return,
        "gross_return":  gross_return,
        "fee":           fee,
        "net_return":    net_return,
    })

    # Drop the first row (NaN from pct_change and diff).
    result = result.dropna()

    # Equity curve: cumulative product of (1 + net_return), starting at 1.
    result["equity"] = (1 + result["net_return"]).cumprod()

    return result
