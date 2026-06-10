"""
strategy/ema_crossover.py — Exponential moving-average crossover signal.

Go long when the fast EMA crosses above the slow EMA; stay flat otherwise.
Identical in structure to ma_crossover but reacts faster to recent price moves
because EWM weights decay exponentially rather than treating all bars equally.

PUBLIC INTERFACE
────────────────
    generate_signals(bars, fast, slow) → pd.Series
"""

import pandas as pd


def generate_signals(bars: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    """
    Return a position signal based on EMA crossover.

    Signal values:
        1.0  — fast EMA > slow EMA: hold long
        0.0  — fast EMA <= slow EMA: stay flat
        NaN  — warmup period (first slow - 1 bars)

    Parameters
    ----------
    bars : pd.DataFrame
        OHLCV data with a DatetimeIndex. Must have a 'close' column.
    fast : int
        Span of the fast EMA.
    slow : int
        Span of the slow EMA. Must be > fast.
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be less than slow ({slow}).")

    close = bars["close"]
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()

    signal = (fast_ema > slow_ema).astype(float)
    signal.iloc[: slow - 1] = float("nan")
    return signal.rename("signal")
