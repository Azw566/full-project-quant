"""
strategy/macd.py — MACD (Moving Average Convergence/Divergence) signal.

Goes long when the MACD line crosses above its signal line; stays flat when it
crosses below. The MACD line is the difference between two EMAs; the signal
line is an EMA of the MACD line itself.

PUBLIC INTERFACE
────────────────
    generate_signals(bars, fast, slow, signal_period) → pd.Series
"""

import pandas as pd


def generate_signals(
    bars: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> pd.Series:
    """
    Return a position signal based on MACD line vs. signal line.

    Signal values:
        1.0  — MACD > signal line: go long
        0.0  — MACD <= signal line: stay flat
        NaN  — warmup period (first slow + signal_period - 2 bars)

    Parameters
    ----------
    bars : pd.DataFrame
        OHLCV data with a DatetimeIndex. Must have a 'close' column.
    fast : int
        Span of the fast EMA (typically 12).
    slow : int
        Span of the slow EMA (typically 26). Must be > fast.
    signal_period : int
        Span of the EMA applied to the MACD line (typically 9).
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be less than slow ({slow}).")

    close = bars["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    signal = (macd_line > signal_line).astype(float)
    warmup = slow + signal_period - 2
    signal.iloc[:warmup] = float("nan")
    return signal.rename("signal")
