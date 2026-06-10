"""
strategy/bollinger_bands.py — Bollinger Band mean-reversion signal.

Enters long when price touches the lower band (price is cheap relative to recent
volatility). Exits to flat when price touches or exceeds the upper band. Holds
between touches — no thrashing in the middle of the band.

PUBLIC INTERFACE
────────────────
    generate_signals(bars, period, num_std) → pd.Series
"""

import pandas as pd


def generate_signals(
    bars: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    """
    Return a position signal based on Bollinger Band touches.

    Signal values:
        1.0  — price <= lower band: go / stay long
        0.0  — price >= upper band: go / stay flat
        NaN  — warmup period (first period - 1 bars)

    Parameters
    ----------
    bars : pd.DataFrame
        OHLCV data with a DatetimeIndex. Must have a 'close' column.
    period : int
        Rolling window for the mean and standard deviation.
    num_std : float
        Width of the bands in standard deviations (typically 2.0).
    """
    close = bars["close"]
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    lower = mid - num_std * std
    upper = mid + num_std * std

    signal = pd.Series(float("nan"), index=close.index)
    signal[close <= lower] = 1.0
    signal[close >= upper] = 0.0
    signal = signal.ffill().fillna(0.0)

    signal[mid.isna()] = float("nan")
    return signal.rename("signal")
