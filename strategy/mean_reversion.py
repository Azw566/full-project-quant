"""
strategy/mean_reversion.py — Z-score mean-reversion signal.

Goes long when price is unusually cheap (z-score below -z_entry), meaning price
is far below its recent mean. Exits to flat once price has recovered back toward
the mean (z-score rises above -z_exit). Uses hysteresis to hold between extremes.

PUBLIC INTERFACE
────────────────
    generate_signals(bars, period, z_entry, z_exit) → pd.Series
"""

import pandas as pd


def generate_signals(
    bars: pd.DataFrame,
    period: int = 20,
    z_entry: float = 1.5,
    z_exit: float = 0.5,
) -> pd.Series:
    """
    Return a position signal based on rolling z-score.

    Signal values:
        1.0  — z < -z_entry: price is z_entry stds below mean → go long
        0.0  — z > -z_exit:  price has recovered to within z_exit stds → go flat
        NaN  — warmup period (first period - 1 bars)

    Parameters
    ----------
    bars : pd.DataFrame
        OHLCV data with a DatetimeIndex. Must have a 'close' column.
    period : int
        Rolling window for mean and standard deviation.
    z_entry : float
        Z-score magnitude to trigger entry (e.g. 1.5 → 1.5 stds below mean).
    z_exit : float
        Z-score magnitude to trigger exit (e.g. 0.5 → price within 0.5 stds of mean).
    """
    if z_exit >= z_entry:
        raise ValueError(
            f"z_exit ({z_exit}) must be less than z_entry ({z_entry}) — "
            "exit threshold should be closer to the mean than the entry threshold."
        )

    close = bars["close"]
    mean = close.rolling(period).mean()
    std = close.rolling(period).std().replace(0, float("nan"))
    z = (close - mean) / std

    signal = pd.Series(float("nan"), index=close.index)
    signal[z < -z_entry] = 1.0   # Price is far below mean: enter long
    signal[z > -z_exit] = 0.0    # Price has reverted enough: go flat
    signal = signal.ffill().fillna(0.0)

    signal[mean.isna()] = float("nan")
    return signal.rename("signal")
