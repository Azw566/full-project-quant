"""
strategy/rsi.py — Relative Strength Index signal.

Enters long when RSI falls below the oversold threshold (price is weak,
mean-reversion expected). Exits to flat when RSI rises above the overbought
threshold. Uses hysteresis — position is held between threshold crossings —
to avoid thrashing in the neutral zone.

PUBLIC INTERFACE
────────────────
    generate_signals(bars, period, oversold, overbought) → pd.Series
"""

import pandas as pd


def generate_signals(
    bars: pd.DataFrame,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pd.Series:
    """
    Return a position signal based on RSI thresholds.

    Signal values:
        1.0  — RSI < oversold: go / stay long
        0.0  — RSI > overbought: go / stay flat
        NaN  — warmup period (first period bars)

    Between the two thresholds the previous position is held (forward-fill).

    Parameters
    ----------
    bars : pd.DataFrame
        OHLCV data with a DatetimeIndex. Must have a 'close' column.
    period : int
        RSI lookback window.
    oversold : float
        RSI level below which we go long (typically 30).
    overbought : float
        RSI level above which we go flat (typically 70).
    """
    if oversold >= overbought:
        raise ValueError(
            f"oversold ({oversold}) must be less than overbought ({overbought})."
        )

    delta = bars["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = 100.0 - 100.0 / (1.0 + rs)

    # Hysteresis: set explicit levels at threshold crossings, forward-fill between.
    signal = pd.Series(float("nan"), index=rsi.index)
    signal[rsi < oversold] = 1.0
    signal[rsi > overbought] = 0.0
    signal = signal.ffill().fillna(0.0)

    # Re-apply NaN for the warmup bars where RSI itself is undefined.
    signal[rsi.isna()] = float("nan")
    return signal.rename("signal")
