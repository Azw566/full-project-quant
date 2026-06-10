"""
strategy/momentum.py — Rate-of-change (momentum) signal.

Goes long when price is higher than it was N bars ago (positive momentum),
stays flat when price is flat or falling. The simplest possible momentum rule.

PUBLIC INTERFACE
────────────────
    generate_signals(bars, period) → pd.Series
"""

import pandas as pd


def generate_signals(bars: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Return a position signal based on N-bar price momentum.

    Signal values:
        1.0  — close > close[period bars ago]: go long
        0.0  — close <= close[period bars ago]: stay flat
        NaN  — warmup period (first period bars)

    Parameters
    ----------
    bars : pd.DataFrame
        OHLCV data with a DatetimeIndex. Must have a 'close' column.
    period : int
        Number of bars to look back for the momentum comparison.
    """
    roc = bars["close"].pct_change(periods=period)
    signal = (roc > 0).astype(float)
    signal[roc.isna()] = float("nan")
    return signal.rename("signal")
