"""
strategy/ma_crossover.py — Moving-average crossover signal.

This is the simplest possible trend-following rule:
    Go long when the fast moving average is above the slow moving average.
    Stay flat otherwise.

It exists to give Phase 1 and Phase 2 a concrete strategy to run.
The signal is intentionally dumb — the system design is the artifact, not the alpha.

PUBLIC INTERFACE
────────────────
    generate_signals(bars, fast, slow) → pd.Series

The function returns a raw signal series. The backtest engine is responsible
for shifting it by one bar before multiplying by returns (to prevent look-ahead).
"""

import pandas as pd


def generate_signals(bars: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """
    Return a position signal based on moving-average crossover.

    Signal values:
        1.0  — fast MA > slow MA: hold long
        0.0  — fast MA <= slow MA: stay flat
        NaN  — not enough history to compute both MAs (warmup period)

    Parameters
    ----------
    bars : pd.DataFrame
        OHLCV data with a DatetimeIndex. Must have a 'close' column.
    fast : int
        Lookback window for the fast moving average (in bars).
    slow : int
        Lookback window for the slow moving average (in bars).
        Must be > fast.

    Notes
    -----
    The first (slow - 1) bars will be NaN — there is not enough history to
    compute the slow MA. The vectorized backtest drops these rows automatically.

    WHY THIS RULE WORKS (conceptually)
    ───────────────────────────────────
    When price is trending up, shorter-term averages rise faster than
    longer-term ones, pushing fast MA above slow MA. This rule captures
    the momentum: buy into trends, exit when the trend reverses.

    The weakness: in choppy, mean-reverting markets it whipsaws — it
    generates many small losses from false breakouts. That's intentional
    for this phase; the point is a working system, not a winning strategy.
    """
    if fast >= slow:
        raise ValueError(
            f"fast ({fast}) must be less than slow ({slow}). "
            "The slow MA needs a longer window to define a trend."
        )

    fast_ma = bars["close"].rolling(fast).mean()
    slow_ma = bars["close"].rolling(slow).mean()

    # Boolean comparison → float (1.0 / 0.0), then mask the warmup period
    signal = (fast_ma > slow_ma).astype(float)
    signal[slow_ma.isna()] = float("nan")

    return signal.rename("signal")
