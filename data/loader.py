"""
data/loader.py — The data layer's public interface.

THE MOST IMPORTANT RULE: the rest of the system (strategy, backtest engine,
portfolio) only ever calls `get_bars()`. Nothing outside this file imports
yfinance. This means we can swap the data source later (a different provider,
a live feed, a database) by changing only this file — the strategy never knows.

That's the first small instance of the North Star principle from the guide:
    "The strategy is blind to where data comes from."
"""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def get_bars(
    symbol: str,
    start: str,
    end: str,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """
    Return daily OHLCV bars for `symbol` between `start` and `end` (inclusive).

    Columns returned: open, high, low, close, volume  (all lowercase)
    Index:            DatetimeIndex named 'date'

    HOW THE CACHE WORKS
    ───────────────────
    The filename encodes the exact request: SPY_2010-01-01_2024-12-31.parquet
    • First call  → downloads from Yahoo Finance, saves to disk, returns data.
    • Later calls → reads from disk, skips the network entirely.

    This means: as long as the cache file exists, every run produces the EXACT
    same DataFrame — same rows, same values, same dtypes. That's reproducibility.

    To force a fresh download (e.g. to extend the date range), simply delete
    the parquet file from data/cache/ and re-run.

    WHY auto_adjust=True
    ────────────────────
    Yahoo Finance provides two kinds of closing prices:
    • 'Close'     — the raw market price on that day.
    • 'Adj Close' — the price adjusted backwards for splits and dividends.

    We always want adjusted prices. Without adjustment, a 4:1 split in 2020
    would show as a sudden -75% drop — your strategy would see a catastrophic
    loss that never happened. auto_adjust=True bakes the adjustment into all
    four OHLC columns, not just Close, so they're all consistent.
    """

    # ── 1. Build the cache file path ────────────────────────────────────────
    # The filename captures all three dimensions of the request so different
    # (symbol, start, end) combinations never collide in the cache folder.
    cache_path = Path(cache_dir) / f"{symbol}_{start}_{end}.parquet"

    # ── 2. Cache hit: return frozen data from disk ───────────────────────────
    if cache_path.exists():
        logger.info("Cache hit  → %s", cache_path)
        df = pd.read_parquet(cache_path)
        return df

    # ── 3. Cache miss: download from Yahoo Finance ───────────────────────────
    logger.info("Cache miss → downloading %s (%s to %s)", symbol, start, end)

    raw = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,   # adjusts OHLC for splits + dividends
        progress=False,     # suppress the yfinance progress bar
    )

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for {symbol} between {start} and {end}. "
            "Check the symbol name and date range."
        )

    # ── 4. Normalise the DataFrame ───────────────────────────────────────────

    # yfinance sometimes returns a MultiIndex column when only one ticker is
    # requested (e.g. ('Close', 'SPY')). Flatten it to just the first level.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Lowercase every column name so the rest of the codebase can write
    # df['close'] instead of guessing whether it's 'Close' or 'CLOSE'.
    raw.columns = [c.lower() for c in raw.columns]

    # Make sure the index is a proper DatetimeIndex and name it 'date'.
    # Some yfinance versions return a plain Index — this normalises it.
    raw.index = pd.to_datetime(raw.index)
    raw.index.name = "date"

    # Drop rows where any OHLCV value is NaN. These occasionally appear at
    # the very start or end of a date range when the market wasn't open.
    n_before = len(raw)
    raw = raw.dropna()
    if len(raw) < n_before:
        logger.warning("Dropped %d NaN rows from %s", n_before - len(raw), symbol)

    # ── 5. Save to parquet and return ────────────────────────────────────────
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(cache_path)
    logger.info("Saved → %s  (%d rows)", cache_path, len(raw))

    return raw
