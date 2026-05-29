"""
tests/test_loader.py — Tests for the data layer.

WHY WE TEST THIS
────────────────
Everything in this project sits on top of the data layer. If get_bars() returns
wrong shapes, NaN values, or non-reproducible data, every single backtest built
on it is silently wrong. We test early and hard.

Run with:  pytest tests/test_loader.py -v
"""

import pandas as pd
import pytest

from data.loader import get_bars

# Use a short, fixed window for tests — fast to download, small to cache.
SYMBOL = "SPY"
START  = "2020-01-01"
END    = "2020-12-31"

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def test_returns_dataframe():
    """get_bars() must return a pandas DataFrame — not a list, dict, or None."""
    df = get_bars(SYMBOL, START, END)
    assert isinstance(df, pd.DataFrame)


def test_has_required_columns():
    """All five OHLCV columns must be present and lowercase."""
    df = get_bars(SYMBOL, START, END)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing column: '{col}'"


def test_no_nan_values():
    """
    No NaN values allowed. A NaN close price would cause a strategy to
    compute nonsense returns without raising an error — a silent bug.
    """
    df = get_bars(SYMBOL, START, END)
    assert not df.isnull().any().any(), "DataFrame contains NaN values"


def test_index_is_datetimeindex():
    """The index must be a DatetimeIndex so time-series operations work correctly."""
    df = get_bars(SYMBOL, START, END)
    assert isinstance(df.index, pd.DatetimeIndex), "Index is not a DatetimeIndex"


def test_date_range_within_bounds():
    """All dates must fall within the requested window."""
    df = get_bars(SYMBOL, START, END)
    assert df.index[0]  >= pd.Timestamp(START), "First bar is before start date"
    assert df.index[-1] <= pd.Timestamp(END),   "Last bar is after end date"


def test_prices_positive():
    """
    All OHLC values must be positive. A negative price is physically impossible
    for equities and signals a data or adjustment bug.
    """
    df = get_bars(SYMBOL, START, END)
    for col in ["open", "high", "low", "close"]:
        assert (df[col] > 0).all(), f"Non-positive values found in '{col}'"


def test_high_gte_low():
    """
    The high of any bar must be >= the low. Violating this means the OHLC
    data is corrupted or columns are misaligned.
    """
    df = get_bars(SYMBOL, START, END)
    assert (df["high"] >= df["low"]).all(), "Found bars where high < low"


def test_reproducibility():
    """
    THE CORE PHASE 0 INVARIANT.

    Two calls with identical arguments must return identical DataFrames.
    The second call reads from the parquet cache; if the data differs,
    something is wrong with the caching logic.

    If this test fails, no backtest result can be trusted — you wouldn't
    know whether a result changed because of a code change or a data change.
    """
    df1 = get_bars(SYMBOL, START, END)
    df2 = get_bars(SYMBOL, START, END)
    assert df1.equals(df2), (
        "Reproducibility broken: two calls with the same arguments returned "
        "different DataFrames. Check the caching logic in data/loader.py."
    )
