"""
tests/test_event_driven.py — Tests for backtest/event_driven.py.

The most important test is the parity check: the event-driven engine must
produce the same output as the vectorized engine on identical inputs.
If it doesn't, there is an accounting bug in one of the two engines.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.event_driven import run
from backtest.vectorized import run as run_vectorized


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_bars(prices: list) -> pd.DataFrame:
    n = len(prices)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open":   prices,
        "high":   prices,
        "low":    prices,
        "close":  prices,
        "volume": [1_000_000] * n,
    }, index=idx)


def _make_signals(values: list) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, name="signal")


# ── Structure tests ────────────────────────────────────────────────────────────

def test_returns_dataframe():
    bars = _make_bars(list(range(1, 101)))
    signals = _make_signals([1.0] * 100)
    result = run(bars, signals, fee_bps=0.0)
    assert isinstance(result, pd.DataFrame)


def test_expected_columns():
    bars = _make_bars([100.0 + i for i in range(50)])
    signals = _make_signals([1.0] * 50)
    result = run(bars, signals, fee_bps=0.0)
    for col in ["position", "market_return", "gross_return", "fee", "net_return", "equity"]:
        assert col in result.columns, f"Missing column: {col}"


def test_no_nan_in_result():
    bars = _make_bars([100.0 + i for i in range(50)])
    signals = _make_signals([float("nan")] * 10 + [1.0] * 40)
    result = run(bars, signals, fee_bps=5.0)
    assert not result.isnull().any().any(), "Result contains NaN values"


# ── PARITY TEST — the heart of Phase 2 ────────────────────────────────────────

def _parity_case(prices, signal_values, fee_bps):
    """Run both engines on the same input and assert the results are identical."""
    bars = _make_bars(prices)
    signals = _make_signals(signal_values)
    vec = run_vectorized(bars, signals, fee_bps=fee_bps)
    evt = run(bars, signals, fee_bps=fee_bps)
    pd.testing.assert_frame_equal(vec, evt, rtol=1e-9, check_like=False)


def test_parity_always_long_no_fees():
    """Simplest case: always long, no fees."""
    prices = [100.0 * (1.001 ** i) for i in range(100)]
    signals = [1.0] * 100
    _parity_case(prices, signals, fee_bps=0.0)


def test_parity_always_long_with_fees():
    prices = [100.0 * (1.001 ** i) for i in range(100)]
    signals = [1.0] * 100
    _parity_case(prices, signals, fee_bps=10.0)


def test_parity_always_flat():
    prices = [100.0 + i for i in range(100)]
    signals = [0.0] * 100
    _parity_case(prices, signals, fee_bps=10.0)


def test_parity_single_position_flip():
    """Flat then long: one trade in the middle."""
    prices = [100.0] * 60
    signals = [0.0] * 30 + [1.0] * 30
    _parity_case(prices, signals, fee_bps=10.0)


def test_parity_alternating_signals():
    """Signals that flip every bar — maximum fee drag."""
    prices = [100.0 + i * 0.1 for i in range(80)]
    signals = [float(i % 2) for i in range(80)]
    _parity_case(prices, signals, fee_bps=5.0)


def test_parity_with_warmup_nans():
    """First 10 signals are NaN (warmup), rest alternate."""
    prices = [100.0 + i * 0.5 for i in range(60)]
    signals = [float("nan")] * 10 + [float(i % 2) for i in range(50)]
    _parity_case(prices, signals, fee_bps=8.0)


def test_parity_declining_market():
    """Trending down while long — tests negative-return accounting."""
    prices = [100.0 * (0.999 ** i) for i in range(100)]
    signals = [1.0] * 100
    _parity_case(prices, signals, fee_bps=10.0)


def test_parity_realistic_ma_crossover():
    """
    Simulate a realistic MA crossover scenario with 200 bars,
    a warmup period, and multiple position changes.
    """
    rng = np.random.default_rng(42)
    prices = list(100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, 200)))
    # Warmup for first 5 bars, then alternating signals
    signals = [float("nan")] * 5 + [float(i % 3 == 0) for i in range(195)]
    _parity_case(prices, signals, fee_bps=10.0)


# ── Correctness tests ──────────────────────────────────────────────────────────

def test_position_is_lagged_signal():
    """position[t] must equal signal[t-1], never signal[t]."""
    n = 20
    bars = _make_bars([100.0] * n)
    signals = _make_signals([0.0] * 10 + [1.0] * 10)
    result = run(bars, signals, fee_bps=0.0)
    positions = result["position"].values
    first_long = next(i for i, p in enumerate(positions) if p == 1.0)
    assert first_long > 0, "Position went long at bar 0 — impossible with a 1-bar lag"


def test_zero_fee_when_position_unchanged():
    bars = _make_bars([100.0 + i * 0.5 for i in range(50)])
    signals = _make_signals([1.0] * 50)
    result = run(bars, signals, fee_bps=10.0)
    interior_fees = result["fee"].iloc[1:]
    assert (interior_fees == 0.0).all()


def test_fee_applied_on_trade():
    bars = _make_bars([100.0] * 30)
    signals = _make_signals([0.0] * 15 + [1.0] * 15)
    result = run(bars, signals, fee_bps=10.0)
    assert (result["fee"] > 0).any()


def test_flat_position_zero_gross_return():
    bars = _make_bars([100.0 + i for i in range(50)])
    signals = _make_signals([0.0] * 50)
    result = run(bars, signals, fee_bps=0.0)
    assert (result["gross_return"].abs() < 1e-12).all()


def test_equity_starts_at_one_plus_first_return():
    bars = _make_bars([100.0 + i for i in range(50)])
    signals = _make_signals([1.0] * 50)
    result = run(bars, signals, fee_bps=0.0)
    assert result["equity"].iloc[0] == pytest.approx(
        1.0 + result["net_return"].iloc[0], rel=1e-9
    )


def test_no_fees_gross_equals_net():
    bars = _make_bars([100.0 + i for i in range(50)])
    signals = _make_signals([1.0] * 50)
    result = run(bars, signals, fee_bps=0.0)
    pd.testing.assert_series_equal(
        result["gross_return"], result["net_return"],
        check_names=False, rtol=1e-10,
    )
