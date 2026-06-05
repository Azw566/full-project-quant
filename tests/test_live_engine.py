"""
tests/test_live_engine.py — Tests for live/engine.py.

All tests are deterministic and require no network connection.
The LiveEngine is driven by hand-crafted bar dicts instead of a real feed.

The critical tests:
    1. After identical inputs, LiveEngine and event_driven.run() produce
       the same results (parity across the backtest/live boundary).
    2. The 1-bar signal lag is preserved in online mode.
    3. Warmup correctly blocks trading until the slow MA window is full.

BOOTSTRAP ALIGNMENT NOTE
─────────────────────────
The event-driven backtest drops its first bars_clean row (the "first-bar
initialization" step).  To produce an identical starting point, the live
engine must be bootstrapped with slow+1 bars:

    deque(maxlen=slow) after slow+1 appends  →  last `slow` closes
    prev_close                               →  bootstrap[-1].close
    portfolio.position                       →  signal[slow-2]  (prev signal)
    pending_signal                           →  signal[slow-1]  (current signal)

The parity tests use generate_signals() for both engines so the signal
logic is identical, then compare column-by-column to 1e-9.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.event_driven import _Broker, _Portfolio, run as run_event_driven
from live.engine import LiveEngine
from strategy.ma_crossover import generate_signals


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_bars_bt(prices: list) -> pd.DataFrame:
    """Build a backtest-style OHLCV DataFrame."""
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="h")
    return pd.DataFrame({
        "open": prices, "high": prices, "low": prices,
        "close": prices, "volume": [1_000_000] * len(prices),
    }, index=idx)


def _make_live_bars(prices: list) -> list[dict]:
    """Build a list of bar dicts suitable for LiveEngine.on_bar()."""
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="h")
    return [
        {"timestamp": ts, "open": p, "high": p, "low": p, "close": p, "volume": 1_000_000}
        for ts, p in zip(idx, prices)
    ]


def _make_engine(fast: int, slow: int, fee_bps: float = 0.0) -> LiveEngine:
    return LiveEngine(_Portfolio(), _Broker(fee_bps), fast=fast, slow=slow)


# ── Warmup tests ───────────────────────────────────────────────────────────────

def test_no_rows_during_warmup():
    """Engine must not record any rows before the slow window is full."""
    engine = _make_engine(fast=3, slow=5)
    bars = _make_live_bars([100.0] * 4)  # one short of slow=5
    for bar in bars:
        engine.on_bar(bar)
    assert engine.get_results().empty


def test_row_count_no_bootstrap():
    """Without bootstrap, rows = n - slow (warmup slow-1 bars + 1 init bar)."""
    fast, slow = 3, 5
    n = 10
    engine = _make_engine(fast=fast, slow=slow)
    bars = _make_live_bars([100.0] * n)
    for bar in bars:
        engine.on_bar(bar)
    assert len(engine.get_results()) == n - slow


def test_initialize_sets_prev_position():
    """
    After bootstrap with slow+1 bars, portfolio.position equals the signal
    computed from the second-to-last bootstrap window (not the latest).
    """
    # Flat prices → fast_ma == slow_ma → signal=0 throughout.
    prices = [100.0] * 6   # slow+1 = 6 bars for slow=5
    portfolio = _Portfolio()
    engine = LiveEngine(portfolio, _Broker(0.0), fast=3, slow=5)
    engine.initialize(_make_live_bars(prices))
    assert portfolio.position == 0.0


def test_initialize_partial_bootstrap_does_not_crash():
    """Partial bootstrap (fewer bars than slow) should warn but not raise."""
    engine = _make_engine(fast=3, slow=5)
    engine.initialize(_make_live_bars([100.0, 101.0]))  # only 2 of 5 needed


# ── Correctness tests ──────────────────────────────────────────────────────────

def test_position_lag_one_bar():
    """
    A signal flip triggered by bar T must not affect the position until bar T+1.
    """
    prices = [100.0] * 5 + [200.0] * 5
    engine = _make_engine(fast=2, slow=4, fee_bps=0.0)
    bars = _make_live_bars(prices)
    for bar in bars:
        engine.on_bar(bar)
    df = engine.get_results()
    if df.empty:
        pytest.skip("Not enough bars")
    first_long = df[df["position"] == 1.0]
    if first_long.empty:
        pytest.skip("Signal never went long")
    first_long_idx = df.index.get_loc(first_long.index[0])
    assert first_long_idx > 0, "Position went long on bar 0 — 1-bar lag broken"


def test_flat_signal_zero_gross_return():
    """Descending prices → always flat → gross return is 0 on every bar."""
    prices = [200.0 - i for i in range(10)]
    engine = _make_engine(fast=2, slow=4, fee_bps=0.0)
    for bar in _make_live_bars(prices):
        engine.on_bar(bar)
    df = engine.get_results()
    if not df.empty:
        assert (df["gross_return"].abs() < 1e-12).all()


def test_fee_only_on_trade():
    """Fee is positive only on bars where position changes."""
    prices = [100.0 + i * 0.1 for i in range(12)]
    engine = _make_engine(fast=2, slow=4, fee_bps=10.0)
    for bar in _make_live_bars(prices):
        engine.on_bar(bar)
    df = engine.get_results()
    if not df.empty:
        no_trade_mask = df["position"].diff().abs() < 1e-12
        assert (df.loc[no_trade_mask, "fee"].iloc[1:] == 0.0).all()


def test_equity_compounds():
    """equity[t] = equity[t-1] × (1 + net_return[t]) for every bar."""
    prices = [100.0 * (1.001 ** i) for i in range(15)]
    engine = _make_engine(fast=2, slow=4, fee_bps=5.0)
    for bar in _make_live_bars(prices):
        engine.on_bar(bar)
    df = engine.get_results()
    if len(df) < 2:
        return
    recomputed = (1 + df["net_return"]).cumprod()
    pd.testing.assert_series_equal(df["equity"], recomputed, rtol=1e-10, check_names=False)


def test_get_results_returns_dataframe():
    assert isinstance(_make_engine(2, 4).get_results(), pd.DataFrame)


def test_get_metrics_none_before_two_bars():
    assert _make_engine(2, 4).get_metrics() is None


# ── Parity tests — live engine must match event-driven backtest ───────────────

def _parity_case(prices: list, fast: int, slow: int, fee_bps: float) -> None:
    """
    Assert that LiveEngine and event_driven.run() produce identical results.

    Both engines use generate_signals() for signal computation so the logic
    is identical.  The live engine is bootstrapped with slow+1 bars to align
    its starting point with the backtest's first-bar initialization step.

    Bootstrap alignment:
        backtest bars_clean starts at index slow (first non-NaN shifted position)
        backtest first-bar-init sets prev_close = bars[slow].close, skips row
        backtest records rows from bars[slow+1 .. n-1]

        live bootstrap = bars[0..slow] (slow+1 bars)
        deque(maxlen=slow) ends up holding bars[1..slow]
        prev_close = bars[slow].close  ← matches backtest
        live records rows from bars[slow+1 .. n-1]  ← same range
    """
    bars_bt   = _make_bars_bt(prices)
    live_bars = _make_live_bars(prices)
    signals   = generate_signals(bars_bt, fast=fast, slow=slow)

    # Backtest
    evt_result = run_event_driven(bars_bt, signals, fee_bps=fee_bps)

    # Live — bootstrap slow+1 bars, then process the rest
    portfolio = _Portfolio()
    broker    = _Broker(fee_bps)
    engine    = LiveEngine(portfolio, broker, fast=fast, slow=slow)
    engine.initialize(live_bars[: slow + 1])
    for bar in live_bars[slow + 1 :]:
        engine.on_bar(bar)

    live_result = engine.get_results()

    assert len(live_result) == len(evt_result), (
        f"Row count mismatch: live={len(live_result)}  evt={len(evt_result)}"
    )

    for col in ["position", "market_return", "gross_return", "fee", "net_return", "equity"]:
        diff = abs(evt_result[col].values - live_result[col].values)
        assert diff.max() < 1e-9, (
            f"Column '{col}' diverges: max_diff={diff.max():.2e}"
        )


def test_parity_always_rising():
    """Rising prices → always long."""
    prices = [100.0 * (1.001 ** i) for i in range(50)]
    _parity_case(prices, fast=3, slow=5, fee_bps=0.0)


def test_parity_always_rising_with_fees():
    prices = [100.0 * (1.001 ** i) for i in range(50)]
    _parity_case(prices, fast=3, slow=5, fee_bps=10.0)


def test_parity_always_declining():
    """Declining prices → always flat."""
    prices = [100.0 * (0.999 ** i) for i in range(50)]
    _parity_case(prices, fast=3, slow=5, fee_bps=10.0)


def test_parity_crossover_fires():
    """Flat then sharply rising — MA crossover fires in the middle."""
    prices = [100.0] * 20 + [200.0 * (1.001 ** i) for i in range(30)]
    _parity_case(prices, fast=3, slow=5, fee_bps=10.0)


def test_parity_multiple_crossovers():
    """Oscillating prices produce multiple golden/death crosses."""
    rng = np.random.default_rng(42)
    prices = list(100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, 60)))
    _parity_case(prices, fast=3, slow=5, fee_bps=8.0)


def test_parity_zero_fees():
    rng = np.random.default_rng(7)
    prices = list(100.0 * np.cumprod(1 + rng.normal(0.001, 0.015, 60)))
    _parity_case(prices, fast=3, slow=5, fee_bps=0.0)


def test_parity_larger_windows():
    """Larger MA windows with a longer price series."""
    rng = np.random.default_rng(13)
    prices = list(100.0 * np.cumprod(1 + rng.normal(0.0003, 0.008, 300)))
    _parity_case(prices, fast=10, slow=30, fee_bps=5.0)
