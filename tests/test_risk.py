"""
tests/test_risk.py — Phase 5 risk-management tests.

Covers the three new features added to _Portfolio and _Broker:
  1. Slippage        — separate half-spread cost in FillEvent and output
  2. Vol targeting   — position scaled by vol_target / realised_vol
  3. Circuit breaker — go flat when drawdown exceeds max_drawdown;
                       resume after cooldown_bars bars
  4. Parity checks   — vectorized and event-driven agree with risk params set
  5. Gap re-bootstrap — LiveEngine calls bootstrap_fn on gap detection
"""

import math

import numpy as np
import pandas as pd
import pytest

from backtest.event_driven import (
    _Broker,
    _Portfolio,
    FillEvent,
    OrderEvent,
    SignalEvent,
    run as run_event_driven,
)
from backtest.vectorized import run as run_vectorized
from live.engine import LiveEngine


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_bars(prices: list) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({
        "open": prices, "high": prices, "low": prices,
        "close": prices, "volume": [1_000_000] * len(prices),
    }, index=idx)


def _make_signals(values: list) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def _make_live_bars(prices: list) -> list[dict]:
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="h")
    return [
        {"timestamp": ts, "open": p, "high": p, "low": p, "close": p, "volume": 1_000_000}
        for ts, p in zip(idx, prices)
    ]


# ── Slippage tests ─────────────────────────────────────────────────────────────

def test_slippage_column_exists():
    """Both engines must always return a 'slippage' column."""
    bars    = _make_bars([100.0 + i for i in range(30)])
    signals = _make_signals([1.0] * 30)
    for result in [
        run_vectorized(bars, signals, fee_bps=10.0),
        run_event_driven(bars, signals, fee_bps=10.0),
    ]:
        assert "slippage" in result.columns


def test_slippage_increases_total_cost():
    """With slippage_bps > 0, net_return < gross_return - fee."""
    bars    = _make_bars([100.0 + i for i in range(30)])
    signals = _make_signals([0.0] * 10 + [1.0] * 20)

    for run_fn in (run_vectorized, run_event_driven):
        r = run_fn(bars, signals, fee_bps=10.0, slippage_bps=5.0)
        # At least one trade occurred — slippage must be positive somewhere.
        assert (r["slippage"] > 0).any()
        # Total cost = fee + slippage; net_return = gross_return - total_cost.
        expected_net = r["gross_return"] - r["fee"] - r["slippage"]
        pd.testing.assert_series_equal(
            r["net_return"], expected_net, rtol=1e-12, check_names=False,
        )


def test_slippage_zero_when_no_trade():
    """When position does not change, slippage is zero."""
    bars    = _make_bars([100.0 + i * 0.5 for i in range(30)])
    signals = _make_signals([1.0] * 30)
    for run_fn in (run_vectorized, run_event_driven):
        r = run_fn(bars, signals, fee_bps=10.0, slippage_bps=20.0)
        # After the first bar (entry trade), position is always 1 → no more trades.
        assert (r["slippage"].iloc[1:] == 0.0).all()


def test_zero_slippage_matches_no_slippage_arg():
    """Explicit slippage_bps=0 must produce the same result as omitting it."""
    bars    = _make_bars([100.0 * (1.002 ** i) for i in range(50)])
    signals = _make_signals([1.0] * 50)
    for run_fn in (run_vectorized, run_event_driven):
        r_default = run_fn(bars, signals, fee_bps=10.0)
        r_zero    = run_fn(bars, signals, fee_bps=10.0, slippage_bps=0.0)
        pd.testing.assert_frame_equal(r_default, r_zero, rtol=1e-12)


# ── Vol targeting tests ────────────────────────────────────────────────────────

def test_vol_targeting_reduces_position():
    """When market vol >> vol_target, scaled position must be < 1."""
    # High-vol random returns: std ≈ 5% per bar → annualised ~80% (daily)
    rng = np.random.default_rng(42)
    prices = list(100.0 * np.cumprod(1 + rng.normal(0.0, 0.05, 100)))
    bars    = _make_bars(prices)
    signals = _make_signals([1.0] * 100)

    vol_target = 0.10   # 10% annualised target, well below 80% realised
    r = run_event_driven(
        bars, signals, fee_bps=0.0,
        vol_target=vol_target, vol_lookback=20, periods_per_year=252.0,
    )
    # After warm-up (20 bars), positions should be < 1.0.
    post_warmup = r["position"].iloc[20:]
    assert (post_warmup < 1.0).all(), (
        "Vol targeting did not scale positions below 1.0 for high-vol prices."
    )


def test_vol_targeting_clips_to_one():
    """When realised vol < vol_target, position must not exceed 1."""
    # Near-flat prices → very low vol → scale would be huge without clipping.
    prices = [100.0 + i * 0.0001 for i in range(80)]
    bars    = _make_bars(prices)
    signals = _make_signals([1.0] * 80)

    r = run_event_driven(
        bars, signals, fee_bps=0.0,
        vol_target=1.0, vol_lookback=10, periods_per_year=252.0,
    )
    assert (r["position"] <= 1.0 + 1e-12).all(), "Position exceeded 1.0 (uncapped leverage)."


def test_vol_targeting_warmup_uses_full_position():
    """Before vol_lookback fills, position must equal the raw signal (no scaling)."""
    rng = np.random.default_rng(7)
    prices = list(100.0 * np.cumprod(1 + rng.normal(0.001, 0.02, 60)))
    bars    = _make_bars(prices)
    signals = _make_signals([1.0] * 60)

    vol_lookback = 20
    r = run_event_driven(
        bars, signals, fee_bps=0.0,
        vol_target=0.10, vol_lookback=vol_lookback, periods_per_year=252.0,
    )
    # The first vol_lookback rows have no vol estimate → position == signal == 1.0.
    warmup_rows = r["position"].iloc[:vol_lookback]
    assert (warmup_rows - 1.0).abs().max() < 1e-12, (
        "Vol targeting scaled positions during warmup (before window was full)."
    )


# ── Circuit breaker tests ──────────────────────────────────────────────────────

def _run_with_breaker(prices, signals_values, max_drawdown=0.10, cooldown_bars=5):
    bars    = _make_bars(prices)
    signals = _make_signals(signals_values)
    return run_event_driven(
        bars, signals, fee_bps=0.0,
        max_drawdown=max_drawdown, cooldown_bars=cooldown_bars,
    )


def test_circuit_breaker_fires():
    """After a large drawdown, position must drop to 0."""
    # Prices: long bull run to build equity, then sharp crash.
    prices = [100.0 * (1.005 ** i) for i in range(30)] + \
             [100.0 * (1.005 ** 29) * (0.97 ** i) for i in range(20)]
    signals_values = [1.0] * 50

    r = _run_with_breaker(prices, signals_values, max_drawdown=0.10)
    # At some point equity must drop >10% → circuit fires → position goes to 0.
    assert (r["position"] == 0.0).any(), "Circuit breaker never fired."


def test_circuit_breaker_stays_flat_during_cooldown():
    """While the circuit is active, position must remain 0 throughout cooldown."""
    prices = [100.0 * (1.005 ** i) for i in range(20)] + \
             [100.0 * (1.005 ** 19) * (0.96 ** i) for i in range(30)]
    signals_values = [1.0] * 50

    cooldown = 10
    r = _run_with_breaker(prices, signals_values, max_drawdown=0.05, cooldown_bars=cooldown)

    fired_idx = r[r["position"] == 0.0].index
    if fired_idx.empty:
        pytest.skip("Circuit never fired — adjust test prices.")

    fire_loc = r.index.get_loc(fired_idx[0])
    # The next `cooldown` rows after the circuit fires must all be 0.
    end_loc = min(fire_loc + cooldown, len(r))
    flat_window = r["position"].iloc[fire_loc:end_loc]
    assert (flat_window == 0.0).all(), (
        "Position was non-zero during circuit-breaker cooldown."
    )


def test_circuit_breaker_resumes_after_cooldown():
    """After cooldown_bars bars flat, the strategy must be able to re-enter."""
    # Bull → crash → long flat period → bull again.
    prices = (
        [100.0 * (1.01 ** i) for i in range(15)]           # bull: build equity
        + [100.0 * (1.01 ** 14) * (0.93 ** i) for i in range(5)]   # crash: trigger breaker
        + [100.0 * (1.01 ** 14) * (0.93 ** 4)] * 30        # flat market: cooldown
        + [100.0 * (1.01 ** 14) * (0.93 ** 4) * (1.01 ** i) for i in range(10)]  # bull again
    )
    signals_values = [1.0] * len(prices)

    cooldown = 5
    r = _run_with_breaker(prices, signals_values, max_drawdown=0.05, cooldown_bars=cooldown)

    # Find the first flat bar after circuit fires.
    fired = r[r["position"] == 0.0]
    if fired.empty:
        pytest.skip("Circuit never fired — adjust test prices.")

    fire_loc = r.index.get_loc(fired.index[0])
    resume_loc = fire_loc + cooldown + 1
    if resume_loc >= len(r):
        pytest.skip("Not enough bars after cooldown to check re-entry.")

    post_cooldown = r["position"].iloc[resume_loc:]
    assert (post_cooldown > 0.0).any(), (
        "Strategy never re-entered after circuit-breaker cooldown expired."
    )


# ── Parity: vectorized == event-driven with risk params ───────────────────────

def _parity_risk(prices, signals_values, fee_bps=5.0, slippage_bps=3.0,
                 vol_target=None, vol_lookback=10, periods_per_year=252.0):
    bars    = _make_bars(prices)
    signals = _make_signals(signals_values)
    kwargs  = dict(
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        vol_target=vol_target,
        vol_lookback=vol_lookback,
        periods_per_year=periods_per_year,
    )
    vec = run_vectorized(bars, signals, **kwargs)
    evt = run_event_driven(bars, signals, **kwargs)
    pd.testing.assert_frame_equal(vec, evt, rtol=1e-9, check_like=False)


def test_parity_with_slippage_only():
    prices         = [100.0 * (1.001 ** i) for i in range(80)]
    signals_values = [0.0] * 20 + [1.0] * 60
    _parity_risk(prices, signals_values, slippage_bps=5.0)


def test_parity_with_vol_targeting():
    rng            = np.random.default_rng(99)
    prices         = list(100.0 * np.cumprod(1 + rng.normal(0.001, 0.02, 120)))
    signals_values = [1.0] * 120
    _parity_risk(
        prices, signals_values,
        fee_bps=5.0, slippage_bps=3.0,
        vol_target=0.15, vol_lookback=10, periods_per_year=252.0,
    )


def test_parity_vol_targeting_multiple_crossovers():
    rng            = np.random.default_rng(17)
    prices         = list(100.0 * np.cumprod(1 + rng.normal(0.0005, 0.015, 150)))
    # Alternating signals — many trades, more interesting slippage/fee interaction.
    signals_values = [float(i % 3 != 0) for i in range(150)]
    _parity_risk(
        prices, signals_values,
        fee_bps=8.0, slippage_bps=4.0,
        vol_target=0.20, vol_lookback=15, periods_per_year=252.0,
    )


# ── Gap re-bootstrap test ──────────────────────────────────────────────────────

def test_gap_triggers_bootstrap_fn():
    """When a gap is detected, bootstrap_fn must be called exactly once."""
    fast, slow = 3, 5
    prices     = [100.0 + i for i in range(slow + 20)]
    live_bars  = _make_live_bars(prices)

    call_count = [0]
    bootstrap_result = live_bars[:slow + 1]

    def mock_bootstrap():
        call_count[0] += 1
        return bootstrap_result

    engine = LiveEngine(
        _Portfolio(), _Broker(0.0), fast=fast, slow=slow,
        bootstrap_fn=mock_bootstrap,
    )
    engine.initialize(live_bars[:slow + 1])

    # Feed normal bars first to establish _bar_interval.
    for bar in live_bars[slow + 1 : slow + 3]:
        engine.on_bar(bar)

    # Inject a bar with a timestamp 3 intervals ahead (simulates a gap).
    gap_bar = dict(live_bars[slow + 3])
    gap_bar["timestamp"] = live_bars[slow + 2]["timestamp"] + pd.Timedelta(hours=3)
    engine.on_bar(gap_bar)

    assert call_count[0] == 1, (
        f"bootstrap_fn called {call_count[0]} times; expected 1."
    )


def test_no_bootstrap_fn_gap_does_not_crash():
    """A gap without bootstrap_fn must log a warning but not raise."""
    fast, slow = 3, 5
    prices     = [100.0 + i for i in range(slow + 10)]
    live_bars  = _make_live_bars(prices)

    engine = LiveEngine(_Portfolio(), _Broker(0.0), fast=fast, slow=slow)
    engine.initialize(live_bars[:slow + 1])

    for bar in live_bars[slow + 1 : slow + 3]:
        engine.on_bar(bar)

    gap_bar = dict(live_bars[slow + 3])
    gap_bar["timestamp"] = live_bars[slow + 2]["timestamp"] + pd.Timedelta(hours=3)
    engine.on_bar(gap_bar)  # must not raise
