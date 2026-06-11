"""
tests/test_multiasset_parity.py — The critical invariant for the multi-asset engine.

INVARIANT: multiasset.run() must reduce to event_driven.run() when N = 1 and
weights ∈ {0, 1}.  If these diverge, there is an accounting bug.

This is the same discipline as the vectorized/event-driven parity test:
a known reduction guarantees the generalisation is bug-free.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.event_driven import run as run_legacy
from backtest.multiasset import run as run_multi


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_bars(prices: list, symbol: str = "SPY") -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices,
         "close": prices, "volume": [1_000_000] * len(prices)},
        index=idx,
    )


def _make_signals(values: list) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, name="signal")


def _parity_check(prices, signal_values, fee_bps, slippage_bps=0.0):
    bars = _make_bars(prices)
    signals = _make_signals(signal_values)

    # Legacy scalar engine
    out_old = run_legacy(bars, signals, fee_bps=fee_bps, slippage_bps=slippage_bps)

    # Multi-asset engine with a 1-column panel + 1-column weights
    panel = bars[["close"]].rename(columns={"close": "SPY"})
    weights = signals.rename("SPY").to_frame()
    out_new = run_multi(panel, weights, fee_bps=fee_bps, slippage_bps=slippage_bps)

    assert len(out_old) == len(out_new), (
        f"Row count mismatch: legacy={len(out_old)}  multi={len(out_new)}"
    )

    np.testing.assert_allclose(
        out_new["equity"].values,
        out_old["equity"].values,
        atol=1e-9,
        err_msg="equity curves diverge between legacy and multi-asset engines",
    )
    np.testing.assert_allclose(
        out_new["net_return"].values,
        out_old["net_return"].values,
        atol=1e-9,
        err_msg="net_return diverges",
    )
    np.testing.assert_allclose(
        out_new["fee"].values,
        out_old["fee"].values,
        atol=1e-9,
        err_msg="fees diverge",
    )


# ── Parity tests ───────────────────────────────────────────────────────────────

def test_reduces_to_legacy_always_long_no_fees():
    prices = [100.0 * (1.001 ** i) for i in range(100)]
    _parity_check(prices, [1.0] * 100, fee_bps=0.0)


def test_reduces_to_legacy_always_long_with_fees():
    prices = [100.0 * (1.001 ** i) for i in range(100)]
    _parity_check(prices, [1.0] * 100, fee_bps=10.0)


def test_reduces_to_legacy_always_flat():
    prices = [100.0 + i for i in range(100)]
    _parity_check(prices, [0.0] * 100, fee_bps=10.0)


def test_reduces_to_legacy_single_flip():
    prices = [100.0] * 60
    _parity_check(prices, [0.0] * 30 + [1.0] * 30, fee_bps=10.0)


def test_reduces_to_legacy_alternating():
    prices = [100.0 + i * 0.1 for i in range(80)]
    signals = [float(i % 2) for i in range(80)]
    _parity_check(prices, signals, fee_bps=5.0)


def test_reduces_to_legacy_with_slippage():
    prices = [100.0 * (1.002 ** i) for i in range(80)]
    signals = [float(i % 3 == 0) for i in range(80)]
    _parity_check(prices, signals, fee_bps=10.0, slippage_bps=5.0)


def test_reduces_to_legacy_declining_market():
    prices = [100.0 * (0.999 ** i) for i in range(100)]
    _parity_check(prices, [1.0] * 100, fee_bps=10.0)


def test_reduces_to_legacy_realistic_ma():
    """200-bar MA-crossover-style scenario with warmup NaNs."""
    rng = np.random.default_rng(42)
    prices = list(100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, 200)))
    signals = [float("nan")] * 5 + [float(i % 3 == 0) for i in range(195)]
    _parity_check(prices, signals, fee_bps=10.0)


# ── Multi-asset specific tests ─────────────────────────────────────────────────

def test_multiasset_two_assets_basic():
    """Smoke test: 2-asset run produces sensible output."""
    n = 100
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    panel = pd.DataFrame(
        {"A": 100.0 * np.cumprod(1 + rng.normal(0.001, 0.01, n)),
         "B": 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))},
        index=idx,
    )
    weights = pd.DataFrame({"A": [0.6] * n, "B": [0.4] * n}, index=idx)
    result = run_multi(panel, weights, fee_bps=10.0)

    assert len(result) > 0
    assert "equity" in result.columns
    assert "net_return" in result.columns
    assert "turnover" in result.columns
    assert not result.isnull().any().any()


def test_multiasset_equity_starts_at_one_plus_first_return():
    """Equity must compound correctly from 1.0."""
    n = 50
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    panel = pd.DataFrame({"A": [100.0 + i for i in range(n)]}, index=idx)
    weights = pd.DataFrame({"A": [1.0] * n}, index=idx)
    result = run_multi(panel, weights, fee_bps=0.0)
    assert result["equity"].iloc[0] == pytest.approx(
        1.0 + result["net_return"].iloc[0], rel=1e-9
    )


def test_multiasset_constant_weights_zero_turnover():
    """Constant weights → zero turnover on all bars after the first trade."""
    n = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(99)
    panel = pd.DataFrame(
        {"A": 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n)),
         "B": 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n))},
        index=idx,
    )
    weights = pd.DataFrame({"A": [0.5] * n, "B": [0.5] * n}, index=idx)
    result = run_multi(panel, weights, fee_bps=10.0)
    # After the initial allocation (first result bar), weights never change
    assert np.allclose(result["turnover"].iloc[1:].values, 0.0, atol=1e-12)
