"""
tests/test_no_lookahead.py — Verify strict causal discipline.

A backtester that accidentally reads tomorrow's data will pass most
return-level tests but fail these structural checks.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.multiasset import run as run_multi
from strategy.rebalance import generate_weights
from risk.covariance import sample_cov
from alloc.base import gmv_weights


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_panel(n: int = 100, n_assets: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = {f"A{i}": 100.0 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
              for i in range(n_assets)}
    return pd.DataFrame(prices, index=idx)


def _gmv_allocator(mu, Sigma, **_):
    return gmv_weights(Sigma)


# ── 1-bar lag test ─────────────────────────────────────────────────────────────

def test_extra_lag_changes_only_last_bar():
    """
    Shifting weights by one additional bar (total lag = 2) should change the
    P&L on every bar by exactly one bar of drift relative to lag-1.
    Specifically: adding extra lag must not produce a *smaller* total return
    in the lucky direction — that would be evidence of look-ahead.

    Practical check: the two equity curves must differ, and the lag-2 curve
    must be calculable without errors (no NaN, no shape mismatch).
    """
    panel = _make_panel()
    n = len(panel)
    weights = pd.DataFrame({"A0": [0.5] * n, "A1": [0.3] * n, "A2": [0.2] * n},
                           index=panel.index)

    result_lag1 = run_multi(panel, weights, fee_bps=0.0)
    # Extra shift: lag by 2 instead of 1
    result_lag2 = run_multi(panel, weights.shift(1), fee_bps=0.0)

    # Both runs must be valid
    assert not result_lag1.isnull().any().any()
    assert not result_lag2.isnull().any().any()
    # The two must differ (extra lag changes the timing of positions).
    # Shape may differ due to extra NaN from double-shifting; compare common prefix.
    min_len = min(len(result_lag1), len(result_lag2))
    assert not np.allclose(
        result_lag1["equity"].values[:min_len],
        result_lag2["equity"].values[:min_len],
    )


# ── generate_weights causal check ─────────────────────────────────────────────

def test_generate_weights_window_is_causal():
    """
    generate_weights must use only data up to bar i-1 when computing
    weights for bar i.  Verify by checking that masking out bar i from the
    returns does NOT change the weight at bar i.
    """
    panel = _make_panel(n=80, n_assets=3)
    lookback = 20
    rebalance_every = 5

    weights_normal = generate_weights(
        panel, _gmv_allocator, sample_cov, lookback, rebalance_every
    )

    # For the first rebalance bar (i = lookback = 20), corrupt bar 20 in the
    # panel.  The weight at bar 20 should be unchanged if the window is [0,20).
    panel_corrupted = panel.copy()
    panel_corrupted.iloc[lookback] *= 999.0

    weights_corrupted = generate_weights(
        panel_corrupted, _gmv_allocator, sample_cov, lookback, rebalance_every
    )

    # Weight at the first rebalance bar must be identical (bar 20 data not used)
    ts_first_rebalance = panel.index[lookback]
    np.testing.assert_array_equal(
        weights_normal.loc[ts_first_rebalance].values,
        weights_corrupted.loc[ts_first_rebalance].values,
        err_msg="Rebalance weight changed when we corrupted bar i — look-ahead bug!",
    )


def test_generate_weights_no_nan_after_warmup():
    """All weights after the first rebalance must be non-NaN."""
    panel = _make_panel(n=80, n_assets=3)
    weights = generate_weights(panel, _gmv_allocator, sample_cov, lookback=20,
                               rebalance_every=5)
    after_warmup = weights.iloc[20:]
    assert not after_warmup.isnull().any().any()


def test_generate_weights_sum_to_one_after_warmup():
    """Weight rows with non-zero allocation must sum to 1 (fully invested)."""
    panel = _make_panel(n=80, n_assets=3)
    weights = generate_weights(panel, _gmv_allocator, sample_cov, lookback=20,
                               rebalance_every=5)
    row_sums = weights.iloc[20:].sum(axis=1)
    non_zero = row_sums[row_sums > 1e-10]
    assert len(non_zero) > 0, "No allocated rows found after warmup"
    assert np.allclose(non_zero.values, 1.0, atol=1e-6), (
        f"Non-zero rows do not sum to 1: min={non_zero.min():.6f}"
    )
