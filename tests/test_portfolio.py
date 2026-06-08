"""
tests/test_portfolio.py — Phase 6 portfolio engine tests.

Covers:
  1. Structure     — return type, required columns, metric keys
  2. Single-asset  — 1-asset portfolio matches run_event_driven exactly
  3. Equal weight  — portfolio return == mean of per-asset returns
  4. Equity        — equity == cumprod(1 + net_return), not avg of equities
  5. Diversification — portfolio vol < avg individual vol (uncorrelated assets)
  6. Risk params   — slippage and vol_target pass through to each asset
  7. Edge cases    — empty dict raises, index intersection, all-flat signals
"""

import numpy as np
import pandas as pd
import pytest

from backtest.event_driven import run as run_event_driven
from backtest.portfolio import PortfolioResult, run_portfolio


# ── Helpers ────────────────────────────────────────────────────────────────────

def _bars(prices: list) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({
        "open":   prices, "high": prices,
        "low":    prices, "close": prices,
        "volume": [1_000_000] * len(prices),
    }, index=idx)


def _sigs(values: list) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx)


# ── 1. Structure ───────────────────────────────────────────────────────────────

def test_returns_portfolio_result():
    result = run_portfolio({"A": (_bars([100 + i for i in range(30)]), _sigs([1.0] * 30))})
    assert isinstance(result, PortfolioResult)


def test_empty_assets_raises():
    with pytest.raises(ValueError):
        run_portfolio({})


def test_combined_has_required_columns():
    result = run_portfolio({"A": (_bars([100 + i for i in range(30)]), _sigs([1.0] * 30))})
    required = {"position", "gross_return", "fee", "slippage", "net_return", "equity"}
    assert required.issubset(result.combined.columns)


def test_metrics_has_symbol_and_portfolio_keys():
    result = run_portfolio({"ASSET_X": (_bars([100 + i for i in range(30)]), _sigs([1.0] * 30))})
    assert "ASSET_X"   in result.metrics
    assert "portfolio" in result.metrics


def test_per_asset_keys_match_input():
    bars = _bars([100 + i for i in range(30)])
    sigs = _sigs([1.0] * 30)
    result = run_portfolio({"AAA": (bars, sigs), "BBB": (bars, sigs)})
    assert set(result.per_asset.keys()) == {"AAA", "BBB"}


def test_metrics_dict_has_standard_keys():
    result = run_portfolio({"A": (_bars([100 + i for i in range(30)]), _sigs([1.0] * 30))})
    for key in ("total_return", "ann_return", "ann_vol", "sharpe", "max_drawdown", "ann_turnover"):
        assert key in result.metrics["portfolio"]


# ── 2. Single-asset parity ─────────────────────────────────────────────────────

def test_single_asset_net_return_matches_standalone():
    prices = [100.0 * (1.003 ** i) for i in range(60)]
    bars   = _bars(prices)
    sigs   = _sigs([0.0] * 10 + [1.0] * 50)

    standalone = run_event_driven(bars, sigs, fee_bps=10.0)
    portfolio  = run_portfolio({"X": (bars, sigs)}, fee_bps=10.0)

    pd.testing.assert_series_equal(
        portfolio.combined["net_return"], standalone["net_return"], rtol=1e-12,
    )


def test_single_asset_equity_matches_standalone():
    prices = [100.0 * (1.003 ** i) for i in range(60)]
    bars   = _bars(prices)
    sigs   = _sigs([0.0] * 10 + [1.0] * 50)

    standalone = run_event_driven(bars, sigs, fee_bps=10.0)
    portfolio  = run_portfolio({"X": (bars, sigs)}, fee_bps=10.0)

    pd.testing.assert_series_equal(
        portfolio.combined["equity"], standalone["equity"], rtol=1e-12,
    )


def test_single_asset_position_matches_standalone():
    prices = [100.0 + i * 0.5 for i in range(50)]
    bars   = _bars(prices)
    sigs   = _sigs([0.0] * 20 + [1.0] * 30)

    standalone = run_event_driven(bars, sigs, fee_bps=5.0)
    portfolio  = run_portfolio({"Z": (bars, sigs)}, fee_bps=5.0)

    pd.testing.assert_series_equal(
        portfolio.combined["position"], standalone["position"], rtol=1e-12,
    )


def test_single_asset_per_asset_matches_standalone():
    """per_asset["X"] must be identical to a direct run_event_driven call."""
    prices = [100.0 * (1.002 ** i) for i in range(50)]
    bars   = _bars(prices)
    sigs   = _sigs([1.0] * 50)

    standalone = run_event_driven(bars, sigs, fee_bps=8.0)
    portfolio  = run_portfolio({"X": (bars, sigs)}, fee_bps=8.0)

    pd.testing.assert_frame_equal(portfolio.per_asset["X"], standalone, rtol=1e-12)


# ── 3. Equal weighting ─────────────────────────────────────────────────────────

def test_two_asset_return_is_mean_of_individual():
    """Portfolio net_return must equal the simple mean of per-asset returns."""
    rng = np.random.default_rng(42)
    n   = 80

    prices_a = list(100.0 * np.cumprod(1 + rng.normal(0.001, 0.01, n)))
    prices_b = list(100.0 * np.cumprod(1 + rng.normal(0.001, 0.01, n)))

    result = run_portfolio({
        "A": (_bars(prices_a), _sigs([1.0] * n)),
        "B": (_bars(prices_b), _sigs([1.0] * n)),
    }, fee_bps=0.0)

    common = result.per_asset["A"].index.intersection(result.per_asset["B"].index)
    expected = (
        result.per_asset["A"].loc[common, "net_return"]
        + result.per_asset["B"].loc[common, "net_return"]
    ) / 2.0

    pd.testing.assert_series_equal(result.combined["net_return"], expected, rtol=1e-12)


def test_three_asset_equal_weighting():
    """Each of 3 assets contributes 1/3; verify on net_return."""
    rng = np.random.default_rng(7)
    n   = 60
    assets = {
        c: (_bars(list(100 * np.cumprod(1 + rng.normal(0.001, 0.01, n)))), _sigs([1.0] * n))
        for c in ("X", "Y", "Z")
    }
    result = run_portfolio(assets, fee_bps=0.0)

    idx = result.combined.index
    expected = sum(result.per_asset[c].loc[idx, "net_return"] for c in ("X", "Y", "Z")) / 3.0
    pd.testing.assert_series_equal(result.combined["net_return"], expected, rtol=1e-12)


# ── 4. Equity compounding ─────────────────────────────────────────────────────

def test_portfolio_equity_is_cumprod_not_avg_of_equities():
    """
    Equity must be (1 + portfolio_net_return).cumprod().
    Averaging individual equity curves gives a different (wrong) result.
    """
    prices = [100 + i for i in range(40)]
    bars   = _bars(prices)
    sigs   = _sigs([1.0] * 40)

    result = run_portfolio({"X": (bars, sigs), "Y": (bars, sigs)}, fee_bps=5.0)

    correct_equity = (1 + result.combined["net_return"]).cumprod()
    pd.testing.assert_series_equal(
        result.combined["equity"], correct_equity, rtol=1e-12, check_names=False,
    )


def test_equity_starts_near_one():
    """First equity value should be 1 + first bar's net return (not 1.0 itself)."""
    prices = [100 + i for i in range(30)]
    bars   = _bars(prices)
    sigs   = _sigs([1.0] * 30)

    result = run_portfolio({"A": (bars, sigs)}, fee_bps=0.0)
    expected_first = 1.0 + result.combined["net_return"].iloc[0]
    assert abs(result.combined["equity"].iloc[0] - expected_first) < 1e-12


# ── 5. Diversification ────────────────────────────────────────────────────────

def test_uncorrelated_assets_reduce_vol():
    """
    Two uncorrelated assets with equal vol → portfolio vol < mean individual vol.
    This is the Markowitz diversification effect.
    """
    rng = np.random.default_rng(99)
    n   = 500

    prices_a = list(100.0 * np.cumprod(1 + rng.normal(0.0, 0.01, n)))
    prices_b = list(100.0 * np.cumprod(1 + rng.normal(0.0, 0.01, n)))

    result = run_portfolio({
        "A": (_bars(prices_a), _sigs([1.0] * n)),
        "B": (_bars(prices_b), _sigs([1.0] * n)),
    }, fee_bps=0.0)

    vol_a   = result.per_asset["A"]["net_return"].std()
    vol_b   = result.per_asset["B"]["net_return"].std()
    vol_ptf = result.combined["net_return"].std()

    assert vol_ptf < (vol_a + vol_b) / 2.0, (
        f"Portfolio vol {vol_ptf:.6f} should be < mean individual vol "
        f"{(vol_a + vol_b) / 2.0:.6f}"
    )


def test_identical_assets_portfolio_matches_standalone():
    """
    Two identical assets: portfolio vol == individual vol (no diversification
    when ρ=1), and portfolio return == individual return.
    """
    prices = [100.0 * (1.002 ** i) for i in range(80)]
    bars   = _bars(prices)
    sigs   = _sigs([1.0] * 80)

    standalone = run_event_driven(bars, sigs, fee_bps=5.0)
    result     = run_portfolio({"A": (bars, sigs), "B": (bars, sigs)}, fee_bps=5.0)

    pd.testing.assert_series_equal(
        result.combined["net_return"], standalone["net_return"], rtol=1e-12,
    )


# ── 6. Risk parameters pass through ───────────────────────────────────────────

def test_slippage_propagates_to_portfolio():
    """Slippage_bps > 0 must produce a non-zero slippage column in combined."""
    prices = [100 + i for i in range(40)]
    bars   = _bars(prices)
    sigs   = _sigs([0.0] * 10 + [1.0] * 30)

    result = run_portfolio({"A": (bars, sigs)}, fee_bps=10.0, slippage_bps=5.0)
    assert (result.combined["slippage"] > 0).any()


def test_zero_slippage_default_matches_explicit_zero():
    prices = [100.0 * (1.002 ** i) for i in range(50)]
    bars   = _bars(prices)
    sigs   = _sigs([1.0] * 50)

    r_default = run_portfolio({"A": (bars, sigs)}, fee_bps=10.0)
    r_zero    = run_portfolio({"A": (bars, sigs)}, fee_bps=10.0, slippage_bps=0.0)
    pd.testing.assert_frame_equal(r_default.combined, r_zero.combined, rtol=1e-12)


# ── 7. Edge cases ─────────────────────────────────────────────────────────────

def test_all_flat_signals_zero_returns():
    """All-flat signals → zero gross_return for every bar."""
    prices = [100 + i for i in range(30)]
    bars   = _bars(prices)
    sigs   = _sigs([0.0] * 30)

    result = run_portfolio({"A": (bars, sigs), "B": (bars, sigs)}, fee_bps=10.0)
    assert (result.combined["gross_return"] == 0.0).all()


def test_fee_nonneg_every_bar():
    """Fee must be >= 0 on every bar."""
    rng    = np.random.default_rng(3)
    prices = list(100 * np.cumprod(1 + rng.normal(0.001, 0.01, 60)))
    bars   = _bars(prices)
    sigs   = _sigs([float(i % 2) for i in range(60)])

    result = run_portfolio({"A": (bars, sigs), "B": (bars, sigs)}, fee_bps=10.0)
    assert (result.combined["fee"] >= 0.0).all()
