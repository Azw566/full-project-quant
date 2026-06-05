"""
tests/test_metrics.py — Unit tests for backtest/metrics.py.

We test metrics on synthetic, manually-computable series so every assertion
can be verified by hand. If these pass, we trust the metrics on real data.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import compute_metrics, TRADING_DAYS_PER_YEAR


def _make_returns(values: list, positions: list = None) -> tuple:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    returns = pd.Series(values, index=idx, name="net_return")
    if positions is None:
        positions = [1.0] * len(values)
    pos = pd.Series(positions, index=idx, name="position")
    return returns, pos


def test_zero_returns_zero_metrics():
    """All-zero daily returns → total return = 0, sharpe = 0."""
    n = TRADING_DAYS_PER_YEAR
    returns, pos = _make_returns([0.0] * n)
    m = compute_metrics(returns, pos)
    assert m["total_return"] == pytest.approx(0.0, abs=1e-10)
    assert m["sharpe"] == pytest.approx(0.0, abs=1e-10)
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-10)


def test_positive_returns_positive_sharpe():
    """Positive mean daily returns with variance → positive Sharpe."""
    # Alternating +0.002 / +0.001 keeps mean positive and std non-zero
    values = [0.002 if i % 2 == 0 else 0.001 for i in range(TRADING_DAYS_PER_YEAR)]
    returns, pos = _make_returns(values)
    m = compute_metrics(returns, pos)
    assert m["sharpe"] > 0


def test_negative_returns_negative_sharpe():
    """Negative mean daily returns with variance → negative Sharpe."""
    values = [-0.002 if i % 2 == 0 else -0.001 for i in range(TRADING_DAYS_PER_YEAR)]
    returns, pos = _make_returns(values)
    m = compute_metrics(returns, pos)
    assert m["sharpe"] < 0


def test_total_return_compounding():
    """
    3 daily returns of 1% should compound to ~3.03%, not exactly 3%.
    Verifies geometric compounding: (1.01)^3 - 1 = 0.030301.
    """
    returns, pos = _make_returns([0.01, 0.01, 0.01])
    m = compute_metrics(returns, pos)
    assert m["total_return"] == pytest.approx((1.01 ** 3) - 1, rel=1e-6)


def test_max_drawdown_is_nonpositive():
    """Max drawdown must always be <= 0 by definition."""
    returns, pos = _make_returns([0.02, -0.05, 0.03, -0.01, 0.04])
    m = compute_metrics(returns, pos)
    assert m["max_drawdown"] <= 0.0


def test_max_drawdown_known_case():
    """
    Manually computed drawdown case:
        Equity: 1.0 → 1.1 → 0.99 → 1.089
        Drawdown from peak 1.1 to trough 0.99: (0.99/1.1 - 1) ≈ -0.1
    """
    returns = pd.Series([0.10, -0.10, 0.10], name="net_return")
    pos = pd.Series([1.0, 1.0, 1.0], name="position")
    m = compute_metrics(returns, pos)
    # Peak equity = 1.10, trough = 1.10 * (1 - 0.10) = 0.99
    expected_dd = (0.99 / 1.10) - 1.0  # ≈ -0.0909
    assert m["max_drawdown"] == pytest.approx(expected_dd, rel=1e-5)


def test_ann_vol_known_case():
    """
    For constant daily returns of σ, ann_vol = σ × √252.
    Using a series with known std validates the formula.
    """
    # Alternating +r and -r gives std ≈ r (for small r, exact for 2-value dist)
    daily_std = 0.01
    n = TRADING_DAYS_PER_YEAR
    values = [daily_std if i % 2 == 0 else -daily_std for i in range(n)]
    returns, pos = _make_returns(values)
    m = compute_metrics(returns, pos)
    expected_vol = pd.Series(values).std(ddof=1) * (TRADING_DAYS_PER_YEAR ** 0.5)
    assert m["ann_vol"] == pytest.approx(expected_vol, rel=1e-6)


def test_turnover_no_changes():
    """If position never changes, annualized turnover should be 0."""
    n = 100
    returns, pos = _make_returns([0.001] * n, positions=[1.0] * n)
    m = compute_metrics(returns, pos)
    assert m["ann_turnover"] == pytest.approx(0.0, abs=1e-10)


def test_turnover_full_flip_every_day():
    """
    Position flips 0→1→0→1 every bar.
    Daily turnover = 1.0 each bar (after first), so ann_turnover ≈ 252.
    """
    n = TRADING_DAYS_PER_YEAR
    positions = [float(i % 2) for i in range(n)]
    returns, pos = _make_returns([0.0] * n, positions=positions)
    m = compute_metrics(returns, pos)
    # diff().abs() on [0,1,0,1,...]: first is NaN, rest are 1.0; mean ≈ 1.0
    assert m["ann_turnover"] == pytest.approx(TRADING_DAYS_PER_YEAR, rel=0.01)


def test_empty_series_raises():
    """Empty input should raise, not silently return garbage."""
    returns = pd.Series([], dtype=float)
    pos = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="empty"):
        compute_metrics(returns, pos)


def test_periods_per_year_changes_annualization():
    """
    The same returns series should produce different ann_vol / ann_return
    depending on periods_per_year.

    daily (252): ann_vol = daily_std × √252
    hourly (8760): ann_vol = daily_std × √8760

    The ratio should equal √(8760/252) ≈ 5.896.
    """
    n = 500
    rng = np.random.default_rng(0)
    values = rng.normal(0.001, 0.01, n).tolist()
    returns, pos = _make_returns(values)

    m_daily  = compute_metrics(returns, pos, periods_per_year=252)
    m_hourly = compute_metrics(returns, pos, periods_per_year=8760)

    ratio = m_hourly["ann_vol"] / m_daily["ann_vol"]
    assert ratio == pytest.approx((8760 / 252) ** 0.5, rel=1e-6)


def test_get_metrics_uses_interval_periods(tmp_path):
    """
    LiveEngine.get_metrics(interval) must use _PERIODS_PER_YEAR[interval],
    not the hardcoded 252.  An hourly session's ann_vol must be √(8760/252) ≈ 5.9×
    larger than a daily session's ann_vol computed from the same returns.
    """
    import pandas as pd
    from live.engine import LiveEngine
    from backtest.event_driven import _Broker, _Portfolio

    fast, slow = 2, 4
    engine = LiveEngine(_Portfolio(), _Broker(0.0), fast=fast, slow=slow)

    # Noisy prices so returns have meaningful variance (uniform prices have ~0 vol)
    rng    = np.random.default_rng(42)
    prices = list(100.0 * np.cumprod(1 + rng.normal(0.002, 0.02, 30)))
    idx    = pd.date_range("2024-01-01", periods=len(prices), freq="h")
    bars   = [{"timestamp": ts, "open": p, "high": p, "low": p, "close": p, "volume": 1e6}
              for ts, p in zip(idx, prices)]
    for bar in bars:
        engine.on_bar(bar)

    m_hourly = engine.get_metrics(interval="1h")
    m_daily  = engine.get_metrics(interval="1d")

    assert m_hourly is not None
    assert m_daily  is not None

    # ann_vol = std × √T; ratio should be √(8760/252)
    expected_ratio = (8760 / 252) ** 0.5
    actual_ratio   = m_hourly["ann_vol"] / m_daily["ann_vol"]
    assert actual_ratio == pytest.approx(expected_ratio, rel=1e-6)
