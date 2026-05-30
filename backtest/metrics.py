"""
backtest/metrics.py — Performance metrics computed from a returns series.

All metrics are computed from scratch — no empyrical dependency — so the
implementation is transparent and testable.

PUBLIC INTERFACE
────────────────
    compute_metrics(returns, positions) → dict
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def compute_metrics(returns: pd.Series, positions: pd.Series) -> dict:
    """
    Compute standard performance metrics from daily strategy returns.

    Parameters
    ----------
    returns : pd.Series
        Daily net returns after fees. Must have no NaN values.
    positions : pd.Series
        Daily position series (0.0 or 1.0). Used to compute turnover.
        Must be aligned to returns.

    Returns
    -------
    dict with keys:
        total_return   — cumulative return over the full period
        ann_return     — geometric annualized return
        ann_vol        — annualized volatility (std of daily returns × √252)
        sharpe         — Sharpe ratio (risk-free rate = 0)
        max_drawdown   — maximum peak-to-trough decline (negative number)
        ann_turnover   — annualized two-way turnover (position changes per year)

    METRIC DEFINITIONS
    ──────────────────
    Ann return:   (1 + total_return)^(252/N) - 1
        Geometric compounding: a 10% return over 2 years is √1.10 - 1 per year,
        not 5%. This matches how investment managers report returns.

    Ann vol:      std(daily_returns) × √252
        Daily volatility scaled to annual. Uses sample std (ddof=1) by convention.
        The √252 comes from the square-root-of-time rule for i.i.d. returns.

    Sharpe ratio: ann_return / ann_vol   (rf = 0)
        Using rf=0 is standard for short-horizon crypto/equity backtests.
        A Sharpe of 1.0 is considered good; >2.0 is excellent and often suspect.

    Max drawdown: min( equity[t] / max(equity[0..t]) - 1 )
        The worst peak-to-trough loss on the equity curve.
        Always <= 0. A max drawdown of -0.20 means -20% from peak.

    Ann turnover: mean(|Δposition|) × 252
        How many times per year the full position flips on average.
        Turnover=2 means the position fully turns over twice a year.
        High turnover × fee costs = strategy death.
    """
    n = len(returns)
    if n == 0:
        raise ValueError("returns series is empty — cannot compute metrics")

    equity = (1 + returns).cumprod()

    total_return = float(equity.iloc[-1] - 1.0)

    years = n / TRADING_DAYS_PER_YEAR
    ann_return = float((1 + total_return) ** (1.0 / years) - 1)

    ann_vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

    sharpe = ann_return / ann_vol if ann_vol > 1e-12 else 0.0

    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1.0
    max_drawdown = float(drawdown.min())

    daily_turnover = positions.diff().abs().mean()
    ann_turnover = float(daily_turnover * TRADING_DAYS_PER_YEAR)

    return {
        "total_return": total_return,
        "ann_return":   ann_return,
        "ann_vol":      ann_vol,
        "sharpe":       sharpe,
        "max_drawdown": max_drawdown,
        "ann_turnover": ann_turnover,
    }
