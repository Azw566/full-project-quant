"""
backtest/metrics.py — Performance metrics computed from a returns series.

All metrics are computed from scratch — no empyrical dependency — so the
implementation is transparent and testable.

PUBLIC INTERFACE
────────────────
    compute_metrics(returns, positions, periods_per_year=252) → dict
"""

import numpy as np
import pandas as pd

# Default annualization factor. Pass periods_per_year explicitly when the
# bar interval is not daily — e.g. 8760 for 1-hour bars, 252 for daily.
TRADING_DAYS_PER_YEAR = 252

# Canonical map from Binance candle interval string → periods per year.
# Single source of truth — import this wherever annualisation is needed.
PERIODS_PER_YEAR: dict[str, float] = {
    "1m":  525_600,
    "5m":  105_120,
    "15m":  35_040,
    "1h":   8_760,
    "4h":   2_190,
    "1d":     252,
}


def compute_metrics(
    returns: pd.Series,
    positions: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """
    Compute standard performance metrics from a strategy returns series.

    Parameters
    ----------
    returns : pd.Series
        Per-bar net returns after fees. Must have no NaN values.
    positions : pd.Series
        Per-bar position series (0.0 or 1.0). Used to compute turnover.
        Must be aligned to returns.
    periods_per_year : int
        Number of bars per year for annualization.
        252  for daily bars (equity default)
        8760 for 1-hour bars
        2190 for 4-hour bars
        Pass the correct value — using the wrong T produces wrong Sharpe/vol/return.

    Returns
    -------
    dict with keys:
        total_return   — cumulative return over the full period
        ann_return     — geometric annualized return
        ann_vol        — annualized volatility (std × √periods_per_year)
        sharpe         — Sharpe ratio (risk-free rate = 0)
        max_drawdown   — maximum peak-to-trough decline (negative number)
        ann_turnover   — annualized two-way turnover (position changes per year)

    METRIC DEFINITIONS
    ──────────────────
    Ann return:   (1 + total_return)^(T/N) - 1   where T = periods_per_year
        Geometric compounding: a 10% return over 2 years is √1.10 - 1 per year.

    Ann vol:      std(returns, ddof=1) × √T
        Uses sample std (Bessel's correction) by convention.

    Sharpe ratio: ann_return / ann_vol   (rf = 0)
        rf=0 is standard for crypto backtests.
        Note: this uses geometric ann_return over arithmetic ann_vol — standard
        quant practice; differs slightly from arithmetic-mean / std × √T.

    Max drawdown: min( equity[t] / max(equity[0..t]) - 1 )
        Always ≤ 0.

    Ann turnover: mean(|Δposition|) × T
        How many full round-trips per year on average.
    """
    n = len(returns)
    if n == 0:
        raise ValueError("returns series is empty — cannot compute metrics")

    equity = (1 + returns).cumprod()

    total_return = float(equity.iloc[-1] - 1.0)

    years = n / periods_per_year
    if total_return <= -1.0:
        # Total wipeout — geometric annualisation would raise a negative base
        # to a fractional exponent. Clamp to -100% annualised.
        ann_return = -1.0
    else:
        ann_return = float((1 + total_return) ** (1.0 / years) - 1)

    ann_vol = float(returns.std(ddof=1) * np.sqrt(periods_per_year))

    sharpe = ann_return / ann_vol if ann_vol > 1e-12 else 0.0

    rolling_max  = equity.cummax()
    drawdown     = equity / rolling_max - 1.0
    max_drawdown = float(drawdown.min())

    daily_turnover = positions.diff().abs().mean()
    ann_turnover   = float(daily_turnover * periods_per_year)

    return {
        "total_return": total_return,
        "ann_return":   ann_return,
        "ann_vol":      ann_vol,
        "sharpe":       sharpe,
        "max_drawdown": max_drawdown,
        "ann_turnover": ann_turnover,
    }
