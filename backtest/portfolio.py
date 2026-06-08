"""
backtest/portfolio.py — Equal-weight multi-asset portfolio engine.

Runs the same MA crossover strategy across N assets independently, then
combines their returns using equal weighting (each asset contributes 1/N).

The per-asset accounting calls run() from backtest.event_driven unchanged —
this is the North Star principle at the portfolio level: individual strategy
engines are identical whether they run alone or inside a portfolio.

PUBLIC INTERFACE
────────────────
    run_portfolio(assets, fee_bps, ...) → PortfolioResult
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.event_driven import run as _run_asset
from backtest.metrics import compute_metrics


@dataclass
class PortfolioResult:
    """Results from a multi-asset equal-weight portfolio backtest."""
    per_asset: dict[str, pd.DataFrame]  # symbol → individual result DataFrame
    combined:  pd.DataFrame             # portfolio-level result (equal-weighted)
    metrics:   dict[str, dict]          # symbol → metrics dict; "portfolio" key for combined


def run_portfolio(
    assets:           dict[str, tuple[pd.DataFrame, pd.Series]],
    fee_bps:          float        = 10.0,
    slippage_bps:     float        = 0.0,
    vol_target:       float | None = None,
    vol_lookback:     int          = 20,
    periods_per_year: float        = 252.0,
    max_drawdown:     float | None = None,
    cooldown_bars:    int          = 20,
) -> PortfolioResult:
    """
    Run an equal-weight portfolio backtest across multiple assets.

    Parameters
    ----------
    assets : dict mapping symbol → (bars, signals)
        Each asset's OHLCV DataFrame and pre-computed signal Series.
        All assets should share the same bar frequency so that
        periods_per_year applies correctly to all of them.
    fee_bps, slippage_bps, vol_target, ... : same as event_driven.run().
        Risk parameters are applied independently to each asset's engine.

    Returns
    -------
    PortfolioResult with:
        per_asset — individual engine output for each symbol
        combined  — portfolio-level DataFrame (equal-weighted columns)
        metrics   — dict keyed by symbol + "portfolio"

    EQUAL WEIGHTING
    ───────────────
    Each of N assets receives weight 1/N. Portfolio net return each bar:
        r_portfolio[t] = mean(r_net_i[t]  for all i)

    Only bars present in ALL assets contribute to the combined result
    (inner join on the time index). This handles minor start-date
    differences due to data availability across assets.

    Portfolio equity is computed by compounding portfolio net returns —
    not by averaging individual equity curves. The former is financially
    correct: it reflects the combined account value growing at the
    portfolio return each period.

    DIVERSIFICATION
    ───────────────
    For two uncorrelated assets with equal individual vol σ, equal-weighting
    gives portfolio vol σ/√2 — the Markowitz free lunch. This benefit shrinks
    as correlation rises, and disappears entirely at ρ=1.
    """
    if not assets:
        raise ValueError("assets dict is empty — nothing to run")

    run_kwargs = dict(
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        vol_target=vol_target,
        vol_lookback=vol_lookback,
        periods_per_year=periods_per_year,
        max_drawdown=max_drawdown,
        cooldown_bars=cooldown_bars,
    )

    # Run each asset independently through the unchanged event-driven engine.
    per_asset: dict[str, pd.DataFrame] = {
        symbol: _run_asset(bars, signals, **run_kwargs)
        for symbol, (bars, signals) in assets.items()
    }

    # Inner join: only bars present in every asset's result.
    common_index = next(iter(per_asset.values())).index
    for df in per_asset.values():
        common_index = common_index.intersection(df.index)

    def _pool(col: str) -> pd.Series:
        """Equal-weighted mean of `col` across all assets on the common index."""
        frame = pd.DataFrame(
            {sym: df.loc[common_index, col] for sym, df in per_asset.items()}
        )
        return frame.mean(axis=1).rename(col)

    net_return   = _pool("net_return")
    gross_return = _pool("gross_return")
    position     = _pool("position")
    fee          = _pool("fee")
    slippage     = _pool("slippage")
    equity       = (1 + net_return).cumprod().rename("equity")

    combined = pd.DataFrame({
        "position":     position,
        "gross_return": gross_return,
        "fee":          fee,
        "slippage":     slippage,
        "net_return":   net_return,
        "equity":       equity,
    })

    T = int(periods_per_year)
    metrics: dict[str, dict] = {
        sym: compute_metrics(df["net_return"], df["position"], periods_per_year=T)
        for sym, df in per_asset.items()
    }
    metrics["portfolio"] = compute_metrics(
        combined["net_return"], combined["position"], periods_per_year=T
    )

    return PortfolioResult(per_asset=per_asset, combined=combined, metrics=metrics)
