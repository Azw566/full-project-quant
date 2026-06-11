"""
main.py — Entry point.

Runs all phases in sequence:
    Phase 0 — data layer verification (reproducibility check)
    Phase 1 — vectorized MA crossover backtest with fees and walk-forward split
    Phase 2 — event-driven backtester parity check
    BTC     — same strategy on BTCUSDT hourly (live instrument validation)
    Phase 6 — multi-asset equal-weight portfolio engine

Usage:
    python main.py
"""

import logging
import sys

# Ensure UTF-8 output on Windows consoles (cp1252 can't encode Greek/math chars).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import yaml

from backtest.event_driven import run as run_event_driven
from backtest.metrics import compute_metrics
from backtest.portfolio import run_portfolio
from backtest.vectorized import run as run_vectorized
from data.loader import get_bars
from feed.binance import get_historical_bars
from strategy import load_strategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def _print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _print_metrics(label: str, metrics: dict) -> None:
    print(f"\n  {label}")
    print(f"    Total return  : {metrics['total_return']:>+.2%}")
    print(f"    Ann return    : {metrics['ann_return']:>+.2%}")
    print(f"    Ann vol       : {metrics['ann_vol']:>.2%}")
    print(f"    Sharpe ratio  : {metrics['sharpe']:>+.2f}")
    print(f"    Max drawdown  : {metrics['max_drawdown']:>.2%}")
    print(f"    Ann turnover  : {metrics['ann_turnover']:>.2f}x")


def run_phase0(cfg: dict) -> None:
    _print_section("PHASE 0 — Data layer (reproducibility check)")

    bars  = get_bars(cfg["symbol"], cfg["start"], cfg["end"], cfg["data_path"])
    bars2 = get_bars(cfg["symbol"], cfg["start"], cfg["end"], cfg["data_path"])

    print(f"\n  Symbol : {cfg['symbol']}")
    print(f"  Shape  : {bars.shape[0]} bars × {bars.shape[1]} columns")
    print(f"  Range  : {bars.index[0].date()}  ->  {bars.index[-1].date()}")
    print(f"\n  {bars.head(3).to_string()}")

    status = "[PASS]" if bars.equals(bars2) else "[FAIL]"
    print(f"\n  {status} Reproducibility check (both calls identical)")


def run_phase1(cfg: dict, bars: pd.DataFrame, strategy_fn) -> None:
    _print_section("PHASE 1 — Vectorized backtest")

    bt_cfg      = cfg["backtest"]
    fee_bps     = bt_cfg["fee_bps"]
    train_ratio = bt_cfg["train_ratio"]

    signals = strategy_fn(bars)

    # ── Walk-forward split ─────────────────────────────────────────────────
    # Split the price data, then backtest each half independently.
    # We report only the out-of-sample (OOS) window — the one the strategy
    # never "saw" during parameter selection.
    n_total = len(bars)
    n_train = int(n_total * train_ratio)

    train_bars    = bars.iloc[:n_train]
    oos_bars      = bars.iloc[n_train:]
    train_signals = signals.iloc[:n_train]
    oos_signals   = signals.iloc[n_train:]

    print(f"\n  Strategy   : {cfg.get('strategy', 'ma_crossover')}")
    print(f"  Fee        : {fee_bps} bps per side")
    print(f"  In-sample  : {train_bars.index[0].date()}  ->  {train_bars.index[-1].date()}  ({n_train} bars)")
    print(f"  OOS        : {oos_bars.index[0].date()}  ->  {oos_bars.index[-1].date()}  ({n_total - n_train} bars)")

    # ── Backtest both windows ──────────────────────────────────────────────
    train_result = run_vectorized(train_bars, train_signals, fee_bps=fee_bps)
    oos_result   = run_vectorized(oos_bars,   oos_signals,   fee_bps=fee_bps)

    # ── Metrics ────────────────────────────────────────────────────────────
    train_metrics = compute_metrics(train_result["net_return"], train_result["position"])
    oos_metrics   = compute_metrics(oos_result["net_return"],   oos_result["position"])

    # Buy-and-hold benchmark (OOS, no fees, always long)
    bh_return = oos_bars["close"].pct_change().dropna()
    bh_pos    = pd.Series(1.0, index=bh_return.index)
    bh_metrics = compute_metrics(bh_return, bh_pos)

    _print_metrics("IN-SAMPLE metrics  (informational only — do not trade on these)", train_metrics)
    _print_metrics("OUT-OF-SAMPLE metrics  (the honest number)", oos_metrics)
    _print_metrics("BUY-AND-HOLD benchmark  (OOS period)", bh_metrics)

    print()
    if oos_metrics["sharpe"] > bh_metrics["sharpe"]:
        print("  Strategy OOS Sharpe > buy-and-hold: strategy added value in this period.")
    else:
        print("  Strategy OOS Sharpe <= buy-and-hold: strategy underperformed passive in this period.")
    print("  (This is expected for a simple rule — the system design is the goal, not the alpha.)\n")


def run_phase2(cfg: dict, bars: pd.DataFrame, strategy_fn) -> None:
    _print_section("PHASE 2 — Event-driven backtest (parity check vs Phase 1)")

    bt_cfg  = cfg["backtest"]
    fee_bps = bt_cfg["fee_bps"]

    signals = strategy_fn(bars)

    vec_result = run_vectorized(bars, signals, fee_bps=fee_bps)
    evt_result = run_event_driven(bars, signals, fee_bps=fee_bps)

    # Compare every column of the two result DataFrames.
    tol = 1e-9
    mismatches = []
    for col in vec_result.columns:
        diff = (vec_result[col] - evt_result[col]).abs()
        if diff.max() > tol:
            mismatches.append(f"{col}: max_diff={diff.max():.2e}")

    if mismatches:
        print("\n  [FAIL] Event-driven results diverge from vectorized:")
        for m in mismatches:
            print(f"    {m}")
        print("\n  This indicates an accounting bug — see notes/phase2_event_driven.txt.")
    else:
        print(f"\n  [PASS] Event-driven matches vectorized within {tol:.0e} on all columns.")
        print(f"         Rows compared : {len(vec_result)}")

    vec_metrics = compute_metrics(vec_result["net_return"], vec_result["position"])
    evt_metrics = compute_metrics(evt_result["net_return"], evt_result["position"])

    _print_metrics("Vectorized engine  (Phase 1 baseline)", vec_metrics)
    _print_metrics("Event-driven engine  (Phase 2)", evt_metrics)
    print()


def run_btc_backtest(cfg: dict, strategy_fn) -> None:
    """
    Run the same MA crossover strategy on BTCUSDT hourly data.

    This is the missing link between the SPY backtest (Phases 1-2) and the
    live system (Phase 3-4): it proves the strategy logic on the actual
    instrument and timescale that the live system trades.

    Data is downloaded from Binance REST and cached to data/cache/ so
    subsequent runs are instant. Uses periods_per_year=8760 for correct
    annualization of hourly bars.
    """
    _print_section("BTC HOURLY BACKTEST — live instrument validation")

    feed_cfg = cfg["feed"]
    bt_cfg   = cfg["backtest"]
    btc_cfg  = cfg.get("btc_backtest", {})

    symbol      = feed_cfg["symbol"]    # BTCUSDT
    interval    = feed_cfg["interval"]  # 1h
    start       = btc_cfg.get("start", "2022-01-01")
    end         = btc_cfg.get("end",   "2024-12-31")
    fee_bps     = bt_cfg["fee_bps"]
    train_ratio = bt_cfg["train_ratio"]

    cache_file = f"{cfg['data_path']}/{symbol}_{interval}.parquet"
    bars       = get_historical_bars(symbol, interval, start, end, cache_path=cache_file)
    signals    = strategy_fn(bars)

    n_total = len(bars)
    n_train = int(n_total * train_ratio)

    train_bars    = bars.iloc[:n_train]
    oos_bars      = bars.iloc[n_train:]
    train_signals = signals.iloc[:n_train]
    oos_signals   = signals.iloc[n_train:]

    print(f"\n  Instrument : {symbol} ({interval} candles)")
    print(f"  Strategy   : {cfg.get('strategy', 'ma_crossover')}")
    print(f"  Fee        : {fee_bps} bps per side")
    print(f"  Period     : {bars.index[0]}  ->  {bars.index[-1]}  ({n_total} bars)")
    print(f"  In-sample  : {train_bars.index[0]}  ->  {train_bars.index[-1]}")
    print(f"  OOS        : {oos_bars.index[0]}  ->  {oos_bars.index[-1]}")

    train_result = run_vectorized(train_bars, train_signals, fee_bps=fee_bps)
    oos_result   = run_vectorized(oos_bars,   oos_signals,   fee_bps=fee_bps)

    # 8760 periods/year for hourly bars — critical for correct Sharpe/vol/return.
    T = 8_760
    train_metrics = compute_metrics(train_result["net_return"], train_result["position"], periods_per_year=T)
    oos_metrics   = compute_metrics(oos_result["net_return"],   oos_result["position"],   periods_per_year=T)

    bh_return  = oos_bars["close"].pct_change().dropna()
    bh_pos     = pd.Series(1.0, index=bh_return.index)
    bh_metrics = compute_metrics(bh_return, bh_pos, periods_per_year=T)

    _print_metrics("IN-SAMPLE metrics  (informational only)", train_metrics)
    _print_metrics("OUT-OF-SAMPLE metrics  (the honest number)", oos_metrics)
    _print_metrics("BUY-AND-HOLD benchmark  (OOS period)", bh_metrics)
    print()


def run_phase6(cfg: dict, bars: pd.DataFrame, strategy_fn) -> None:
    """
    Phase 6 — Multi-asset equal-weight portfolio engine.

    Runs the MA crossover strategy on two assets simultaneously:
      - SPY (the real equity data from Phase 1)
      - A synthetic "SPY-B": SPY returns + small independent noise

    SPY-B is generated deterministically so results are reproducible.
    Its purpose is to give the portfolio engine two distinct but correlated
    assets — exactly the regime where diversification provides partial benefit.

    The key thing to observe: portfolio vol is lower than the average of the
    two individual vols, even when the assets are correlated. This is the
    Markowitz free lunch. When assets are fully correlated (ρ=1), the benefit
    disappears; when uncorrelated (ρ=0), portfolio vol falls to 1/√2 of
    individual vol.
    """
    _print_section("PHASE 6 — Portfolio engine (multi-asset equal weight)")

    bt_cfg  = cfg["backtest"]
    fee_bps = bt_cfg["fee_bps"]

    # Build SPY-B: SPY daily returns + 30%-of-vol independent noise.
    rng          = np.random.default_rng(42)
    spy_returns  = bars["close"].pct_change().dropna()
    noise        = rng.normal(0.0, spy_returns.std() * 0.3, len(spy_returns))
    returns_b    = spy_returns.values + noise
    closes_b     = bars["close"].iloc[0] * np.cumprod(1 + returns_b)
    bars_b       = bars.copy()
    bars_b["close"] = np.concatenate([[bars["close"].iloc[0]], closes_b])
    for col in ("open", "high", "low"):
        bars_b[col] = bars_b["close"]

    sigs_a = strategy_fn(bars)
    sigs_b = strategy_fn(bars_b)

    assets = {
        "SPY":   (bars,   sigs_a),
        "SPY-B": (bars_b, sigs_b),
    }

    result = run_portfolio(assets, fee_bps=fee_bps)

    ptf_m = result.metrics["portfolio"]
    a_m   = result.metrics["SPY"]
    b_m   = result.metrics["SPY-B"]

    print(f"\n  Strategy   : {cfg.get('strategy', 'ma_crossover')}")
    print(f"  Fee        : {fee_bps} bps per side")
    print(f"  Assets     : SPY + SPY-B (equal weight 50/50)")
    print(f"  SPY-B      : SPY returns + 30%-of-vol independent noise (rho ~ 0.95)")
    print(f"  Bars       : {len(result.combined)} (inner join)")

    _print_metrics("SPY   (standalone)", a_m)
    _print_metrics("SPY-B (standalone)", b_m)
    _print_metrics("PORTFOLIO (equal weight)", ptf_m)

    avg_vol    = (a_m["ann_vol"] + b_m["ann_vol"]) / 2
    vol_reduc  = 1.0 - ptf_m["ann_vol"] / avg_vol
    avg_sharpe = (a_m["sharpe"] + b_m["sharpe"]) / 2

    print(f"\n  Vol reduction from diversification : {vol_reduc:+.1%}")
    print(f"  Portfolio Sharpe : {ptf_m['sharpe']:+.2f}   avg individual : {avg_sharpe:+.2f}")
    if ptf_m["sharpe"] >= avg_sharpe - 0.05:
        print("  Diversification maintained or improved risk-adjusted returns.")
    print()


def main() -> None:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    bars        = get_bars(cfg["symbol"], cfg["start"], cfg["end"], cfg["data_path"])
    strategy_fn = load_strategy(cfg)

    run_phase0(cfg)
    run_phase1(cfg, bars, strategy_fn)
    run_phase2(cfg, bars, strategy_fn)
    run_btc_backtest(cfg, strategy_fn)
    run_phase6(cfg, bars, strategy_fn)


if __name__ == "__main__":
    main()
