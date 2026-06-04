"""
main.py — Entry point.

Runs both phases in sequence:
    Phase 0 — data layer verification (reproducibility check)
    Phase 1 — vectorized MA crossover backtest with fees and walk-forward split

Usage:
    python main.py
"""

import logging

import pandas as pd
import yaml

from data.loader import get_bars
from strategy.ma_crossover import generate_signals
from backtest.vectorized import run as run_vectorized
from backtest.event_driven import run as run_event_driven
from backtest.metrics import compute_metrics

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


def run_phase1(cfg: dict, bars: pd.DataFrame) -> None:
    _print_section("PHASE 1 — Vectorized backtest (MA crossover)")

    bt_cfg      = cfg["backtest"]
    fast        = bt_cfg["fast_ma"]
    slow        = bt_cfg["slow_ma"]
    fee_bps     = bt_cfg["fee_bps"]
    train_ratio = bt_cfg["train_ratio"]

    # ── Signal generation ──────────────────────────────────────────────────
    signals = generate_signals(bars, fast=fast, slow=slow)

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

    print(f"\n  Strategy   : {fast}/{slow}-day MA crossover")
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


def run_phase2(cfg: dict, bars: pd.DataFrame) -> None:
    _print_section("PHASE 2 — Event-driven backtest (parity check vs Phase 1)")

    bt_cfg  = cfg["backtest"]
    fast    = bt_cfg["fast_ma"]
    slow    = bt_cfg["slow_ma"]
    fee_bps = bt_cfg["fee_bps"]

    signals = generate_signals(bars, fast=fast, slow=slow)

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


def main() -> None:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    bars = get_bars(cfg["symbol"], cfg["start"], cfg["end"], cfg["data_path"])

    run_phase0(cfg)
    run_phase1(cfg, bars)
    run_phase2(cfg, bars)


if __name__ == "__main__":
    main()
