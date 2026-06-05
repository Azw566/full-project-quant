"""
live/runner.py — Entry point for the live paper trading session.

Usage (from the repo root):
    python live/runner.py
    python -m live.runner

What happens:
    1. Load config.yaml.
    2. Bootstrap the last `slow_ma` closed 1h BTCUSDT candles from the
       Binance REST API (sync, ~0.5s).
    3. Initialize the LiveEngine with the bootstrap data.
    4. Connect to the Binance WebSocket and process candles indefinitely,
       printing a line for every closed candle.
    5. On Ctrl+C: disconnect cleanly, print a session summary, and exit.

This is a paper-only session — no real orders are placed.
All position changes are simulated fills at the candle's closing price.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import yaml

# Ensure the repo root is on the path when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.event_driven import _Broker, _Portfolio
from feed.binance import BinanceFeed
from live.engine import LiveEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Session summary ────────────────────────────────────────────────────────────

def _print_summary(engine: LiveEngine, cfg: dict) -> None:
    feed_cfg = cfg["feed"]
    print("\n" + "=" * 62)
    print("  LIVE SESSION SUMMARY")
    print("=" * 62)

    df = engine.get_results()
    if df.empty:
        print("  No completed bars recorded (session ended during warmup).")
        print()
        return

    print(f"\n  Symbol   : {feed_cfg['symbol']}")
    print(f"  Interval : {feed_cfg['interval']}")
    print(f"  Bars     : {len(df)}")
    if not df.empty:
        print(f"  From     : {df.index[0]}")
        print(f"  To       : {df.index[-1]}")

    metrics = engine.get_metrics(feed_cfg["interval"])
    if metrics:
        print(f"\n  Total return  : {metrics['total_return']:>+.2%}")
        print(f"  Ann return    : {metrics['ann_return']:>+.2%}")
        print(f"  Ann vol       : {metrics['ann_vol']:>.2%}")
        print(f"  Sharpe ratio  : {metrics['sharpe']:>+.2f}")
        print(f"  Max drawdown  : {metrics['max_drawdown']:>.2%}")
        print(f"  Ann turnover  : {metrics['ann_turnover']:>.2f}x")
    else:
        print("\n  Not enough bars to compute metrics (need >= 2).")
    print()


# ── Main async loop ────────────────────────────────────────────────────────────

async def _stream(feed: BinanceFeed, engine: LiveEngine) -> None:
    """Drive the event loop: feed bars into the engine as candles close."""
    try:
        async for bar in feed.stream():
            engine.on_bar(bar)
    except asyncio.CancelledError:
        pass


def main() -> None:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    feed_cfg = cfg["feed"]
    bt_cfg   = cfg["backtest"]

    symbol   = feed_cfg["symbol"]
    interval = feed_cfg["interval"]
    fast     = bt_cfg["fast_ma"]
    slow     = bt_cfg["slow_ma"]
    fee_bps  = bt_cfg["fee_bps"]

    feed      = BinanceFeed(symbol, interval, warmup_bars=slow + 1)
    portfolio = _Portfolio()
    broker    = _Broker(fee_bps)
    engine    = LiveEngine(portfolio, broker, fast=fast, slow=slow)

    # Synchronous bootstrap — blocks for ~0.5 s, runs before event loop.
    bootstrap_bars = feed.bootstrap()
    engine.initialize(bootstrap_bars)

    logger.info("=" * 62)
    logger.info("Live paper session  |  %s %s  |  Press Ctrl+C to stop", symbol, interval)
    logger.info("=" * 62)

    try:
        asyncio.run(_stream(feed, engine))
    except KeyboardInterrupt:
        pass
    finally:
        _print_summary(engine, cfg)


if __name__ == "__main__":
    main()
