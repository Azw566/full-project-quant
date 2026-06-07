"""
live/runner.py — Entry point for the live trading session.

Usage (from the repo root):
    python live/runner.py              # paper mode (simulated fills)
    python live/runner.py --testnet    # testnet mode (real Binance testnet orders)

What happens:
    1. Load config.yaml.
    2. Bootstrap the last `slow_ma` closed 1h BTCUSDT candles from the
       Binance REST API (sync, ~0.5s).
    3. Initialize the LiveEngine with the bootstrap data.
    4. Connect to the Binance WebSocket and process candles indefinitely,
       printing a line for every closed candle.
    5. On Ctrl+C: disconnect cleanly, print a session summary, and exit.

Paper mode: all fills are simulated at the candle's closing price.
Testnet mode: real market orders are submitted to the Binance Spot Testnet.
    Requires BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET env vars.
    Get a free key pair at https://testnet.binance.vision/
"""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Live trading session")
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Submit real orders to the Binance Spot Testnet (requires API keys in env)",
    )
    args = parser.parse_args()

    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    feed_cfg = cfg["feed"]
    bt_cfg   = cfg["backtest"]
    risk_cfg = cfg.get("risk", {})

    symbol   = feed_cfg["symbol"]
    interval = feed_cfg["interval"]
    fast     = bt_cfg["fast_ma"]
    slow     = bt_cfg["slow_ma"]
    fee_bps  = bt_cfg["fee_bps"]

    # Risk parameters from config (all optional — defaults disable each feature).
    vol_target      = risk_cfg.get("vol_target",    None)
    vol_lookback    = risk_cfg.get("vol_lookback",  20)
    max_drawdown    = risk_cfg.get("max_drawdown",  None)
    cooldown_bars   = risk_cfg.get("cooldown_bars", 20)
    slippage_bps    = risk_cfg.get("slippage_bps",  0.0)

    # Annualisation factor for vol targeting — must match the candle interval.
    _PERIODS: dict[str, float] = {
        "1m": 525_600, "5m": 105_120, "15m": 35_040,
        "1h": 8_760,   "4h": 2_190,   "1d": 252,
    }
    periods_per_year = _PERIODS.get(interval, 252)

    feed = BinanceFeed(symbol, interval, warmup_bars=slow + 1)

    portfolio = _Portfolio(
        vol_target=vol_target,
        vol_lookback=vol_lookback,
        periods_per_year=periods_per_year,
        max_drawdown=max_drawdown,
        cooldown_bars=cooldown_bars,
    )

    if args.testnet:
        from execution.binance_broker import TestnetBroker, load_credentials
        api_key, api_secret = load_credentials()
        order_qty = cfg.get("testnet", {}).get("order_qty", 0.001)
        # Real fills already include spread — no simulated slippage on top.
        broker = TestnetBroker(api_key, api_secret, symbol=symbol, order_qty=order_qty)
        broker.sync_clock()
        logger.info("Broker: Binance Spot Testnet  |  order_qty=%.4f BTC", order_qty)
    else:
        broker = _Broker(fee_bps, slippage_bps)
        logger.info(
            "Broker: simulated paper  |  fee_bps=%.1f  slippage_bps=%.1f",
            fee_bps, slippage_bps,
        )

    engine = LiveEngine(
        portfolio, broker, fast=fast, slow=slow,
        bootstrap_fn=feed.bootstrap,
    )

    # Synchronous bootstrap — blocks for ~0.5 s, runs before event loop.
    bootstrap_bars = feed.bootstrap()
    engine.initialize(bootstrap_bars)

    mode = "TESTNET" if args.testnet else "PAPER"
    logger.info("=" * 62)
    logger.info("Live %s session  |  %s %s  |  Press Ctrl+C to stop", mode, symbol, interval)
    logger.info("=" * 62)

    try:
        asyncio.run(_stream(feed, engine))
    except KeyboardInterrupt:
        pass
    finally:
        _print_summary(engine, cfg)


if __name__ == "__main__":
    main()
