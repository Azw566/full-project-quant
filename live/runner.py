"""
live/runner.py — Entry point for the live trading session.

Usage (from the repo root):
    python live/runner.py                      # single-asset paper mode
    python live/runner.py --testnet            # single-asset testnet mode
    python live/runner.py --portfolio          # multi-asset paper mode

Single-asset mode trades the symbol set in config.yaml → feed.symbol.
Portfolio mode trades all symbols listed in config.yaml → portfolio.symbols
with one independent LiveEngine per symbol. Results are printed per-asset and
as a simple equal-weight aggregate at session end.

What happens:
    1. Load config.yaml.
    2. Bootstrap the last `slow_ma` closed candles from the Binance REST API
       (sync, ~0.5s per symbol).
    3. Initialize each LiveEngine with its bootstrap data.
    4. Connect to Binance WebSocket streams and process candles indefinitely,
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
from backtest.metrics import PERIODS_PER_YEAR
from feed.binance import BinanceFeed
from live.engine import LiveEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Session summary ────────────────────────────────────────────────────────────

# ── Session summary helpers ────────────────────────────────────────────────────

def _print_engine_summary(symbol: str, engine: LiveEngine, interval: str) -> None:
    df = engine.get_results()
    if df.empty:
        print(f"  {symbol:10s}: no completed bars (session ended during warmup).")
        return

    metrics = engine.get_metrics(interval)
    print(f"\n  {symbol}")
    print(f"    Bars          : {len(df)}")
    print(f"    From / To     : {df.index[0]}  →  {df.index[-1]}")
    if metrics:
        print(f"    Total return  : {metrics['total_return']:>+.2%}")
        print(f"    Ann return    : {metrics['ann_return']:>+.2%}")
        print(f"    Ann vol       : {metrics['ann_vol']:>.2%}")
        print(f"    Sharpe ratio  : {metrics['sharpe']:>+.2f}")
        print(f"    Max drawdown  : {metrics['max_drawdown']:>.2%}")
        print(f"    Ann turnover  : {metrics['ann_turnover']:>.2f}x")
    else:
        print("    (not enough bars to compute metrics)")


def _print_summary(engines: dict[str, LiveEngine], cfg: dict) -> None:
    feed_cfg = cfg["feed"]
    interval = feed_cfg["interval"]

    print("\n" + "=" * 62)
    print("  LIVE SESSION SUMMARY")
    print("=" * 62)

    for symbol, engine in engines.items():
        _print_engine_summary(symbol, engine, interval)

    # Portfolio aggregate when running multiple assets.
    if len(engines) > 1:
        import pandas as pd
        net_returns = {}
        for sym, eng in engines.items():
            df = eng.get_results()
            if not df.empty:
                net_returns[sym] = df["net_return"]

        if net_returns:
            combined = pd.DataFrame(net_returns).dropna()
            if not combined.empty:
                ptf_ret = combined.mean(axis=1)
                equity  = (1 + ptf_ret).cumprod()
                total_r = float(equity.iloc[-1] - 1.0)
                print(f"\n  PORTFOLIO (equal weight, {len(net_returns)} assets)")
                print(f"    Total return  : {total_r:>+.2%}")
                print(f"    Final equity  : {equity.iloc[-1]:.6f}")

    print()


# ── Async stream helpers ───────────────────────────────────────────────────────

async def _stream_one(feed: BinanceFeed, engine: LiveEngine) -> None:
    """Stream one asset into its engine until cancelled."""
    try:
        async for bar in feed.stream():
            engine.on_bar(bar)
    except asyncio.CancelledError:
        pass


async def _stream_all(feed_engine_pairs: list[tuple[BinanceFeed, LiveEngine]]) -> None:
    """Run multiple asset streams concurrently; cancel all on KeyboardInterrupt."""
    tasks = [
        asyncio.create_task(_stream_one(feed, engine))
        for feed, engine in feed_engine_pairs
    ]
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# ── Engine factory ─────────────────────────────────────────────────────────────



def _build_engine(
    symbol: str,
    interval: str,
    fast: int,
    slow: int,
    fee_bps: float,
    risk_cfg: dict,
    testnet: bool,
    testnet_order_qty: float,
    periods_per_year: float,
) -> tuple[BinanceFeed, LiveEngine]:
    feed = BinanceFeed(symbol, interval, warmup_bars=slow + 1)

    portfolio = _Portfolio(
        vol_target=risk_cfg.get("vol_target",    None),
        vol_lookback=risk_cfg.get("vol_lookback",  20),
        periods_per_year=periods_per_year,
        max_drawdown=risk_cfg.get("max_drawdown",  None),
        cooldown_bars=risk_cfg.get("cooldown_bars", 20),
    )

    if testnet:
        from execution.binance_broker import TestnetBroker, load_credentials
        api_key, api_secret = load_credentials()
        broker = TestnetBroker(
            api_key, api_secret, symbol=symbol, order_qty=testnet_order_qty,
        )
        broker.sync_clock()
        logger.info("Broker[%s]: Binance Testnet  |  qty=%.4f", symbol, testnet_order_qty)
    else:
        slippage_bps = risk_cfg.get("slippage_bps", 0.0)
        broker = _Broker(fee_bps, slippage_bps)
        logger.info(
            "Broker[%s]: paper  |  fee=%.1f bps  slip=%.1f bps",
            symbol, fee_bps, slippage_bps,
        )

    engine = LiveEngine(portfolio, broker, fast=fast, slow=slow, bootstrap_fn=feed.bootstrap)
    return feed, engine


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Live trading session")
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Submit real orders to the Binance Spot Testnet (requires API keys in env)",
    )
    parser.add_argument(
        "--portfolio",
        action="store_true",
        help="Run all symbols listed in config.yaml → portfolio.symbols concurrently",
    )
    args = parser.parse_args()

    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    feed_cfg = cfg["feed"]
    bt_cfg   = cfg["backtest"]
    risk_cfg = cfg.get("risk", {})

    interval         = feed_cfg["interval"]
    fast             = bt_cfg["fast_ma"]
    slow             = bt_cfg["slow_ma"]
    fee_bps          = bt_cfg["fee_bps"]
    periods_per_year = PERIODS_PER_YEAR.get(interval, 252)
    testnet_qty      = cfg.get("testnet", {}).get("order_qty", 0.001)

    # Determine which symbols to trade.
    if args.portfolio:
        symbols = cfg.get("portfolio", {}).get("symbols", [feed_cfg["symbol"]])
        if not symbols:
            logger.error("portfolio.symbols is empty in config.yaml — aborting.")
            return
    else:
        symbols = [feed_cfg["symbol"]]

    # Build one feed + engine per symbol.
    feed_engine_pairs: list[tuple[BinanceFeed, LiveEngine]] = []
    engines: dict[str, LiveEngine] = {}

    for sym in symbols:
        feed, engine = _build_engine(
            symbol=sym,
            interval=interval,
            fast=fast,
            slow=slow,
            fee_bps=fee_bps,
            risk_cfg=risk_cfg,
            testnet=args.testnet,
            testnet_order_qty=testnet_qty,
            periods_per_year=periods_per_year,
        )
        bootstrap_bars = feed.bootstrap()
        engine.initialize(bootstrap_bars)
        feed_engine_pairs.append((feed, engine))
        engines[sym] = engine

    mode = "TESTNET" if args.testnet else "PAPER"
    sym_str = ", ".join(symbols)
    logger.info("=" * 62)
    logger.info(
        "Live %s  |  %s  |  %s  |  Press Ctrl+C to stop",
        mode, sym_str, interval,
    )
    logger.info("=" * 62)

    try:
        asyncio.run(_stream_all(feed_engine_pairs))
    except KeyboardInterrupt:
        pass
    finally:
        _print_summary(engines, cfg)


if __name__ == "__main__":
    main()
