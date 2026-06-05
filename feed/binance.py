"""
feed/binance.py — Binance market-data feed: REST bootstrap + WebSocket stream.

Two responsibilities:
    1. bootstrap() — synchronous REST call to fetch the last N closed candles
       before the event loop starts. Used to warm up the rolling MA window.
    2. stream()    — async generator that yields one bar dict per closed candle
       from the live WebSocket. Reconnects automatically on disconnect.

No API key is required — Binance provides public market data unauthenticated.

PUBLIC INTERFACE
────────────────
    feed = BinanceFeed(symbol, interval, warmup_bars)
    bars = feed.bootstrap()              # call before asyncio.run()
    async for bar in feed.stream(): ...  # yields closed candles indefinitely

Bar dict schema (same from both bootstrap and stream):
    timestamp : pd.Timestamp (candle close time, UTC)
    open      : float
    high      : float
    low       : float
    close     : float
    volume    : float
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from collections.abc import AsyncIterator
from pathlib import Path

import pandas as pd
import websockets

logger = logging.getLogger(__name__)

_REST_BASE = "https://api.binance.com/api/v3"
_WS_BASE   = "wss://stream.binance.com:9443/ws"


class BinanceFeed:
    def __init__(self, symbol: str, interval: str, warmup_bars: int) -> None:
        """
        Parameters
        ----------
        symbol      : Binance trading pair, e.g. "BTCUSDT"
        interval    : Candle interval string, e.g. "1h", "4h", "1d"
        warmup_bars : How many historical bars to fetch at startup.
                      Should be >= slow_ma to guarantee a full MA window.
        """
        self._symbol_rest = symbol.upper()
        self._symbol_ws   = symbol.lower()
        self._interval    = interval
        self._warmup_bars = warmup_bars

    # ── Bootstrap (sync) ──────────────────────────────────────────────────────

    def bootstrap(self) -> list[dict]:
        """
        Fetch the last `warmup_bars` closed candles from the Binance REST API.

        Runs synchronously — call this before starting the async event loop.
        Takes ~0.5–1s depending on network latency.

        REST endpoint: GET /api/v3/klines?symbol=BTCUSDT&interval=1h&limit=N
        Returns the N most recent completed candles in ascending time order.
        """
        url = (
            f"{_REST_BASE}/klines"
            f"?symbol={self._symbol_rest}"
            f"&interval={self._interval}"
            f"&limit={self._warmup_bars}"
        )
        logger.info(
            "Bootstrap: fetching %d × %s bars for %s ...",
            self._warmup_bars, self._interval, self._symbol_rest,
        )
        with urllib.request.urlopen(url, timeout=15) as resp:
            raw: list = json.loads(resp.read())

        bars = [_parse_rest_kline(row) for row in raw]
        logger.info(
            "Bootstrap complete: %d bars  |  last close=%.2f at %s",
            len(bars), bars[-1]["close"], bars[-1]["timestamp"],
        )
        return bars

    # ── Live stream (async) ───────────────────────────────────────────────────

    async def stream(self) -> AsyncIterator[dict]:
        """
        Yield one bar dict per closed candle from the Binance WebSocket.

        Connects to:
            wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}

        Only yields bars where the candle is confirmed closed (k.x == True).
        Partial (in-progress) candles are silently ignored.

        Reconnects automatically on disconnect with exponential backoff
        (1 s → 2 s → 4 s → ... → 60 s maximum).
        """
        url = f"{_WS_BASE}/{self._symbol_ws}@kline_{self._interval}"
        backoff = 1

        while True:
            try:
                logger.info("WebSocket: connecting to %s", url)
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    backoff = 1  # reset after a successful connection
                    logger.info("WebSocket: connected. Waiting for candles to close...")
                    async for raw in ws:
                        msg = json.loads(raw)
                        kline = msg.get("k")
                        if kline and kline["x"]:   # x = True: candle is closed
                            yield _parse_ws_kline(kline)

            except (websockets.ConnectionClosed, ConnectionError, OSError) as exc:
                logger.warning(
                    "WebSocket disconnected: %s. Reconnecting in %ds ...", exc, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)




# ── Historical download (paginated REST) ──────────────────────────────────────

def get_historical_bars(
    symbol:     str,
    interval:   str,
    start:      str,
    end:        str,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """
    Fetch full historical OHLCV bars from Binance REST API with pagination.

    Parameters
    ----------
    symbol     : Binance pair, e.g. "BTCUSDT"
    interval   : Candle interval, e.g. "1h", "4h", "1d"
    start, end : ISO date strings, e.g. "2022-01-01"
    cache_path : If given, save/load a parquet cache at this path.
                 Subsequent calls skip the download entirely.

    Returns
    -------
    pd.DataFrame with DatetimeIndex (UTC, close-time) and columns:
        open, high, low, close, volume
    — same schema as data/loader.get_bars() so both feed directly into
      generate_signals() and the backtest engines.

    PAGINATION
    ──────────
    Binance's /api/v3/klines returns at most 1000 bars per request.
    We loop, advancing startTime past the last batch's close_time until
    we reach end. For 3 years of 1h data (~26,000 bars) this is ~27 calls.
    """
    if cache_path:
        p = Path(cache_path)
        if p.exists():
            logger.info("Loading %s %s history from cache: %s", symbol, interval, cache_path)
            return pd.read_parquet(p)

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms   = int(pd.Timestamp(end,   tz="UTC").timestamp() * 1000)
    sym      = symbol.upper()

    logger.info("Downloading %s %s history from %s to %s ...", sym, interval, start, end)

    all_rows: list = []
    cursor         = start_ms

    while cursor < end_ms:
        url = (
            f"{_REST_BASE}/klines"
            f"?symbol={sym}"
            f"&interval={interval}"
            f"&startTime={cursor}"
            f"&endTime={end_ms}"
            f"&limit=1000"
        )
        with urllib.request.urlopen(url, timeout=15) as resp:
            batch: list = json.loads(resp.read())

        if not batch:
            break

        all_rows.extend(batch)
        cursor = int(batch[-1][6]) + 1   # advance past last close_time

    if not all_rows:
        raise ValueError(f"No data returned for {sym} {interval} {start}→{end}")

    bars = [_parse_rest_kline(row) for row in all_rows]
    df   = (
        pd.DataFrame(bars)
        .set_index("timestamp")[["open", "high", "low", "close", "volume"]]
    )

    logger.info("Downloaded %d bars for %s %s", len(df), sym, interval)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info("Cached to %s", cache_path)

    return df


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_rest_kline(row: list) -> dict:
    """
    Parse one element from the /klines REST response.

    Response schema (each element is an array):
        [0]  open_time  (ms)
        [1]  open
        [2]  high
        [3]  low
        [4]  close
        [5]  volume
        [6]  close_time (ms)   ← we use this as the canonical timestamp
        ...
    """
    return {
        "timestamp": pd.Timestamp(int(row[6]), unit="ms", tz="UTC"),
        "open":      float(row[1]),
        "high":      float(row[2]),
        "low":       float(row[3]),
        "close":     float(row[4]),
        "volume":    float(row[5]),
    }


def _parse_ws_kline(k: dict) -> dict:
    """
    Parse the 'k' object from a WebSocket kline event.

    WebSocket 'k' fields:
        T  close_time (ms)
        o  open
        h  high
        l  low
        c  close
        v  volume
        x  is_closed (bool)
    """
    return {
        "timestamp": pd.Timestamp(int(k["T"]), unit="ms", tz="UTC"),
        "open":      float(k["o"]),
        "high":      float(k["h"]),
        "low":       float(k["l"]),
        "close":     float(k["c"]),
        "volume":    float(k["v"]),
    }
