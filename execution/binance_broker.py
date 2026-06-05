"""
execution/binance_broker.py — Binance testnet execution broker.

Implements the same on_order(OrderEvent) → FillEvent interface as
backtest.event_driven._Broker, but submits real market orders to the
Binance Spot Testnet instead of simulating fills.

CREDENTIALS
───────────
Get a free API key pair at https://testnet.binance.vision/ (GitHub login).
Set as environment variables before running with --testnet:

    BINANCE_TESTNET_API_KEY=<your_key>
    BINANCE_TESTNET_API_SECRET=<your_secret>

ORDER SIZE
──────────
The strategy's position delta is always 0.0 or ±1.0 (binary long/flat).
We map that to a fixed order_qty (BTC) per trade:
    delta =  1.0 → BUY  order_qty BTC
    delta = -1.0 → SELL order_qty BTC
    delta =  0.0 → no order placed

Using a fixed BTC quantity avoids the BUY/SELL quoteOrderQty asymmetry on
Binance spot (quoteOrderQty is only supported for BUY side).

FEE NORMALIZATION
─────────────────
The exchange returns commission in BTC (buy) or USDT (sell). We normalize
to a position-unit fraction so it's comparable to the simulated broker:

    fee = total_commission_usdt / total_trade_value_usdt

This matches simulated fee = abs(delta) * fee_rate when both charge ~10 bps.

CLOCK DRIFT
───────────
Binance rejects signed requests whose timestamp is outside recvWindow of
server time. Call broker.sync_clock() once at startup to calibrate the
local-to-server offset. recvWindow=10000 (10 s) provides additional buffer.

BLOCKING NOTE
─────────────
on_order() uses requests (synchronous HTTP). In the asyncio stream loop
this blocks the event loop for ~100-500 ms per order. For a 1-hour candle
strategy that is negligible — there's an hour between orders. A
production system would use httpx[async] instead.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from urllib.parse import urlencode

import requests

from backtest.event_driven import FillEvent, OrderEvent

logger = logging.getLogger(__name__)

_BASE_URL        = "https://testnet.binance.vision"
_ORDER_ENDPOINT  = "/api/v3/order"


class TestnetBroker:
    """
    Real execution broker wired to the Binance Spot Testnet.

    Drop-in replacement for backtest.event_driven._Broker: same
    on_order(OrderEvent) → FillEvent signature, no other interface changes.
    LiveEngine works with either broker without modification.
    """

    def __init__(
        self,
        api_key:    str,
        api_secret: str,
        symbol:     str   = "BTCUSDT",
        order_qty:  float = 0.001,
    ) -> None:
        self._api_secret     = api_secret
        self._symbol         = symbol
        self._order_qty      = order_qty
        self._session        = requests.Session()
        self._session.headers.update({"X-MBX-APIKEY": api_key})
        self._time_offset_ms: int = 0  # calibrated by sync_clock()

    # ── Public interface ───────────────────────────────────────────────────────

    def on_order(self, event: OrderEvent) -> FillEvent:
        """
        Execute an order on the Binance testnet.

        Returns a no-op FillEvent immediately if delta is zero.
        Otherwise places a MARKET order and waits for the FILLED status
        before returning — market orders on testnet settle in milliseconds.
        """
        if event.delta == 0.0:
            return FillEvent(timestamp=event.timestamp, delta=0.0, fee=0.0)

        side = "BUY" if event.delta > 0 else "SELL"
        logger.info(
            "%s | Placing %s MARKET %.8f %s on testnet",
            event.timestamp, side, self._order_qty, self._symbol,
        )

        data = self._place_market_order(side, self._order_qty)
        fee  = self._extract_fee(data)

        executed_qty  = float(data["executedQty"])
        quote_qty     = float(data["cummulativeQuoteQty"])
        avg_price     = quote_qty / executed_qty if executed_qty else 0.0
        logger.info(
            "%s | Fill confirmed  avgPrice=%.2f  executedQty=%.8f  fee=%.4f%%",
            event.timestamp, avg_price, executed_qty, fee * 100,
        )

        return FillEvent(
            timestamp=event.timestamp,
            delta=event.delta,
            fee=fee * abs(event.delta),
        )

    def sync_clock(self) -> None:
        """
        Calibrate local clock against Binance server time.

        Call once at startup. Without this, a machine whose clock drifts by
        more than recvWindow (10 s) will have every signed order rejected
        with error -1021 (timestamp outside recvWindow).
        """
        resp = requests.get(f"{_BASE_URL}/api/v3/time", timeout=5)
        resp.raise_for_status()
        server_ms          = int(resp.json()["serverTime"])
        local_ms           = int(time.time() * 1000)
        self._time_offset_ms = server_ms - local_ms
        logger.info("Clock synced with Binance server: offset=%+d ms", self._time_offset_ms)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _ms_timestamp(self) -> int:
        """Current time in milliseconds, corrected for server clock offset."""
        return int(time.time() * 1000) + self._time_offset_ms

    def _place_market_order(self, side: str, qty: float) -> dict:
        """Submit a signed MARKET order and return the JSON response."""
        params: dict = {
            "symbol":      self._symbol,
            "side":        side,
            "type":        "MARKET",
            "quantity":    f"{qty:.8f}",
            "timestamp":   self._ms_timestamp(),
            "recvWindow":  10_000,   # 10-second window; guards against residual drift
        }
        params["signature"] = _sign(params, self._api_secret)

        resp = self._session.post(
            _BASE_URL + _ORDER_ENDPOINT,
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status")
        if status == "PARTIALLY_FILLED":
            logger.warning(
                "Order %s partially filled: executedQty=%s of %s requested. "
                "Proceeding with partial fill — position model remains abstract.",
                data.get("orderId"), data.get("executedQty"), f"{qty:.8f}",
            )
        elif status != "FILLED":
            raise RuntimeError(
                f"Order not filled: status={status!r}  orderId={data.get('orderId')}"
            )
        return data

    @staticmethod
    def _extract_fee(response: dict) -> float:
        """
        Return total commission as a fraction of trade value (0.001 = 10 bps).

        Binance reports one fill entry per matched trade lot. Commissions on
        a BUY are denominated in the base asset (BTC); commissions on a SELL
        are denominated in the quote asset (USDT). We convert BTC commissions
        to USDT using each fill's execution price before dividing.
        """
        total_commission_usdt = 0.0
        total_trade_usdt      = 0.0

        for fill in response.get("fills", []):
            fill_price = float(fill["price"])
            fill_qty   = float(fill["qty"])
            commission = float(fill["commission"])
            asset      = fill["commissionAsset"]

            total_trade_usdt += fill_price * fill_qty

            if asset == "BTC":
                total_commission_usdt += commission * fill_price
            else:
                # USDT, BNB (with discount), or other stablecoins — treat as USDT.
                total_commission_usdt += commission

        if total_trade_usdt == 0.0:
            return 0.0
        return total_commission_usdt / total_trade_usdt


# ── Module-level helpers ───────────────────────────────────────────────────────

def _sign(params: dict, secret: str) -> str:
    """HMAC-SHA256 signature over the URL-encoded parameter string."""
    query_string = urlencode(params)
    return hmac.new(
        secret.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def load_credentials() -> tuple[str, str]:
    """
    Read API key and secret from environment variables.

    Raises ValueError at startup so the error surfaces before the first
    order fires, not mid-session.
    """
    key    = os.environ.get("BINANCE_TESTNET_API_KEY",    "")
    secret = os.environ.get("BINANCE_TESTNET_API_SECRET", "")
    if not key or not secret:
        raise ValueError(
            "Binance testnet credentials not set.\n"
            "Export BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET "
            "before running with --testnet.\n"
            "Get a free key pair at https://testnet.binance.vision/"
        )
    return key, secret
