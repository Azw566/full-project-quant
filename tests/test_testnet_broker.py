"""
tests/test_testnet_broker.py — Unit tests for the Binance testnet broker.

All HTTP calls are mocked — no network access required.
Tests verify:
    • delta=0 → no HTTP call, FillEvent fee=0
    • delta=1.0 / -1.0 → correct side, fee computed from fills
    • PARTIALLY_FILLED → warning, not RuntimeError
    • non-FILLED/PARTIALLY_FILLED status → RuntimeError
    • _extract_fee with BTC commissions (converted via fill price)
    • _extract_fee with USDT commissions (used directly)
    • _extract_fee with multiple fills
    • _extract_fee with no fills → 0.0
    • _sign produces deterministic HMAC-SHA256
    • sync_clock sets _time_offset_ms from server response
    • _ms_timestamp uses the calibrated offset
    • load_credentials raises ValueError when env vars absent
"""

from __future__ import annotations

import hashlib
import hmac
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backtest.event_driven import FillEvent, OrderEvent
from execution.binance_broker import (
    TestnetBroker,
    _sign,
    load_credentials,
)


# ── Test fixtures ──────────────────────────────────────────────────────────────

_TS = pd.Timestamp("2024-01-15 14:00:00", tz="UTC")

_BUY_RESPONSE = {
    "symbol":               "BTCUSDT",
    "orderId":              1001,
    "status":               "FILLED",
    "side":                 "BUY",
    "executedQty":          "0.00100000",
    "cummulativeQuoteQty":  "42.00000000",   # 0.001 BTC * 42000 USDT
    "fills": [
        {
            "price":            "42000.00",
            "qty":              "0.00100000",
            "commission":       "0.00000100",  # 0.001 BTC * 10 bps = 0.000001 BTC
            "commissionAsset":  "BTC",
        }
    ],
}

_SELL_RESPONSE = {
    "symbol":               "BTCUSDT",
    "orderId":              1002,
    "status":               "FILLED",
    "side":                 "SELL",
    "executedQty":          "0.00100000",
    "cummulativeQuoteQty":  "42.00000000",
    "fills": [
        {
            "price":            "42000.00",
            "qty":              "0.00100000",
            "commission":       "0.04200000",  # 0.042 USDT = 10 bps of 42.0
            "commissionAsset":  "USDT",
        }
    ],
}

_PARTIAL_RESPONSE = {
    "symbol":               "BTCUSDT",
    "orderId":              1003,
    "status":               "PARTIALLY_FILLED",
    "executedQty":          "0.00050000",
    "cummulativeQuoteQty":  "21.00000000",   # 0.0005 BTC × 42000
    "fills": [
        {
            "price":           "42000.00",
            "qty":             "0.00050000",
            "commission":      "0.00000050",
            "commissionAsset": "BTC",
        }
    ],
}

_REJECTED_RESPONSE = {
    "symbol":  "BTCUSDT",
    "orderId": 1004,
    "status":  "REJECTED",
}


def _make_broker() -> TestnetBroker:
    return TestnetBroker(
        api_key="testkey",
        api_secret="testsecret",
        symbol="BTCUSDT",
        order_qty=0.001,
    )


def _mock_post(response_json: dict):
    """Return a mock that satisfies resp.raise_for_status() + resp.json()."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = response_json
    mock_resp.raise_for_status.return_value = None
    return mock_resp


# ── No-op on zero delta ────────────────────────────────────────────────────────

def test_zero_delta_no_http_call():
    broker = _make_broker()
    event  = OrderEvent(timestamp=_TS, delta=0.0)
    with patch.object(broker._session, "post") as mock_post:
        fill = broker.on_order(event)
    mock_post.assert_not_called()
    assert fill.delta == 0.0
    assert fill.fee   == 0.0
    assert fill.timestamp == _TS


# ── BUY order ─────────────────────────────────────────────────────────────────

def test_buy_order_places_correct_side():
    broker = _make_broker()
    event  = OrderEvent(timestamp=_TS, delta=1.0)
    with patch.object(broker._session, "post", return_value=_mock_post(_BUY_RESPONSE)) as mock_post:
        broker.on_order(event)
    params = mock_post.call_args.kwargs["params"]
    assert params["side"] == "BUY"
    assert float(params["quantity"]) == pytest.approx(0.001)


def test_buy_order_fill_event_delta():
    broker = _make_broker()
    event  = OrderEvent(timestamp=_TS, delta=1.0)
    with patch.object(broker._session, "post", return_value=_mock_post(_BUY_RESPONSE)):
        fill = broker.on_order(event)
    assert fill.delta     == 1.0
    assert fill.timestamp == _TS


def test_buy_order_fee_from_btc_commission():
    # BTC commission 0.000001 BTC at 42000 USDT = 0.042 USDT commission
    # trade value = 0.001 BTC * 42000 = 42.0 USDT
    # fee fraction = 0.042 / 42.0 = 0.001 (10 bps) * abs(delta=1.0) = 0.001
    broker = _make_broker()
    event  = OrderEvent(timestamp=_TS, delta=1.0)
    with patch.object(broker._session, "post", return_value=_mock_post(_BUY_RESPONSE)):
        fill = broker.on_order(event)
    assert fill.fee == pytest.approx(0.001, rel=1e-9)


# ── SELL order ────────────────────────────────────────────────────────────────

def test_sell_order_places_correct_side():
    broker = _make_broker()
    event  = OrderEvent(timestamp=_TS, delta=-1.0)
    with patch.object(broker._session, "post", return_value=_mock_post(_SELL_RESPONSE)) as mock_post:
        broker.on_order(event)
    params = mock_post.call_args.kwargs["params"]
    assert params["side"] == "SELL"


def test_sell_order_fee_from_usdt_commission():
    # USDT commission 0.042 USDT, trade value 42.0 USDT → fee = 0.001 (10 bps)
    # abs(delta=-1.0) = 1.0, so FillEvent.fee = 0.001 * 1.0 = 0.001
    broker = _make_broker()
    event  = OrderEvent(timestamp=_TS, delta=-1.0)
    with patch.object(broker._session, "post", return_value=_mock_post(_SELL_RESPONSE)):
        fill = broker.on_order(event)
    assert fill.fee   == pytest.approx(0.001, rel=1e-9)
    assert fill.delta == -1.0


# ── Partial fill and rejected status ──────────────────────────────────────────

def test_partial_fill_logs_warning_not_raises(caplog):
    """PARTIALLY_FILLED must log a warning and return a FillEvent — not crash."""
    import logging
    broker = _make_broker()
    event  = OrderEvent(timestamp=_TS, delta=1.0)
    with patch.object(broker._session, "post", return_value=_mock_post(_PARTIAL_RESPONSE)):
        with caplog.at_level(logging.WARNING):
            fill = broker.on_order(event)
    assert fill is not None
    assert "partially" in caplog.text.lower()


def test_rejected_order_raises():
    broker = _make_broker()
    event  = OrderEvent(timestamp=_TS, delta=1.0)
    with patch.object(broker._session, "post", return_value=_mock_post(_REJECTED_RESPONSE)):
        with pytest.raises(RuntimeError, match="REJECTED"):
            broker.on_order(event)


# ── _extract_fee unit tests ────────────────────────────────────────────────────

def test_extract_fee_btc_commission():
    response = {
        "fills": [
            {
                "price":           "40000.00",
                "qty":             "0.00100000",
                "commission":      "0.00000100",  # 0.001 BTC * 40000 = 0.04 USDT
                "commissionAsset": "BTC",
            }
        ]
    }
    # trade = 40.0 USDT, commission = 0.04 USDT → 0.04/40.0 = 0.001
    fee = TestnetBroker._extract_fee(response)
    assert fee == pytest.approx(0.001, rel=1e-9)


def test_extract_fee_usdt_commission():
    response = {
        "fills": [
            {
                "price":           "40000.00",
                "qty":             "0.00100000",
                "commission":      "0.04000000",
                "commissionAsset": "USDT",
            }
        ]
    }
    fee = TestnetBroker._extract_fee(response)
    assert fee == pytest.approx(0.001, rel=1e-9)


def test_extract_fee_multiple_fills():
    response = {
        "fills": [
            {
                "price":           "40000.00",
                "qty":             "0.00060000",
                "commission":      "0.00000060",
                "commissionAsset": "BTC",
            },
            {
                "price":           "40010.00",
                "qty":             "0.00040000",
                "commission":      "0.00000040",
                "commissionAsset": "BTC",
            },
        ]
    }
    # fill1: trade=24.0 USDT, commission=0.024 USDT
    # fill2: trade=16.004 USDT, commission=0.016004 USDT
    # total: trade=40.004 USDT, commission=0.040004 USDT
    # fee = 0.040004 / 40.004 ≈ 0.001 (very close to 10 bps)
    fee = TestnetBroker._extract_fee(response)
    assert fee == pytest.approx(0.040004 / 40.004, rel=1e-9)


def test_extract_fee_no_fills():
    fee = TestnetBroker._extract_fee({"fills": []})
    assert fee == 0.0


def test_extract_fee_missing_fills_key():
    fee = TestnetBroker._extract_fee({})
    assert fee == 0.0


# ── _sign ─────────────────────────────────────────────────────────────────────

def test_sign_deterministic():
    params = {"symbol": "BTCUSDT", "side": "BUY", "timestamp": 1700000000000}
    assert _sign(params, "secret") == _sign(params, "secret")


def test_sign_hmac_sha256():
    params = {"symbol": "BTCUSDT", "side": "BUY", "timestamp": 1700000000000}
    from urllib.parse import urlencode
    expected = hmac.new(
        b"secret",
        urlencode(params).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    assert _sign(params, "secret") == expected


def test_sign_different_secrets_differ():
    params = {"symbol": "BTCUSDT", "timestamp": 1700000000000}
    assert _sign(params, "secret_a") != _sign(params, "secret_b")


# ── sync_clock / _ms_timestamp ────────────────────────────────────────────────

def test_sync_clock_sets_offset():
    broker    = _make_broker()
    fake_server_ms = int(time.time() * 1000) + 500   # server is 500ms ahead
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"serverTime": fake_server_ms}
    mock_resp.raise_for_status.return_value = None
    with patch("requests.get", return_value=mock_resp):
        broker.sync_clock()
    assert abs(broker._time_offset_ms - 500) < 100   # within 100ms tolerance


def test_ms_timestamp_uses_offset():
    broker = _make_broker()
    broker._time_offset_ms = 1_000   # 1 second ahead
    ts = broker._ms_timestamp()
    assert abs(ts - (int(time.time() * 1000) + 1_000)) < 100


def test_ms_timestamp_zero_offset_by_default():
    broker = _make_broker()
    assert broker._time_offset_ms == 0
    ts = broker._ms_timestamp()
    assert abs(ts - int(time.time() * 1000)) < 100


# ── load_credentials ──────────────────────────────────────────────────────────

def test_load_credentials_missing_both(monkeypatch):
    monkeypatch.delenv("BINANCE_TESTNET_API_KEY",    raising=False)
    monkeypatch.delenv("BINANCE_TESTNET_API_SECRET", raising=False)
    with pytest.raises(ValueError, match="BINANCE_TESTNET_API_KEY"):
        load_credentials()


def test_load_credentials_missing_secret(monkeypatch):
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY",    "somekey")
    monkeypatch.delenv("BINANCE_TESTNET_API_SECRET", raising=False)
    with pytest.raises(ValueError):
        load_credentials()


def test_load_credentials_missing_key(monkeypatch):
    monkeypatch.delenv("BINANCE_TESTNET_API_KEY",    raising=False)
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "somesecret")
    with pytest.raises(ValueError):
        load_credentials()


def test_load_credentials_both_present(monkeypatch):
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY",    "mykey")
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "mysecret")
    key, secret = load_credentials()
    assert key    == "mykey"
    assert secret == "mysecret"
