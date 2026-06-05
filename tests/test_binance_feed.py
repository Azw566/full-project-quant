"""
tests/test_binance_feed.py — Unit tests for feed/binance.py parsers.

No network access required — all tests operate on captured JSON fixtures.
Tests verify:
    • _parse_rest_kline uses close_time (row[6]), not open_time (row[0])
    • _parse_ws_kline uses close_time (k["T"]), not open_time (k["t"])
    • All numeric fields are float
    • Timestamp is tz-aware UTC
    • Both parsers produce the same schema
"""

from __future__ import annotations

import pandas as pd
import pytest

from feed.binance import _parse_rest_kline, _parse_ws_kline


# ── Fixtures ───────────────────────────────────────────────────────────────────

# Minimal Binance REST kline row (only the fields we use).
# Full schema: [open_time, open, high, low, close, volume, close_time, ...]
_REST_ROW = [
    1638747600000,   # [0] open_time  (2021-12-05 23:00 UTC)
    "57000.00",      # [1] open
    "57500.00",      # [2] high
    "56800.00",      # [3] low
    "57200.00",      # [4] close
    "123.456",       # [5] volume
    1638751199999,   # [6] close_time (2021-12-05 23:59:59.999 UTC)
    "7060000.00",    # [7] quote asset volume (ignored)
]

# Minimal Binance WebSocket kline object (the 'k' field of a kline event).
_WS_KLINE = {
    "t": 1638747600000,   # open_time  (same as above)
    "T": 1638751199999,   # close_time (same as above)
    "o": "57000.00",
    "h": "57500.00",
    "l": "56800.00",
    "c": "57200.00",
    "v": "123.456",
    "x": True,            # is_closed
}

_EXPECTED_CLOSE_TS = pd.Timestamp(1638751199999, unit="ms", tz="UTC")


# ── REST parser ────────────────────────────────────────────────────────────────

def test_rest_timestamp_is_close_time():
    bar = _parse_rest_kline(_REST_ROW)
    assert bar["timestamp"] == _EXPECTED_CLOSE_TS, (
        "REST parser must use close_time (row[6]), not open_time (row[0])"
    )


def test_rest_timestamp_is_utc():
    bar = _parse_rest_kline(_REST_ROW)
    assert bar["timestamp"].tzinfo is not None
    assert str(bar["timestamp"].tzinfo) == "UTC"


def test_rest_ohlcv_values():
    bar = _parse_rest_kline(_REST_ROW)
    assert bar["open"]   == pytest.approx(57000.00)
    assert bar["high"]   == pytest.approx(57500.00)
    assert bar["low"]    == pytest.approx(56800.00)
    assert bar["close"]  == pytest.approx(57200.00)
    assert bar["volume"] == pytest.approx(123.456)


def test_rest_all_numeric_fields_are_float():
    bar = _parse_rest_kline(_REST_ROW)
    for field in ("open", "high", "low", "close", "volume"):
        assert isinstance(bar[field], float), f"{field} should be float"


# ── WebSocket parser ───────────────────────────────────────────────────────────

def test_ws_timestamp_is_close_time():
    bar = _parse_ws_kline(_WS_KLINE)
    assert bar["timestamp"] == _EXPECTED_CLOSE_TS, (
        "WS parser must use close_time (k['T']), not open_time (k['t'])"
    )


def test_ws_timestamp_is_utc():
    bar = _parse_ws_kline(_WS_KLINE)
    assert bar["timestamp"].tzinfo is not None
    assert str(bar["timestamp"].tzinfo) == "UTC"


def test_ws_ohlcv_values():
    bar = _parse_ws_kline(_WS_KLINE)
    assert bar["open"]   == pytest.approx(57000.00)
    assert bar["high"]   == pytest.approx(57500.00)
    assert bar["low"]    == pytest.approx(56800.00)
    assert bar["close"]  == pytest.approx(57200.00)
    assert bar["volume"] == pytest.approx(123.456)


def test_ws_all_numeric_fields_are_float():
    bar = _parse_ws_kline(_WS_KLINE)
    for field in ("open", "high", "low", "close", "volume"):
        assert isinstance(bar[field], float), f"{field} should be float"


# ── Schema parity ──────────────────────────────────────────────────────────────

def test_rest_and_ws_produce_same_schema():
    """Both parsers must return the same keys so LiveEngine can handle either."""
    rest_bar = _parse_rest_kline(_REST_ROW)
    ws_bar   = _parse_ws_kline(_WS_KLINE)
    assert set(rest_bar.keys()) == set(ws_bar.keys())


def test_rest_and_ws_produce_same_values_for_same_candle():
    """Given the same candle, both parsers must produce identical numeric values."""
    rest_bar = _parse_rest_kline(_REST_ROW)
    ws_bar   = _parse_ws_kline(_WS_KLINE)
    assert rest_bar["timestamp"] == ws_bar["timestamp"]
    for field in ("open", "high", "low", "close", "volume"):
        assert rest_bar[field] == pytest.approx(ws_bar[field])
