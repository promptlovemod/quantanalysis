from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from fastapi.testclient import TestClient

import qua7_api


def _frame() -> pd.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    index = pd.date_range(start=start, periods=30, freq="D")
    close = [
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,
        103.0,
        102.0,
        101.0,
        100.0,
        99.0,
        98.0,
        99.0,
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,
        105.0,
        104.0,
        103.0,
        102.0,
        101.0,
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,
        105.0,
        106.0,
        107.0,
    ]
    return pd.DataFrame(
        {
            "Open": [value - 0.25 for value in close],
            "High": [value + 0.75 for value in close],
            "Low": [value - 1.0 for value in close],
            "Close": close,
            "Volume": [1000.0 + idx for idx in range(len(close))],
        },
        index=index,
    )


client = TestClient(qua7_api.app)


def test_backtest_endpoint_is_reproducible_and_returns_trade_log(monkeypatch):
    monkeypatch.setattr(qua7_api, "load_ticker_frame", lambda ticker, period: _frame())

    response_a = client.get("/backtest/AAPL", params={"fast_window": 3, "slow_window": 5, "limit": 30})
    response_b = client.get("/backtest/AAPL", params={"fast_window": 3, "slow_window": 5, "limit": 30})

    assert response_a.status_code == 200
    assert response_b.status_code == 200
    payload_a = response_a.json()
    payload_b = response_b.json()
    assert payload_a["summary"] == payload_b["summary"]
    assert payload_a["trade_log"] == payload_b["trade_log"]
    assert payload_a["summary"]["engine_version"] == "qua9-backtester/v1"
    assert payload_a["summary"]["strategy"] == "sma_crossover"
    assert payload_a["summary"]["trade_count"] == len(payload_a["trade_log"])
    assert any(trade["action"] == "BUY" for trade in payload_a["trade_log"])
    assert any(trade["action"] == "SELL" for trade in payload_a["trade_log"])
    assert payload_a["latency"]["p95_target_ms"] == 2000.0


def test_backtest_endpoint_rejects_invalid_window_order(monkeypatch):
    monkeypatch.setattr(qua7_api, "load_ticker_frame", lambda ticker, period: _frame())

    response = client.get("/backtest/AAPL", params={"fast_window": 5, "slow_window": 5})

    assert response.status_code == 400
    assert "fast_window must be smaller than slow_window" in response.json()["detail"]
