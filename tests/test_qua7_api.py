from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
from fastapi.testclient import TestClient

import qua7_api


def _frame() -> pd.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    index = pd.date_range(start=start, periods=120, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0 + idx for idx in range(120)],
            "High": [101.0 + idx for idx in range(120)],
            "Low": [99.0 + idx for idx in range(120)],
            "Close": [100.5 + idx for idx in range(120)],
            "Volume": [1000.0 + idx for idx in range(120)],
        },
        index=index,
    )


client = TestClient(qua7_api.app)


def test_analysis_endpoint_returns_traceable_indicator_payload(monkeypatch):
    monkeypatch.setattr(qua7_api, "load_ticker_frame", lambda ticker, period: _frame())

    response = client.get(
        "/analysis/AAPL",
        params={
            "limit": 80,
            "indicators": "sma,ema,rsi,macd",
            "sma_window": 10,
            "ema_window": 12,
            "rsi_period": 14,
            "macd_fast": 10,
            "macd_slow": 21,
            "macd_signal": 7,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "AAPL"
    assert payload["count"] == 80
    assert payload["parameters"]["indicators"] == ["sma", "ema", "rsi", "macd"]
    assert set(payload["data"]["indicators"].keys()) == {"sma", "ema", "rsi", "macd"}
    assert payload["latency"]["p95_target_ms"] == 300.0
    assert payload["latency"]["p95_target_met"] in (True, False)


def test_analysis_endpoint_honors_since_until(monkeypatch):
    frame = _frame()
    monkeypatch.setattr(qua7_api, "load_ticker_frame", lambda ticker, period: frame)

    since = (frame.index[20].to_pydatetime() + timedelta()).isoformat()
    until = (frame.index[40].to_pydatetime() + timedelta()).isoformat()

    response = client.get(
        "/analysis/AAPL",
        params={"since": since, "until": until, "indicators": "sma,ema,rsi"},
    )

    assert response.status_code == 200
    payload = response.json()
    timestamps = [datetime.fromisoformat(row["timestamp"]) for row in payload["data"]["ohlcv"]]
    assert all(ts >= datetime.fromisoformat(since) for ts in timestamps)
    assert all(ts <= datetime.fromisoformat(until) for ts in timestamps)


def test_analysis_endpoint_requires_three_indicators(monkeypatch):
    monkeypatch.setattr(qua7_api, "load_ticker_frame", lambda ticker, period: _frame())
    response = client.get("/analysis/AAPL", params={"indicators": "sma,ema"})
    assert response.status_code == 400
    assert "At least three indicators" in response.json()["detail"]


def test_analysis_ui_route_exists():
    response = client.get("/analysis-ui")
    assert response.status_code == 200
    assert "Single-Ticker Analysis" in response.text
