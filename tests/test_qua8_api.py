from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from fastapi.testclient import TestClient

import qua7_api


def _frame(offset: float) -> pd.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    index = pd.date_range(start=start, periods=120, freq="D")
    close = [100.0 + offset + (idx * (1.0 + offset / 50.0)) + ((idx % 5) * 0.25) for idx in range(120)]
    return pd.DataFrame(
        {
            "Open": [value - 0.5 for value in close],
            "High": [value + 0.75 for value in close],
            "Low": [value - 1.0 for value in close],
            "Close": close,
            "Volume": [1000.0 + idx + (offset * 10.0) for idx in range(120)],
        },
        index=index,
    )


client = TestClient(qua7_api.app)


def test_compare_endpoint_supports_ten_symbols_and_returns_correlation(monkeypatch):
    frames = {f"TICK{idx}": _frame(float(idx)) for idx in range(10)}
    monkeypatch.setattr(qua7_api, "load_ticker_frame", lambda ticker, period: frames[ticker])

    response = client.get(
        "/compare",
        params={
            "symbols": ",".join(frames.keys()),
            "limit": 90,
            "period": "2y",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 10
    assert payload["parameters"]["symbols"] == list(frames.keys())
    assert len(payload["data"]["ticker_metrics"]) == 10
    assert set(payload["data"]["correlation"].keys()) == {"pearson", "spearman"}
    pearson = payload["data"]["correlation"]["pearson"]
    assert pearson["TICK0"]["TICK0"] == 1.0
    assert set(pearson.keys()) == set(frames.keys())
    assert payload["range"]["aligned_observations"] == 90
    assert payload["latency"]["p95_target_ms"] == 2000.0


def test_compare_endpoint_rejects_single_symbol():
    response = client.get("/compare", params={"symbols": "AAPL"})

    assert response.status_code == 400
    assert "at least two distinct symbols" in response.json()["detail"]
