# -*- coding: utf-8 -*-
"""QUA-7 single-ticker analysis API/UI for the real Quant Analyzer project."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import time
from typing import Deque, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

from analyzer import DataFetcher
from utils.backtesting import run_sma_crossover_backtest
from utils.multi_ticker_analysis import build_multi_ticker_analysis, parse_symbols
from utils.technical_indicators import (
    exponential_moving_average,
    moving_average_convergence_divergence,
    relative_strength_index,
    simple_moving_average,
)


ANALYSIS_P95_TARGET_MS = 300.0
COMPARE_P95_TARGET_MS = 2000.0
BACKTEST_P95_TARGET_MS = 2000.0


class LatencyTracker:
    """Stores recent latency samples and calculates percentiles."""

    def __init__(self, max_samples: int = 500):
        self._samples: Deque[float] = deque(maxlen=max_samples)

    def record(self, latency_ms: float) -> None:
        self._samples.append(float(latency_ms))

    def percentile(self, percentile: float) -> Optional[float]:
        if not self._samples:
            return None
        ordered = sorted(self._samples)
        index = max(0, min(len(ordered) - 1, int((percentile / 100.0) * (len(ordered) - 1))))
        return ordered[index]


latency_tracker = LatencyTracker()
compare_latency_tracker = LatencyTracker()
backtest_latency_tracker = LatencyTracker()


app = FastAPI(
    title="Quant Analyzer QUA-7 API",
    description="Single-ticker analysis API/UI with parameterized indicators",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_iso_datetime(raw: Optional[str], field_name: str) -> Optional[datetime]:
    if raw is None:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name} format. Use ISO format.") from exc


def _to_utc_epoch(dt: datetime) -> float:
    normalized = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return normalized.timestamp()


def _parse_indicators(raw: Optional[str]) -> List[str]:
    if raw is None or not raw.strip():
        return ["sma", "ema", "rsi", "macd"]
    selected = [item.strip().lower() for item in raw.split(",") if item.strip()]
    allowed = {"sma", "ema", "rsi", "macd"}
    invalid = [item for item in selected if item not in allowed]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unsupported indicator(s): {', '.join(sorted(set(invalid)))}")
    if len(selected) < 3:
        raise HTTPException(status_code=400, detail="At least three indicators are required for this analysis endpoint.")
    return selected


def _series_to_list(series: pd.Series) -> List[Optional[float]]:
    output: List[Optional[float]] = []
    for value in series.tolist():
        output.append(None if pd.isna(value) else float(value))
    return output


def _compute_sma(close: pd.Series, window: int) -> List[Optional[float]]:
    return _series_to_list(simple_moving_average(close, window))


def _compute_ema(close: pd.Series, window: int) -> List[Optional[float]]:
    return _series_to_list(exponential_moving_average(close, window))


def _compute_rsi(close: pd.Series, period: int) -> List[Optional[float]]:
    return _series_to_list(relative_strength_index(close, period))


def _compute_macd(close: pd.Series, fast: int, slow: int, signal: int) -> Dict[str, List[Optional[float]]]:
    macd_result = moving_average_convergence_divergence(close, fast, slow, signal)
    return {
        "macd": _series_to_list(macd_result.macd),
        "signal": _series_to_list(macd_result.signal),
        "histogram": _series_to_list(macd_result.histogram),
    }


def _frame_to_payload(df: pd.DataFrame) -> List[Dict[str, float | str]]:
    payload: List[Dict[str, float | str]] = []
    for index, row in df.iterrows():
        payload.append(
            {
                "timestamp": pd.Timestamp(index).to_pydatetime().isoformat(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
            }
        )
    return payload


def load_ticker_frame(ticker: str, period: str) -> pd.DataFrame:
    fetcher = DataFetcher(ticker=ticker.upper(), period=period)
    frame, _info = fetcher._fetch_ohlcv(ticker.upper(), reuse_primary=False)
    if frame is None or frame.empty:
        raise HTTPException(status_code=404, detail=f"No OHLCV data available for {ticker.upper()}")
    return frame.sort_index()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/analysis/{ticker}")
async def analyze_single_ticker(
    ticker: str,
    period: str = Query("10y", description="Fetch period for the backing OHLCV pull"),
    limit: int = Query(180, ge=20, le=5000, description="Number of candles to include after filtering"),
    since: Optional[str] = Query(None, description="Start time in ISO format"),
    until: Optional[str] = Query(None, description="End time in ISO format"),
    indicators: Optional[str] = Query("sma,ema,rsi,macd", description="Comma-separated list from sma, ema, rsi, macd"),
    sma_window: int = Query(20, ge=1, le=500),
    ema_window: int = Query(20, ge=1, le=500),
    rsi_period: int = Query(14, ge=1, le=500),
    macd_fast: int = Query(12, ge=1, le=500),
    macd_slow: int = Query(26, ge=2, le=500),
    macd_signal: int = Query(9, ge=1, le=500),
):
    started = time.perf_counter()
    normalized_ticker = ticker.upper()
    since_dt = _parse_iso_datetime(since, "since")
    until_dt = _parse_iso_datetime(until, "until")
    if since_dt and until_dt and since_dt > until_dt:
        raise HTTPException(status_code=400, detail="since must be less than or equal to until")
    selected_indicators = _parse_indicators(indicators)
    if "macd" in selected_indicators and macd_fast >= macd_slow:
        raise HTTPException(status_code=400, detail="macd_fast must be smaller than macd_slow")

    try:
        frame = load_ticker_frame(normalized_ticker, period)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if since_dt is not None:
        frame = frame[frame.index.map(lambda value: _to_utc_epoch(pd.Timestamp(value).to_pydatetime())) >= _to_utc_epoch(since_dt)]
    if until_dt is not None:
        frame = frame[frame.index.map(lambda value: _to_utc_epoch(pd.Timestamp(value).to_pydatetime())) <= _to_utc_epoch(until_dt)]
    frame = frame.tail(limit)

    close = frame["Close"]
    indicator_payload: Dict[str, object] = {}
    try:
        if "sma" in selected_indicators:
            indicator_payload["sma"] = _compute_sma(close, sma_window)
        if "ema" in selected_indicators:
            indicator_payload["ema"] = _compute_ema(close, ema_window)
        if "rsi" in selected_indicators:
            indicator_payload["rsi"] = _compute_rsi(close, rsi_period)
        if "macd" in selected_indicators:
            indicator_payload["macd"] = _compute_macd(close, macd_fast, macd_slow, macd_signal)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Indicator computation failed: {exc}") from exc

    duration_ms = (time.perf_counter() - started) * 1000.0
    latency_tracker.record(duration_ms)
    p95_ms = latency_tracker.percentile(95)

    ohlcv_payload = _frame_to_payload(frame)
    return {
        "ticker": normalized_ticker,
        "count": len(ohlcv_payload),
        "requested_indicators": selected_indicators,
        "parameters": {
            "ticker": normalized_ticker,
            "period": period,
            "limit": limit,
            "since": since_dt.isoformat() if since_dt else None,
            "until": until_dt.isoformat() if until_dt else None,
            "indicators": selected_indicators,
            "sma_window": sma_window,
            "ema_window": ema_window,
            "rsi_period": rsi_period,
            "macd_fast": macd_fast,
            "macd_slow": macd_slow,
            "macd_signal": macd_signal,
        },
        "range": {
            "returned_start": ohlcv_payload[0]["timestamp"] if ohlcv_payload else None,
            "returned_end": ohlcv_payload[-1]["timestamp"] if ohlcv_payload else None,
        },
        "data": {
            "ohlcv": ohlcv_payload,
            "indicators": indicator_payload,
        },
        "latency": {
            "request_ms": round(duration_ms, 2),
            "analysis_endpoint_p95_ms": round(p95_ms, 2) if p95_ms is not None else None,
            "p95_target_ms": ANALYSIS_P95_TARGET_MS,
            "p95_target_met": (p95_ms <= ANALYSIS_P95_TARGET_MS) if p95_ms is not None else None,
        },
    }


@app.get("/compare")
async def compare_multiple_tickers(
    symbols: str = Query(..., description="Comma-separated ticker list"),
    period: str = Query("10y", description="Fetch period for the backing OHLCV pull"),
    limit: int = Query(180, ge=20, le=5000, description="Number of candles to retain per symbol after filtering"),
    since: Optional[str] = Query(None, description="Inclusive start time in ISO format"),
    until: Optional[str] = Query(None, description="Inclusive end time in ISO format"),
):
    started = time.perf_counter()
    since_dt = _parse_iso_datetime(since, "since")
    until_dt = _parse_iso_datetime(until, "until")
    if since_dt and until_dt and since_dt > until_dt:
        raise HTTPException(status_code=400, detail="since must be less than or equal to until")

    try:
        normalized_symbols = parse_symbols(symbols)
        payload = build_multi_ticker_analysis(
            symbols=normalized_symbols,
            period=period,
            limit=limit,
            since=since_dt,
            until=until_dt,
            load_frame=load_ticker_frame,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    duration_ms = (time.perf_counter() - started) * 1000.0
    compare_latency_tracker.record(duration_ms)
    p95_ms = compare_latency_tracker.percentile(95)
    payload["parameters"] = {
        "symbols": normalized_symbols,
        "period": period,
        "limit": limit,
        "since": since_dt.isoformat() if since_dt else None,
        "until": until_dt.isoformat() if until_dt else None,
    }
    payload["latency"] = {
        "request_ms": round(duration_ms, 2),
        "compare_endpoint_p95_ms": round(p95_ms, 2) if p95_ms is not None else None,
        "p95_target_ms": COMPARE_P95_TARGET_MS,
        "p95_target_met": (p95_ms <= COMPARE_P95_TARGET_MS) if p95_ms is not None else None,
    }
    return payload


@app.get("/backtest/{ticker}")
async def backtest_single_ticker(
    ticker: str,
    period: str = Query("10y", description="Fetch period for the backing OHLCV pull"),
    limit: int = Query(180, ge=20, le=5000, description="Number of candles to retain after filtering"),
    since: Optional[str] = Query(None, description="Inclusive start time in ISO format"),
    until: Optional[str] = Query(None, description="Inclusive end time in ISO format"),
    fast_window: int = Query(20, ge=1, le=500),
    slow_window: int = Query(50, ge=2, le=500),
    initial_cash: float = Query(10000.0, gt=0.0, description="Initial capital for the deterministic equity curve"),
):
    started = time.perf_counter()
    normalized_ticker = ticker.upper()
    since_dt = _parse_iso_datetime(since, "since")
    until_dt = _parse_iso_datetime(until, "until")
    if since_dt and until_dt and since_dt > until_dt:
        raise HTTPException(status_code=400, detail="since must be less than or equal to until")

    try:
        frame = load_ticker_frame(normalized_ticker, period)
        if since_dt is not None:
            frame = frame[frame.index >= pd.Timestamp(since_dt)]
        if until_dt is not None:
            frame = frame[frame.index <= pd.Timestamp(until_dt)]
        frame = frame.tail(limit)
        result = run_sma_crossover_backtest(
            frame,
            fast_window=fast_window,
            slow_window=slow_window,
            initial_cash=initial_cash,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    duration_ms = (time.perf_counter() - started) * 1000.0
    backtest_latency_tracker.record(duration_ms)
    p95_ms = backtest_latency_tracker.percentile(95)
    payload = result.to_dict()
    payload["ticker"] = normalized_ticker
    payload["parameters"] = {
        "ticker": normalized_ticker,
        "period": period,
        "limit": limit,
        "since": since_dt.isoformat() if since_dt else None,
        "until": until_dt.isoformat() if until_dt else None,
        "fast_window": fast_window,
        "slow_window": slow_window,
        "initial_cash": initial_cash,
    }
    payload["range"] = {
        "returned_start": frame.index[0].to_pydatetime().isoformat() if not frame.empty else None,
        "returned_end": frame.index[-1].to_pydatetime().isoformat() if not frame.empty else None,
        "observations": int(len(frame)),
    }
    payload["latency"] = {
        "request_ms": round(duration_ms, 2),
        "backtest_endpoint_p95_ms": round(p95_ms, 2) if p95_ms is not None else None,
        "p95_target_ms": BACKTEST_P95_TARGET_MS,
        "p95_target_met": (p95_ms <= BACKTEST_P95_TARGET_MS) if p95_ms is not None else None,
    }
    return payload


@app.get("/analysis-ui", response_class=HTMLResponse)
async def analysis_ui() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Quant Analyzer QUA-7</title>
  <style>
    :root {
      --bg: #0b1220;
      --bg2: #14304a;
      --card: #f4f7fb;
      --ink: #102030;
      --muted: #617181;
      --accent: #ff8a00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: radial-gradient(circle at top left, #27537a 0%, transparent 36%),
                  linear-gradient(135deg, var(--bg), var(--bg2));
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--ink);
      padding: 24px;
    }
    main {
      max-width: 980px;
      margin: 0 auto;
      background: var(--card);
      border-radius: 20px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
      padding: 20px;
    }
    h1 { margin: 0 0 12px; }
    p { color: var(--muted); }
    form {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      align-items: end;
    }
    label {
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 12px;
      color: var(--muted);
    }
    input {
      border: 1px solid #ccd6df;
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 14px;
    }
    button {
      border: 0;
      border-radius: 10px;
      padding: 11px 14px;
      background: linear-gradient(120deg, var(--accent), #ffc24a);
      font-weight: 700;
      cursor: pointer;
    }
    pre {
      margin-top: 18px;
      background: #0f1c2a;
      color: #eaf1f8;
      border-radius: 12px;
      padding: 16px;
      overflow: auto;
      min-height: 280px;
    }
  </style>
</head>
<body>
  <main>
    <h1>Single-Ticker Analysis</h1>
    <p>Use the existing Quant Analyzer market-data path and request parameterized indicators in one call.</p>
    <form id="analysis-form">
      <label>Ticker<input name="ticker" value="AAPL" required /></label>
      <label>Period<input name="period" value="10y" required /></label>
      <label>Limit<input name="limit" type="number" value="180" min="20" max="5000" /></label>
      <label>Indicators<input name="indicators" value="sma,ema,rsi,macd" /></label>
      <label>SMA<input name="sma_window" type="number" value="20" min="1" /></label>
      <label>EMA<input name="ema_window" type="number" value="20" min="1" /></label>
      <label>RSI<input name="rsi_period" type="number" value="14" min="1" /></label>
      <label>MACD Fast<input name="macd_fast" type="number" value="12" min="1" /></label>
      <label>MACD Slow<input name="macd_slow" type="number" value="26" min="2" /></label>
      <label>MACD Signal<input name="macd_signal" type="number" value="9" min="1" /></label>
      <button type="submit">Run Analysis</button>
    </form>
    <pre id="result">Run a query to inspect the JSON response.</pre>
  </main>
  <script>
    const form = document.getElementById("analysis-form");
    const result = document.getElementById("result");

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const data = new FormData(form);
      const ticker = data.get("ticker");
      const params = new URLSearchParams();
      for (const [key, value] of data.entries()) {
        if (key === "ticker" || String(value).trim() === "") continue;
        params.set(key, value);
      }
      const url = `/analysis/${encodeURIComponent(ticker)}?${params.toString()}`;
      result.textContent = `Fetching ${url} ...`;
      try {
        const response = await fetch(url);
        const payload = await response.json();
        result.textContent = JSON.stringify(payload, null, 2);
      } catch (error) {
        result.textContent = String(error);
      }
    });
  </script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
