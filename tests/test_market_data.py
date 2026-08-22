from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from utils.market_data import (
    BatchFetchAudit,
    MarketDataService,
    RateLimiter,
    RetryConfig,
    format_batch_audit_report,
    load_symbol_universe,
)


def _frame(rows: int = 600) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0 + idx for idx in range(rows)],
            "High": [101.0 + idx for idx in range(rows)],
            "Low": [99.0 + idx for idx in range(rows)],
            "Close": [100.5 + idx for idx in range(rows)],
            "Volume": [1000.0 + idx for idx in range(rows)],
        },
        index=index,
    )


@dataclass
class StubProvider:
    name: str
    responses: list[object]

    def available(self) -> bool:
        return True

    def fetch(self, symbol: str, period: str) -> pd.DataFrame | None:
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_market_data_service_falls_back_after_retry():
    saved: list[tuple[str, pd.DataFrame]] = []
    primary = StubProvider("Tiingo", [RuntimeError("temporary outage"), None])
    secondary = StubProvider("yfinance", [_frame()])
    sleep_calls: list[float] = []
    service = MarketDataService(
        providers=[primary, secondary],
        cache_inspect=lambda symbol: (None, "miss", {}),
        cache_save=lambda symbol, frame: saved.append((symbol, frame)),
        retry_config=RetryConfig(max_attempts=2, backoff_seconds=0.25),
        rate_limiters={},
        sleeper=lambda seconds: sleep_calls.append(seconds),
    )

    frame, info = service.fetch_daily_ohlcv("AAPL", "2y")

    assert len(frame) == 600
    assert info["source"] == "yfinance"
    assert info["cache_state"] == "miss"
    assert [attempt["status"] for attempt in info["provider_attempts"]] == ["error", "empty", "success"]
    assert sleep_calls == [0.25]
    assert saved and saved[0][0] == "AAPL"


def test_market_data_service_uses_cache_without_provider_calls():
    cached_frame = _frame(rows=520)
    provider = StubProvider("Tiingo", [RuntimeError("should not be used")])
    service = MarketDataService(
        providers=[provider],
        cache_inspect=lambda symbol: (cached_frame, "hit", {"latest_fetch": "2026-04-18T00:00:00"}),
        cache_save=lambda symbol, frame: None,
    )

    frame, info = service.fetch_daily_ohlcv("MSFT", "2y")

    assert frame.equals(cached_frame)
    assert info["source"] == "cache"
    assert info["provider_attempts"] == []
    assert len(provider.responses) == 1
    assert str(provider.responses[0]) == "should not be used"


def test_rate_limiter_sleeps_until_slot_available():
    timeline = iter([0.0, 0.1, 0.5])
    sleep_calls: list[float] = []
    limiter = RateLimiter(
        min_interval_seconds=0.5,
        monotonic=lambda: next(timeline),
        sleeper=lambda seconds: sleep_calls.append(seconds),
    )

    first_sleep = limiter.acquire()
    second_sleep = limiter.acquire()

    assert first_sleep == 0.0
    assert second_sleep == 0.4
    assert sleep_calls == [0.4]


def test_batch_audit_flags_stale_history_and_failures():
    frames = {
        "AAPL": (_frame(rows=520), {"cache_state": "hit", "source": "cache", "latest_fetch": "2026-04-18T00:00:00"}),
        "MSFT": (_frame(rows=300), {"cache_state": "hit", "source": "Tiingo", "latest_fetch": "2026-04-17T00:00:00"}),
    }
    service = MarketDataService(
        providers=[],
        cache_inspect=lambda symbol: (None, "miss", {}),
        cache_save=lambda symbol, frame: None,
    )

    def _fetch(symbol: str, period: str):
        if symbol == "NVDA":
            raise RuntimeError("provider failure")
        return frames[symbol]

    service.fetch_daily_ohlcv = _fetch  # type: ignore[method-assign]
    audit = service.batch_audit(["AAPL", "MSFT", "NVDA"], period="2y", max_age_hours=12)

    assert audit.total_symbols == 3
    assert audit.succeeded_symbols == 2
    assert audit.insufficient_history_symbols == ["MSFT"]
    assert audit.failed_symbols == ["NVDA"]
    assert audit.provider_usage == {"cache": 1, "Tiingo": 1}
    assert audit.overall_status == "FAIL"


def test_load_symbol_universe_respects_comments_and_limit(tmp_path: Path):
    universe_path = tmp_path / "universe.txt"
    universe_path.write_text("AAPL\n# comment\n msft  \nTSLA # inline comment\nNVDA\n", encoding="utf-8")

    symbols = load_symbol_universe(universe_path, max_symbols=3)

    assert symbols == ["AAPL", "MSFT", "TSLA"]


def test_format_batch_audit_report_renders_status_and_exceptions():
    audit = BatchFetchAudit(
        total_symbols=3,
        succeeded_symbols=2,
        requested_symbols=["AAPL", "MSFT", "NVDA"],
        failed_symbols=["NVDA"],
        insufficient_history_symbols=["MSFT"],
        provider_usage={"cache": 1, "Tiingo": 1},
        period="2y",
        max_age_hours=6,
        min_history_rows=504,
        min_history_span_days=729,
        generated_at="2026-04-18T16:00:00+00:00",
    )

    report = format_batch_audit_report(audit, universe_name="benchmark_universe_full.txt")

    assert "# Market Data Freshness Report" in report
    assert "- Universe: benchmark_universe_full.txt" in report
    assert "- Overall status: FAIL" in report
    assert "- History gate: >=504 rows or >=729 calendar days" in report
    assert "- Failed symbols: NVDA" in report
    assert "- Insufficient history symbols: MSFT" in report


def test_batch_audit_accepts_two_year_calendar_span_even_below_row_floor():
    frame = _frame(rows=501)
    frame.index = pd.date_range("2024-04-18", "2026-04-17", periods=501)
    service = MarketDataService(
        providers=[],
        cache_inspect=lambda symbol: (None, "miss", {}),
        cache_save=lambda symbol, frame: None,
    )

    service.fetch_daily_ohlcv = lambda symbol, period: (  # type: ignore[method-assign]
        frame,
        {"cache_state": "hit", "source": "cache", "latest_fetch": "2099-04-18T00:00:00+00:00"},
    )
    audit = service.batch_audit(
        ["MSFT"],
        period="2y",
        max_age_hours=6,
        min_history_rows=504,
        min_history_span_days=729,
    )

    assert audit.insufficient_history_symbols == []
    assert audit.overall_status == "PASS"
