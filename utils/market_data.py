"""Market data provider orchestration for daily OHLCV retrieval.

This module centralizes provider selection, retry/backoff, lightweight
rate-limiting, and batch readiness auditing for market data workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
import json
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_MIN_ROWS = 50
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 0.5
RATE_LIMIT_BACKOFF_SECONDS = 5.0
TIINGO_TIMEOUT_SECONDS = 15
TRADING_DAYS_PER_YEAR = 252
TWO_YEAR_HISTORY_ROWS = 2 * TRADING_DAYS_PER_YEAR
# Daily market data is typically validated against the latest completed
# trading session, so a two-year window often spans 729 calendar days when
# the current wall-clock date is one day ahead of the last market close.
TWO_YEAR_HISTORY_SPAN_DAYS = (365 * 2) - 1


NormalizeFrame = Callable[[object], Optional[pd.DataFrame]]
CacheInspect = Callable[[str], tuple]
CacheSave = Callable[[str, pd.DataFrame], None]


class MarketDataProvider(Protocol):
    """Interface for provider-specific OHLCV fetchers."""

    name: str

    def available(self) -> bool:
        """Return True when the provider is configured and callable."""

    def fetch(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Return normalized OHLCV rows or None when the provider has no data."""


@dataclass
class RetryConfig:
    """Retry policy for transient provider failures."""

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS
    rate_limit_backoff_seconds: float = RATE_LIMIT_BACKOFF_SECONDS


@dataclass
class ProviderAttempt:
    """Structured record of one provider call outcome."""

    provider: str
    attempt: int
    status: str
    detail: str = ""
    slept_seconds: float = 0.0


@dataclass
class BatchFetchAudit:
    """Summary of a batch market-data readiness run."""

    total_symbols: int
    succeeded_symbols: int
    requested_symbols: list[str] = field(default_factory=list)
    stale_symbols: list[str] = field(default_factory=list)
    insufficient_history_symbols: list[str] = field(default_factory=list)
    failed_symbols: list[str] = field(default_factory=list)
    provider_usage: dict[str, int] = field(default_factory=dict)
    period: str = "2y"
    max_age_hours: int = 24
    min_history_rows: int = TWO_YEAR_HISTORY_ROWS
    min_history_span_days: int | None = None
    generated_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )

    @property
    def overall_status(self) -> str:
        """Return PASS when the batch meets all readiness checks."""
        if self.failed_symbols or self.stale_symbols or self.insufficient_history_symbols:
            return "FAIL"
        return "PASS"

    def to_dict(self) -> dict:
        """Serialize the audit into a JSON-safe dictionary."""
        return {
            "total_symbols": self.total_symbols,
            "succeeded_symbols": self.succeeded_symbols,
            "requested_symbols": list(self.requested_symbols),
            "stale_symbols": list(self.stale_symbols),
            "insufficient_history_symbols": list(self.insufficient_history_symbols),
            "failed_symbols": list(self.failed_symbols),
            "provider_usage": dict(self.provider_usage),
            "period": self.period,
            "max_age_hours": self.max_age_hours,
            "min_history_rows": self.min_history_rows,
            "min_history_span_days": self.min_history_span_days,
            "generated_at": self.generated_at,
            "overall_status": self.overall_status,
        }


class RateLimiter:
    """Minimum-interval rate limiter for provider calls."""

    def __init__(
        self,
        min_interval_seconds: float,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._next_allowed_at = 0.0

    def acquire(self) -> float:
        """Block until the next provider slot is available and return sleep time."""
        now = self._monotonic()
        if now < self._next_allowed_at:
            sleep_for = self._next_allowed_at - now
            self._sleeper(sleep_for)
            now = self._monotonic()
        else:
            sleep_for = 0.0
        self._next_allowed_at = now + self._min_interval_seconds
        return sleep_for


class TiingoDailyProvider:
    """Daily adjusted OHLCV provider backed by the Tiingo REST API."""

    name = "Tiingo"
    _BASE_URL = "https://api.tiingo.com/tiingo/daily"
    _PERIOD_UNITS_TO_DAYS = {"d": 1, "w": 7, "mo": 30, "y": 365}

    def __init__(self, api_key: str, normalizer: NormalizeFrame) -> None:
        self._api_key = api_key.strip()
        self._normalizer = normalizer

    def available(self) -> bool:
        """Return True when a Tiingo API key is configured."""
        return bool(self._api_key)

    def fetch(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch adjusted OHLCV rows for one symbol."""
        if not self.available():
            return None

        start_date, end_date = self._period_to_dates(period)
        request_url = (
            f"{self._BASE_URL}/{symbol.upper()}/prices"
            f"?startDate={start_date}&endDate={end_date}"
            f"&resampleFreq=daily&sort=date&token={self._api_key}"
        )
        request = Request(request_url, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=TIINGO_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode())

        if not isinstance(payload, list) or not payload:
            return None

        frame = pd.DataFrame(
            {
                "date": [row["date"][:10] for row in payload],
                "Open": [row.get("adjOpen") or row.get("open", 0) for row in payload],
                "High": [row.get("adjHigh") or row.get("high", 0) for row in payload],
                "Low": [row.get("adjLow") or row.get("low", 0) for row in payload],
                "Close": [row.get("adjClose") or row.get("close", 0) for row in payload],
                "Volume": [row.get("adjVolume") or row.get("volume", 0) for row in payload],
            }
        )
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date")
        frame.index.name = None
        return self._normalizer(frame)

    @classmethod
    def _period_to_dates(cls, period: str) -> tuple[str, str]:
        """Convert a yfinance-style period string into Tiingo date bounds."""
        end_date = dt.date.today()
        numeric_part = int("".join(filter(str.isdigit, period)) or 1)
        unit_part = "".join(filter(str.isalpha, period.lower()))
        span_days = numeric_part * cls._PERIOD_UNITS_TO_DAYS.get(unit_part, 365)
        start_date = end_date - dt.timedelta(days=span_days)
        return str(start_date), str(end_date)


class YFinanceDailyProvider:
    """Daily OHLCV provider backed by yfinance."""

    name = "yfinance"

    def __init__(self, yf_module: object | None, normalizer: NormalizeFrame) -> None:
        self._yf = yf_module
        self._normalizer = normalizer

    def available(self) -> bool:
        """Return True when the yfinance module is importable."""
        return self._yf is not None

    def fetch(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV rows for one symbol using the most stable yfinance path."""
        if not self.available():
            return None
        try:
            frame = self._normalizer(
                self._yf.download(symbol, period=period, progress=False, auto_adjust=True)
            )
        except Exception:
            frame = None
        if frame is not None and not frame.empty:
            return frame
        try:
            history = self._yf.Ticker(symbol).history(period=period, auto_adjust=True)
            return self._normalizer(history)
        except Exception:
            return None


class MarketDataService:
    """Coordinates cache-aware OHLCV fetches across multiple providers."""

    def __init__(
        self,
        providers: Iterable[MarketDataProvider],
        cache_inspect: CacheInspect,
        cache_save: CacheSave,
        retry_config: RetryConfig | None = None,
        rate_limiters: Optional[dict[str, RateLimiter]] = None,
        min_rows: int = DEFAULT_MIN_ROWS,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._providers = list(providers)
        self._cache_inspect = cache_inspect
        self._cache_save = cache_save
        self._retry_config = retry_config or RetryConfig()
        self._rate_limiters = rate_limiters or {}
        self._min_rows = int(min_rows)
        self._sleeper = sleeper

    def fetch_daily_ohlcv(self, symbol: str, period: str) -> tuple[pd.DataFrame, dict]:
        """Fetch one symbol with cache reuse, retries, provider fallback, and audit info."""
        cached_frame, cache_state, meta = self._cache_inspect(symbol)
        if cached_frame is not None:
            info = dict(meta)
            info.update(
                {
                    "cache_state": "hit",
                    "source": "cache",
                    "provider_attempts": [],
                }
            )
            return cached_frame, info

        attempts: list[ProviderAttempt] = []
        for provider in self._providers:
            if not provider.available():
                continue
            frame = self._fetch_with_provider(provider, symbol, period, attempts)
            if frame is None:
                continue
            self._cache_save(symbol, frame)
            info = dict(meta)
            info.update(
                {
                    "cache_state": cache_state,
                    "source": provider.name,
                    "provider_attempts": [attempt.__dict__ for attempt in attempts],
                }
            )
            return frame, info

        raise RuntimeError(f"Could not fetch data for {symbol.upper()}")

    def batch_audit(
        self,
        symbols: Iterable[str],
        period: str,
        max_age_hours: int,
        min_history_rows: int = TWO_YEAR_HISTORY_ROWS,
        min_history_span_days: int | None = None,
    ) -> BatchFetchAudit:
        """Fetch and validate a symbol batch against freshness and history thresholds."""
        stale_symbols: list[str] = []
        insufficient_history_symbols: list[str] = []
        failed_symbols: list[str] = []
        provider_usage: dict[str, int] = {}
        succeeded_symbols = 0

        cutoff = pd.Timestamp(dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=max_age_hours))
        normalized_symbols = [symbol.upper() for symbol in symbols]
        for symbol in normalized_symbols:
            try:
                frame, info = self.fetch_daily_ohlcv(symbol, period)
            except Exception:
                failed_symbols.append(symbol)
                continue

            succeeded_symbols += 1
            provider_usage[info["source"]] = provider_usage.get(info["source"], 0) + 1
            latest_fetch = info.get("latest_fetch")
            if latest_fetch and pd.to_datetime(latest_fetch, utc=True) < cutoff:
                stale_symbols.append(symbol)
            if not _meets_history_requirement(
                frame=frame,
                min_history_rows=min_history_rows,
                min_history_span_days=min_history_span_days,
            ):
                insufficient_history_symbols.append(symbol)

        return BatchFetchAudit(
            total_symbols=len(normalized_symbols),
            succeeded_symbols=succeeded_symbols,
            requested_symbols=normalized_symbols,
            stale_symbols=stale_symbols,
            insufficient_history_symbols=insufficient_history_symbols,
            failed_symbols=failed_symbols,
            provider_usage=provider_usage,
            period=period,
            max_age_hours=max_age_hours,
            min_history_rows=min_history_rows,
            min_history_span_days=min_history_span_days,
        )

    def _fetch_with_provider(
        self,
        provider: MarketDataProvider,
        symbol: str,
        period: str,
        attempts: list[ProviderAttempt],
    ) -> Optional[pd.DataFrame]:
        """Execute one provider with retry/backoff semantics."""
        limiter = self._rate_limiters.get(provider.name)
        for attempt_number in range(1, self._retry_config.max_attempts + 1):
            slept_seconds = limiter.acquire() if limiter is not None else 0.0
            try:
                frame = provider.fetch(symbol, period)
                if frame is None or frame.empty or len(frame) < self._min_rows:
                    attempts.append(
                        ProviderAttempt(
                            provider=provider.name,
                            attempt=attempt_number,
                            status="empty",
                            detail="insufficient rows",
                            slept_seconds=slept_seconds,
                        )
                    )
                    break
                attempts.append(
                    ProviderAttempt(
                        provider=provider.name,
                        attempt=attempt_number,
                        status="success",
                        slept_seconds=slept_seconds,
                    )
                )
                return frame
            except Exception as exc:
                is_rate_limited = self._is_rate_limit_error(exc)
                attempts.append(
                    ProviderAttempt(
                        provider=provider.name,
                        attempt=attempt_number,
                        status="rate_limited" if is_rate_limited else "error",
                        detail=str(exc),
                        slept_seconds=slept_seconds,
                    )
                )
                if attempt_number >= self._retry_config.max_attempts:
                    break
                delay_seconds = (
                    self._retry_config.rate_limit_backoff_seconds
                    if is_rate_limited
                    else self._retry_config.backoff_seconds * attempt_number
                )
                self._sleeper(delay_seconds)
        return None

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Return True when an exception looks like provider rate limiting."""
        if isinstance(exc, HTTPError) and exc.code == 429:
            return True
        message = str(exc).lower()
        return "429" in message or "rate limit" in message


def load_symbol_universe(path: str | Path, max_symbols: Optional[int] = None) -> list[str]:
    """Read a text watchlist and return normalized symbols in file order."""
    raw_lines = Path(path).read_text(encoding="utf-8").splitlines()
    symbols: list[str] = []
    for raw_line in raw_lines:
        stripped = raw_line.split("#", 1)[0].strip().upper()
        if stripped:
            symbols.append(stripped)
        if max_symbols is not None and len(symbols) >= max_symbols:
            break
    return symbols


def format_batch_audit_report(audit: BatchFetchAudit, universe_name: str) -> str:
    """Render a Markdown summary for human review or alert delivery."""
    provider_usage = ", ".join(
        f"{provider}={count}" for provider, count in sorted(audit.provider_usage.items())
    ) or "none"
    history_gate = f">={audit.min_history_rows} rows"
    if audit.min_history_span_days is not None:
        history_gate = f"{history_gate} or >={audit.min_history_span_days} calendar days"
    lines = [
        "# Market Data Freshness Report",
        "",
        f"- Universe: {universe_name}",
        f"- Generated at: {audit.generated_at}",
        f"- Period: {audit.period}",
        f"- Freshness SLA: {audit.max_age_hours}h",
        f"- History gate: {history_gate}",
        f"- Requested symbols: {audit.total_symbols}",
        f"- Succeeded symbols: {audit.succeeded_symbols}",
        f"- Provider usage: {provider_usage}",
        f"- Overall status: {audit.overall_status}",
        "",
        "## Exceptions",
        "",
        f"- Failed symbols: {_format_symbol_list(audit.failed_symbols)}",
        f"- Stale symbols: {_format_symbol_list(audit.stale_symbols)}",
        f"- Insufficient history symbols: {_format_symbol_list(audit.insufficient_history_symbols)}",
    ]
    return "\n".join(lines) + "\n"


def _format_symbol_list(symbols: list[str], max_items: int = 15) -> str:
    """Format a symbol list compactly for reports."""
    if not symbols:
        return "none"
    if len(symbols) <= max_items:
        return ", ".join(symbols)
    visible = ", ".join(symbols[:max_items])
    return f"{visible}, ... (+{len(symbols) - max_items} more)"


def _meets_history_requirement(
    frame: pd.DataFrame,
    min_history_rows: int,
    min_history_span_days: int | None,
) -> bool:
    """Return True when a frame satisfies either row-count or date-span history gates."""
    if len(frame) >= min_history_rows:
        return True
    if min_history_span_days is None or frame.empty:
        return False
    span_days = int((frame.index.max() - frame.index.min()).days)
    return span_days >= min_history_span_days
