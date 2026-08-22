"""QUA-8 multi-ticker comparison helpers for the Quant Analyzer API."""

from __future__ import annotations

from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
MIN_SYMBOLS_PER_QUERY = 2
# Keep the endpoint comfortably above the >=10-ticker acceptance gate while
# preventing unbounded payload growth from accidental oversized requests.
MAX_SYMBOLS_PER_QUERY = 25


FrameLoader = Callable[[str, str], pd.DataFrame]


def parse_symbols(raw_symbols: str) -> list[str]:
    """Parse, normalize, and validate a comma-separated symbol list.

    Args:
        raw_symbols: Comma-separated ticker symbols from the API surface.

    Returns:
        A de-duplicated list of uppercase ticker symbols in first-seen order.

    Assumptions:
        Symbol parsing is deterministic and order-preserving. No future data is
        involved; this function performs input validation only.
    """

    if raw_symbols is None or not raw_symbols.strip():
        raise ValueError("symbols must contain at least two comma-separated tickers")

    normalized_symbols: list[str] = []
    seen_symbols: set[str] = set()
    for raw_symbol in raw_symbols.split(","):
        symbol = raw_symbol.strip().upper()
        if not symbol or symbol in seen_symbols:
            continue
        normalized_symbols.append(symbol)
        seen_symbols.add(symbol)

    if len(normalized_symbols) < MIN_SYMBOLS_PER_QUERY:
        raise ValueError("at least two distinct symbols are required")
    if len(normalized_symbols) > MAX_SYMBOLS_PER_QUERY:
        raise ValueError(f"at most {MAX_SYMBOLS_PER_QUERY} symbols are allowed per request")
    return normalized_symbols


def build_multi_ticker_analysis(
    *,
    symbols: list[str],
    period: str,
    limit: int,
    since: datetime | None,
    until: datetime | None,
    load_frame: FrameLoader,
) -> dict:
    """Build QUA-8 comparison metrics and return-correlation payloads.

    Args:
        symbols: Normalized ticker symbols.
        period: Historical fetch period routed to the existing data layer.
        limit: Maximum number of rows to retain per symbol after filtering.
        since: Optional inclusive lower timestamp bound.
        until: Optional inclusive upper timestamp bound.
        load_frame: Callback that returns OHLCV data for one symbol.

    Returns:
        A JSON-safe dictionary with aligned price metrics and Pearson/Spearman
        return correlations.

    Assumptions:
        Each metric uses only the rows returned by ``load_frame`` after
        historical filtering. Alignment is performed on the overlapping
        timestamps across all requested symbols; no future values are created or
        backfilled.
    """

    close_frames: dict[str, pd.Series] = {}
    ticker_metrics: list[dict] = []
    for symbol in symbols:
        raw_frame = load_frame(symbol, period)
        prepared_frame = _prepare_frame(raw_frame, symbol=symbol, limit=limit, since=since, until=until)
        close_frames[symbol] = prepared_frame["Close"].rename(symbol)
        ticker_metrics.append(_build_symbol_metric(symbol=symbol, frame=prepared_frame))

    aligned_prices = pd.concat(close_frames.values(), axis=1, join="inner").sort_index()
    if aligned_prices.empty or len(aligned_prices) < 2:
        raise ValueError("requested symbols do not have enough overlapping history for comparison")
    aligned_returns = aligned_prices.pct_change().dropna(how="any")
    if aligned_returns.empty:
        raise ValueError("requested symbols do not have enough overlapping return history for correlation")

    return {
        "symbols": symbols,
        "count": len(symbols),
        "range": {
            "aligned_start": aligned_prices.index[0].to_pydatetime().isoformat(),
            "aligned_end": aligned_prices.index[-1].to_pydatetime().isoformat(),
            "aligned_observations": int(len(aligned_prices)),
        },
        "data": {
            "ticker_metrics": ticker_metrics,
            "correlation": {
                "pearson": _correlation_to_dict(aligned_returns.corr(method="pearson")),
                "spearman": _correlation_to_dict(aligned_returns.corr(method="spearman")),
            },
        },
    }


def _prepare_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    limit: int,
    since: datetime | None,
    until: datetime | None,
) -> pd.DataFrame:
    """Filter and validate one symbol frame for multi-ticker comparison."""

    if frame is None or frame.empty:
        raise ValueError(f"no OHLCV data available for {symbol}")

    prepared = frame.sort_index()
    if since is not None:
        prepared = prepared[prepared.index >= pd.Timestamp(since)]
    if until is not None:
        prepared = prepared[prepared.index <= pd.Timestamp(until)]
    prepared = prepared.tail(limit)
    if prepared.empty:
        raise ValueError(f"no OHLCV data remains for {symbol} after filtering")

    close_values = pd.to_numeric(prepared["Close"], errors="coerce")
    finite_close_values = close_values.dropna()
    if finite_close_values.empty:
        raise ValueError(f"close series is empty for {symbol}")
    if not np.isfinite(finite_close_values).all():
        raise ValueError(f"close series contains non-finite values for {symbol}")
    prepared = prepared.copy()
    prepared["Close"] = close_values.astype(float)
    return prepared


def _build_symbol_metric(symbol: str, frame: pd.DataFrame) -> dict:
    """Build per-symbol return and volatility metrics from historical closes."""

    close = frame["Close"]
    returns = close.pct_change().dropna()
    annualized_volatility_pct = 0.0
    average_daily_return_pct = 0.0
    if not returns.empty:
        annualized_volatility_pct = float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)
        average_daily_return_pct = float(returns.mean() * 100.0)

    cumulative_return_pct = float(((close.iloc[-1] / close.iloc[0]) - 1.0) * 100.0)
    return {
        "symbol": symbol,
        "observations": int(len(frame)),
        "start": frame.index[0].to_pydatetime().isoformat(),
        "end": frame.index[-1].to_pydatetime().isoformat(),
        "start_close": float(close.iloc[0]),
        "end_close": float(close.iloc[-1]),
        "latest_close": float(close.iloc[-1]),
        "cumulative_return_pct": round(cumulative_return_pct, 6),
        "average_daily_return_pct": round(average_daily_return_pct, 6),
        "annualized_volatility_pct": round(annualized_volatility_pct, 6),
    }


def _correlation_to_dict(correlation: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Convert a correlation matrix to a JSON-safe nested dictionary."""

    output: dict[str, dict[str, float | None]] = {}
    for row_symbol, row in correlation.iterrows():
        row_output: dict[str, float | None] = {}
        for column_symbol, value in row.items():
            row_output[str(column_symbol)] = None if pd.isna(value) else round(float(value), 6)
        output[str(row_symbol)] = row_output
    return output
