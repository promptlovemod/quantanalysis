#!/usr/bin/env python3
"""Generate a top-universe market-data freshness report for QUA-4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyzer import DATA_CACHE_TTL_HOURS, DataCache, HAS_YF, TIINGO_API_KEY, _normalize_ohlcv_frame, yf
from utils.market_data import (
    MarketDataService,
    RateLimiter,
    RetryConfig,
    TiingoDailyProvider,
    TWO_YEAR_HISTORY_ROWS,
    TWO_YEAR_HISTORY_SPAN_DAYS,
    YFinanceDailyProvider,
    format_batch_audit_report,
    load_symbol_universe,
)


DEFAULT_UNIVERSE_FILE = PROJECT_ROOT / "benchmark_universe_full.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "market_data"
DEFAULT_SYMBOL_LIMIT = 50
DEFAULT_FRESHNESS_HOURS = DATA_CACHE_TTL_HOURS


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the freshness-report job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-file", type=Path, default=DEFAULT_UNIVERSE_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--period", default="2y")
    parser.add_argument("--max-symbols", type=int, default=DEFAULT_SYMBOL_LIMIT)
    parser.add_argument("--freshness-hours", type=int, default=DEFAULT_FRESHNESS_HOURS)
    parser.add_argument("--min-history-rows", type=int, default=TWO_YEAR_HISTORY_ROWS)
    parser.add_argument("--min-history-span-days", type=int, default=TWO_YEAR_HISTORY_SPAN_DAYS)
    return parser.parse_args()


def build_market_data_service() -> MarketDataService:
    """Construct the production market-data service used by the analyzer."""
    providers = [
        TiingoDailyProvider(TIINGO_API_KEY, _normalize_ohlcv_frame),
        YFinanceDailyProvider(yf if HAS_YF else None, _normalize_ohlcv_frame),
    ]
    return MarketDataService(
        providers=providers,
        cache_inspect=DataCache.inspect,
        cache_save=DataCache.save,
        retry_config=RetryConfig(),
        rate_limiters={
            "Tiingo": RateLimiter(min_interval_seconds=0.25),
            "yfinance": RateLimiter(min_interval_seconds=0.10),
        },
    )


def main() -> int:
    """Run the top-universe freshness report and write JSON/Markdown artifacts."""
    args = parse_args()
    universe_symbols = load_symbol_universe(args.universe_file, max_symbols=args.max_symbols)
    audit = build_market_data_service().batch_audit(
        symbols=universe_symbols,
        period=args.period,
        max_age_hours=args.freshness_hours,
        min_history_rows=args.min_history_rows,
        min_history_span_days=args.min_history_span_days,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_base = args.output_dir / "qua4_top50_market_data_freshness"
    json_path = report_base.with_suffix(".json")
    md_path = report_base.with_suffix(".md")

    json_path.write_text(json.dumps(audit.to_dict(), indent=2), encoding="utf-8")
    md_path.write_text(
        format_batch_audit_report(audit, universe_name=args.universe_file.name),
        encoding="utf-8",
    )

    print(f"Status: {audit.overall_status}")
    print(f"Symbols: {audit.succeeded_symbols}/{audit.total_symbols}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    return 0 if audit.overall_status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
