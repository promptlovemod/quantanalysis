"""Pre-fetch OHLCV history for a benchmark universe into the local DuckDB cache.

Run BEFORE `run_all.py --benchmark ...` when using --parallel workers:
parallel children then serve every ticker from cache instead of racing
providers concurrently (which triggers rate-limit failures).

Usage:
    python tools/prefetch_benchmark_data.py [watchlist.txt] [--period 10y]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("analyzer", ROOT / "analyzer.py")
analyzer = importlib.util.module_from_spec(spec)
sys.modules["analyzer"] = analyzer
spec.loader.exec_module(analyzer)

import logging

_prefetch_logger = logging.getLogger("prefetch")
if not _prefetch_logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S"))
    _prefetch_logger.addHandler(_handler)
    _prefetch_logger.setLevel(logging.INFO)
analyzer.log = _prefetch_logger


def load_watchlist(path: Path) -> list[str]:
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.split("#")[0].strip().upper()
        if token and token not in symbols:
            symbols.append(token)
    return symbols


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("watchlist", nargs="?", default="benchmark_universe.txt")
    parser.add_argument("--period", default="10y")
    parser.add_argument("--delay-seconds", type=float, default=1.5)
    args = parser.parse_args()

    watchlist_path = Path(args.watchlist)
    if not watchlist_path.exists():
        print(f"watchlist not found: {watchlist_path}")
        return 2
    symbols = load_watchlist(watchlist_path)
    print(f"pre-fetching {len(symbols)} symbols (period={args.period})")

    ok, failed = [], []
    for i, sym in enumerate(symbols, 1):
        try:
            fetcher = analyzer.DataFetcher(sym, period=args.period)
            df = fetcher.fetch()
            rows = len(df) if df is not None else 0
            state = (fetcher._primary_info or {}).get("source", "?")
            print(f"[{i:>2}/{len(symbols)}] {sym:<6} rows={rows:<6} source={state}")
            (ok if rows > 0 else failed).append(sym)
        except Exception as exc:
            print(f"[{i:>2}/{len(symbols)}] {sym:<6} FAILED: {exc}")
            failed.append(sym)
        time.sleep(args.delay_seconds)

    print(f"\ndone: {len(ok)} ok, {len(failed)} failed")
    if failed:
        print("failed:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
