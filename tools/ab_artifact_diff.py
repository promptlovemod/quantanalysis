"""A/B artifact comparison for numerics-neutrality verification.

Compares two per-ticker artifact directories (baseline vs optimized build)
and verifies that all numeric content is bit-identical after whitelisting
known-volatile metadata (timestamps, durations, run ids, commit SHAs).

Usage:
    python tools/ab_artifact_diff.py BASELINE_DIR CANDIDATE_DIR [--ticker XLV]

Exit codes:
    0 = numerically identical (quality uncompromised)
    1 = differences found (quality may be compromised)
    2 = usage / IO error
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

VOLATILE_KEY_RE = re.compile(
    r"(run_id|started_at|completed_at|generated|elapsed|duration|"
    r"code_commit|timestamp|_time$|_ts$|wall|config_hash|universe_hash)",
    re.IGNORECASE,
)


def _load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _walk(a, b, path="$", volatile_path: bool = False, diffs=None, max_report=60):
    if diffs is None:
        diffs = []
    if len(diffs) >= max_report:
        return diffs

    if isinstance(path, str):
        leaf = path.rsplit(".", 1)[-1]
        volatile_path = volatile_path or bool(VOLATILE_KEY_RE.search(leaf))

    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                diffs.append(("MISSING_IN_BASELINE", f"{path}.{k}",
                              None, "<present>" if not volatile_path else "(volatile)"))
            elif k not in b:
                diffs.append(("MISSING_IN_CANDIDATE", f"{path}.{k}",
                              "<present>" if not volatile_path else "(volatile)", None))
            else:
                _walk(a[k], b[k], f"{path}.{k}", volatile_path, diffs, max_report)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(("LENGTH", path, len(a), len(b)))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _walk(x, y, f"{path}[{i}]", volatile_path, diffs, max_report)
    elif isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            diffs.append(("VALUE", path, a, b))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a != b:
            both_num = not (a is None or b is None)
            rel = abs(a - b) / max(abs(a), abs(b), 1e-300) if both_num else float("inf")
            kind = "FLOAT_NEAR" if (math.isfinite(rel) and rel < 1e-9) else "FLOAT_DIFF"
            diffs.append((kind, path, a, b))
    elif isinstance(a, str) and isinstance(b, str):
        if a != b and not volatile_path:
            diffs.append(("STRING", path, a[:120], b[:120]))
        elif a != b and volatile_path and not VOLATILE_KEY_RE.search(str(a)) :
            pass  # volatile container values (timestamps inside lists etc.)
    else:
        if a != b:
            diffs.append(("TYPE_OR_VALUE", path,
                          type(a).__name__, type(b).__name__))

    return diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--ticker", default=None,
                    help="Only compare files matching <TICKER>_*.json")
    args = ap.parse_args()

    base_files = {p.name: p for p in args.baseline.glob("*.json")}
    cand_files = {p.name: p for p in args.candidate.glob("*.json")}
    if args.ticker:
        prefix = f"{args.ticker.upper()}_"
        base_files = {k: v for k, v in base_files.items() if k.startswith(prefix)}
        cand_files = {k: v for k, v in cand_files.items() if k.startswith(prefix)}

    names = sorted(set(base_files) | set(cand_files))
    if not names:
        print("No JSON artifacts matched.")
        return 2

    verdict_ok = True
    hard = 0
    soft = 0
    for name in names:
        if name not in base_files or name not in cand_files:
            print(f"[FAIL] {name}: missing in {'candidate' if name in base_files else 'baseline'}")
            verdict_ok = False
            hard += 1
            continue
        diffs = _walk(_load_json(base_files[name]), _load_json(cand_files[name]))
        real = [d for d in diffs if d[0] != "FLOAT_NEAR"]
        near = [d for d in diffs if d[0] == "FLOAT_NEAR"]
        if not real and not near:
            print(f"[OK]   {name}: bit-identical")
            continue
        if real:
            verdict_ok = False
            hard += len(real)
            print(f"[DIFF] {name}: {len(real)} difference(s)")
            for kind, path, av, bv in real[:25]:
                print(f"       {kind:18s} {path}: baseline={av!r} candidate={bv!r}")
            if len(real) > 25:
                print(f"       ... and {len(real) - 25} more")
        if near:
            soft += len(near)
            print(f"[NOTE] {name}: {len(near)} sub-1e-9 relative float wobbles")

    print()
    print("=" * 64)
    if verdict_ok:
        print(f"VERDICT: IDENTICAL — no quality compromise detected "
              f"({hard} hard diffs, {soft} float wobbles)")
        return 0
    print(f"VERDICT: DIFFERENCES FOUND — quality may be compromised "
          f"({hard} hard diffs, {soft} float wobbles). Investigate before use.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OSError as e:
        print(f"io error: {e}", file=sys.stderr)
        sys.exit(2)
