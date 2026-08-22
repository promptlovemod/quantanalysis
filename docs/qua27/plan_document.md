# QUA-27: Benchmark Gate Measurement-Integrity Remediation Plan

## Objective
Make the benchmark quality gate measure the models that were actually deployed, and distinguish "missing evidence" from "measured failure", before drawing any further conclusions about alpha.

## Phase 0 Forensic Findings (2026-08-22)

### F1 — Evidence-scope mixing (CRITICAL, gate verdict partially invalid)
- The gate aggregates `signal.walkforward_backtest.wf_sharpe` and `signal.cpcv.sharpe_p5` across all tickers unconditionally (`utils/portfolio_tools.py`, quality-gate builder).
- Per `evidence_scope` in current signal JSONs (e.g. AAPL 2026-04-23): when the selected candidate is DL-family (PatchTST/TiDE/TFT...), `walkforward_scope` and `cpcv_scope` are `tree_family_diagnostics`.
- Consequence: for every DL-selected ticker, the gate scored tree models that were never deployed. The March FAIL on `min_median_wf_sharpe` / `min_positive_wf_share` / `min_median_cpcv_p5` mixes two different evidence populations.

### F2 — reliability_score_mean = 0.0 is an artifact-availability artifact
- Reliability score lives only in `<TICKER>_diagnostics.json` (`analyzer.py` ~L10961–10971).
- 0 of 61 archived ticker folders contain a diagnostics JSON; current `_mean_or_none` / `_meets_lower` return None → check fails, but the March build reported an exact 0.0 mean — i.e., missing artifacts were coerced into a measured-zero.
- Gate conflated "diagnostics artifact missing" with "models unreliable".

### F3 — No version control (CRITICAL for reproducibility)
- Neither workspace folder is a git repo. `experiment_runs.jsonl` records config hashes (v14.0) but not code state. The exact code that produced the March FAIL cannot be reproduced.

### F4 — Static-backtest selection-on-test risk
- `backtest()` at `analyzer.py:8537` picks `best = max(..., key=tree.results[k]['f1'])` (L8544). If those F1 values are computed on the holdout being backtested, the static surface is optimistically selected. README declares selected-candidate holdout as the primary static surface; verify this path complies.

### F5 — Execution assumptions (mild optimism)
- Same-bar close execution: `ret = close.pct_change().shift(-1)` with signal built on features through bar-t close (analyzer.py L8568, L5353, L8795). Signal at t trades at t-close → next close; strict practice executes at t+1.
- Costs ARE modeled (10 bps one-way default + volume-adaptive bin) — good. But WF trade counts (~934 for SPY ≈ 1.9/day) imply turnover-driven cost drag dominates at these horizons.

### What already passes (do not break)
- Success rate 91.5%, seed stability 100%, buy recall OK, conformal abstention working as designed (qhat=1.0 → set_size=3 → ABSTAIN).

## Remediation Phases

### P1 — Gate truthfulness fixes (utils/portfolio_tools.py, run_all.py)
1. Scope-aware aggregation: include `wf_sharpe` / `cpcv_p5` in gate metrics ONLY when `evidence_scope.walkforward_scope == "selected_candidate"` (resp. cpcv); otherwise count under a separate non-evidentiary bucket.
2. Availability vs measurement: per-metric coverage counts; missing artifact ⇒ metric null + explicit `data_availability` block; checks evaluated against missing data must emit `insufficient_evidence`, never a silent numeric zero.
3. Unit tests for both behaviors.

### P2 — Reproducibility baseline
1. `git init`, `.gitignore` (reports/, data_cache/, __pycache__/, caches), initial commit tagged `pre-qua27-baseline`.
2. Record git commit SHA into run metadata / `experiment_runs.jsonl`.

### P3 — Execution-realism audit (analyzer.py — requires senior review per QUA-26 policy)
1. Verify/fix selected-candidate static backtest selection (F4).
2. Enforce t+1 execution lag option; report turnover per WF fold alongside Sharpe.

#### P3 status (2026-08-22)
- F4 resolved by inspection: `bt` precedence at analyzer.py:10045 uses the selected candidate's `holdout_backtest`; legacy `backtest()` best-by-F1 path is fallback-only and labeled (`static_holdout_backtest` vs `selected_candidate_holdout_backtest`, distinguished in run_all.py:2191). Residual: fallback surface still self-selects on holdout F1 — accepted, labeled, not evidentiary for deployment decisions.
- Execution lag implemented: new CONFIG `execution_lag_bars = 1` (default). Applied at all three return-construction sites: `_forward_returns_from_close` (selected-candidate + calibration paths), legacy `backtest()` (analyzer.py ~8570), walkforward (~8797). With lag=1 a signal decided on close-t earns close[t+1]→close[t+2]; removes same-bar-close optimism. Historical comparability with March artifacts intentionally broken.
- Per-fold turnover: aggregate `wf_trades` already reported per ticker; per-fold granularity deferred as non-blocking.

### P4 — Statistical hygiene
1. Multiple-testing ledger: record N candidates tested per run; apply deflated-Sharpe / Bonferroni-style note to gate interpretation.
2. Pre-register ONE primary model configuration before any re-run.

### P5 — Re-run benchmark gate on frozen config
- Ex-ante success criterion: all 8 checks evaluated on valid, available evidence (not tuned thresholds). PASS or FAIL accepted honestly; no threshold adjustment until then.

## Exit Criteria Mapping

| Failed check (Mar) | Valid after |
|---|---|
| min_positive_wf_share | P1 scope filter + P3 execution lag |
| min_median_wf_sharpe | P1 + P3 |
| min_median_cpcv_p5 | P1 scope filter |
| min_reliability_score_mean | P2 availability fix (re-measure, don't assume) |
| max_median_ece | Calibration work post-P5 if still failing |

## Governance
- P1/P2 are mechanical: delegate to implementation agents, review diffs.
- P3/P4 touch trading logic: senior/cloud review required (QUA-26 policy).
- Do not add models anywhere in this plan.
