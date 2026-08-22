# QUA-27 Pre-Registration — Benchmark Gate Re-Run

Declared BEFORE the P5 re-run. This file is committed before any new benchmark artifacts are generated.

## Frozen configuration
- Code: commit at time of re-run (SHA recorded in `run_metadata.code_commit` of `benchmark_quality.json`; must be >= 05f1e5d).
- Universe: `benchmark_universe.txt` (59 tickers), unchanged.
- Thresholds: repo defaults in `utils/portfolio_tools.py` DEFAULT_BENCHMARK_THRESHOLDS — unchanged.
- Execution realism: `execution_lag_bars = 1`, costs per existing volume-adaptive model.
- Compile policy: `dl_compile_mode = 'max-autotune'` (unchanged behavior), with persistent Triton/Inductor cache (`.inductor_cache/`) shared across worker processes — kernel caching only, no numeric-policy change.
- Throughput: benchmark may run via existing `run_all.py --parallel N --gpu-jobs M`; ticker-level interleaving does not alter per-ticker seeds or artifacts.
- Seed: 42. No hyperparameter, threshold, or universe edits between this declaration and verdict.

## Primary hypothesis (one only)
After removing evidence-scope mixing (P1) and same-bar execution optimism (P3), deployed-model walk-forward evidence across the universe has median net-of-cost WF Sharpe > 0, i.e. gate check `min_median_wf_sharpe` PASSES on valid evidence only.

## Secondary endpoints (descriptive, not success criteria)
- `min_positive_wf_share`, `min_median_cpcv_p5`, `max_median_ece`, reliability availability (`data_availability`).

## Decision rule
1. Gate evaluated ONLY on scope-eligible metrics (`data_availability.available == true`).
2. PASS / MARGINAL / FAIL accepted as-is. **No threshold will be adjusted after seeing results.**
3. FAIL on primary → conclusion is "no deployable daily-frequency edge under current features/models", triggering a feature/label research cycle — not tuning.

## Multiple-testing interpretation
Gate now reports `multiple_testing` ledger. With ~13 configured models × 59 tickers, expected max |Sharpe| under the null ≈ √(2·ln N) > 3 for independent trials; candidate correlation lowers this, but any single-ticker Sharpe below ~2 after deflation is indistinguishable from selection luck. The gate's cross-sectional medians are the pre-registered defense against this.

## What would change my mind post hoc
Nothing that involves editing thresholds. Only: (a) evidence-availability bugs discovered in artifacts (documented, fixed, full re-run), or (b) data errors (documented, full re-run). Both must be recorded in docs/qua27/.
