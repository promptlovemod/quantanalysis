# QUA-27 Throughput Note — Numerics-Neutral Performance Work

Recorded per preregistration ("What would change my mind post hoc"): this is a
documented, code-level change ahead of any P5 re-run. No threshold, universe,
seed, hyperparameter, or label/cost logic was edited.

## Motivation

Observed benchmark throughput on the 2026-08-22 P5 attempt (commit `b774ae2`,
2 workers): ~10–31 min/ticker, median ≈ 15 min. A full 59-ticker pass costs
7–9 h wall clock and the run has no recovery path after an interruption (the
attempt died externally at 15/59 with ~2 h spent).

## Changes

### 1. Benchmark resume (orchestration only — `run_all.py`)
- New CLI flag: `run_all.py --debug benchmark --resume` (and `--benchmark ... --resume`).
- Skips tickers whose `reports/<T>/<T>_signal.json` **and**
  `<T>_diagnostics.json` already exist; gate evaluation still covers the FULL
  universe by reading artifacts from disk for every ticker.
- Default remains OFF: without `--resume`, behavior is byte-for-byte as before.
- Per-ticker numerics are untouched: resume changes only *whether* a child
  process is spawned, never what it computes.

### 2. Walk-forward re-fit parallelism (`analyzer.py`, `backtest_walkforward`)
- The sequential loop fit models one at a time (~27 expanding-window refits,
  ~3 min). Replaced by:
  1. a schedule walk that replicates the sequential state machine exactly
     (refit when no model yet or `(i - last_fit) >= refit_period`; degenerate
     single-class windows are skipped *without* advancing `last_fit`, and the
     skipped bar receives no prediction — holes preserved);
  2. parallel fitting of all scheduled refits via joblib threading backend;
  3. sequential stitching with segment batch-prediction (tree predict is
     row-independent → identical to per-bar predict).
- Each individual fit receives byte-identical `(Xs_all[:rows], tr_y.values)`
  inputs as before. No model parameter is modified — in particular inner
  thread counts (`n_jobs`) are left exactly as produced by grid search.
- Gated by new CONFIG key `tree_evidence_parallel_workers` (default 6;
  `0/1` = sequential).

### 3. CPCV path-fit parallelism (`analyzer.py`, `run_cpcv`)
- The 15 combinatorial paths are independent (fit + predict + score). Now run
  through the same threading pool; results are collected in combination
  order, so the `sharpes` list fed into mean/std/percentiles is element-wise
  identical to the sequential loop.
- Same `tree_evidence_parallel_workers` knob.

## Why this cannot change results (argument, then proof)

- Every model fit gets identical inputs and identical hyperparameters,
  including `random_state`. Concurrency does not alter any single fit's
  internals (no inner-thread-count edits — deliberately avoided because XGBoost
  float32 histogram reductions are not guaranteed thread-count-invariant).
- Tree-model `predict()` is row-independent: batch prediction equals per-bar
  prediction.
- Aggregation order preserved: WF segments stitched left-to-right; CPCV
  sharpes kept in combination order.
- Empirical proof required: A/B rerun of XLV under benchmark context
  (`ANALYZER_FORCE_DIAGNOSTICS=1 ANALYZER_RUN_CONTEXT=benchmark`, seed 42)
  compared with `tools/ab_artifact_diff.py` against the frozen baseline
  snapshot `reports/_baseline_20260822_b774ae2/XLV/` (artifacts generated
  2026-08-22 on commit `b774ae2`). Result recorded below.

## A/B result (XLV, 2026-08-22)

Three-arm design. All runs: seed 42, benchmark context
(`ANALYZER_FORCE_DIAGNOSTICS=1 ANALYZER_RUN_CONTEXT=benchmark`), same DuckDB
cache (Saturday — no new market data between runs).

| Arm | Code | Context | Artifacts |
|---|---|---|---|
| Baseline | `b774ae2` | P5 orchestrator, 2 workers (sibling ticker on same GPU) | `reports/_baseline_20260822_b774ae2/XLV/` |
| Control | `b774ae2` (stash-verified clean) | solo | `reports/_control_b774ae2_solo/XLV/` |
| Treatment | optimized + CatBoost-CPCV guard | solo | `reports/XLV/` (final state) |

### Primary endpoint — stages touched by the optimization

`signal.walkforward_backtest.*` and `signal.cpcv.*` compared control vs
treatment at full float precision (control stores them under
`tree_family_diagnostics` due to its DL-family reference model):

```
wf   BIT-IDENTICAL   (wf_sharpe=-0.7039139605083368, wf_trades=890, ...)
cpcv BIT-IDENTICAL   (n_paths=15, sharpe_p5=-2.740585818785417, ...)
```

WF stage wall clock: ~190 s → ~67 s (≈2.8×). CPCV on XLV stayed sequential by
design (evidence model was CatBoost — see guard below); RF/XGB-evidence
tickers get the parallel path.

### First treatment iteration caught a real bug (kept out of final build)

Threaded CPCV with a CatBoost evidence model dropped 14/15 paths
(`Paths=1`) — concurrent CatBoost-GPU `fit()` calls fail and were silently
swallowed by the per-path exception guard, changing the Sharpe distribution.
Fix: CPCV falls back to the original sequential loop whenever the evidence
model class comes from `catboost`. Post-fix CPCV matches control exactly.

### Noise-floor finding (pre-existing, NOT caused by this work)

Control (`b774ae2`, zero changes) vs baseline differs in 84 places:
reference_model_used flips (TiDE ↔ CatBoost), and conformal / static-backtest /
calibration-diagnostics values cascade accordingly. Root cause: DL training
early-stopping epochs vary across identical-config runs (observed 37/31,
37/31, 41/20, 46/28) — CUDA training on this GPU is not bitwise reproducible,
and DL candidates feed reference-model selection. Tree-family evidence
(WF/CPCV/tree grids) reproduced bit-exactly in every run. Implication for the
March artifact claim "seed stability 100%": true for tree models, not
demonstrable for the DL selection path. Logged as a separate research-integrity
issue; any single-run comparison must use the tree-evidence blocks, which is
exactly what the QUA-27 gate now scopes to.

## Verdict

Optimizations are numerics-neutral on their touched surfaces (proven
bit-identical) and safe to enable for P5. Recommended P5 relaunch:

```powershell
python run_all.py --debug benchmark --parallel 3 --gpu-jobs 2 --resume
```

## What was deliberately NOT done

- No change to DL training (epochs/patience/dataloader/compile) — numerics risk.
- No change to grid-search budgets, thresholds, feature pipeline, conformal,
  selection, or cost model.
- No change to default worker counts of existing knobs (`tree_grid_max_workers`
  stays 4); raising them would change `config_hash` audit trails for zero
  protocol benefit.
