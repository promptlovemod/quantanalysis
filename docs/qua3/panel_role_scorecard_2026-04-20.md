# Panel Role Scorecard

Date: April 20, 2026
Issue: QUA-10
Owner: CTO

This document adapts the board-supplied four-role interview framework into a
 concrete panel scorecard for the founding-engineer hiring flow.

## Purpose
- Preserve the existing founding-engineer hiring bar.
- Add system-specific role lenses so split-panel feedback can be compared on a
  consistent structure.
- Force explicit rejection conditions for critical failure modes.

## Global Decision Rules
- No final `hire` recommendation if any interviewer records a hard no-hire
  condition.
- No final `hire` recommendation if the candidate scores `1` in a
  mission-critical competency for the role they are strongest in.
- Use this panel scorecard together with:
  - `interview_scorecard.md`
  - `founding_engineer_hiring_rubric.md`
  - `interview_batch_debrief_2026-04-19.md`

## Role 1: ML & Signal Engineer
Focus:
- Ensemble stacking
- HMM regime detection
- Conformal prediction
- Execution-aware signal gating

Hard no-hire condition:
- Automatic no-hire if the candidate shows lookahead blind spots on the HMM
  section.

Competencies to score:
- Leakage awareness and temporal validation discipline
- HMM/regime-model reasoning
- Conformal prediction and uncertainty calibration
- Signal gating under execution constraints
- Model-stack integration judgment

Panel notes:
- Watch for casual leakage assumptions, weak chronology control, or inability to
  explain why HMM-derived state labels can leak future information.

## Role 2: Quant Research Engineer
Focus:
- Fundamentals and speculative-growth diagnostics
- Monte Carlo scenario analysis
- Portfolio analytics: Sharpe, Calmar, drawdown
- Panel and benchmark flows

Competencies to score:
- Research-method rigor
- Scenario/risk interpretation
- Portfolio-metric fluency
- Benchmark and panel-workflow understanding
- Communication of uncertainty and limitations

Panel notes:
- Watch for overconfident alpha claims, weak caveats around simulation limits,
  or poor understanding of benchmark versus production decision quality.

## Role 3: Data & Pipeline Engineer
Focus:
- Point-in-time correctness
- Walk-forward orchestration
- DuckDB/Tiingo/yfinance data layer
- Repo audit flows

Weights:
- Point-in-time correctness: 30%
- Remaining competencies: split across the rest of the role score

Hard no-hire condition:
- Any score of `1` on point-in-time correctness is an automatic no-hire.

Competencies to score:
- Point-in-time correctness
- Walk-forward and backfill orchestration
- Data-provider reliability handling
- Storage/cache integrity
- Auditability and reproducibility

Panel notes:
- This is the most important systems-protection lens for the current codebase.
- Watch for confusion around as-of joins, revisions, stale data, or benchmark
  contamination from future information.

## Role 4: Systems & Delivery Engineer
Focus:
- HTML dashboard architecture
- Telegram notification pipeline with rate limiting and alert suppression
- Flow orchestration with partial-failure handling
- Observability

Competencies to score:
- Delivery reliability
- UI/report architecture judgment
- Notification/control-plane safety
- Partial-failure handling
- Observability and runbook mindset

Panel notes:
- Watch for candidates who can build isolated components but cannot describe how
  to keep the full system stable under partial failure.

## Consolidated Panel Scorecard

For each candidate, record:
- Role 1 overall:
- Role 2 overall:
- Role 3 overall:
- Role 4 overall:
- Hard no-hire triggered: `yes` | `no`
- Best-fit role:
- Overall recommendation: `advance` | `hold` | `reject`

## Split-Panel Decision Framework
- If one panelist recommends `reject` based on a hard no-hire rule, default to
  `reject` unless factual misunderstanding is proven.
- If panel scores differ by one level but no hard rule is triggered, prefer
  `hold` and collect one targeted follow-up signal.
- If the candidate is strongest in Role 3 or Role 4 and also clears the global
  founding-engineer bar, treat that as the highest-priority path for this
  company stage.

## Current CTO Recommendation
- Use Role 3 and Role 4 as the primary decision lenses for the founding
  engineer.
- Treat Role 1 and Role 2 as differentiators, not the core hiring bar, unless
  the company explicitly decides to bias the hire toward research/modeling
  depth over systems execution.
