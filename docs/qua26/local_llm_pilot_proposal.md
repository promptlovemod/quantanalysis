# QUA-27: Local LLM Pilot Proposal

## Purpose
Approve a narrow local-LLM pilot that reduces premium-model usage for routine CEO/CTO organizational work without exposing the trading system to weaker reasoning on high-stakes technical decisions.

## Decision Requested
Approve a single-agent pilot using a local model for executive support work only.

## Recommended Pilot
### Pilot role
`Local Executive Ops Analyst`

### Why this role first
- The highest-confidence savings opportunity is organizational work, not production trading logic.
- The current `QUA-26` analysis found no cloud-LLM application code inside the Quant Analyzer repository to migrate directly.
- A narrow support role creates measurable cost data without increasing data-integrity risk in the ML pipeline, backtester, or signal path.

### Specific use case
- First assignment class: low-ambiguity executive-support requests such as issue summarization, repo lookups, dependency mapping, meeting-prep briefs, and test-log condensation.
- These tasks are repetitive, text-heavy, and easy to review, which makes them suitable for a local-model pilot while keeping cloud-model capacity focused on architecture and technical judgment.

## Ollama Model Recommendation
### Recommended default
`qwen2.5:14b`

### Rationale
- Ollama lists `qwen2.5:14b` at roughly 9.0 GB, which is materially easier to host locally than larger frontier-class open models while still offering stronger instruction following and structured-output behavior than very small models.
- Ollama documents Qwen2.5 with up to 128K context support and improved long-text, structured-data, and JSON handling. Those strengths fit executive-support workflows better than pure code-generation optimization.
- For this pilot, consistency on summaries, routing notes, and draft documents matters more than peak coding benchmark performance.

### Hardware-constrained fallback
`llama3.1:8b`

### Fallback rationale
- Ollama exposes `llama3.1` in an 8B size class, which is more realistic if the operator wants lower memory pressure or faster first-pass latency on modest hardware.
- This fallback should be treated as a cost-control option, not the default recommendation, because the pilot benefits from stronger structured instruction following when review load is the real bottleneck.

## Scope
### In scope
- Inbox triage and task summarization
- Daily status rollups
- Documentation drafting and cleanup
- Repository inventory and file mapping
- Test-log and CI-log condensation
- Evidence gathering for later CEO/CTO review

### Out of scope
- Changes to production Python trading logic
- Market-data ingestion, cache behavior, or DuckDB schema work
- Feature engineering, walk-forward validation, HMM regime detection, conformal prediction, or portfolio logic
- Final technical decisions on reliability, rollout risk, or model quality

## Operating Policy
### Local-model first
- Ticket triage
- Status reporting
- Docs maintenance
- Search-heavy research tasks with explicit instructions
- Mechanical summarization and formatting work

### Local-model allowed only with cloud or human review
- Docs-only repository edits
- Small isolated test additions
- CI triage summaries
- Draft prioritization proposals

### Cloud-model only
- Architecture and planning decisions
- Ambiguous prioritization or staffing trade-offs
- Any task that touches production trading code or data-integrity controls
- Any task where lookahead safety is uncertain
- Final sign-off on production-impacting work

## Guardrails
- The pilot agent cannot independently close issues that touch trading logic, backtesting logic, market data, or risk policy.
- All code-impacting output requires cloud-model or human review before merge or completion.
- The pilot agent must not own tasks involving lookahead-bias review, walk-forward split safety, or feature availability at inference time.
- The pilot starts with one agent only. Expansion requires a separate approval decision.

## Success Criteria
- Premium-model usage decreases for approved organizational task classes.
- No production incidents or data-integrity regressions are attributable to pilot output.
- Review overhead remains lower than the savings from reduced premium-model use.
- The pilot produces consistent, usable summaries and documentation with low rework.

## Expected Cost Profile
### Direct model cost
- Marginal per-request model cost on the local Ollama runtime is expected to be near zero relative to paid cloud usage, excluding host amortization and electricity.

### Practical savings expectation
- The realistic savings target is not 100% elimination of premium-model spend. The target is to offload routine executive-support traffic while preserving cloud-model review on higher-risk work.
- A reasonable pilot expectation is that approved organizational tasks shifted to the local agent reduce cloud usage for that task class substantially, while total savings are partially offset by review time and local infrastructure overhead.

### Measurement approach
- Track the count of tasks handled by the local agent versus cloud agents during the pilot window.
- Estimate avoided cloud spend by comparing routed local tasks against the average cloud-model cost of the same task class before pilot launch.
- Subtract any measured increase in reviewer time or infrastructure expense before declaring net savings.

## Proposed Rollout
1. Approve the pilot operating model.
2. Create one local-model agent with a narrow executive-ops prompt.
3. Route only approved task classes to that agent.
4. Require review on any repository-editing work.
5. Review pilot results after a short trial window and decide whether to expand, narrow, or stop.

## Implementation Timeline
### Week 1
- Obtain CEO/board approval and confirm the supported Ollama adapter configuration.
- Finalize the pilot prompt, routing rules, and review/escape-hatch policy.

### Week 2
- Create the `Local Executive Ops Analyst` agent.
- Start with one narrow queue: summarization, repo lookup, and documentation support.

### Week 3
- Expand to the full approved executive-support task list if review burden stays acceptable.
- Record task counts, review effort, and any failure modes.

### Week 4
- Produce a pilot readout with observed savings, review burden, and any policy violations.
- Decide whether to expand the pilot, keep it narrow, or shut it down.

## Required Dependencies
- CEO or board approval to create the new agent, or delegation of `canCreateAgents=true` to an authorized operator
- Confirmation of the supported local adapter configuration from the platform owner
- Written routing rules for local-model vs cloud-model work before first assignment

## Recommendation
Approve the pilot as a conservative hybrid-routing experiment. This captures likely savings on low-risk executive support work while preserving premium-model capacity for planning, architecture, and trading-system judgment where failure cost is materially higher than token cost.
