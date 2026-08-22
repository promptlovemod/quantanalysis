# QUA-26: Local LLM Migration Plan

## Objective
Reduce cloud-model usage for CEO/CTO organizational work that does not require frontier-model reasoning, while preserving complex cloud models for planning, architecture, and other high-ambiguity decisions.

## Current-State Findings
- The Quant Analyzer repository does not currently contain cloud-LLM application code to swap to Ollama or another local provider.
- The request therefore applies to the Paperclip agent/task-routing layer and to CEO/CTO workload allocation, not to `analyzer.py`, the feature pipeline, or the backtester.
- The current CTO agent is configured with `adapterType="codex_local"` and does not have permission to create new agents.
- The current CTO agent also cannot read Paperclip agent-configuration reflection docs, so adapter-specific local-model settings cannot be self-served from this seat.
- The explicit goal is not "move code work to weaker models." The goal is "move routine organizational work off premium models so premium models remain available for complex planning and judgment-heavy work."

## Scope Decision
This issue should be treated as an executive-workload routing project, not a production-model refactor.

### In Scope
- Define which CEO/CTO work categories are safe to route to local models.
- Define which executive and technical decisions still require premium cloud models.
- Recommend the first pilot agent role for organizational support work.
- Prepare a board/CEO-ready approval path for agent creation.

### Out of Scope
- Changing the market-data cache, DuckDB schema, or provider logic.
- Changing feature engineering, walk-forward boundaries, HMM regime logic, conformal calibration, or ensemble selection.
- Allowing unsupervised local-model edits to production trading logic.

## Recommended Migration Approach
### Option A: Conservative hybrid routing (recommended)
- Create one local-model agent for routine CEO/CTO support work such as inbox triage, issue summarization, documentation drafting, repo inventory, status rollups, and formatting-heavy edits.
- Reserve premium cloud models for complex planning, ambiguous delegation, architecture trade-offs, and high-stakes technical judgment.
- Require cloud-model review before the local agent can close any issue that touches trading logic, backtesting logic, data integrity paths, or portfolio decisions.
- Expand scope only after a measured pilot.

### Option B: Mixed worker pool
- Create several local-model workers for bounded implementation tasks and reserve cloud models for review and escalation.
- This saves more cloud usage but increases the risk of weak reasoning on ambiguous engineering work.

### Option C: Broad migration
- Move most agent work to local models by default and escalate only failures.
- This is not recommended for this company because backtest integrity failures and silent reasoning errors are more expensive than token savings.

## CEO/CTO Workload Routing Policy
### Local-model first
- Inbox triage and comment summarization.
- Ticket triage and status rollups.
- Documentation drafting and cleanup.
- Repo inventory and file mapping.
- Test-log condensation and repetitive reporting.
- Mechanical searches and evidence gathering for a later executive review.

### Local-model allowed with cloud review required
- Docs-only file changes.
- Small isolated test additions.
- Non-critical scaffolding work with explicit file ownership.
- CI failure triage where the fix is later reviewed by a cloud-model agent or human.
- Draft prioritization proposals that a CEO/CTO cloud-model pass will approve or reject.

### Cloud-model only
- Cross-issue prioritization when trade-offs are ambiguous or politically sensitive.
- Complex planning, decomposition, or architecture work.
- Agent-prompt design for roles that will influence engineering direction.
- Final technical judgment on system reliability, data integrity, and rollout risk.
- Any change touching `analyzer.py`, `monte_carlo.py`, `portfolio_tools.py`, market-data providers, cache behavior, walk-forward split logic, feature generation, model gating, conformal logic, or execution/risk policy.
- Any task where lookahead safety is uncertain.
- Final sign-off on production changes affecting signal quality or data integrity.

## First Pilot Recommendation
Create one local-model agent for executive support work rather than trading-system development.

Suggested first role:
- `Local Executive Ops Analyst`

Suggested first-task types:
- issue summarization
- inbox triage
- daily status rollups
- documentation maintenance
- repo search and dependency mapping
- test and CI log condensation

Success gate for expanding beyond the pilot:
- zero integrity regressions caused by local-model output
- measured reduction in cloud-model usage for approved task classes
- acceptable review burden on cloud/human reviewers

## Concrete Pilot Proposal For CEO Approval
### Specific pilot use case
- Launch one `Local Executive Ops Analyst` agent for low-ambiguity CEO/CTO support work only.
- Initial task classes:
  - issue summarization
  - inbox triage
  - repo search and dependency mapping
  - documentation drafting and cleanup
  - test-log and CI-log condensation
- The pilot explicitly excludes production trading-code ownership, market-data work, backtesting logic, feature engineering, portfolio logic, and any task where lookahead safety is in question.

### Ollama model selection rationale
- Recommended default: `qwen2.5:14b`
- Reasoning:
  - Ollama lists `qwen2.5:14b` at roughly 9 GB, which makes it materially easier to host locally than larger open-weight models while still preserving enough capability for structured executive-support work.
  - Ollama documents Qwen2.5 as strong on long-context instruction following, structured outputs, and JSON-style formatting, which matters more for triage, summaries, and routing notes than raw coding strength.
  - This role is deliberately optimized for reliability on repetitive organizational tasks rather than ambiguous engineering decisions.
- Hardware-constrained fallback: `llama3.1:8b`
  - Use only if available hardware or latency constraints make the 14B option impractical.
  - Treat the fallback as a capacity compromise, not the preferred pilot target.

### Expected cost savings
- Assumption set:
  - The current premium-model spend includes a meaningful share of repetitive organizational traffic from CEO/CTO workflows.
  - The pilot agent handles only approved executive-support tasks and all higher-risk work remains on premium models.
- Under those assumptions, the expected savings are:
  - 60-80% reduction in cloud-model spend for the approved executive-support task class moved to the pilot.
  - 15-30% reduction in total CEO/CTO organizational-model spend, depending on how much of the current workload is actually routine enough to reroute.
- Savings should be treated as provisional until measured against:
  - local compute overhead
  - reviewer time
  - any rework caused by lower-quality first drafts

### Success metrics
- Zero data-integrity regressions, backtesting-integrity regressions, or production incidents attributable to pilot output.
- At least 60% of approved pilot task volume completed by the local agent without escalation on first pass.
- Net reduction in premium-model usage for the approved task class after accounting for review overhead.
- Reviewer rejection or rewrite rate stays low enough that the pilot remains cheaper than keeping the same task class fully on premium models.

### Implementation timeline
- Week 1:
  - CEO/board approves the pilot scope.
  - Platform owner confirms the supported Ollama adapter configuration.
  - Routing policy and escalation rules are written down.
- Week 2:
  - Create the `Local Executive Ops Analyst` agent.
  - Start with a narrow queue of summarization, repo lookup, and documentation tasks.
- Week 3:
  - Expand to the full approved executive-support queue if rejection and rework stay acceptable.
  - Track task counts, review burden, and failure modes.
- Week 4:
  - Produce a pilot readout with savings, review overhead, and any policy violations.
  - Decide whether to expand, keep narrow, or stop the pilot.

## Required Approval And Dependencies
- CEO or board must create the new agent directly or grant `canCreateAgents=true` to an appropriate operator.
- Platform owner must provide the allowed local adapter type and supported configuration for Ollama or any other local runtime.
- Routing rules for CEO/CTO work must be written down before the first local-model agent receives work.

## Minimal Implementation Slice
1. Approve Option A as the operating model.
2. Create one local-model agent with a narrow executive-ops/documentation prompt.
3. Restrict that agent to local-model-first task classes only.
4. Require cloud review on any task that touches production Python files.
5. Review pilot results after a short trial window before expanding scope.

## Risks And Mitigations
### Risk: silent weak reasoning on ambiguous technical work
- Mitigation: keep the first pilot on routine CEO/CTO support work and require review on code-impacting work.

### Risk: local-model output drifts into trading-system decisions
- Mitigation: explicitly forbid direct ownership of data integrity, backtesting, and model-quality tasks.

### Risk: premium-model savings are erased by poor task routing
- Mitigation: define a clear cloud-only bucket for complex planning, architecture, and ambiguous prioritization before the pilot starts.

### Risk: governance dead-end because current role cannot hire agents
- Mitigation: route the approval request to CEO/board with this memo as the decision artifact.

## Success Metrics
- Cloud-model usage decreases for approved CEO/CTO support task categories.
- No production incidents, data-leakage regressions, or backtesting-integrity regressions are introduced by the pilot.
- Review overhead remains lower than the cloud-token savings.
