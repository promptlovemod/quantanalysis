# Founding Engineer Hiring Rubric

Date: April 18, 2026
Owner: CTO
Source issue: QUA-10

## Role Summary
Hire a founding backend/data platform engineer for quant analytics infrastructure.

This is not a generic quant-research role. The first engineer must be able to
take a working but script-centric Python analysis system and harden it into a
reliable service and data platform without breaking output parity.

## Mission
- Own ingestion reliability, persistence contracts, and API/service evolution.
- Establish engineering quality baselines: CI, testing, observability, and
  runbooks.
- Deliver the technical foundation behind QUA-4 through QUA-6 and enable QUA-7
  through QUA-9.

## Must-Have Skills
- Strong production Python engineering.
- API/service design experience with FastAPI or an equivalent Python web stack.
- SQL and PostgreSQL data modeling; TimescaleDB experience preferred.
- Time-series or event-driven data pipeline experience.
- External API integration reliability: retries, rate limits, stale data
  handling, and partial-failure recovery.
- Test discipline with deterministic fixtures and regression coverage.
- AWS deployment fundamentals.

## Strong Preferences
- Financial market data experience.
- Experience migrating monolithic scripts into modular services.
- Familiarity with ML-adjacent production systems and artifact validation.
- Comfort operating in early-stage 0-to-1 environments with broad ownership.

## Explicit Non-Targets
- Pure quant researcher with weak software-engineering habits.
- Pure frontend/product engineer.
- Infra-only platform engineer who cannot ship Python application code directly.

## Codebase-Specific Requirements
- Must be able to stabilize existing script entry points such as `analyzer.py`,
  `fundamental.py`, `monte_carlo.py`, `panel_runner.py`, and `run_all.py`.
- Must preserve current analysis/reporting behavior while extracting cleaner
  service and data boundaries.
- Must understand the current local DuckDB cache model and design a clean path
  toward Postgres/Timescale-backed persistence.
- Must treat diagnostics, reproducibility, and artifact validation as first-
  class requirements, not secondary cleanup work.

## Interview Plan
1. Intro: 30 minutes
2. System design: 60 minutes
3. Practical technical: 60-90 minutes
4. Founder final: 45 minutes
5. References

## Scorecard Categories
- 0-to-1 ownership
- Ingestion/system design reliability
- Coding/testing quality
- Product/domain reasoning
- Collaboration/communication

## Hiring Bar
- No category below 2/4.
- At least three categories at 3/4 or above.
- System design and coding/testing must both be at least 3/4.

## Compensation Guardrails
- Base salary: USD 140,000-175,000
- Equity: 0.75%-1.50%
- Sign-on/equipment ceiling: USD 10,000

## CEO Summary
Prioritize reliability, systems design, and Python ownership ahead of pure
finance pedigree. Finance-domain familiarity is valuable, but it should not
outweigh evidence that the candidate can turn this codebase into a dependable
product foundation.
