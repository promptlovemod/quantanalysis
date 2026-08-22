# QUA-3 Stack Decision Memo

Date: April 18, 2026
Owner: CTO

## Decisions
- Runtime: Python 3.12, FastAPI.
- Data: PostgreSQL 16 + TimescaleDB; S3-compatible raw snapshots.
- CI/CD: GitHub Actions (`lint`, `test`, `build`, `deploy-staging`).
- Deploy target: AWS ECS Fargate (staging first).
- Secrets: `.env` for dev; AWS Secrets Manager for staging/prod.

## Deferred
- Event bus adoption (Kafka/SQS) until throughput requires it.
- Data lake split until analytics volume outgrows Postgres.
