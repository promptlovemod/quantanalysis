# Environment and Secrets Runbook

Environments:
- dev (local `.env`)
- staging (secrets manager)
- prod (secrets manager)

Required secrets:
- MARKET_DATA_PROVIDER_API_KEY
- MARKET_DATA_PROVIDER_API_SECRET
- DATABASE_URL
- REDIS_URL (optional)
- APP_SIGNING_KEY

Policy:
- Never commit secrets.
- Rotate credentials every 90 days.
- Remove access immediately on offboarding.
