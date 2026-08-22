# Repository Controls Baseline

This document defines the required repository controls for `main`.

## Branch Protection (`main`)

Apply these rules in repository settings:

- Require a pull request before merging.
- Require approvals: 1 minimum.
- Require review from Code Owners.
- Dismiss stale pull request approvals when new commits are pushed.
- Require status checks to pass before merging.
- Required checks:
  - `lint`
  - `test`
  - `build`
- Require branches to be up to date before merging.
- Restrict force pushes.
- Restrict branch deletion.

## CODEOWNERS

`CODEOWNERS` is enforced for repository-wide ownership with stricter ownership for workflow and sensitive paths.

## CI/CD Workflows

- `CI` workflow (`.github/workflows/ci.yml`): required checks for lint, tests, and build artifact generation.
- `Deploy Staging` workflow (`.github/workflows/deploy-staging.yml`): runs on `main` push or manual dispatch, packages release bundle, then triggers deployment if `STAGING_DEPLOY_WEBHOOK_URL` is configured.

## Required Secrets/Environment

- Repository secret `STAGING_DEPLOY_WEBHOOK_URL` (optional but required for actual deployment trigger).
- Optional GitHub `staging` environment protection rules (recommended reviewers and wait timer).
