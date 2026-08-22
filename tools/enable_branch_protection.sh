#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-}"
if [[ -z "$REPO" ]]; then
  echo "usage: $0 <owner/repo>"
  exit 1
fi

echo "Applying branch protection to $REPO (branch: main)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/github_branch_protection.py" apply "${REPO}"

echo "Branch protection update complete."
