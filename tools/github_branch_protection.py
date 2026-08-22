#!/usr/bin/env python3
"""Manage GitHub branch protection for the Quant Analyzer repository.

This CLI applies and inspects the documented protection policy for `main`.

Environment:
    GITHUB_TOKEN: GitHub token with repository administration permission.

Usage:
    python tools/github_branch_protection.py apply <owner/repo>
    python tools/github_branch_protection.py check <owner/repo>
    python tools/github_branch_protection.py apply <owner/repo> --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import requests

GITHUB_API_BASE_URL = "https://api.github.com"
BRANCH_NAME = "main"
REQUIRED_STATUS_CHECKS = ("lint", "test", "build")
REQUIRED_APPROVING_REVIEW_COUNT = 1
REQUEST_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class GitHubBranchProtectionClient:
    """Small GitHub REST client for branch protection operations.

    Attributes:
        token: GitHub personal access token or app token.
        api_base_url: Base GitHub API URL.
        timeout_seconds: Request timeout for all HTTP calls.
    """

    token: str
    api_base_url: str = GITHUB_API_BASE_URL
    timeout_seconds: int = REQUEST_TIMEOUT_SECONDS

    def _headers(self) -> dict[str, str]:
        """Return headers required for GitHub branch protection endpoints."""

        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def apply_branch_protection(
        self,
        repo: str,
        branch: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply branch protection settings to a repository branch.

        Args:
            repo: Repository in `owner/name` form.
            branch: Branch name to protect.
            payload: GitHub API branch protection document.

        Returns:
            Parsed JSON response from GitHub.
        """

        response = requests.put(
            f"{self.api_base_url}/repos/{repo}/branches/{branch}/protection",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def get_branch_protection(self, repo: str, branch: str) -> dict[str, Any]:
        """Fetch branch protection settings for a repository branch.

        Args:
            repo: Repository in `owner/name` form.
            branch: Branch name to inspect.

        Returns:
            Parsed JSON response from GitHub.
        """

        response = requests.get(
            f"{self.api_base_url}/repos/{repo}/branches/{branch}/protection",
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()


def build_branch_protection_payload() -> dict[str, Any]:
    """Build the canonical branch protection payload for `main`.

    Returns:
        JSON-serializable payload matching the documented repository controls.
    """

    return {
        "required_status_checks": {
            "strict": True,
            "contexts": list(REQUIRED_STATUS_CHECKS),
        },
        "enforce_admins": True,
        "required_pull_request_reviews": {
            "dismiss_stale_reviews": True,
            "require_code_owner_reviews": True,
            "required_approving_review_count": REQUIRED_APPROVING_REVIEW_COUNT,
        },
        "restrictions": None,
        "required_linear_history": True,
        "allow_force_pushes": False,
        "allow_deletions": False,
        "block_creations": False,
        "required_conversation_resolution": True,
        "lock_branch": False,
        "allow_fork_syncing": False,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the branch protection CLI.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(
        description="Apply or inspect GitHub branch protection for main.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_name in ("apply", "check"):
        subparser = subparsers.add_parser(command_name)
        subparser.add_argument("repo", help="Repository in owner/name form.")
        subparser.add_argument(
            "--branch",
            default=BRANCH_NAME,
            help=f"Branch to manage. Defaults to {BRANCH_NAME}.",
        )
        subparser.add_argument(
            "--api-base-url",
            default=GITHUB_API_BASE_URL,
            help="GitHub API base URL.",
        )
        subparser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print the payload without calling the GitHub API.",
        )

    return parser.parse_args()


def require_token() -> str:
    """Read the GitHub token from the environment.

    Returns:
        Token value from `GITHUB_TOKEN`.

    Raises:
        SystemExit: If the token is not configured.
    """

    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        raise SystemExit("error: GITHUB_TOKEN is required")
    return token


def main() -> int:
    """Execute the CLI entrypoint.

    Returns:
        Process exit code.
    """

    args = parse_args()

    if args.command == "apply":
        payload = build_branch_protection_payload()
        if args.dry_run:
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0

        client = GitHubBranchProtectionClient(
            token=require_token(),
            api_base_url=args.api_base_url,
        )
        response = client.apply_branch_protection(args.repo, args.branch, payload)
        print(json.dumps(response, indent=2, sort_keys=True))
        return 0

    if args.dry_run:
        print(
            json.dumps(
                {
                    "repo": args.repo,
                    "branch": args.branch,
                    "note": "check requires a live GitHub API call",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    client = GitHubBranchProtectionClient(
        token=require_token(),
        api_base_url=args.api_base_url,
    )
    response = client.get_branch_protection(args.repo, args.branch)
    print(json.dumps(response, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
