"""GitHub OAuth 2.0 Web Flow client and organization verification service."""

from __future__ import annotations

import logging
import urllib.parse
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_API_BASE = "https://api.github.com"


class GitHubOAuthError(Exception):
    """Raised when GitHub OAuth exchange or user identity retrieval fails."""


class GitHubOrgUnauthorizedError(GitHubOAuthError):
    """Raised when the authenticated user does not belong to the allowed organization."""


class GitHubOAuthService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def get_authorization_url(self, state: str) -> str:
        """Construct the GitHub OAuth authorization URL with required scopes."""
        if not self.settings.github_client_id:
            raise GitHubOAuthError("GITHUB_CLIENT_ID is not configured in backend settings.")

        # Request user profile and organization membership read scopes
        scopes = "read:user user:email read:org"
        params = {
            "client_id": self.settings.github_client_id,
            "redirect_uri": self.settings.github_redirect_uri,
            "scope": scopes,
            "state": state,
            "allow_signup": "false",
            "prompt": "select_account",
        }
        return f"{GITHUB_AUTH_URL}?{urllib.parse.urlencode(params)}"

    async def exchange_code_for_token(self, code: str) -> str:
        """Exchange the temporary authorization code server-side for a GitHub user access token."""
        if not self.settings.github_client_id or not self.settings.github_client_secret:
            raise GitHubOAuthError("GitHub OAuth credentials (CLIENT_ID or CLIENT_SECRET) are missing.")

        payload = {
            "client_id": self.settings.github_client_id,
            "client_secret": self.settings.github_client_secret,
            "code": code,
            "redirect_uri": self.settings.github_redirect_uri,
        }
        headers = {"Accept": "application/json"}

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(GITHUB_TOKEN_URL, json=payload, headers=headers)
            if resp.status_code != 200:
                logger.error("GitHub token exchange failed status=%d body=%s", resp.status_code, resp.text)
                raise GitHubOAuthError(f"GitHub token exchange returned HTTP {resp.status_code}")

            data = resp.json()
            if "error" in data:
                err_desc = data.get("error_description", data.get("error"))
                logger.error("GitHub token exchange returned error: %s", err_desc)
                raise GitHubOAuthError(f"GitHub OAuth error: {err_desc}")

            token = data.get("access_token")
            if not token:
                raise GitHubOAuthError("No access_token found in GitHub response.")
            return str(token)

    async def get_user_profile(self, access_token: str) -> dict[str, Any]:
        """Fetch the authenticated user's profile from GitHub API."""
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "LQMS-AI-Audit-Agent",
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{GITHUB_API_BASE}/user", headers=headers)
            if resp.status_code != 200:
                logger.error("GitHub user profile fetch failed status=%d", resp.status_code)
                raise GitHubOAuthError(f"Failed to fetch user profile (HTTP {resp.status_code})")
            return resp.json()

    async def get_user_emails(self, access_token: str) -> list[dict[str, Any]]:
        """Fetch user emails to determine primary email address."""
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "LQMS-AI-Audit-Agent",
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{GITHUB_API_BASE}/user/emails", headers=headers)
                if resp.status_code == 200:
                    return resp.json()
        except Exception as exc:
            logger.warning("Could not fetch user emails: %s", exc)
        return []

    async def verify_org_membership(self, access_token: str, username: str) -> tuple[bool, str]:
        """Verify that the user belongs to the configured company organization.
        
        Returns (is_member, org_name).
        """
        allowed_org = (self.settings.github_allowed_org or "").strip()
        if not allowed_org:
            # If no restriction is configured in development, allow the user and record default
            return True, "default"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "LQMS-AI-Audit-Agent",
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            # Check direct membership in allowed_org
            org_url = f"{GITHUB_API_BASE}/orgs/{allowed_org}/members/{username}"
            resp = await client.get(org_url, headers=headers)
            if resp.status_code in (204, 200):
                return True, allowed_org

            # Also check list of user's organizations
            user_orgs_resp = await client.get(f"{GITHUB_API_BASE}/user/orgs", headers=headers)
            if user_orgs_resp.status_code == 200:
                orgs = user_orgs_resp.json()
                for org in orgs:
                    if str(org.get("login", "")).lower() == allowed_org.lower():
                        return True, allowed_org

            logger.warning(
                "User '%s' is NOT a member of configured organization '%s'",
                username,
                allowed_org,
            )
            return False, allowed_org


_oauth_service = GitHubOAuthService()


def get_github_oauth_service() -> GitHubOAuthService:
    return _oauth_service
