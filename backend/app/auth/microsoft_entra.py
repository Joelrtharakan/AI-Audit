"""Microsoft Entra ID delegated OAuth 2.0 (authorization-code flow) client.

Replaces the former GitHub OAuth sign-in. Authenticates a work/school user and
acquires a delegated Microsoft Graph access token carrying the scopes required by
the Microsoft 365 Copilot Chat API (`POST /beta/copilot/conversations`).

The Chat API supports **delegated permissions only** -- there is no application /
daemon flow -- and every listed scope is required for the call to succeed:

    Sites.Read.All, Mail.Read, People.Read.All, OnlineMeetingTranscript.Read.All,
    Chat.Read, ChannelMessage.Read.All, ExternalItem.Read.All

All are `.All` scopes and therefore require tenant administrator consent. The
authenticated user must also hold a Microsoft 365 Copilot add-on license.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

# Microsoft Graph delegated scopes required by the M365 Copilot Chat API.
# Source: https://learn.microsoft.com/microsoft-365/copilot/extensibility/api/ai-services/chat/copilotroot-post-conversations
COPILOT_CHAT_GRAPH_SCOPES: tuple[str, ...] = (
    "Sites.Read.All",
    "Mail.Read",
    "People.Read.All",
    "OnlineMeetingTranscript.Read.All",
    "Chat.Read",
    "ChannelMessage.Read.All",
    "ExternalItem.Read.All",
)

# `offline_access` -> refresh token; the OIDC scopes are implied by MSAL but
# listed for clarity. MSAL rejects reserved scopes in the list it is *given*,
# so only pass the resource scopes to acquire_token_*; the authorization URL
# may include the OIDC scopes.
_AUTH_URL_SCOPES: tuple[str, ...] = COPILOT_CHAT_GRAPH_SCOPES + ("offline_access",)


class MicrosoftAuthError(Exception):
    """Raised when Entra token acquisition or identity retrieval fails."""


class MicrosoftTenantUnauthorizedError(MicrosoftAuthError):
    """Raised when the authenticated user's tenant is not in the configured allow-list."""


def _authority(tenant_id: str) -> str:
    tid = (tenant_id or "").strip() or "organizations"
    return f"https://login.microsoftonline.com/{tid}"


class MicrosoftEntraService:
    """Thin wrapper around `msal.ConfidentialClientApplication` for the web app flow."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def _app(self) -> Any:
        try:
            import msal
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise MicrosoftAuthError(
                "msal is not installed. Install via `pip install msal`."
            ) from exc

        if not self.settings.microsoft_client_id or not self.settings.microsoft_client_secret:
            raise MicrosoftAuthError(
                "Microsoft Entra credentials (MICROSOFT_CLIENT_ID / MICROSOFT_CLIENT_SECRET) are not configured."
            )

        return msal.ConfidentialClientApplication(
            client_id=self.settings.microsoft_client_id,
            client_credential=self.settings.microsoft_client_secret,
            authority=_authority(self.settings.microsoft_tenant_id),
            # Authority is a fixed, tenant-pinned login.microsoftonline.com URL;
            # skip the network round-trips for instance/authority discovery.
            instance_discovery=False,
        )

    # -- Authorization request ------------------------------------------------

    def get_authorization_url(self, state: str) -> str:
        """Build the Entra `/authorize` URL for the delegated authorization-code flow."""
        if not self.settings.microsoft_client_id:
            raise MicrosoftAuthError("MICROSOFT_CLIENT_ID is not configured in backend settings.")
        app = self._app()
        return app.get_authorization_request_url(
            scopes=list(_AUTH_URL_SCOPES),
            state=state,
            redirect_uri=self.settings.microsoft_redirect_uri,
            prompt="select_account",
        )

    # -- Token acquisition --------------------------------------------------

    def _normalize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        if "access_token" not in result:
            err = result.get("error_description") or result.get("error") or "unknown error"
            logger.error("Entra token acquisition failed: %s", err)
            raise MicrosoftAuthError(f"Microsoft Entra token acquisition failed: {err}")
        return {
            "access_token": str(result["access_token"]),
            "refresh_token": str(result.get("refresh_token", "")),
            "expires_in": int(result.get("expires_in", 0)),
            "id_token_claims": dict(result.get("id_token_claims", {}) or {}),
        }

    def acquire_token_by_authorization_code(self, code: str) -> dict[str, Any]:
        """Exchange the authorization code for a delegated Graph access + refresh token."""
        app = self._app()
        result = app.acquire_token_by_authorization_code(
            code=code,
            scopes=list(COPILOT_CHAT_GRAPH_SCOPES),
            redirect_uri=self.settings.microsoft_redirect_uri,
        )
        return self._normalize_result(result)

    def acquire_token_by_refresh_token(self, refresh_token: str) -> dict[str, Any]:
        """Silently refresh an expired delegated access token."""
        if not refresh_token:
            raise MicrosoftAuthError("No refresh token available to refresh the Microsoft session.")
        app = self._app()
        result = app.acquire_token_by_refresh_token(
            refresh_token=refresh_token,
            scopes=list(COPILOT_CHAT_GRAPH_SCOPES),
        )
        return self._normalize_result(result)

    # -- Identity / tenant gate ------------------------------------------

    @staticmethod
    def get_user_profile(id_token_claims: dict[str, Any]) -> dict[str, Any]:
        """Extract a safe identity from ID token claims (no extra Graph call needed)."""
        claims = id_token_claims or {}
        upn = str(claims.get("preferred_username") or claims.get("upn") or claims.get("email") or "").strip()
        return {
            "user_id": str(claims.get("oid") or claims.get("sub") or "").strip(),
            "user_principal_name": upn,
            "name": str(claims.get("name") or upn).strip(),
            "email": str(claims.get("email") or upn).strip(),
            "tenant_id": str(claims.get("tid") or "").strip(),
        }

    def verify_tenant(self, tenant_id: str) -> tuple[bool, str]:
        """Check the token tenant against the optional allow-list. Returns (ok, tenant_id)."""
        allowed = (self.settings.microsoft_allowed_tenant_id or "").strip()
        if not allowed:
            return True, tenant_id or "organizations"
        if (tenant_id or "").strip().lower() == allowed.lower():
            return True, tenant_id
        logger.warning("User tenant '%s' is not the configured allowed tenant '%s'", tenant_id, allowed)
        return False, allowed


_entra_service = MicrosoftEntraService()


def get_microsoft_entra_service() -> MicrosoftEntraService:
    return _entra_service
