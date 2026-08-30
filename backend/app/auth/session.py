"""Server-side session management for authenticated LQMS users.

Stores the delegated Microsoft Graph access token (and refresh token) encrypted
at rest; the plaintext token is only decrypted on demand for a Microsoft 365
Copilot Chat API call and is never returned to the frontend.
"""

from __future__ import annotations

import dataclasses
import secrets
import time
from typing import Any

from app.auth.security import decrypt_token, encrypt_token
from app.config import get_settings


@dataclasses.dataclass
class LQMSUserSession:
    session_id: str
    user_id: str
    user_principal_name: str
    name: str
    email: str
    avatar_url: str
    organization: str  # Entra tenant id/name, or GitHub org (or "")
    copilot_enabled: bool
    encrypted_access_token: str
    encrypted_refresh_token: str
    created_at: float
    expires_at: float
    # Which identity provider authenticated this session -- "microsoft" (Entra
    # -> M365 Copilot) or "github" (GitHub OAuth -> GitHub Copilot). Decides
    # which Copilot backend this session's investigations use.
    auth_provider: str = "microsoft"

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def get_decrypted_token(self) -> str:
        """Decrypt the delegated Microsoft Graph access token for a Copilot Chat API call."""
        return decrypt_token(self.encrypted_access_token)

    def get_decrypted_refresh_token(self) -> str:
        """Decrypt the delegated refresh token for a silent token refresh."""
        return decrypt_token(self.encrypted_refresh_token)

    def update_tokens(self, *, access_token: str, refresh_token: str = "") -> None:
        """Re-encrypt and store a freshly refreshed token pair in place."""
        self.encrypted_access_token = encrypt_token(access_token)
        if refresh_token:
            self.encrypted_refresh_token = encrypt_token(refresh_token)

    def to_safe_dict(self) -> dict[str, Any]:
        """Return safe user profile for frontend consumption -- NEVER returns a token."""
        return {
            "authenticated": True,
            "auth_provider": self.auth_provider,
            "user_id": self.user_id,
            "user_principal_name": self.user_principal_name,
            # `github_login` retained as an alias so the existing dashboard UI
            # (which still keys on it) keeps rendering the signed-in identity.
            "github_login": self.user_principal_name,
            "name": self.name or self.user_principal_name,
            "email": self.email,
            "avatar_url": self.avatar_url,
            "organization": self.organization,
            "copilot_enabled": self.copilot_enabled,
            "expires_at": int(self.expires_at),
        }


class SessionStore:
    """Thread-safe in-memory session store with automatic expiration cleanup."""

    def __init__(self) -> None:
        self._sessions: dict[str, LQMSUserSession] = {}

    def create_session(
        self,
        *,
        user_id: str,
        user_principal_name: str,
        name: str,
        email: str,
        avatar_url: str,
        organization: str,
        copilot_enabled: bool,
        plain_access_token: str,
        plain_refresh_token: str = "",
        auth_provider: str = "microsoft",
    ) -> LQMSUserSession:
        settings = get_settings()
        session_id = secrets.token_urlsafe(32)
        now = time.time()
        expires_at = now + (settings.session_expiry_hours * 3600)

        session = LQMSUserSession(
            session_id=session_id,
            user_id=user_id,
            user_principal_name=user_principal_name,
            name=name,
            email=email,
            avatar_url=avatar_url,
            organization=organization,
            copilot_enabled=copilot_enabled,
            encrypted_access_token=encrypt_token(plain_access_token),
            encrypted_refresh_token=encrypt_token(plain_refresh_token),
            created_at=now,
            expires_at=expires_at,
            auth_provider=auth_provider,
        )
        self._sessions[session_id] = session
        self._cleanup_expired()
        return session

    def get_session(self, session_id: str) -> LQMSUserSession | None:
        if not session_id:
            return None
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.is_expired:
            self.delete_session(session_id)
            return None
        return session

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired_keys = [k for k, s in self._sessions.items() if s.expires_at < now]
        for k in expired_keys:
            self._sessions.pop(k, None)


_global_session_store = SessionStore()


def get_session_store() -> SessionStore:
    return _global_session_store
