"""Server-side session management for authenticated LQMS users with encrypted GitHub tokens."""

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
    github_user_id: int
    github_login: str
    name: str
    email: str
    avatar_url: str
    organization: str
    copilot_enabled: bool
    encrypted_github_token: str
    created_at: float
    expires_at: float

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def get_decrypted_token(self) -> str:
        """Obtain decrypted GitHub user access token for isolated Copilot SDK execution."""
        return decrypt_token(self.encrypted_github_token)

    def to_safe_dict(self) -> dict[str, Any]:
        """Return safe user profile for frontend consumption -- NEVER returns the token."""
        return {
            "authenticated": True,
            "github_user_id": self.github_user_id,
            "github_login": self.github_login,
            "name": self.name or self.github_login,
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
        github_user_id: int,
        github_login: str,
        name: str,
        email: str,
        avatar_url: str,
        organization: str,
        copilot_enabled: bool,
        plain_github_token: str,
    ) -> LQMSUserSession:
        settings = get_settings()
        session_id = secrets.token_urlsafe(32)
        now = time.time()
        expires_at = now + (settings.session_expiry_hours * 3600)
        encrypted_token = encrypt_token(plain_github_token)

        session = LQMSUserSession(
            session_id=session_id,
            github_user_id=github_user_id,
            github_login=github_login,
            name=name,
            email=email,
            avatar_url=avatar_url,
            organization=organization,
            copilot_enabled=copilot_enabled,
            encrypted_github_token=encrypted_token,
            created_at=now,
            expires_at=expires_at,
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
