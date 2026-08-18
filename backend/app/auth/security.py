"""Cryptographic security utilities for OAuth state signing and token encryption at rest."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from app.config import get_settings


def _get_fernet() -> Fernet:
    settings = get_settings()
    # Derive 32-byte urlsafe base64 key from session_secret
    key_bytes = hashlib.sha256(settings.session_secret.encode("utf-8")).digest()
    fernet_key = base64.urlsafe_b64encode(key_bytes)
    return Fernet(fernet_key)


def encrypt_token(plain_token: str) -> str:
    """Encrypt a GitHub user access token for secure server-side storage."""
    if not plain_token:
        return ""
    fernet = _get_fernet()
    encrypted_bytes = fernet.encrypt(plain_token.encode("utf-8"))
    return encrypted_bytes.decode("utf-8")


def decrypt_token(encrypted_token: str) -> str:
    """Decrypt an encrypted GitHub access token on demand for Copilot sessions."""
    if not encrypted_token:
        return ""
    fernet = _get_fernet()
    try:
        decrypted_bytes = fernet.decrypt(encrypted_token.encode("utf-8"))
        return decrypted_bytes.decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("Failed to decrypt GitHub access token (invalid secret or corrupted ciphertext).") from exc


def generate_oauth_state(return_to: str = "") -> str:
    """Generate a signed cryptographic OAuth state token containing timestamp and nonce."""
    settings = get_settings()
    nonce = secrets.token_urlsafe(16)
    timestamp = int(time.time())
    payload = json.dumps({"nonce": nonce, "ts": timestamp, "ret": return_to})
    payload_b64 = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("utf-8")

    sig = hmac.new(
        settings.session_secret.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return f"{payload_b64}.{sig}"


def validate_oauth_state(state: str, max_age_seconds: int = 600) -> tuple[bool, str]:
    """Validate OAuth state signature and ensure it has not expired or been tampered with.
    
    Returns (is_valid, return_to_url).
    """
    if not state or "." not in state:
        return False, ""

    settings = get_settings()
    try:
        payload_b64, sig = state.rsplit(".", 1)
        expected_sig = hmac.new(
            settings.session_secret.encode("utf-8"),
            payload_b64.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(sig, expected_sig):
            return False, ""

        payload_bytes = base64.urlsafe_b64decode(payload_b64.encode("utf-8"))
        data: dict[str, Any] = json.loads(payload_bytes.decode("utf-8"))

        ts = data.get("ts", 0)
        if time.time() - ts > max_age_seconds:
            return False, ""

        return True, str(data.get("ret", ""))
    except Exception:
        return False, ""
