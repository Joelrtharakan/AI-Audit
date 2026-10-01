"""GitHub Copilot authentication via LiteLLM's NATIVE credential store.

LiteLLM's ``github_copilot`` provider authenticates with the OAuth *device flow*
and caches credentials on disk (``GITHUB_COPILOT_TOKEN_DIR`` /
``GITHUB_COPILOT_ACCESS_TOKEN_FILE`` / ``GITHUB_COPILOT_API_KEY_FILE``; default
``~/.config/litellm/github_copilot``). This module reuses LiteLLM's own
``Authenticator`` (so every path / URL / client-id environment override LiteLLM
honours is honoured here too) -- it does NOT implement a second OAuth client or
a second credential store.

SECRETS: nothing in this module logs, returns or serializes a token. Status
functions report only AVAILABLE / REFRESH_REQUIRED / MISSING.

NOTE on blocking: LiteLLM's authenticator is synchronous and its interactive
``get_access_token()`` blocks (prints a code, polls for a minute). So the device
flow is started explicitly -- the code is fetched with one short request, then
polling runs in a worker thread and writes the token into LiteLLM's own token
file. A normal request never blocks on authorization; it fails fast instead.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

AVAILABLE = "AVAILABLE"
REFRESH_REQUIRED = "REFRESH_REQUIRED"  # access token cached; short-lived Copilot key refreshes on use
MISSING = "MISSING"

_lock = threading.Lock()
_pending: dict[str, Any] | None = None  # public device-flow info only (never a token)


def _authenticator():
    from litellm.llms.github_copilot.authenticator import Authenticator

    return Authenticator()


def _paths() -> tuple[str, str]:
    a = _authenticator()
    return a.access_token_file, a.api_key_file


def read_access_token() -> str | None:
    """The cached OAuth access token (a SECRET -- callers must never log it)."""
    try:
        access_file, _ = _paths()
        with open(access_file, "r") as f:
            tok = f.read().strip()
        return tok or None
    except Exception:  # noqa: BLE001
        return None


def auth_status() -> dict[str, Any]:
    """Credential availability for the CURRENT process, without a network call and
    without exposing credential content."""
    try:
        access_file, api_key_file = _paths()
    except Exception as exc:  # noqa: BLE001
        return {"status": MISSING, "token_source": "oauth_cache", "detail": f"authenticator unavailable ({type(exc).__name__})"}

    has_access = bool(read_access_token())
    key_valid = False
    try:
        with open(api_key_file, "r") as f:
            info = json.load(f)
        key_valid = bool(info.get("token")) and float(info.get("expires_at", 0)) > datetime.now().timestamp()
    except Exception:  # noqa: BLE001
        key_valid = False

    if has_access and key_valid:
        status = AVAILABLE
    elif has_access:
        status = REFRESH_REQUIRED
    else:
        status = MISSING
    return {
        "status": status,
        "token_source": "oauth_cache",
        "token_dir": os.path.dirname(access_file),  # a directory path, not a credential
        "device_authorization_pending": _pending is not None,
    }


def credentials_usable() -> bool:
    return auth_status()["status"] != MISSING


def start_device_flow() -> dict[str, Any]:
    """Begin LiteLLM's native device authorization without blocking the caller.

    Returns the public device-flow info the user needs (verification URL + user
    code -- these are meant to be displayed, they are not credentials). Polling
    runs in a daemon thread and stores the resulting token in LiteLLM's own token
    file. A flow already in progress is reused, never duplicated."""
    global _pending
    with _lock:
        if _pending is not None and _pending["expires_at"] > time.time():
            return {k: v for k, v in _pending.items() if k != "device_code"}
        a = _authenticator()
        info = a._get_device_code()  # one short HTTP request; raises GetDeviceCodeError on failure
        expires_in = int(info.get("expires_in", 900) or 900)
        _pending = {
            "verification_uri": info["verification_uri"],
            "user_code": info["user_code"],
            "expires_in": expires_in,
            "expires_at": time.time() + expires_in,
            "device_code": info["device_code"],
        }
        threading.Thread(target=_poll_until_authorized, args=(a, dict(_pending)), daemon=True,
                         name="github-copilot-device-auth").start()
        logger.info("GitHub Copilot device authorization started (awaiting user approval).")
        return {k: v for k, v in _pending.items() if k != "device_code"}


def _poll_until_authorized(authenticator, pending: dict[str, Any]) -> None:
    global _pending
    try:
        while time.time() < pending["expires_at"]:
            try:
                token = authenticator._poll_for_access_token(pending["device_code"])
            except Exception as exc:  # noqa: BLE001 - LiteLLM raises on its own 1-minute poll window
                if "Timed out" in str(exc):
                    continue
                logger.warning("GitHub Copilot device authorization failed (%s).", type(exc).__name__)
                return
            fd = os.open(authenticator.access_token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(token)
            logger.info("GitHub Copilot device authorization completed; credentials cached by LiteLLM.")
            return
        logger.warning("GitHub Copilot device authorization expired before approval.")
    finally:
        with _lock:
            _pending = None


def missing_credentials_message() -> str:
    return "The GitHub Copilot session could not be initialized. Sign in with GitHub and try again."


def provider_auth_status(settings) -> dict[str, Any]:
    """Copilot PROVIDER credential status (distinct from the application's GitHub
    login). Presence only -- no secret is read into the result."""
    if (settings.copilot_github_token or "").strip():
        source = "app_login_token" if settings.copilot_use_app_oauth_token else "configured_token"
        return {"status": AVAILABLE, "credential_source": source}
    st = auth_status()
    return {
        "status": st["status"],
        "credential_source": "oauth_cache" if st["status"] != MISSING else "none",
        "device_authorization_pending": st.get("device_authorization_pending", False),
    }
