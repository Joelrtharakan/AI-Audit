"""Microsoft Entra ID delegated sign-in + session tests (replaces test_github_oauth.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.auth.microsoft_entra import (
    COPILOT_CHAT_GRAPH_SCOPES,
    MicrosoftAuthError,
    MicrosoftEntraService,
)
from app.auth.security import decrypt_token, encrypt_token, generate_oauth_state, validate_oauth_state
from app.auth.session import SessionStore
from app.config import Settings


# --- OAuth state (unchanged CSRF machinery) ---------------------------------

def test_oauth_state_roundtrip_and_tamper_rejected():
    state = generate_oauth_state(return_to="http://localhost:5510/index.html")
    ok, ret = validate_oauth_state(state)
    assert ok and ret == "http://localhost:5510/index.html"
    assert validate_oauth_state(state[:-4] + "0000")[0] is False


# --- token encryption at rest ---------------------------------------------

def test_delegated_token_encrypted_roundtrip():
    enc = encrypt_token("delegated-graph-token")
    assert enc and enc != "delegated-graph-token"
    assert decrypt_token(enc) == "delegated-graph-token"


# --- authorization URL ---------------------------------------------------

def _svc(**overrides):
    settings = Settings(
        microsoft_client_id="client-123",
        microsoft_client_secret="secret-abc",
        microsoft_tenant_id="tenant-xyz",
        **overrides,
    )
    svc = MicrosoftEntraService()
    svc.settings = settings
    return svc


def test_authorization_url_requests_all_required_graph_scopes():
    svc = _svc()
    fake_app = MagicMock()
    fake_app.get_authorization_request_url.return_value = "https://login.microsoftonline.com/tenant-xyz/oauth2/v2.0/authorize?..."
    with patch.object(svc, "_app", return_value=fake_app):
        svc.get_authorization_url(state="s1")
    kwargs = fake_app.get_authorization_request_url.call_args.kwargs
    assert kwargs["state"] == "s1"
    assert kwargs["redirect_uri"] == svc.settings.microsoft_redirect_uri
    for scope in COPILOT_CHAT_GRAPH_SCOPES:
        assert scope in kwargs["scopes"]
    assert "offline_access" in kwargs["scopes"]


def test_missing_credentials_raises():
    svc = MicrosoftEntraService()
    svc.settings = Settings(microsoft_client_id="", microsoft_client_secret="")
    with pytest.raises(MicrosoftAuthError):
        svc.get_authorization_url(state="s1")


# --- code exchange -----------------------------------------------------

def test_acquire_token_by_authorization_code_success():
    svc = _svc()
    fake_app = MagicMock()
    fake_app.acquire_token_by_authorization_code.return_value = {
        "access_token": "at-1",
        "refresh_token": "rt-1",
        "expires_in": 3600,
        "id_token_claims": {"oid": "u1", "preferred_username": "amy@contoso.com", "name": "Amy", "tid": "tenant-xyz"},
    }
    with patch.object(svc, "_app", return_value=fake_app):
        result = svc.acquire_token_by_authorization_code("code-1")
    assert result["access_token"] == "at-1"
    profile = svc.get_user_profile(result["id_token_claims"])
    assert profile["user_principal_name"] == "amy@contoso.com"
    assert profile["tenant_id"] == "tenant-xyz"


def test_acquire_token_failure_raises():
    svc = _svc()
    fake_app = MagicMock()
    fake_app.acquire_token_by_authorization_code.return_value = {
        "error": "invalid_grant",
        "error_description": "AADSTS70008 expired",
    }
    with patch.object(svc, "_app", return_value=fake_app):
        with pytest.raises(MicrosoftAuthError):
            svc.acquire_token_by_authorization_code("bad-code")


# --- tenant allow-list gate -------------------------------------------

def test_tenant_gate_allows_when_unset():
    assert _svc(microsoft_allowed_tenant_id="").verify_tenant("any-tenant") == (True, "any-tenant")


def test_tenant_gate_blocks_foreign_tenant():
    ok, _ = _svc(microsoft_allowed_tenant_id="tenant-xyz").verify_tenant("other-tenant")
    assert ok is False


# --- session store never leaks token --------------------------------

def test_session_safe_dict_never_contains_token():
    store = SessionStore()
    session = store.create_session(
        user_id="u1",
        user_principal_name="amy@contoso.com",
        name="Amy",
        email="amy@contoso.com",
        avatar_url="",
        organization="tenant-xyz",
        copilot_enabled=True,
        plain_access_token="super-secret-token",
        plain_refresh_token="refresh-secret",
    )
    safe = session.to_safe_dict()
    assert "super-secret-token" not in str(safe)
    assert "refresh-secret" not in str(safe)
    assert session.get_decrypted_token() == "super-secret-token"
    assert session.get_decrypted_refresh_token() == "refresh-secret"


def test_session_token_refresh_in_place():
    store = SessionStore()
    session = store.create_session(
        user_id="u1", user_principal_name="a", name="a", email="a", avatar_url="",
        organization="t", copilot_enabled=True, plain_access_token="old", plain_refresh_token="r1",
    )
    session.update_tokens(access_token="new", refresh_token="r2")
    assert session.get_decrypted_token() == "new"
    assert session.get_decrypted_refresh_token() == "r2"
