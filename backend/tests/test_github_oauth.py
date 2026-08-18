"""Comprehensive test suite for GitHub Copilot Enterprise OAuth Integration."""

from __future__ import annotations

import base64
import json
import time
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth.github_oauth import GitHubOAuthError, GitHubOAuthService, get_github_oauth_service
from app.auth.security import decrypt_token, encrypt_token, generate_oauth_state, validate_oauth_state
from app.auth.session import SessionStore, get_session_store
from app.config import get_settings
from app.main import create_app
from app.services.llm.base import LLMResponse
from app.services.llm.factory import get_llm_provider
from app.services.llm.providers.github_copilot_provider import GitHubCopilotProvider
from app.services.llm.providers.ollama_provider import OllamaProvider


# -----------------------------------------------------------------------------
# 1. OAuth State Generation & Security Tests
# -----------------------------------------------------------------------------

def test_oauth_state_generation_and_valid_verification():
    state = generate_oauth_state(return_to="http://localhost:5510/index.html")
    assert isinstance(state, str)
    assert "." in state

    is_valid, ret_url = validate_oauth_state(state)
    assert is_valid is True
    assert ret_url == "http://localhost:5510/index.html"


def test_oauth_state_tampering_rejected():
    state = generate_oauth_state()
    payload_b64, sig = state.rsplit(".", 1)

    # Tamper with the payload
    tampered_payload = base64.urlsafe_b64encode(b'{"nonce":"attacker","ts":999999999}').decode("utf-8")
    tampered_state = f"{tampered_payload}.{sig}"

    is_valid, _ = validate_oauth_state(tampered_state)
    assert is_valid is False


def test_oauth_state_expired_rejected():
    # Construct an expired state token
    old_ts = int(time.time()) - 1000  # 1000 seconds old
    payload = json.dumps({"nonce": "old_nonce", "ts": old_ts, "ret": ""})
    payload_b64 = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("utf-8")

    import hashlib
    import hmac
    settings = get_settings()
    sig = hmac.new(
        settings.session_secret.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    expired_state = f"{payload_b64}.{sig}"
    is_valid, _ = validate_oauth_state(expired_state, max_age_seconds=600)
    assert is_valid is False


# -----------------------------------------------------------------------------
# 2. Token Encryption at Rest Tests
# -----------------------------------------------------------------------------

def test_token_encryption_and_decryption():
    raw_token = "ghp_MockEnterpriseGitHubAccessTokenSecret12345"
    encrypted = encrypt_token(raw_token)

    assert encrypted != raw_token
    assert "ghp_" not in encrypted

    decrypted = decrypt_token(encrypted)
    assert decrypted == raw_token


def test_token_decryption_with_corrupt_data_raises_value_error():
    with pytest.raises(ValueError, match="Failed to decrypt"):
        decrypt_token("invalid_corrupted_encrypted_payload")


# -----------------------------------------------------------------------------
# 3. Session Store & Token Concealment Tests
# -----------------------------------------------------------------------------

def test_session_creation_and_safe_dict_never_exposes_token():
    store = SessionStore()
    raw_token = "ghp_SecretTokenShouldNeverBeExposedInApiOrLogs"

    session = store.create_session(
        github_user_id=12345,
        github_login="auditor_jane",
        name="Jane Auditor",
        email="jane@company.com",
        avatar_url="https://github.com/avatars/jane.png",
        organization="acme-corp",
        copilot_enabled=True,
        plain_github_token=raw_token,
    )

    assert session.session_id is not None
    assert session.get_decrypted_token() == raw_token

    safe_dict = session.to_safe_dict()
    assert safe_dict["authenticated"] is True
    assert safe_dict["github_login"] == "auditor_jane"
    assert safe_dict["name"] == "Jane Auditor"
    assert safe_dict["organization"] == "acme-corp"
    assert safe_dict["copilot_enabled"] is True

    # Critical security invariant: No token or encrypted token in safe output
    assert "token" not in safe_dict
    assert "encrypted_github_token" not in safe_dict
    assert raw_token not in str(safe_dict)


def test_session_expiration_cleanup():
    store = SessionStore()
    session = store.create_session(
        github_user_id=999,
        github_login="temp_user",
        name="Temp",
        email="temp@test.com",
        avatar_url="",
        organization="test-org",
        copilot_enabled=True,
        plain_github_token="ghp_test",
    )
    # Manually expire the session
    session.expires_at = time.time() - 10

    lookup = store.get_session(session.session_id)
    assert lookup is None


# -----------------------------------------------------------------------------
# 4. FastAPI Endpoints (/api/auth/github/login, callback, /me, logout)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_auth_me_unauthenticated_returns_safe_false():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/auth/me")
        assert resp.status_code == 200
        data = resp.json()
        assert data["authenticated"] is False
        assert data["github_login"] == ""


@pytest.mark.asyncio
async def test_auth_flow_login_generates_redirect():
    settings = get_settings()
    settings.github_client_id = "test_client_id_123"

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", follow_redirects=False) as client:
        resp = await client.get("/api/auth/github/login")
        assert resp.status_code == 307
        location = resp.headers.get("location", "")
        assert "github.com/login/oauth/authorize" in location
        assert "client_id=test_client_id_123" in location
        assert "state=" in location


@pytest.mark.asyncio
async def test_auth_callback_with_authorized_org_creates_session_and_sets_cookie():
    settings = get_settings()
    settings.github_client_id = "test_client_id"
    settings.github_client_secret = "test_secret"
    settings.github_allowed_org = "acme-corp"

    state = generate_oauth_state()

    mock_oauth_service = AsyncMock()
    mock_oauth_service.exchange_code_for_token.return_value = "ghp_ValidTokenForAcmeEmployee"
    mock_oauth_service.get_user_profile.return_value = {
        "id": 5555,
        "login": "acme_auditor",
        "name": "Acme Auditor",
        "avatar_url": "https://avatars.githubusercontent.com/u/5555",
        "email": "auditor@acme-corp.com",
    }
    mock_oauth_service.get_user_emails.return_value = []
    mock_oauth_service.verify_org_membership.return_value = (True, "acme-corp")

    app = create_app()
    transport = ASGITransport(app=app)

    with patch("app.routers.auth.get_github_oauth_service", return_value=mock_oauth_service):
        async with AsyncClient(transport=transport, base_url="http://test", follow_redirects=False) as client:
            resp = await client.get(f"/api/auth/github/callback?code=mock_code_123&state={state}")
            assert resp.status_code == 302
            assert "auth=success" in resp.headers.get("location", "")

            cookies = resp.cookies
            session_cookie = cookies.get(settings.session_cookie_name)
            assert session_cookie is not None

            # Now test /api/auth/me with the session cookie
            client.cookies.set(settings.session_cookie_name, session_cookie)
            me_resp = await client.get("/api/auth/me")
            assert me_resp.status_code == 200
            me_data = me_resp.json()
            assert me_data["authenticated"] is True
            assert me_data["github_login"] == "acme_auditor"
            assert me_data["organization"] == "acme-corp"
            assert "ghp_" not in str(me_data)  # Zero token exposure


@pytest.mark.asyncio
async def test_auth_callback_unauthorized_org_rejects_and_creates_no_session():
    settings = get_settings()
    settings.github_allowed_org = "acme-corp"

    state = generate_oauth_state()

    mock_oauth_service = AsyncMock()
    mock_oauth_service.exchange_code_for_token.return_value = "ghp_ExternalUserToken"
    mock_oauth_service.get_user_profile.return_value = {
        "id": 9999,
        "login": "external_user",
        "name": "External User",
        "avatar_url": "",
        "email": "user@other.com",
    }
    mock_oauth_service.get_user_emails.return_value = []
    mock_oauth_service.verify_org_membership.return_value = (False, "acme-corp")

    app = create_app()
    transport = ASGITransport(app=app)

    with patch("app.routers.auth.get_github_oauth_service", return_value=mock_oauth_service):
        async with AsyncClient(transport=transport, base_url="http://test", follow_redirects=False) as client:
            resp = await client.get(f"/api/auth/github/callback?code=mock_code_ext&state={state}")
            assert resp.status_code == 302
            location = resp.headers.get("location", "")
            assert "unauthorized_org" in location
            assert resp.cookies.get(settings.session_cookie_name) is None


@pytest.mark.asyncio
async def test_logout_destroys_session():
    store = get_session_store()
    session = store.create_session(
        github_user_id=123,
        github_login="user_to_logout",
        name="User",
        email="",
        avatar_url="",
        organization="acme",
        copilot_enabled=True,
        plain_github_token="ghp_xyz",
    )
    session_id = session.session_id

    app = create_app()
    transport = ASGITransport(app=app)
    settings = get_settings()

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        client.cookies.set(settings.session_cookie_name, session_id)
        logout_resp = await client.post("/api/auth/logout")
        assert logout_resp.status_code == 200

        # Verify session is destroyed from server memory
        assert store.get_session(session_id) is None


# -----------------------------------------------------------------------------
# 5. Multi-User Copilot Session Isolation & Provider Switching Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multi_user_copilot_session_isolation():
    """Verify that User A (Token A) and User B (Token B) maintain separate client/session identities."""
    provider = GitHubCopilotProvider(model="auto", timeout_seconds=10.0)

    # Mock _get_shared_copilot_client
    captured_tokens = []

    async def mock_get_client(github_token, log_level):
        captured_tokens.append(github_token)
        mock_client = AsyncMock()
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.data.content = json.dumps({"status": "ok", "user": github_token})
        mock_session.send_and_wait.return_value = mock_response
        mock_client.create_session.return_value = mock_session
        return mock_client

    with patch("app.services.llm.providers.github_copilot_provider._get_shared_copilot_client", side_effect=mock_get_client):
        # User A investigation
        res_a = await provider.generate(
            node="extraction",
            prompt="investigate finding A",
            user_token="ghp_UserAToken_111",
            user_id="user_a",
        )
        # User B investigation
        res_b = await provider.generate(
            node="extraction",
            prompt="investigate finding B",
            user_token="ghp_UserBToken_222",
            user_id="user_b",
        )

        assert "ghp_UserAToken_111" in captured_tokens
        assert "ghp_UserBToken_222" in captured_tokens
        assert captured_tokens[0] != captured_tokens[1]


def test_provider_switching_development_ollama_vs_production_copilot():
    settings = get_settings()

    settings.llm_provider = "ollama"
    dev_provider = get_llm_provider("ollama")
    assert isinstance(dev_provider, OllamaProvider)

    settings.llm_provider = "copilot"
    prod_provider = get_llm_provider("copilot")
    assert isinstance(prod_provider, GitHubCopilotProvider)
