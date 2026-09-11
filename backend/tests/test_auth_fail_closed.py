"""Phase 1 -- fail-closed internal API authentication.

No production configuration may expose the API merely because the internal API
key is absent or equals a development sentinel.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from app import auth
from app.config import Settings


def _call(settings, header: str = ""):
    """Invoke the dependency with a specific Settings object."""
    orig = auth.get_settings
    auth.get_settings = lambda: settings
    try:
        return asyncio.run(auth.require_internal_api_key(x_internal_api_key=header))
    finally:
        auth.get_settings = orig


# --- development / test: missing key -> local convenience bypass ---------------

@pytest.mark.parametrize("env", ["development", "test"])
@pytest.mark.parametrize("key", ["", "dev", "devkey123", "development"])
def test_dev_missing_key_allows_local_calls(env, key):
    s = Settings(environment=env, internal_api_key=key)
    assert _call(s) is None
    assert _call(s, "anything") is None


# --- development: a real key is still enforced --------------------------------

def test_dev_real_key_is_enforced():
    s = Settings(environment="development", internal_api_key="real-secret-123")
    assert _call(s, "real-secret-123") is None
    with pytest.raises(HTTPException) as ei:
        _call(s, "wrong")
    assert ei.value.status_code == 401
    with pytest.raises(HTTPException) as ei:
        _call(s, "")
    assert ei.value.status_code == 401


# --- production: sentinel / missing key -> hard misconfiguration error --------

@pytest.mark.parametrize("env", ["production", "prod", "PRODUCTION"])
@pytest.mark.parametrize("key", ["", "dev", "devkey123", "development", "changeme"])
def test_production_missing_secret_fails_closed(env, key):
    s = Settings(environment=env, internal_api_key=key)
    for header in ("", "dev", "devkey123", "guessed-value"):
        with pytest.raises(HTTPException) as ei:
            _call(s, header)
        assert ei.value.status_code == 500  # configuration error, not 401


# --- production: real key configured -> normal enforcement -------------------

def test_production_real_key_enforced():
    s = Settings(environment="production", internal_api_key="prod-secret-abc")
    assert _call(s, "prod-secret-abc") is None
    with pytest.raises(HTTPException) as ei:
        _call(s, "")
    assert ei.value.status_code == 401
    with pytest.raises(HTTPException) as ei:
        _call(s, "prod-secret-xyz")
    assert ei.value.status_code == 401


def test_is_production_and_sentinel_helpers():
    assert Settings(environment="production").is_production is True
    assert Settings(environment="development").is_production is False
    assert Settings(internal_api_key="").internal_api_key_is_sentinel is True
    assert Settings(internal_api_key="devkey123").internal_api_key_is_sentinel is True
    assert Settings(internal_api_key="a-real-secret").internal_api_key_is_sentinel is False
