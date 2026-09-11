"""Phase 8 + 9 -- production configuration hygiene and startup fail-fast.

Production must refuse to start when a security-critical setting is missing or
a development placeholder. Development/test config still starts (with warnings).
A runtime LLM outage must NOT be a boot failure.
"""
from __future__ import annotations

import pytest

from app.config import Settings
from app.startup_checks import ConfigurationError, validate_configuration


def _prod(**over) -> Settings:
    base = dict(
        environment="production",
        internal_api_key="a-real-strong-secret-value",
        session_secret="x" * 40,
        allowed_origins="https://lqms.example.com",
        analysis_cache_max_entries=512,
        analysis_cache_ttl_seconds=3600.0,
    )
    base.update(over)
    return Settings(**base)


def test_valid_production_config_starts():
    assert validate_configuration(_prod()) == []


@pytest.mark.parametrize("key", ["", "dev", "devkey123", "changeme"])
def test_production_missing_internal_api_key_fails_fast(key):
    with pytest.raises(ConfigurationError) as ei:
        validate_configuration(_prod(internal_api_key=key))
    assert "INTERNAL_API_KEY" in str(ei.value)


@pytest.mark.parametrize("secret", [
    "dev-secret-key-change-in-production-min-32-chars",
    "dev-session-secret-change-in-prod-min-32-chars",
    "short",
    "please-change-this-in-production-now!!",  # contains "change"
])
def test_production_placeholder_or_short_session_secret_fails_fast(secret):
    with pytest.raises(ConfigurationError) as ei:
        validate_configuration(_prod(session_secret=secret))
    assert "SESSION_SECRET" in str(ei.value)


def test_production_localhost_cors_fails_fast():
    with pytest.raises(ConfigurationError) as ei:
        validate_configuration(_prod(allowed_origins="https://ok.example.com,http://localhost:5510"))
    assert "ALLOWED_ORIGINS" in str(ei.value)


def test_production_empty_cors_fails_fast():
    with pytest.raises(ConfigurationError):
        validate_configuration(_prod(allowed_origins=""))


def test_invalid_cache_bounds_fail_in_any_environment():
    with pytest.raises(ConfigurationError):
        validate_configuration(_prod(analysis_cache_max_entries=0))
    # non-production: logged, not raised
    problems = validate_configuration(Settings(environment="development", analysis_cache_ttl_seconds=0))
    assert any("TTL" in p for p in problems)


def test_development_placeholders_start_with_warnings_only():
    s = Settings(environment="development", internal_api_key="", session_secret="dev")
    # does not raise
    problems = validate_configuration(s)
    # dev doesn't check auth/session placeholders -> only cache checks run (all fine)
    assert problems == []


def test_deterministic_fallback_config_is_not_a_boot_failure():
    # no LLM provider credentials configured at all -> still a valid deployment
    s = _prod(llm_provider="ollama")
    assert validate_configuration(s) == []
