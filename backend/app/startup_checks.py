"""Fail-fast production configuration validation (spec Phase 8 + 9).

Called once at application startup. In production (``ENVIRONMENT=production``)
a security-critical misconfiguration raises ``ConfigurationError`` so the
process refuses to start rather than come up insecure or misrepresenting AI
availability.

Deliberately does NOT require an LLM/provider to be reachable: the architecture
supports a deterministic safety fallback, and a runtime provider outage must
not be turned into a boot failure. Only *deployment misconfiguration* fails
here.
"""

from __future__ import annotations

import logging

from app.config import Settings

logger = logging.getLogger(__name__)

_DEV_SESSION_SECRETS = {
    "dev-secret-key-change-in-production-min-32-chars",
    "dev-session-secret-change-in-prod-min-32-chars",
    "changeme",
    "",
}


class ConfigurationError(RuntimeError):
    """Raised when a production deployment is misconfigured in a way that would
    be insecure or misleading to start."""


def validate_configuration(settings: Settings) -> list[str]:
    """Return a list of problems. Empty == OK. Raises in production if non-empty.

    In non-production environments the problems are logged as warnings and the
    app is allowed to start (local-dev convenience).
    """
    problems: list[str] = []

    # --- cache bounds (all environments) --------------------------------
    if settings.analysis_cache_max_entries < 1:
        problems.append("ANALYSIS_CACHE_MAX_ENTRIES must be >= 1")
    if settings.analysis_cache_ttl_seconds <= 0:
        problems.append("ANALYSIS_CACHE_TTL_SECONDS must be > 0")

    if settings.is_production:
        # --- authentication -------------------------------------------
        if settings.internal_api_key_is_sentinel:
            problems.append(
                "INTERNAL_API_KEY is unset or a development placeholder; a real "
                "secret is required when ENVIRONMENT=production"
            )
        # --- session signing secret ----------------------------------
        secret = settings.session_secret.strip()
        if secret in _DEV_SESSION_SECRETS or "change" in secret.lower():
            problems.append("SESSION_SECRET is a development placeholder; set a real secret")
        if len(secret) < 32:
            problems.append("SESSION_SECRET must be at least 32 characters")
        # --- CORS -----------------------------------------------------
        origins = settings.allowed_origins_list
        if not origins:
            problems.append("ALLOWED_ORIGINS must be set explicitly in production")
        if any("localhost" in o or "127.0.0.1" in o for o in origins):
            problems.append("ALLOWED_ORIGINS contains a localhost origin in production")

    if problems:
        if settings.is_production:
            raise ConfigurationError(
                "Refusing to start: production configuration problems:\n  - "
                + "\n  - ".join(problems)
            )
        for p in problems:
            logger.warning("Configuration warning (non-production): %s", p)

    return problems
