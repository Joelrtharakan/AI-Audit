"""Authentication package for LQMS AI Gateway."""

from __future__ import annotations

import hmac
import logging

from fastapi import Header, HTTPException, status

from app.config import get_settings

logger = logging.getLogger(__name__)


async def require_internal_api_key(x_internal_api_key: str = Header(default="")) -> None:
    """Fail-closed guard for the internal API surface.

    Production (``ENVIRONMENT=production``): a real ``INTERNAL_API_KEY`` MUST be
    configured and every request MUST present a matching header. A missing or
    development-placeholder key is a server misconfiguration (HTTP 500), never
    an implicit "authentication disabled".

    Development / test: if no real key is configured, local UI calls are
    permitted without a header (explicit, environment-scoped convenience). If a
    real key *is* configured, it is enforced exactly as in production.
    """
    settings = get_settings()
    expected = settings.internal_api_key

    if settings.internal_api_key_is_sentinel:
        if settings.is_production:
            logger.error(
                "INTERNAL_API_KEY is unset or a development placeholder while "
                "ENVIRONMENT=production -- refusing the request. Configure a real secret."
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server authentication is not configured.",
            )
        # Development / test convenience: no real secret -> allow local calls.
        return

    if not x_internal_api_key or not hmac.compare_digest(x_internal_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-Internal-Api-Key header.",
        )
