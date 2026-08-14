import hmac

from fastapi import Header, HTTPException, status

from app.config import get_settings


async def require_internal_api_key(x_internal_api_key: str = Header(default="")) -> None:
    settings = get_settings()
    expected = settings.internal_api_key

    if not expected:
        # Misconfigured server: fail closed rather than silently accepting all requests.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is missing INTERNAL_API_KEY configuration.",
        )

    if not x_internal_api_key or not hmac.compare_digest(x_internal_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-Internal-Api-Key header.",
        )
