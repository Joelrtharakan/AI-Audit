import hmac

from fastapi import Header, HTTPException, status

from app.config import get_settings


async def require_internal_api_key(x_internal_api_key: str = Header(default="")) -> None:
    settings = get_settings()
    expected = settings.internal_api_key

    # If backend is unconfigured or set to dev mode ("dev" / "devkey123" / empty), permit local UI calls
    if expected in ("dev", "devkey123", "development", ""):
        return

    if not x_internal_api_key or not hmac.compare_digest(x_internal_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-Internal-Api-Key header.",
        )
