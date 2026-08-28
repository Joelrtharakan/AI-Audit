"""FastAPI Authentication router -- Microsoft Entra ID delegated sign-in for LQMS.

Replaces the former GitHub OAuth flow. A work/school user signs in with Microsoft
Entra ID; the resulting delegated Microsoft Graph token (carrying the scopes
required by the Microsoft 365 Copilot Chat API) is encrypted into the server-side
session and used per-request for Copilot calls.
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import Any

from fastapi import APIRouter, Cookie, Header, Request, Response, status
from fastapi.responses import RedirectResponse

from app.auth.microsoft_entra import MicrosoftAuthError, get_microsoft_entra_service
from app.auth.security import generate_oauth_state, validate_oauth_state
from app.auth.session import LQMSUserSession, get_session_store
from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


def get_current_user_session(
    lqms_session: str | None = Cookie(default=None),
    authorization: str | None = Header(default=None),
) -> LQMSUserSession | None:
    """Retrieve the active LQMS user session from cookie or Authorization header."""
    session_id = lqms_session
    if not session_id and authorization and authorization.startswith("Bearer "):
        session_id = authorization[7:].strip()

    if not session_id:
        return None

    return get_session_store().get_session(session_id)


def apply_user_copilot_token(request: Request) -> LQMSUserSession | None:
    """Load the signed-in user's delegated Graph token into settings for this request.

    Mirrors the former GitHub-token plumbing: the per-user token from the session
    is placed on ``settings.microsoft_copilot_access_token`` so the factory-built
    ``MicrosoftCopilotProvider`` picks it up. Returns the session (or ``None``).
    """
    settings = get_settings()
    user_session = get_current_user_session(
        lqms_session=request.cookies.get(settings.session_cookie_name),
        authorization=request.headers.get("authorization"),
    )
    if user_session is not None:
        token = user_session.get_decrypted_token()
        if token:
            settings.microsoft_copilot_access_token = token
    return user_session


async def refresh_user_copilot_token(user_session: LQMSUserSession) -> bool:
    """Attempt one silent delegated-token refresh; re-store on the session. Returns success."""
    refresh_token = user_session.get_decrypted_refresh_token()
    if not refresh_token:
        return False
    try:
        result = get_microsoft_entra_service().acquire_token_by_refresh_token(refresh_token)
    except MicrosoftAuthError as exc:
        logger.warning("Silent Microsoft token refresh failed: %s", exc)
        return False
    user_session.update_tokens(
        access_token=result["access_token"],
        refresh_token=result.get("refresh_token", ""),
    )
    get_settings().microsoft_copilot_access_token = result["access_token"]
    return True


@router.get("/microsoft/login")
async def microsoft_login(request: Request, return_to: str = "") -> Response:
    """Initiate the Microsoft Entra ID delegated authorization-code flow."""
    settings = get_settings()
    entra = get_microsoft_entra_service()

    ret_url = return_to or settings.frontend_dashboard_url
    state = generate_oauth_state(return_to=ret_url)

    try:
        auth_url = entra.get_authorization_url(state=state)
    except MicrosoftAuthError as exc:
        logger.warning("Microsoft Entra sign-in not configured: %s", exc)
        msg = (
            "Microsoft sign-in is not yet configured. Set MICROSOFT_TENANT_ID, "
            "MICROSOFT_CLIENT_ID and MICROSOFT_CLIENT_SECRET in backend/.env"
        )
        target = f"{settings.frontend_login_url}?auth=error&error=missing_config&message={urllib.parse.quote(msg)}"
        return RedirectResponse(url=target, status_code=status.HTTP_302_FOUND)

    response = RedirectResponse(url=auth_url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)
    response.set_cookie(
        key="lqms_oauth_state",
        value=state,
        max_age=600,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return response


@router.get("/microsoft/callback")
async def microsoft_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
    lqms_oauth_state: str | None = Cookie(default=None),
) -> Response:
    """Handle the Entra redirect: exchange code, verify tenant, establish a session."""
    settings = get_settings()
    entra = get_microsoft_entra_service()
    session_store = get_session_store()

    login_page = settings.frontend_login_url
    dashboard_page = settings.frontend_dashboard_url

    if error:
        err_msg = error_description or error
        logger.warning("Microsoft Entra authorization error: %s", err_msg)
        target = f"{login_page}?auth=error&error=oauth_denied&message={urllib.parse.quote(err_msg)}"
        return RedirectResponse(url=target, status_code=status.HTTP_302_FOUND)

    if not state or not validate_oauth_state(state)[0]:
        logger.warning("Invalid or expired OAuth state parameter rejected.")
        target = f"{login_page}?auth=error&error=invalid_state&message={urllib.parse.quote('Invalid or expired OAuth state')}"
        return RedirectResponse(url=target, status_code=status.HTTP_302_FOUND)

    if not code:
        logger.warning("OAuth callback missing code parameter.")
        target = f"{login_page}?auth=error&error=missing_code&message={urllib.parse.quote('Missing authorization code')}"
        return RedirectResponse(url=target, status_code=status.HTTP_302_FOUND)

    try:
        token_result = entra.acquire_token_by_authorization_code(code=code)
        profile = entra.get_user_profile(token_result["id_token_claims"])
    except MicrosoftAuthError as exc:
        logger.error("Microsoft Entra token exchange failed: %s", exc)
        target = f"{login_page}?auth=error&error=exchange_failed&message={urllib.parse.quote('Failed to authenticate with Microsoft')}"
        return RedirectResponse(url=target, status_code=status.HTTP_302_FOUND)

    is_allowed, tenant = entra.verify_tenant(profile.get("tenant_id", ""))
    if not is_allowed:
        logger.warning(
            "Access denied: user '%s' tenant '%s' not in allowed tenant '%s'",
            profile.get("user_principal_name"),
            profile.get("tenant_id"),
            settings.microsoft_allowed_tenant_id,
        )
        msg = f"Your Microsoft account ({profile.get('user_principal_name')}) is not authorized for this application."
        target = f"{login_page}?auth=error&error=unauthorized_tenant&message={urllib.parse.quote(msg)}"
        return RedirectResponse(url=target, status_code=status.HTTP_302_FOUND)

    session = session_store.create_session(
        user_id=profile.get("user_id", ""),
        user_principal_name=profile.get("user_principal_name", ""),
        name=profile.get("name", ""),
        email=profile.get("email", ""),
        avatar_url="",
        organization=tenant,
        copilot_enabled=settings.microsoft_copilot_enabled,
        plain_access_token=token_result["access_token"],
        plain_refresh_token=token_result.get("refresh_token", ""),
    )

    logger.info(
        "Microsoft Entra sign-in successful: user='%s' tenant='%s' session_id=%s",
        profile.get("user_principal_name"),
        tenant,
        session.session_id[:8],
    )

    target = f"{dashboard_page}?auth=success"
    response = RedirectResponse(url=target, status_code=status.HTTP_302_FOUND)
    response.set_cookie(
        key=settings.session_cookie_name,
        value=session.session_id,
        max_age=settings.session_expiry_hours * 3600,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    response.delete_cookie("lqms_oauth_state")
    return response


@router.get("/me")
async def get_current_user_profile(
    lqms_session: str | None = Cookie(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Return safe public user session information -- NEVER exposes the access token."""
    user_session = get_current_user_session(
        lqms_session=lqms_session,
        authorization=authorization,
    )
    if user_session is None:
        return {
            "authenticated": False,
            "user_principal_name": "",
            "github_login": "",
            "name": "",
            "avatar_url": "",
            "organization": "",
            "copilot_enabled": False,
        }

    return user_session.to_safe_dict()


@router.post("/logout")
async def logout(
    response: Response,
    lqms_session: str | None = Cookie(default=None),
) -> dict[str, Any]:
    """Destroy the local LQMS user session and clear authentication cookies."""
    settings = get_settings()
    store = get_session_store()
    if lqms_session:
        store.delete_session(lqms_session)

    response.delete_cookie(key=settings.session_cookie_name)
    response.delete_cookie(key="lqms_oauth_state")
    return {"status": "ok", "message": "Successfully logged out of LQMS"}


# Backwards-compatible aliases: the previous GitHub routes now point at Microsoft.
@router.get("/github/login", include_in_schema=False)
async def _legacy_github_login(request: Request, return_to: str = "") -> Response:
    return await microsoft_login(request, return_to=return_to)
