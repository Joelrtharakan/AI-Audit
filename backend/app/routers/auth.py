"""FastAPI Authentication & GitHub OAuth Router for LQMS."""

from __future__ import annotations

import logging
import urllib.parse
from typing import Any

from fastapi import APIRouter, Cookie, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse

from app.auth.github_oauth import (
    GitHubOAuthError,
    GitHubOrgUnauthorizedError,
    get_github_oauth_service,
)
from app.auth.security import generate_oauth_state, validate_oauth_state
from app.auth.session import LQMSUserSession, get_session_store
from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


def get_current_user_session(
    lqms_session: str | None = Cookie(default=None),
    authorization: str | None = Header(default=None),
) -> LQMSUserSession | None:
    """Dependency to retrieve the active LQMS user session from cookie or Authorization header."""
    session_id = lqms_session
    if not session_id and authorization and authorization.startswith("Bearer "):
        session_id = authorization[7:].strip()

    if not session_id:
        return None

    store = get_session_store()
    return store.get_session(session_id)


@router.get("/github/login")
async def github_login(request: Request, return_to: str = "") -> Response:
    """Initiate GitHub OAuth 2.0 web flow with signed CSRF state."""
    settings = get_settings()
    oauth_service = get_github_oauth_service()

    ret_url = return_to or settings.frontend_dashboard_url
    state = generate_oauth_state(return_to=ret_url)

    try:
        auth_url = oauth_service.get_authorization_url(state=state)
    except GitHubOAuthError as exc:
        logger.warning("GitHub OAuth not configured: %s", exc)
        redirect_target = f"{settings.frontend_login_url}?auth=error&error=missing_config&message={urllib.parse.quote('GitHub OAuth is not yet configured. Please set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET in backend/.env')}"
        return RedirectResponse(url=redirect_target, status_code=status.HTTP_302_FOUND)

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


@router.get("/github/callback")
async def github_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
    lqms_oauth_state: str | None = Cookie(default=None),
) -> Response:
    """Handle GitHub OAuth authorization callback, exchange code, verify organization, and establish session."""
    settings = get_settings()
    oauth_service = get_github_oauth_service()
    session_store = get_session_store()

    login_page = settings.frontend_login_url
    dashboard_page = settings.frontend_dashboard_url

    # Check if user cancelled or GitHub returned error
    if error:
        err_msg = error_description or error
        logger.warning("GitHub OAuth authorization error: %s", err_msg)
        redirect_target = f"{login_page}?auth=error&error=oauth_denied&message={urllib.parse.quote(err_msg)}"
        return RedirectResponse(url=redirect_target, status_code=status.HTTP_302_FOUND)

    # Validate state parameter for CSRF prevention
    if not state or not validate_oauth_state(state)[0]:
        logger.warning("Invalid or expired OAuth state parameter rejected.")
        redirect_target = f"{login_page}?auth=error&error=invalid_state&message={urllib.parse.quote('Invalid or expired OAuth state')}"
        return RedirectResponse(url=redirect_target, status_code=status.HTTP_302_FOUND)

    if not code:
        logger.warning("OAuth callback missing code parameter.")
        redirect_target = f"{login_page}?auth=error&error=missing_code&message={urllib.parse.quote('Missing authorization code')}"
        return RedirectResponse(url=redirect_target, status_code=status.HTTP_302_FOUND)

    # Server-side code exchange
    try:
        access_token = await oauth_service.exchange_code_for_token(code=code)
        profile = await oauth_service.get_user_profile(access_token=access_token)
    except Exception as exc:
        logger.error("GitHub OAuth exchange or profile fetch failed: %s", exc)
        redirect_target = f"{login_page}?auth=error&error=exchange_failed&message={urllib.parse.quote('Failed to authenticate with GitHub')}"
        return RedirectResponse(url=redirect_target, status_code=status.HTTP_302_FOUND)

    username = str(profile.get("login", "")).strip()
    user_id = int(profile.get("id", 0))
    display_name = str(profile.get("name") or username).strip()
    avatar_url = str(profile.get("avatar_url", "")).strip()
    email = str(profile.get("email") or "").strip()

    if not email:
        emails = await oauth_service.get_user_emails(access_token=access_token)
        primary = next((e.get("email") for e in emails if e.get("primary")), None)
        email = primary or (emails[0].get("email") if emails else "")

    # Company Organization Verification
    is_member, org_name = await oauth_service.verify_org_membership(
        access_token=access_token,
        username=username,
    )
    if not is_member:
        logger.warning(
            "Access denied: User '%s' is not a member of approved organization '%s'",
            username,
            settings.github_allowed_org,
        )
        msg = f"Your GitHub account (@{username}) is not authorized for this application (must belong to {settings.github_allowed_org})."
        redirect_target = f"{login_page}?auth=error&error=unauthorized_org&message={urllib.parse.quote(msg)}"
        return RedirectResponse(url=redirect_target, status_code=status.HTTP_302_FOUND)

    # Create Authenticated Session with Encrypted Token
    session = session_store.create_session(
        github_user_id=user_id,
        github_login=username,
        name=display_name,
        email=email,
        avatar_url=avatar_url,
        organization=org_name,
        copilot_enabled=True,
        plain_github_token=access_token,
    )

    logger.info(
        "GitHub OAuth successful: user='%s' org='%s' session_id=%s",
        username,
        org_name,
        session.session_id[:8],
    )

    # Redirect to dashboard with session cookie
    redirect_target = f"{dashboard_page}?auth=success"
    response = RedirectResponse(url=redirect_target, status_code=status.HTTP_302_FOUND)
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
    session: LQMSUserSession | None = Cookie(default=None),
    lqms_session: str | None = Cookie(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Return safe public user session information -- NEVER exposes GitHub access token."""
    user_session = get_current_user_session(
        lqms_session=lqms_session,
        authorization=authorization,
    )
    if user_session is None:
        return {
            "authenticated": False,
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
    """Destroy local LQMS user session and clear authentication cookies."""
    settings = get_settings()
    store = get_session_store()
    if lqms_session:
        store.delete_session(lqms_session)

    response.delete_cookie(key=settings.session_cookie_name)
    response.delete_cookie(key="lqms_oauth_state")
    return {"status": "ok", "message": "Successfully logged out of LQMS"}
