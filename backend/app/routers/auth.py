"""FastAPI Authentication router -- dual SSO for LQMS.

Two identity providers coexist:
  * Microsoft Entra ID  (/api/auth/microsoft/*)  -> Microsoft 365 Copilot backend
  * GitHub OAuth         (/api/auth/github/*)     -> GitHub Copilot backend

Whichever provider a user signs in with is recorded on the session
(`auth_provider`) and determines which Copilot backend their investigations use
(see `apply_user_copilot_token`). The per-user access token is encrypted into the
server-side session and used per-request; it is never returned to the frontend.
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import Any

from fastapi import APIRouter, Cookie, Header, Request, Response, status
from fastapi.responses import RedirectResponse

from app.auth.github_oauth import GitHubOAuthError, get_github_oauth_service
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
            if user_session.auth_provider == "github":
                settings.copilot_github_token = token
                settings.llm_provider = "github_copilot"
                from app.services.llm.providers.github_copilot_provider import reset_copilot_clients
                reset_copilot_clients()
            else:
                settings.microsoft_copilot_access_token = token
                settings.llm_provider = "microsoft_copilot"
    return user_session


async def refresh_user_copilot_token(user_session: LQMSUserSession) -> bool:
    """Attempt one silent delegated-token refresh; re-store on the session. Returns success."""
    # GitHub OAuth web-flow tokens do not carry a refresh token; nothing to do.
    if user_session.auth_provider == "github":
        return False
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
            "auth_provider": "",
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


# ---------------------------------------------------------------------------
# GitHub OAuth 2.0 web flow  ->  GitHub Copilot backend
# ---------------------------------------------------------------------------


@router.get("/github/login")
async def github_login(request: Request, return_to: str = "") -> Response:
    """Initiate the GitHub OAuth 2.0 web flow (signed CSRF state)."""
    settings = get_settings()
    oauth = get_github_oauth_service()

    ret_url = return_to or settings.frontend_dashboard_url
    state = generate_oauth_state(return_to=ret_url)

    try:
        auth_url = oauth.get_authorization_url(state=state)
    except GitHubOAuthError as exc:
        logger.warning("GitHub OAuth not configured: %s", exc)
        msg = (
            "GitHub sign-in is not yet configured. Set GITHUB_CLIENT_ID and "
            "GITHUB_CLIENT_SECRET in backend/.env"
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


@router.get("/github/callback")
async def github_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
    lqms_oauth_state: str | None = Cookie(default=None),
) -> Response:
    """Handle the GitHub redirect: exchange code, verify org, establish a session."""
    settings = get_settings()
    oauth = get_github_oauth_service()
    session_store = get_session_store()

    login_page = settings.frontend_login_url
    dashboard_page = settings.frontend_dashboard_url

    if error:
        err_msg = error_description or error
        logger.warning("GitHub OAuth authorization error: %s", err_msg)
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
        access_token = await oauth.exchange_code_for_token(code=code)
        profile = await oauth.get_user_profile(access_token=access_token)
    except Exception as exc:  # noqa: BLE001 - normalized to a user-safe redirect
        logger.error("GitHub OAuth exchange or profile fetch failed: %s", exc)
        target = f"{login_page}?auth=error&error=exchange_failed&message={urllib.parse.quote('Failed to authenticate with GitHub')}"
        return RedirectResponse(url=target, status_code=status.HTTP_302_FOUND)

    username = str(profile.get("login", "")).strip()
    email = str(profile.get("email") or "").strip()
    if not email:
        emails = await oauth.get_user_emails(access_token=access_token)
        primary = next((e.get("email") for e in emails if e.get("primary")), None)
        email = primary or (emails[0].get("email") if emails else "")

    is_member, org_name = await oauth.verify_org_membership(access_token=access_token, username=username)
    if not is_member:
        logger.warning("GitHub user '%s' is not a member of allowed org '%s'", username, settings.github_allowed_org)
        msg = f"Your GitHub account (@{username}) is not authorized for this application."
        target = f"{login_page}?auth=error&error=unauthorized_org&message={urllib.parse.quote(msg)}"
        return RedirectResponse(url=target, status_code=status.HTTP_302_FOUND)

    session = session_store.create_session(
        auth_provider="github",
        user_id=str(profile.get("id", "")),
        user_principal_name=username,
        name=str(profile.get("name") or username).strip(),
        email=email,
        avatar_url=str(profile.get("avatar_url", "")).strip(),
        organization=org_name if org_name != "default" else "",
        copilot_enabled=True,
        plain_access_token=access_token,
        plain_refresh_token="",
    )

    logger.info("GitHub OAuth sign-in successful: user='%s' org='%s' session_id=%s", username, org_name, session.session_id[:8])

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
