from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Any

from app.config import get_settings
from app.services import llm_metrics
from app.services.llm import get_llm_provider

router = APIRouter(tags=["health"])

_MICROSOFT_PROVIDER_ALIASES = ("microsoft_copilot", "microsoft-copilot", "m365_copilot", "m365-copilot", "copilot")
_GITHUB_PROVIDER_ALIASES = ("github_copilot", "github-copilot")


class SwitchProviderRequest(BaseModel):
    provider: str
    model: str | None = None
    token: str | None = None


def _active_model(settings) -> str:
    from app.services.llm.execution import resolve_execution_config
    try:
        return resolve_execution_config().model
    except Exception:  # noqa: BLE001
        return settings.llm_model or settings.ollama_model


def _is_microsoft(provider: str) -> bool:
    return provider.strip().lower() in _MICROSOFT_PROVIDER_ALIASES


def _is_github(provider: str) -> bool:
    return provider.strip().lower() in _GITHUB_PROVIDER_ALIASES


def _has_any_copilot_token(settings, user_session) -> bool:
    return bool(
        settings.microsoft_copilot_access_token
        or settings.copilot_github_token
        or (user_session and user_session.copilot_enabled)
    )


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/health/llm")
async def health_llm() -> dict:
    settings = get_settings()
    provider = get_llm_provider()
    status = await provider.check_health()
    has_token = bool(settings.microsoft_copilot_access_token)

    return {
        "status": "ok" if status.get("available") else "degraded",
        "provider": settings.llm_provider,
        "model": _active_model(settings),
        "available": status.get("available", False),
        "has_microsoft_token": has_token,
        "has_copilot_token": has_token,  # backwards-compatible alias for the dashboard UI
        **status,
    }


@router.get("/api/v1/provider")
@router.get("/health/provider")
async def get_provider_status(request: Request) -> dict[str, Any]:
    """Get active LLM provider, current model, readiness, and list of supported providers."""
    settings = get_settings()

    from app.routers.auth import apply_user_copilot_token
    user_session = apply_user_copilot_token(request)

    provider = get_llm_provider()
    status = await provider.check_health()
    has_token = _has_any_copilot_token(settings, user_session)

    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "model": _active_model(settings),
        "available": status.get("available", False) if settings.llm_provider == "ollama" else has_token,
        "has_microsoft_token": bool(settings.microsoft_copilot_access_token),
        "has_github_token": bool(settings.copilot_github_token),
        "has_copilot_token": has_token,
        "authenticated": bool(user_session),
        "auth_provider": user_session.auth_provider if user_session else None,
        "details": status,
        "supported_providers": [
            {
                "id": "ollama",
                "name": "Ollama (Local / Development)",
                "description": "Local offline inference with Qwen3:8b",
                "model": settings.ollama_model,
            },
            {
                "id": "microsoft_copilot",
                "name": "Microsoft 365 Copilot",
                "description": "Microsoft Graph Copilot Chat API via LiteLLM (Entra ID sign-in)",
                "model": "m365-copilot",
                "has_token": bool(settings.microsoft_copilot_access_token),
            },
            {
                "id": "github_copilot",
                "name": "GitHub Copilot",
                "description": "GitHub Copilot SDK (GitHub OAuth sign-in)",
                "model": settings.copilot_model,
                "has_token": bool(settings.copilot_github_token),
            },
        ],
    }


@router.post("/api/v1/provider")
@router.post("/health/provider")
async def switch_provider(payload: SwitchProviderRequest, request: Request) -> dict[str, Any]:
    """Dynamically switch the active LLM provider (Ollama vs Microsoft 365 Copilot)."""
    settings = get_settings()
    requested = payload.provider.strip().lower()

    if requested not in ("ollama", *_MICROSOFT_PROVIDER_ALIASES, *_GITHUB_PROVIDER_ALIASES, "groq", "openrouter", "gemini"):
        return {
            "status": "error",
            "message": f"Unsupported provider: '{payload.provider}'. Must be 'ollama', 'microsoft_copilot', or 'github_copilot'.",
            "provider": settings.llm_provider,
        }

    target = "github_copilot" if _is_github(requested) else ("microsoft_copilot" if _is_microsoft(requested) else requested)
    settings.llm_provider = target

    from app.routers.auth import apply_user_copilot_token
    user_session = apply_user_copilot_token(request)
    # apply_user_copilot_token may re-set llm_provider from the session; an
    # explicit unauthenticated switch request still wins.
    settings.llm_provider = target
    if user_session is None and payload.token is not None:
        if _is_github(target):
            settings.copilot_github_token = payload.token.strip()
        elif _is_microsoft(target):
            settings.microsoft_copilot_access_token = payload.token.strip()

    _provider_default = {
        "ollama": settings.ollama_model,
        "github_copilot": settings.copilot_model or "auto",
        "microsoft_copilot": "m365-chat",
        "groq": settings.groq_model,
        "gemini": settings.gemini_model,
        "openrouter": settings.openrouter_model,
    }.get(target, settings.ollama_model)
    if payload.model:
        settings.llm_model = payload.model.strip()
        if target == "ollama":
            settings.ollama_model = payload.model.strip()
        elif _is_github(target):
            settings.copilot_model = payload.model.strip()
    else:
        # No explicit model on a provider switch -> reset LLM_MODEL to the new
        # provider's default so a stale model from the previous provider does
        # not leak into the new route.
        settings.llm_model = _provider_default

    provider = get_llm_provider()
    status = await provider.check_health()
    has_token = _has_any_copilot_token(settings, user_session)

    return {
        "status": "ok",
        "message": f"Successfully switched active LLM provider to '{target}'",
        "provider": target,
        "model": _active_model(settings),
        "available": status.get("available", False) if target == "ollama" else has_token,
        "has_microsoft_token": bool(settings.microsoft_copilot_access_token),
        "has_github_token": bool(settings.copilot_github_token),
        "has_copilot_token": has_token,
        "authenticated": bool(user_session),
        "details": status,
    }


@router.get("/health/llm-metrics")
async def health_llm_metrics() -> dict:
    """Read-only operational metrics for the LLM synthesis boundary."""
    return {"status": "ok", "metrics": llm_metrics.aggregated()}
