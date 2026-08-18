from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Any

from app.config import get_settings
from app.services import llm_metrics
from app.services.llm import get_llm_provider

router = APIRouter(tags=["health"])


class SwitchProviderRequest(BaseModel):
    provider: str
    model: str | None = None
    token: str | None = None


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/health/llm")
async def health_llm() -> dict:
    settings = get_settings()
    provider = get_llm_provider()
    status = await provider.check_health()
    active_model = settings.copilot_model if settings.llm_provider == "copilot" else settings.ollama_model

    return {
        "status": "ok" if status.get("available") else "degraded",
        "provider": settings.llm_provider,
        "model": active_model,
        "available": status.get("available", False),
        "has_copilot_token": bool(settings.copilot_github_token),
        **status,
    }


@router.get("/api/v1/provider")
@router.get("/health/provider")
async def get_provider_status(request: Request) -> dict[str, Any]:
    """Get active LLM provider, current model, readiness, and list of supported providers."""
    settings = get_settings()

    from app.routers.auth import get_current_user_session
    user_session = get_current_user_session(
        lqms_session=request.cookies.get(settings.session_cookie_name),
        authorization=request.headers.get("authorization"),
    )
    if user_session:
        user_token = user_session.get_decrypted_token()
        if user_token:
            settings.copilot_github_token = user_token

    provider = get_llm_provider()
    status = await provider.check_health()
    active_model = settings.copilot_model if settings.llm_provider == "copilot" else settings.ollama_model
    has_token = bool(settings.copilot_github_token or (user_session and user_session.copilot_enabled))

    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "model": active_model,
        "available": status.get("available", False) if settings.llm_provider == "ollama" else has_token,
        "has_copilot_token": has_token,
        "authenticated": bool(user_session),
        "details": status,
        "supported_providers": [
            {
                "id": "ollama",
                "name": "Ollama (Local / Development)",
                "description": "Local offline inference with Qwen3:8b",
                "model": settings.ollama_model,
            },
            {
                "id": "copilot",
                "name": "GitHub Copilot SDK (Production)",
                "description": "Official GitHub Copilot agent runtime",
                "model": settings.copilot_model,
                "has_token": has_token,
            },
        ],
    }


@router.post("/api/v1/provider")
@router.post("/health/provider")
async def switch_provider(payload: SwitchProviderRequest, request: Request) -> dict[str, Any]:
    """Dynamically switch the active LLM provider (Ollama vs Copilot) and optionally set token."""
    from app.services.llm.providers.github_copilot_provider import reset_copilot_clients
    settings = get_settings()
    requested = payload.provider.strip().lower()

    if requested not in ("ollama", "copilot", "github_copilot", "github-copilot", "groq", "openrouter", "gemini"):
        return {
            "status": "error",
            "message": f"Unsupported provider: '{payload.provider}'. Must be 'ollama' or 'copilot'.",
            "provider": settings.llm_provider,
        }

    target = "copilot" if requested in ("github_copilot", "github-copilot") else requested
    settings.llm_provider = target

    from app.routers.auth import get_current_user_session
    user_session = get_current_user_session(
        lqms_session=request.cookies.get(settings.session_cookie_name),
        authorization=request.headers.get("authorization"),
    )
    if user_session:
        user_token = user_session.get_decrypted_token()
        if user_token:
            settings.copilot_github_token = user_token
            reset_copilot_clients()
    elif payload.token is not None:
        settings.copilot_github_token = payload.token.strip()
        reset_copilot_clients()

    if payload.model:
        if target == "copilot":
            settings.copilot_model = payload.model
        elif target == "ollama":
            settings.ollama_model = payload.model

    provider = get_llm_provider()
    status = await provider.check_health()
    active_model = settings.copilot_model if settings.llm_provider == "copilot" else settings.ollama_model
    has_token = bool(settings.copilot_github_token or (user_session and user_session.copilot_enabled))

    return {
        "status": "ok",
        "message": f"Successfully switched active LLM provider to '{target}'",
        "provider": target,
        "model": active_model,
        "available": status.get("available", False) if target == "ollama" else has_token,
        "has_copilot_token": has_token,
        "authenticated": bool(user_session),
        "details": status,
    }


@router.get("/health/llm-metrics")
async def health_llm_metrics() -> dict:
    """Read-only operational metrics for the LLM synthesis boundary."""
    return {"status": "ok", "metrics": llm_metrics.aggregated()}
