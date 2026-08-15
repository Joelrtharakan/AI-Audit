from fastapi import APIRouter
from app.config import get_settings
from app.services.ollama_client import OllamaClient

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/health/llm")
async def health_llm() -> dict:
    settings = get_settings()
    if settings.llm_provider == "ollama":
        client = OllamaClient()
        status = await client.check_health()
        return {
            "provider": "ollama",
            "model": status.get("model"),
            "available": status.get("available", False),
            "model_installed": status.get("model_installed", False),
            "installed_models": status.get("installed_models", []),
        }
    return {
        "provider": settings.llm_provider,
        "available": True,
    }
