from fastapi import APIRouter
from app.config import get_settings
from app.services import llm_metrics
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


@router.get("/health/llm-metrics")
async def health_llm_metrics() -> dict:
    """Read-only operational metrics for the LLM synthesis boundary
    (app.services.llm_metrics). Reads the existing in-memory aggregate
    state only -- performs no Ollama call, no investigation, no
    computation beyond `llm_metrics.aggregated()` itself (Sections 1/9:
    integrate into the existing health router rather than a second
    metrics architecture; never block on a live provider call the way
    /health/llm intentionally does). Contains only counts, latencies, and
    token numbers -- never finding text, evidence text, claims, prompts,
    or LLM responses (Section 2); `llm_metrics.record_execution` never
    accepts that content in the first place, so there is nothing here to
    filter out.
    """
    return {"status": "ok", "metrics": llm_metrics.aggregated()}
