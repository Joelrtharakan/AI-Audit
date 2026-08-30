import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.auth import require_internal_api_key
from app.models.analysis import AnalyzeFindingResponse
from app.models.requests import AnalyzeFindingRequest
from app.services.finding_analysis_service import FindingAnalysisService
from app.services.llm_client import LLMError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analyze"], dependencies=[Depends(require_internal_api_key)])


def get_finding_analysis_service() -> FindingAnalysisService:
    return FindingAnalysisService()


@router.post("/analyze-finding", response_model=AnalyzeFindingResponse)
async def analyze_finding(
    payload: AnalyzeFindingRequest,
    request: Request,
    service: FindingAnalysisService = Depends(get_finding_analysis_service),
) -> AnalyzeFindingResponse:
    """Sprint 1 LLM-first entry point: reasons about a brand-new finding with no
    RAG dependency. Never writes to the LQMS; always returns a clean error on
    LLM failure rather than partial/guessed data."""
    from app.config import get_settings
    settings = get_settings()

    from app.routers.auth import apply_user_copilot_token
    apply_user_copilot_token(request)

    # Freeze ONE provider + ONE model for this analysis request.
    from app.services.llm.execution import (
        UnknownProviderError,
        begin_request,
        resolve_execution_config,
    )
    try:
        exec_config = resolve_execution_config(
            provider=(payload.llm_provider or settings.llm_provider),
            model=(getattr(payload, "llm_model", "") or None),
        )
    except UnknownProviderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    rid = begin_request(exec_config)
    logger.info("ANALYZE START request_id=%s route=%s", rid, exec_config.public_dict())

    try:
        return await service.analyze(payload)
    except LLMError as exc:
        logger.error("Finding analysis failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="AI service unavailable. Existing form data has not been changed.",
        ) from exc
