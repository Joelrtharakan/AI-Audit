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
    if payload.llm_provider:
        settings.llm_provider = payload.llm_provider.strip().lower()

    # Check user session for Copilot token
    from app.routers.auth import get_current_user_session
    user_session = get_current_user_session(
        lqms_session=request.cookies.get(settings.session_cookie_name),
        authorization=request.headers.get("authorization"),
    )
    if user_session:
        user_token = user_session.get_decrypted_token()
        if user_token:
            from app.services.llm.providers.github_copilot_provider import reset_copilot_clients
            settings.copilot_github_token = user_token
            reset_copilot_clients()
    try:
        return await service.analyze(payload)
    except LLMError as exc:
        logger.error("Finding analysis failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="AI service unavailable. Existing form data has not been changed.",
        ) from exc
