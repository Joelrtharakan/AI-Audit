"""POST /api/v1/investigate — the agentic investigation endpoint.

Accepts a full finding/CA context, runs the LangGraph investigation agent,
and returns:
  - The investigation report
  - A 5-field CA draft (requires auditor review)
  - The agent trace
  - The final agent state

Security:
  - Same X-Internal-Api-Key auth as /api/v1/analyze-finding
  - All finding data is passed as observation input, never as instructions
  - Write-permission boundary enforced in permissions.py (not in prompt)
  - Agent never modifies the production LQMS
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException

from app.agent.cache import compute_cache_key, get_cached_analysis, set_cached_analysis
from app.agent.graph import get_agent_graph
from app.agent.state import AgentState
from app.auth import require_internal_api_key
from app.config import get_settings
from app.models.agent import (
    AgentFinalState,
    AgentTraceStep,
    AiMetadata,
    InvestigateRequest,
    InvestigateResponse,
)
from app.services.llm_client import (
    AllLLMProvidersUnavailableError,
    LLMError,
    NoLLMProviderConfiguredError,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["investigate"],
    dependencies=[Depends(require_internal_api_key)],
)


@router.post("/investigate", response_model=InvestigateResponse)
async def investigate_finding(payload: InvestigateRequest) -> InvestigateResponse:
    """Run the agentic investigation pipeline on a finding.

    The agent:
      1. Assesses observation quality and extracts structured data
      2. Plans which LQMS tools to call (if any)
      3. Executes tools and records evidence
      4. Performs RCA, 5-Why, impact, and CAPA analysis
      5. Self-reviews the analysis
      6. Generates an investigation report and 5-field CA draft

    The agent NEVER modifies the LQMS. Final authority rests with the auditor.
    """
    settings = get_settings()

    # Check cache first for duplicate request instant response
    depts_str = ",".join(payload.departments) if payload.departments else ""
    cache_key = compute_cache_key(
        payload.finding_text, depts_str, payload.audit_criteria or ""
    )
    cached = get_cached_analysis(cache_key)
    if cached:
        logger.info("Cache HIT for finding investigation: %s", cache_key[:12])
        return InvestigateResponse(**cached)

    graph = get_agent_graph()

    # Initial state
    initial_state: AgentState = {
        "request": payload,
        "iteration_count": 0,
        "tool_call_count": 0,
        "critic_iteration": 0,
        "observation_quality": None,
        "extraction": None,
        "investigation_plan": None,
        "needs_investigation": False,
        "planned_tools": [],
        "completed_tools": [],
        "current_tool": None,
        "tool_results": {},
        "evidence_ledger": [],
        "evidence_gaps": [],
        "root_cause": None,
        "contributing_factors": [],
        "five_why": None,
        "impact_assessment": None,
        "capa_analysis": None,
        "critic_approved": False,
        "critic_feedback": None,
        "critic_send_back": False,
        "report": None,
        "ca_draft": None,
        "final_state": None,
        "trace": [AgentTraceStep.ok("Investigation agent started")],
        "errors": [],
    }

    try:
        final_state_dict = await asyncio.wait_for(
            graph.ainvoke(initial_state),
            timeout=settings.agent_overall_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.error("Investigation agent timed out after %ss", settings.agent_overall_timeout_seconds)
        raise HTTPException(
            status_code=504,
            detail=(
                f"Investigation agent timed out after {settings.agent_overall_timeout_seconds}s. "
                "Existing LQMS form data has not been changed."
            ),
        )
    except NoLLMProviderConfiguredError as exc:
        logger.error("Investigation agent error: No LLM provider is configured: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={
                "status": "degraded",
                "investigation_completed": False,
                "reason": "no_llm_provider_configured",
                "message": str(exc),
            },
        )
    except AllLLMProvidersUnavailableError as exc:
        logger.error("Investigation agent error: All LLM providers unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={
                "status": "degraded",
                "investigation_completed": False,
                "reason": "all_llm_providers_unavailable",
                "providers": exc.provider_statuses,
            },
        )
    except LLMError as exc:
        logger.error("Investigation agent LLM failure: %s", exc)
        raise HTTPException(
            status_code=502,
            detail={
                "status": "degraded",
                "investigation_completed": False,
                "reason": "llm_error",
                "message": str(exc),
            },
        )
    except Exception as exc:
        logger.error("Investigation agent unexpected error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "degraded",
                "investigation_completed": False,
                "reason": "unexpected_error",
                "message": str(exc),
            },
        )

    model_name = (
        settings.ollama_model
        if settings.llm_provider == "ollama"
        else settings.openrouter_model
    )

    resp = InvestigateResponse(
        final_state=final_state_dict.get("final_state") or AgentFinalState.INVESTIGATION_REQUIRED,
        report=final_state_dict.get("report"),
        ca_draft=final_state_dict.get("ca_draft"),
        trace=final_state_dict.get("trace", []),
        ai_metadata=AiMetadata(
            model=model_name,
            prompt_version=settings.analysis_prompt_version,
            generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            suggestion_id=str(uuid.uuid4()),
        ),
    )

    # Store in cache
    set_cached_analysis(cache_key, resp.model_dump())
    return resp
