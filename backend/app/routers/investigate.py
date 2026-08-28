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
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request

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

# (marker text, label) -- matched against AgentTraceStep.message to bound
# each node's wall-clock span using the trace's own timestamps, rather than
# adding a second parallel timing-collection mechanism through every node.
_PERF_MARKERS: list[tuple[str, str]] = [
    ("Finding extracted", "Understanding/Extraction"),
    ("Deterministic semantic extraction recovered", "Understanding/Extraction"),
    ("Extraction LLM call failed", "Understanding/Extraction"),
    ("No LQMS tool investigation needed", "Investigation Planning"),
    ("Investigation plan created", "Investigation Planning"),
    ("Consolidated core synthesis completed", "Core Synthesis"),
    ("Core synthesis transitioned to deterministic", "Core Synthesis"),
    ("compact recovery call produced this analysis", "Core Synthesis"),
    ("Core synthesis error", "Core Synthesis"),
    ("Deterministic critic firewall", "Critic"),
    ("Critic review", "Critic"),
    ("Critic node failed", "Critic"),
    ("Investigation report generated", "Report Generation"),
    ("Final evidence verification", "Final Validation"),
]


def _log_performance_summary(trace: list, pipeline_ms: int, analysis_mode: str | None) -> None:
    """PHASE 16: logs a PERFORMANCE SUMMARY derived from the trace's own
    timestamps (every AgentTraceStep already carries one) -- avoids adding a
    second, parallel timing-collection path through every node just to
    produce this log line."""
    if not trace:
        logger.info("PERFORMANCE SUMMARY total_pipeline_ms=%d analysis_mode=%s (no trace)", pipeline_ms, analysis_mode)
        return
    try:
        timestamps = [dt.datetime.fromisoformat(t.timestamp) for t in trace if t.timestamp]
    except Exception:
        timestamps = []
    spans: dict[str, int] = {}
    prev_time = timestamps[0] if timestamps else None
    prev_label = "Understanding/Extraction"
    for step, ts in zip(trace, timestamps):
        label = next((lbl for marker, lbl in _PERF_MARKERS if marker in step.message), None)
        if label and prev_time is not None:
            spans[prev_label] = spans.get(prev_label, 0) + int((ts - prev_time).total_seconds() * 1000)
            prev_time = ts
            prev_label = label
    lines = [f"  {name}: {ms} ms" for name, ms in spans.items()]
    logger.info(
        "PERFORMANCE SUMMARY\n%s\n  Total: %d ms\n  analysis_mode: %s",
        "\n".join(lines) if lines else "  (insufficient trace markers to break down)",
        pipeline_ms,
        analysis_mode,
    )


@router.post("/investigate", response_model=InvestigateResponse)
async def investigate_finding(
    payload: InvestigateRequest,
    request: Request,
) -> InvestigateResponse:
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
    if payload.llm_provider:
        settings.llm_provider = payload.llm_provider.strip().lower()

    # Resolve delegated Microsoft Graph token from the authenticated session first,
    # falling back to an explicit token in the payload (local-dev convenience).
    from app.routers.auth import apply_user_copilot_token
    user_session = apply_user_copilot_token(request)
    if user_session is None and payload.microsoft_copilot_access_token:
        settings.microsoft_copilot_access_token = payload.microsoft_copilot_access_token.strip()

    # Check cache first for duplicate request instant response. Keyed on
    # model + prompt_version too, so a model swap or prompt edit can't
    # silently serve a result generated under different configuration.
    depts_str = ",".join(payload.departments) if payload.departments else ""
    if settings.llm_provider == "ollama":
        cache_model = settings.ollama_model
    elif settings.llm_provider in ("microsoft_copilot", "microsoft-copilot", "m365_copilot", "copilot"):
        # The M365 Copilot Chat API exposes no model identifier; use a stable
        # literal so the cache key still changes if the provider changes.
        cache_model = "m365-copilot"
    else:
        cache_model = settings.openrouter_model
    cache_key = compute_cache_key(
        payload.finding_text,
        depts_str,
        payload.audit_criteria or "",
        model=cache_model,
        prompt_version=settings.analysis_prompt_version,
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

    pipeline_t0 = time.monotonic()
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

    pipeline_ms = int((time.monotonic() - pipeline_t0) * 1000)
    _log_performance_summary(final_state_dict.get("trace", []), pipeline_ms, final_state_dict.get("analysis_mode"))

    # Section 20/23 fail-closed enforcement: final_evidence_verification_node
    # sets validation_failure when a BLOCKER invariant fired or the
    # structural quality gate graded FAIL. Previously this flag was written
    # to AgentState and never read anywhere outside app/agent/ — a failed
    # gate had zero effect on the HTTP response, which was serialized
    # exactly as if validation had passed. This was audited and confirmed
    # by source inspection (validation_failure had zero references outside
    # app/agent/ before this fix), not assumed.
    _validation_failure = final_state_dict.get("validation_failure")
    if _validation_failure:
        logger.error("Investigation failed structural validation gate: %s", _validation_failure)
        raise HTTPException(
            status_code=422,
            detail={
                "status": "validation_failed",
                "investigation_completed": False,
                "reason": "structural_quality_gate_failed",
                "message": _validation_failure,
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

    # Store in cache -- but never a DEGRADED result. Caching a transient
    # provider failure would otherwise permanently poison every subsequent
    # request for the same finding, even once Ollama recovers (this is
    # exactly the "Cache HIT for finding investigation" trap: a prior
    # DEGRADED response getting replayed forever instead of being retried).
    report = resp.report
    if report is not None and getattr(report, "analysis_mode", "LLM") == "DEGRADED":
        logger.info("Not caching DEGRADED analysis for finding: %s", cache_key[:12])
    else:
        set_cached_analysis(cache_key, resp.model_dump())
    return resp
