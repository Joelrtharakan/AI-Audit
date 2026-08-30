"""End-to-end tests confirming the LLM router's provider-selection metadata
(provider_used, fallback_used, provider_attempts) surfaces correctly on the
final InvestigationReport, via the actual live graph node
(core_synthesis_node) -- not a mocked-away shortcut. Complements
test_llm_router.py (pure router unit tests) and
test_analysis_mode_degraded.py (existing DEGRADED-mode coverage).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.skip(reason="Multi-provider fan-out/failover router removed from the inference path: one investigation = one provider + one model via the LiteLLM boundary. See app/services/llm/execution.py + providers/litellm_provider.py.")

from app.models.agent import (
    AgentTraceStep,
    CanonicalFindingState,
    EvidenceItem,
    EvidenceStatus,
    InvestigateRequest,
)
from app.services.groq_client import GroqRateLimitedError
from app.services.llm_router import get_last_call_metadata, reset_circuits_for_testing


@pytest.fixture(autouse=True)
def _reset_router_state():
    reset_circuits_for_testing()
    yield
    reset_circuits_for_testing()


def _build_state(finding_text: str) -> dict:
    ledger = [EvidenceItem(claim="the required review was not completed", source="AUDITOR_FINDING", status=EvidenceStatus.VERIFIED)]
    canonical = CanonicalFindingState(
        raw_finding=finding_text,
        observed_deviation="the required review — not completed",
        deviation_condition="not completed",
        facts=["the required review was not completed"],
        affected_objects=["the required review"],
    )
    return {
        "request": InvestigateRequest(finding_text=finding_text),
        "iteration_count": 0,
        "tool_call_count": 0,
        "critic_iteration": 0,
        "observation_quality": None,
        "extraction": None,
        "canonical_finding_state": canonical,
        "investigation_plan": None,
        "needs_investigation": False,
        "planned_tools": [],
        "completed_tools": [],
        "current_tool": None,
        "tool_results": {},
        "evidence_ledger": ledger,
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
        "trace": [AgentTraceStep.ok("Test started")],
        "errors": [],
    }


_LLM_RESPONSE = json.dumps({
    "root_cause": {"status": "NOT_ESTABLISHED", "category": "TO_BE_CONFIRMED", "candidate_hypotheses": [], "narrative": "Not established."},
    "five_why": {"steps": [], "is_complete": False, "status_note": "Stopped at evidence boundary"},
    "impact_assessment": {"status": "IMPACT_REQUIRES_ASSESSMENT", "areas": []},
    "capa": {"status": "INVESTIGATION_REQUIRED", "potential_areas": [], "recommended_investigation": []},
    "contributing_factors": [],
    "ca_draft": {
        "immediate_action": "Review the record.",
        "root_cause": "NOT_ESTABLISHED",
        "root_cause_category": "TO_BE_CONFIRMED",
        "preventive_action": "Pending.",
        "impact_analysis": "Pending.",
    },
})


@pytest.mark.asyncio
async def test_core_synthesis_records_provider_used_via_router():
    """core_synthesis_node reads the router's ContextVar metadata right
    after its own chat_completion call -- verify that metadata reaches the
    node's returned state (get_llm_client mocked at the module level here,
    as in every other core_synthesis test, but returning a real-shaped
    router-metadata-populated client is exactly what happens when the
    router itself is the thing get_llm_client() returns in production)."""
    from app.agent.nodes.core_synthesis import core_synthesis_node

    state = _build_state("The required review was not completed for the reporting period.")

    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get_client, \
         patch("app.services.llm_router._last_call_metadata") as mock_ctxvar:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = _LLM_RESPONSE
        mock_get_client.return_value = mock_client
        mock_ctxvar.get.return_value = {
            "provider_used": "openrouter",
            "fallback_used": True,
            "provider_attempts": ["groq", "openrouter"],
        }
        result = await core_synthesis_node(state)

    assert result["analysis_mode"] == "LLM"
    assert result["provider_used"] == "openrouter"
    assert result["fallback_used"] is True
    assert result["provider_attempts"] == ["groq", "openrouter"]


@pytest.mark.asyncio
async def test_full_graph_surfaces_provider_metadata_on_report():
    """Runs the real router (not mocked away) through the actual
    get_agent_graph(), with only the provider CLIENTS mocked at the httpx
    boundary equivalent (_build_provider_client) -- so this exercises the
    router's real failover logic end-to-end inside the live graph, not just
    core_synthesis_node in isolation."""
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text="The required review was not completed for the reporting period."
    )
    initial_state = {
        "request": request,
        "iteration_count": 0,
        "tool_call_count": 0,
        "critic_iteration": 0,
        "evidence_ledger": [],
        "evidence_gaps": [],
        "trace": [],
        "errors": [],
        "completed_tools": [],
        "planned_tools": [],
        "tool_results": {},
        "contributing_factors": [],
    }

    groq_mock = AsyncMock()
    groq_mock.chat_completion.side_effect = GroqRateLimitedError("429", retry_after=30.0)
    openrouter_mock = AsyncMock()
    openrouter_mock.chat_completion.return_value = _LLM_RESPONSE

    def build(name):
        return {"groq": groq_mock, "openrouter": openrouter_mock}.get(name, AsyncMock())

    with patch("app.services.llm_router._provider_configured", side_effect=lambda n: n in ("groq", "openrouter")), \
         patch("app.services.llm_router._build_provider_client", side_effect=build):
        from app.config import get_settings
        settings = get_settings()
        orig = settings.llm_provider
        settings.llm_provider = "groq"
        try:
            result = await graph.ainvoke(initial_state)
        finally:
            settings.llm_provider = orig

    report = result.get("report")
    assert report is not None
    assert report.analysis_mode == "LLM"
    assert report.provider_used == "openrouter"
    assert report.fallback_used is True
    assert "openrouter" in report.provider_attempts
    # NOTE: this does NOT assert "groq" in report.provider_attempts. The
    # graph makes several LLM calls before core_synthesis (understanding,
    # investigation planning); the FIRST of those opens Groq's circuit, so
    # by the time core_synthesis makes ITS call, Groq is already known-down
    # and is correctly skipped without a fresh attempt -- report.
    # provider_attempts reflects core_synthesis's own call specifically.
    # This is Section 13's requirement in action ("the following node must
    # NOT independently attempt Groq again"), not a gap: Groq WAS attempted
    # once for this request (by an earlier node), just not redundantly by
    # every single node.
    assert groq_mock.chat_completion.call_count <= 1, "Groq must be attempted at most once per request, not once per node"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_live_groq_openrouter_unavailable_gemini_real_success():
    """The critical live test (Section 19): Groq and OpenRouter are made
    unavailable deterministically (mocked to fail instantly, per the task's
    own instruction not to hammer real rate-limited providers), and the
    REAL Gemini API handles the request -- unmocked, live network call.
    Requires GOOGLE_API_KEY configured; deselected by default via
    `-m "not integration"`.

    This proves the exact chain: Groq failure -> OpenRouter failure ->
    Gemini REAL SUCCESS -> analysis_mode=LLM (not DEGRADED)."""
    from app.agent.graph import get_agent_graph
    from app.services.openrouter_client import OpenRouterInvalidResponseError

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text="The required weekly review record was not completed for the reporting period."
    )
    initial_state = {
        "request": request,
        "iteration_count": 0,
        "tool_call_count": 0,
        "critic_iteration": 0,
        "evidence_ledger": [],
        "evidence_gaps": [],
        "trace": [],
        "errors": [],
        "completed_tools": [],
        "planned_tools": [],
        "tool_results": {},
        "contributing_factors": [],
    }

    groq_mock = AsyncMock()
    groq_mock.chat_completion.side_effect = GroqRateLimitedError("429", retry_after=60.0)
    openrouter_mock = AsyncMock()
    openrouter_mock.chat_completion.side_effect = OpenRouterInvalidResponseError("empty completion")

    real_build = None
    from app.services.llm_router import _build_provider_client as _real_build_provider_client
    real_build = _real_build_provider_client

    def build(name):
        if name == "groq":
            return groq_mock
        if name == "openrouter":
            return openrouter_mock
        return real_build(name)  # gemini: real client, real network call

    with patch("app.services.llm_router._provider_configured", side_effect=lambda n: True), \
         patch("app.services.llm_router._build_provider_client", side_effect=build):
        from app.config import get_settings
        settings = get_settings()
        orig = settings.llm_provider
        settings.llm_provider = "groq"
        try:
            result = await graph.ainvoke(initial_state)
        finally:
            settings.llm_provider = orig

    report = result.get("report")
    assert report is not None
    assert report.analysis_mode == "LLM", (
        f"Gemini succeeded for this call but analysis_mode was not LLM (was {report.analysis_mode}); "
        "a provider failure must never be reported as an analytical failure"
    )
    assert report.provider_used == "gemini"
    assert report.fallback_used is True
    assert report.root_cause is not None
