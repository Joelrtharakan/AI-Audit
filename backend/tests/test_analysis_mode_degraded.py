"""End-to-end tests for the analysis_mode / DETERMINISTIC CAUSAL DEGRADATION
architecture, run through the LIVE graph node (core_synthesis_node) with the
LLM call forced to fail — verifying the deterministic fallback path, not any
specific finding's wording. Findings below span unrelated QMS domains
(inspection, document control, calibration, training, environmental
monitoring) to demonstrate the fallback generalizes rather than being tuned
to one case.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.models.agent import (
    AgentTraceStep,
    CanonicalFindingState,
    EvidenceItem,
    EvidenceStatus,
    InvestigateRequest,
)


def _build_state(finding_text: str, verified: list[str], reported: list[str]) -> dict:
    ledger = [EvidenceItem(claim=c, source="AUDITOR_FINDING", status=EvidenceStatus.VERIFIED) for c in verified]
    ledger += [EvidenceItem(claim=c, source="REPORTED_STATEMENT", status=EvidenceStatus.REPORTED) for c in reported]
    canonical = CanonicalFindingState(
        raw_finding=finding_text,
        observed_deviation=verified[0] if verified else "not extracted",
        deviation_condition="not completed",
        facts=verified,
        reported_statements=reported,
        affected_objects=verified[:1],
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


DEGRADED_MODE_DOMAINS = [
    (
        "The scheduled inspection of the storage area was not conducted for the required period.",
        ["the scheduled inspection was not conducted for the required period"],
        ["the area supervisor confirmed the inspection was skipped"],
    ),
    (
        "The document control record shows the wrong revision was used on the production floor.",
        ["the document control record shows the wrong revision was used on the production floor"],
        [],
    ),
    (
        "The calibration for the measuring instrument was overdue by several weeks.",
        ["the calibration for the measuring instrument was overdue by several weeks"],
        [],
    ),
    (
        "The training record for the new analyst did not contain a trainer sign-off.",
        ["the training record for the new analyst did not contain a trainer sign-off"],
        [],
    ),
    (
        "The environmental monitoring check for the cleanroom was not performed during the shift.",
        ["the environmental monitoring check for the cleanroom was not performed during the shift"],
        ["the shift lead confirmed the monitoring check was missed"],
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("finding_text,verified,reported", DEGRADED_MODE_DOMAINS)
async def test_degraded_mode_never_fabricates_root_cause(finding_text, verified, reported):
    from app.agent.nodes.core_synthesis import core_synthesis_node

    state = _build_state(finding_text, verified, reported)

    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.side_effect = RuntimeError("simulated LLM/429 failure")
        mock_get_client.return_value = mock_client
        result = await core_synthesis_node(state)

    # LLM call failed but the deterministic analytical engine produced a
    # complete, evidence-grounded result -> analysis_mode="DETERMINISTIC",
    # never "DEGRADED" and never mislabeled "DEGRADED MODE" in prose.
    # DEGRADED is reserved for when the deterministic engine ALSO fails to
    # produce a safe result.
    assert result["analysis_mode"] == "DETERMINISTIC"
    root_cause = result["root_cause"]
    assert root_cause.status == "NOT_ESTABLISHED"
    assert root_cause.category == "TO_BE_CONFIRMED"
    assert root_cause.narrative
    assert "DEGRADED MODE" not in root_cause.narrative

    five_why = result["five_why"]
    assert len(five_why.steps) >= 1
    for step in five_why.steps:
        assert step.answer != "Analysis unavailable"
        assert "[object Object]" not in (step.answer or "")

    # CAPA must stay pending, never presented as a finalized action.
    capa = result["capa_analysis"]
    assert capa.status.value == "INVESTIGATION_REQUIRED"

    impact = result["impact_assessment"]
    assert impact.narrative
    assert "DEGRADED MODE" not in (impact.narrative or "")


@pytest.mark.asyncio
async def test_degraded_mode_preserves_reported_mechanism_in_five_why():
    """When a REPORTED statement establishes the mechanism, degraded mode
    must still surface it as a distinct Why-step rather than only the raw
    observation — this is the deterministic-degradation requirement, not
    just 'don't crash'."""
    from app.agent.nodes.core_synthesis import core_synthesis_node

    finding_text = "The scheduled inspection of the storage area was not conducted. The area supervisor confirmed the inspection was skipped."
    state = _build_state(
        finding_text,
        verified=["the scheduled inspection was not conducted"],
        reported=["the area supervisor confirmed the inspection was skipped"],
    )

    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.side_effect = RuntimeError("simulated failure")
        mock_get_client.return_value = mock_client
        result = await core_synthesis_node(state)

    five_why = result["five_why"]
    assert len(five_why.steps) == 3
    joined_answers = " ".join(s.answer or "" for s in five_why.steps).lower()
    assert "supervisor confirmed" in joined_answers or "skipped" in joined_answers
    assert five_why.steps[-1].status == "UNKNOWN"


@pytest.mark.asyncio
async def test_llm_success_path_sets_analysis_mode_llm():
    """The normal (non-degraded) path must be distinguishable from degraded
    mode via analysis_mode, not just inferred from content."""
    import json

    from app.agent.nodes.core_synthesis import core_synthesis_node

    finding_text = "The required weekly review record was not completed for the reporting period."
    state = _build_state(finding_text, verified=["the required weekly review record was not completed"], reported=[])

    llm_response = json.dumps({
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

    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client
        result = await core_synthesis_node(state)

    assert result["analysis_mode"] == "LLM"


@pytest.mark.asyncio
async def test_full_graph_survives_total_llm_outage_and_stays_deterministic():
    """Worst-case resilience test: every LLM-calling node fails (simulating
    a total provider outage / persistent 429s), not just core_synthesis.
    The graph must run to completion and report DETERMINISTIC mode (the
    deterministic evidence-grounded engine still produced a safe, complete
    result) rather than crash, silently present a normal-looking LLM report,
    or mislabel a successful deterministic result as DEGRADED."""
    from app.agent.graph import get_agent_graph
    from app.models.agent import InvestigateRequest

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text="The required activity was missed. The technician confirmed it was missed."
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

    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_cs, \
         patch("app.agent.nodes.understanding.get_llm_client") as mock_u, \
         patch("app.services.extraction.get_llm_client") as mock_e, \
         patch("app.agent.nodes.investigation_planner.get_llm_client") as mock_p, \
         patch("app.agent.nodes.critic.get_llm_client") as mock_c:
        for mock in (mock_cs, mock_u, mock_e, mock_p, mock_c):
            client = AsyncMock()
            client.chat_completion.side_effect = RuntimeError("simulated total provider outage")
            mock.return_value = client
        result = await graph.ainvoke(initial_state)

    report = result.get("report")
    assert report is not None, "graph must complete and produce a report even under total LLM outage"
    assert report.analysis_mode == "DETERMINISTIC"
    assert report.root_cause.status == "NOT_ESTABLISHED"
    assert report.human_review_required is True
    # Degraded-mode CAPA must stay pending, never present a finalized action.
    assert report.capa.status.value == "INVESTIGATION_REQUIRED"
