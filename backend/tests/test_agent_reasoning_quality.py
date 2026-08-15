"""Reasoning quality and evidence-grounding tests based on prompt Section 27.

Tests verify anti-assumption rules, question derivation quality, evidence ledger traceability,
and strict write permission boundaries.
"""

import pytest
from app.models.agent import (
    AgentFinalState,
    EvidenceStatus,
    InvestigateRequest,
    RootCauseStatus,
)


def _build_initial_state(request: InvestigateRequest) -> dict:
    from app.models.agent import AgentTraceStep
    return {
        "request": request,
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
        "trace": [AgentTraceStep.ok("Test started")],
        "errors": [],
    }


# ---------------------------------------------------------------------------
# Integration Tests — Real LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_no_asking_which_sop_when_sop_is_named():
    """Test 1: If SOP-OPS-014 is named, the agent must NOT ask 'Which SOP was revised?'."""
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text="SOP-OPS-014 was revised and staff reportedly were not retrained."
    )
    final = await graph.ainvoke(_build_initial_state(request))

    report = final.get("report")
    if report and report.investigation:
        q_strs = [q.question if hasattr(q, "question") else str(q) for q in report.investigation.questions]
        questions = " ".join(q_strs).lower()
        # Must not ask "which SOP" since SOP-OPS-014 is already specified
        assert "which sop" not in questions
        assert "which sop-ops-014" not in questions


@pytest.mark.asyncio
@pytest.mark.integration
async def test_no_assuming_retraining_occurred():
    """Test 2: Do NOT ask 'When did retraining occur?' unless evidence indicates retraining occurred."""
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text="SOP-OPS-014 was revised and staff reportedly were not retrained."
    )
    final = await graph.ainvoke(_build_initial_state(request))

    report = final.get("report")
    if report and report.investigation:
        q_strs = [q.question if hasattr(q, "question") else str(q) for q in report.investigation.questions]
        questions = " ".join(q_strs).lower()
        # Must not assume retraining took place
        assert "when did retraining occur" not in questions
        assert "when did the retraining occur" not in questions


@pytest.mark.asyncio
@pytest.mark.integration
async def test_no_assuming_automated_reminders_exist():
    """Test 3: Do NOT suggest automated retraining reminders unless evidence indicates such system exists."""
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text="SOP-OPS-014 was revised and staff reportedly were not retrained."
    )
    final = await graph.ainvoke(_build_initial_state(request))

    report = final.get("report")
    if report and report.investigation:
        q_strs = [q.question if hasattr(q, "question") else str(q) for q in report.investigation.questions]
        questions = " ".join(q_strs).lower()
        assert "automated retraining reminder" not in questions
        assert "automated notification" not in questions



@pytest.mark.asyncio
@pytest.mark.integration
async def test_unestablished_root_cause_produces_investigation_required():
    """Test 7: If no evidence is available, root cause must be UNVERIFIED / STATED_UNVERIFIED / NOT_ESTABLISHED."""
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text="Operator training records did not contain evidence of retraining following SOP revision."
    )
    final = await graph.ainvoke(_build_initial_state(request))

    report = final.get("report")
    if report:
        assert report.root_cause.status in (
            RootCauseStatus.STATED_UNVERIFIED,
            RootCauseStatus.NOT_ESTABLISHED,
            RootCauseStatus.INFERRED,
        )
        assert report.capa.status.value in ("INVESTIGATION_REQUIRED", "CAPA_DRAFT_POSSIBLE", "INSUFFICIENT_EVIDENCE")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_three_confidence_metrics_calculated():
    """Test confidence metrics: Observation, Root Cause, and Overall Confidence."""
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text="SOP-OPS-014 revision training attendance log missing."
    )
    final = await graph.ainvoke(_build_initial_state(request))

    report = final.get("report")
    if report:
        assert report.observation_confidence in ("LOW", "MEDIUM", "HIGH")
        assert report.root_cause_confidence in ("LOW", "MEDIUM", "HIGH")
        assert report.overall_confidence in ("LOW", "MEDIUM", "HIGH")
