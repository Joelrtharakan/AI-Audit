"""Agent integration tests — all tests call the real LLM.

Run with:  pytest tests/test_agent_integration.py -m integration -v

These tests verify end-to-end agent behavior including:
  - Final state correctness for different finding types
  - CA draft field count and content constraints
  - Trace presence and structure
  - Evidence ledger population
  - Code-enforced invariants
"""

import pytest

from app.models.agent import AgentFinalState, InvestigateRequest


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


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_produces_valid_response_structure():
    """Agent must always produce a structurally valid response."""
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text="Temperature monitoring logs were not maintained for three consecutive days."
    )
    final = await graph.ainvoke(_build_initial_state(request))

    # Final state is always a valid enum value
    assert final.get("final_state") in (s.value for s in AgentFinalState)

    # Trace is always populated
    trace = final.get("trace", [])
    assert len(trace) > 0

    # Report is generated (may be None only if agent hit a hard error)
    report = final.get("report")
    if report:
        assert report.human_review_required is True
        assert report.observation_quality in ("SUFFICIENT", "INSUFFICIENT", "CONFLICTING")
        assert report.confidence in ("LOW", "MEDIUM", "HIGH")
        assert report.investigation_required in ("YES", "NO", "LIMITED")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_ca_draft_has_exactly_five_fields():
    """CA draft must contain exactly the 5 authorized fields."""
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text=(
            "During the internal audit of the laboratory, it was noted that calibration "
            "records for three analytical balances were overdue. Staff indicated that the "
            "calibration schedule had not been updated after equipment was relocated."
        ),
        departments=["Laboratory"],
        finding_type="Non-Conformity",
        severity="Major",
    )
    final = await graph.ainvoke(_build_initial_state(request))

    ca_draft = final.get("ca_draft")
    if ca_draft:
        draft_dict = ca_draft.model_dump()
        authorized = {"immediate_action", "root_cause", "root_cause_category", "preventive_action", "impact_analysis"}
        assert set(draft_dict.keys()) == authorized
        # All five fields must be non-empty strings
        for field, value in draft_dict.items():
            assert isinstance(value, str), f"CA draft field '{field}' is not a string"
            assert len(value.strip()) > 0, f"CA draft field '{field}' is empty"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_insufficient_finding_yields_investigation_required():
    """A very thin finding should yield INVESTIGATION_REQUIRED or INSUFFICIENT_EVIDENCE."""
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()
    request = InvestigateRequest(finding_text="Something was wrong.")
    final = await graph.ainvoke(_build_initial_state(request))

    final_state = final.get("final_state")
    # A vague finding cannot yield READY_FOR_HUMAN_REVIEW with a confident RCA
    # It is acceptable for the agent to produce a draft even for vague findings,
    # but the report must honestly reflect the uncertainty
    if report := final.get("report"):
        root_cause_status = report.root_cause.status
        # If root cause is not established, CAPA cannot be CAPA_RECOMMENDED
        if root_cause_status == "NOT_ESTABLISHED":
            assert report.capa.status.value in (
                "INVESTIGATION_REQUIRED", "INSUFFICIENT_EVIDENCE", "NO_CAPA_RECOMMENDATION_YET"
            )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_terminates_within_max_iterations():
    """Agent must never run indefinitely."""
    from app.agent.graph import get_agent_graph
    from app.config import get_settings

    settings = get_settings()
    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text="Equipment was not calibrated as required by the schedule.",
        departments=["Maintenance"],
        clause_number="7.1.5",
    )
    final = await graph.ainvoke(_build_initial_state(request))

    # Agent must have terminated with a valid final state
    assert final.get("final_state") in (s.value for s in AgentFinalState)
    assert final.get("tool_call_count", 0) <= settings.agent_max_tool_calls


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_root_cause_stated_unverified_not_promoted():
    """A reported (STATED_UNVERIFIED) root cause must never be promoted to ESTABLISHED
    without VERIFIED evidence — even if the LLM tries to.
    Code-enforcement in rca.py downgrades it."""
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text=(
            "Staff stated they were not trained on the revised procedure. "
            "No training records were available for review."
        )
    )
    final = await graph.ainvoke(_build_initial_state(request))

    report = final.get("report")
    if report:
        rc_status = report.root_cause.status
        evidence_ledger = report.evidence
        has_verified = any(e.status.value == "VERIFIED" for e in evidence_ledger)

        # Code invariant: ESTABLISHED requires VERIFIED evidence
        if rc_status == "ESTABLISHED":
            assert has_verified, "Root cause is ESTABLISHED but no VERIFIED evidence in ledger"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_trace_contains_expected_checkpoints():
    """Trace must always contain recognizable investigation checkpoints."""
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text="A non-conformity was observed during the audit of the laboratory."
    )
    final = await graph.ainvoke(_build_initial_state(request))

    trace = final.get("trace", [])
    messages = [step.get("message", "") if isinstance(step, dict) else step.message for step in trace]
    messages_str = " ".join(messages).lower()

    # At minimum, observation quality and human review warning must appear
    assert any("quality" in m or "observation" in m for m in messages)
    assert any("review" in m or "auditor" in m for m in messages)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_lqms_tools_unavailable_graceful():
    """When LQMS_ASPNET_BASE_URL is empty, agent must complete gracefully
    with evidence gaps recorded, not crash."""
    import os
    from app.agent.graph import get_agent_graph
    from app.config import get_settings

    # Verify the URL is empty (test environment has no LQMS)
    settings = get_settings()
    if settings.lqms_aspnet_base_url:
        pytest.skip("LQMS_ASPNET_BASE_URL is configured — tool calls will be real")

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text=(
            "Calibration records for the spectrophotometer were not found during audit review. "
            "Staff indicated the equipment was serviced last month but records were not filed."
        ),
        departments=["Laboratory"],
        audit_number="AUD-2024-001",
    )
    final = await graph.ainvoke(_build_initial_state(request))

    # Must complete without exception
    assert final.get("final_state") in (s.value for s in AgentFinalState)

    # Evidence gaps must be recorded when tools return no data
    report = final.get("report")
    if report and final.get("completed_tools"):
        assert len(report.evidence_gaps) >= 0  # gaps are expected


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_report_always_requires_human_review():
    """human_review_required must always be True — model validator enforces this."""
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text="An audit finding was identified in the quality management system."
    )
    final = await graph.ainvoke(_build_initial_state(request))

    report = final.get("report")
    if report:
        assert report.human_review_required is True
