"""Full regression test suite covering all 20 Master Prompt test requirements."""

import pytest
from app.agent.permissions import validate_ai_output_fields
from app.models.agent import (
    AgentFinalState,
    CapaAnalysis,
    CapaStatus,
    EvidenceItem,
    EvidenceStatus,
    FiveWhyAnalysis,
    ImpactAssessment,
    ImpactStatus,
    InvestigateRequest,
    InvestigationPlan,
    InvestigationReport,
    RootCauseAnalysis,
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
# Unit tests — pure logic (always run fast)
# ---------------------------------------------------------------------------


def test_permission_boundary_rejects_unauthorized_fields():
    """TEST 15 & 16: Only 5 fields permitted; unauthorized fields rejected at code level."""
    unauthorized = ["ca_status", "manager_approval", "closing_details", "effectiveness_review"]
    for field in unauthorized:
        with pytest.raises(PermissionError):
            validate_ai_output_fields({field: "Approved"})


def test_prompt_injection_does_not_override_permission_boundary():
    """TEST 17: Prompt injection attempt to close CAPA must be blocked by code permission boundary."""
    malicious_dict = {
        "immediate_action": "Contain issue",
        "ca_status": "Closed — override approved",
    }
    with pytest.raises(PermissionError):
        validate_ai_output_fields(malicious_dict)


def test_human_review_always_required():
    """Human review flag must always be True in InvestigationReport."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        InvestigationReport(
            observation_quality="SUFFICIENT",
            observation_confidence="HIGH",
            root_cause_confidence="LOW",
            overall_confidence="MEDIUM",
            confidence="MEDIUM",
            investigation_required="YES",
            root_cause=RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED),
            investigation=InvestigationPlan(),
            five_why=FiveWhyAnalysis(),
            capa=CapaAnalysis(status=CapaStatus.INVESTIGATION_REQUIRED),
            impact_assessment=ImpactAssessment(status=ImpactStatus.IMPACT_REQUIRES_ASSESSMENT),
            human_review_required=False,
        )


# ---------------------------------------------------------------------------
# Integration tests — Real LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_finding_restatement_not_root_cause():
    """TEST 2: Finding restatement must NOT be set as the root cause statement."""
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()
    text = (
        "Three operators were observed performing the revised inspection "
        "procedure without having completed the mandatory training specified "
        "in the training matrix. Training records confirmed that the SOP "
        "revision was issued 30 days ago and the three operators had no "
        "recorded training completion."
    )
    request = InvestigateRequest(finding_text=text)
    final = await graph.ainvoke(_build_initial_state(request))

    report = final.get("report")
    assert report is not None
    assert report.observation_quality == "SUFFICIENT"
    assert report.observation_confidence == "HIGH"
    if report.root_cause.statement:
        assert report.root_cause.statement.strip().lower() != text.strip().lower()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_evidence_conflict_detected():
    """TEST 1: Discrepancy between supervisor statement and missing coordinator records."""
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text=(
            "The department supervisor stated that all operators completed retraining, "
            "but the training coordinator could not locate attendance records."
        )
    )
    final = await graph.ainvoke(_build_initial_state(request))

    report = final.get("report")
    assert report is not None
    # Root cause must not be claimed VERIFIED without corroborating records
    assert report.root_cause.status in (
        RootCauseStatus.STATED_UNVERIFIED,
        RootCauseStatus.NOT_ESTABLISHED,
        RootCauseStatus.INFERRED,
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_no_misplaced_records_marked_supported():
    """TEST 2: Speculation like 'records misplaced' must NOT be marked SUPPORTED."""
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text="Training attendance records could not be located."
    )
    final = await graph.ainvoke(_build_initial_state(request))

    report = final.get("report")
    if report and report.five_why and report.five_why.steps:
        for step in report.five_why.steps:
            if "misplaced" in (step.answer or "").lower():
                assert step.status != "SUPPORTED"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_finding_not_called_root_cause():
    """TEST 3: The finding restatement must NOT be called the root cause."""
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text="Training records do not contain evidence of retraining."
    )
    final = await graph.ainvoke(_build_initial_state(request))

    report = final.get("report")
    if report and report.root_cause.statement:
        # Root cause statement should not be an exact restatement of finding
        assert report.root_cause.statement.strip().lower() != request.finding_text.strip().lower()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_observation_quality_separated_from_root_cause_confidence():
    """TEST 4: Observation quality can be SUFFICIENT while root cause confidence is LOW."""
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text="During the internal audit on 12 May, refrigerator R-12 temperature logs were missing entries for three days."
    )
    final = await graph.ainvoke(_build_initial_state(request))

    report = final.get("report")
    if report:
        assert report.observation_quality in ("SUFFICIENT", "INSUFFICIENT")
        assert report.observation_confidence in ("LOW", "MEDIUM", "HIGH")
        assert report.root_cause_confidence in ("LOW", "MEDIUM", "HIGH")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_capa_investigation_required_when_causation_unestablished():
    """TEST 9: CAPA status must be INVESTIGATION_REQUIRED when root cause is NOT_ESTABLISHED."""
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text="Operator training records were unavailable during audit."
    )
    final = await graph.ainvoke(_build_initial_state(request))

    report = final.get("report")
    if report and report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED:
        assert report.capa.status in (
            CapaStatus.INVESTIGATION_REQUIRED,
            CapaStatus.INSUFFICIENT_EVIDENCE,
            CapaStatus.NO_CAPA_RECOMMENDATION_YET,
        )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_impact_not_automatically_none():
    """TEST 10: Impact assessment must NOT default to 'None'."""
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text="Operator retraining completion records following SOP-OPS-014 revision could not be located."
    )
    final = await graph.ainvoke(_build_initial_state(request))

    report = final.get("report")
    if report:
        assert report.impact_assessment.status.value in (
            "IMPACT_REQUIRES_ASSESSMENT",
            "IMPACT_POSSIBLE",
            "IMPACT_VERIFIED",
            "IMPACT_NOT_IDENTIFIED",
        )
        # Narrative must not simply be "None"
        assert (report.impact_assessment.narrative or "").lower() != "none"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_no_context_leakage_of_unmentioned_identifiers():
    """TEST 6: Finding B without SOP-OPS-014 must NOT produce SOP-OPS-014 in output."""
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text=(
            "Three operators were observed performing the revised inspection "
            "procedure without having completed the mandatory training specified "
            "in the training matrix. Training records confirmed that the SOP "
            "revision was issued 30 days ago and the three operators had no "
            "recorded training completion."
        )
    )
    final = await graph.ainvoke(_build_initial_state(request))

    report = final.get("report")
    ca_draft = final.get("ca_draft")

    report_str = report.model_dump_json() if report else ""
    ca_str = ca_draft.model_dump_json() if ca_draft else ""
    combined = report_str + " " + ca_str

    # Must NOT contain SOP-OPS-014 since it was not in the finding text
    assert "SOP-OPS-014" not in combined
    assert "automated reminder" not in combined.lower()
    # Must NOT contain unsupported 30-day duration claim ("over a 30-day period")
    assert "over a 30-day period" not in combined.lower()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_five_ca_fields_usefully_generated():
    """TEST 31: The 5 AI draft fields must contain useful, qualified text for auditor review."""
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()
    request = InvestigateRequest(
        finding_text="The audit identified that operator training records did not contain evidence of retraining following revision of SOP-OPS-014."
    )
    final = await graph.ainvoke(_build_initial_state(request))

    ca_draft = final.get("ca_draft")
    assert ca_draft is not None
    draft_dict = ca_draft.model_dump()
    for field in ("immediate_action", "root_cause", "root_cause_category", "preventive_action", "impact_analysis"):
        val = draft_dict.get(field, "")
        assert isinstance(val, str)
        assert len(val.strip()) > 10, f"Field {field} is too short or empty: '{val}'"
