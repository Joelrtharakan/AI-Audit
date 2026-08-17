"""Critical regression cases (A, C, D from the analytical-validation-firewall
hardening request), run through the LIVE graph nodes in the order the actual
graph executes them: core_synthesis_node -> final_evidence_verification_node
(critic and generate_report sit between them in the real graph but don't
mutate root_cause/five_why/capa content, so chaining these two directly
still exercises the real validation path these cases care about).

Each case mocks the LLM to return exactly the kind of output a
non-compliant model might produce, to prove the deterministic validators
catch it regardless of prompt compliance.
"""

from __future__ import annotations

import json
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
        # Real production states always have finding_subject populated by
        # understand_finding_node before core_synthesis ever runs -- these
        # hand-built test states previously omitted it, which only matters
        # now that final_evidence_verification's causal-proposition
        # eligibility layer falls back to re-deriving a subject from raw
        # text when it's missing (a fallback meant for genuinely
        # unstructured input, not a substitute for normal extraction).
        finding_subject=verified[0] if verified else "not extracted",
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


async def _run_synthesis_and_verification(state: dict, llm_response: str) -> dict:
    from app.agent.nodes.core_synthesis import core_synthesis_node
    from app.agent.nodes.final_evidence_verification import final_evidence_verification_node

    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client
        state = await core_synthesis_node(state)

    return await final_evidence_verification_node(state)


@pytest.mark.asyncio
async def test_case_a_confirmed_missed_check_stays_not_established():
    """Case A: 'The technician confirmed the check was missed.' Mechanism
    is established (REPORTED), but that is NOT a root cause -- root cause
    must stay NOT_ESTABLISHED, and no fabricated systemic cause (e.g. a
    forced ESTABLISHED status) may appear even if the mocked LLM tries."""
    finding_text = "The technician confirmed the check was missed."
    state = _build_state(
        finding_text,
        verified=["the required check was not completed"],
        reported=["the technician confirmed the check was missed"],
    )

    llm_response = json.dumps({
        "root_cause": {
            "status": "VERIFIED",  # non-compliant model overclaiming certainty
            "category": "HUMAN_ERROR",
            "candidate_hypotheses": [],
            "narrative": "The technician missed the check.",
        },
        "five_why": {
            "steps": [
                {"level": 1, "question": "Why was the check not completed?", "answer": "The check was not completed.", "status": "VERIFIED"},
            ],
            "is_complete": True,
            "status_note": "Complete",
        },
        "impact_assessment": {"status": "IMPACT_REQUIRES_ASSESSMENT", "areas": []},
        "capa": {"status": "INVESTIGATION_REQUIRED", "potential_areas": [], "recommended_investigation": []},
        "contributing_factors": [],
        "ca_draft": {
            "immediate_action": "Assess the affected check record.",
            "root_cause": "The technician missed the check.",
            "root_cause_category": "HUMAN_ERROR",
            "preventive_action": "Reinforce check completion.",
            "impact_analysis": "Scope requires auditor assessment.",
        },
    })

    result = await _run_synthesis_and_verification(state, llm_response)

    root_cause = result["root_cause"]
    # The analytical validator must downgrade an ESTABLISHED-like claim that
    # only a REPORTED (not VERIFIED-for-cause) statement supports.
    assert root_cause.status != "VERIFIED"

    # The 5-Why chain skipped the explicitly available mechanism (the
    # technician's account) -- the validator must have repaired it in.
    five_why = result["five_why"]
    joined = " ".join((s.answer or "") for s in five_why.steps).lower()
    assert "technician" in joined or "missed" in joined


@pytest.mark.asyncio
async def test_case_c_verified_task_assignment_failure_can_stay_established():
    """Case C: 'Audit trail confirms the responsible user was never
    assigned the task.' This IS a VERIFIED fact directly stating the cause
    -- the validator must NOT blanket-downgrade an ESTABLISHED status when
    real VERIFIED evidence actually supports it."""
    finding_text = "Audit trail confirms the responsible user was never assigned the task."
    state = _build_state(
        finding_text,
        verified=["the audit trail confirms the responsible user was never assigned the task"],
        reported=[],
    )

    llm_response = json.dumps({
        "root_cause": {
            "status": "VERIFIED",
            "category": "TASK_ASSIGNMENT",
            "candidate_hypotheses": [],
            "narrative": "The audit trail confirms the task was never assigned to a responsible user.",
        },
        "five_why": {"steps": [], "is_complete": True, "status_note": "Complete"},
        "impact_assessment": {"status": "IMPACT_REQUIRES_ASSESSMENT", "areas": []},
        "capa": {"status": "INVESTIGATION_REQUIRED", "potential_areas": [], "recommended_investigation": []},
        "contributing_factors": [],
        "ca_draft": {
            "immediate_action": "Assign the task to a responsible user.",
            "root_cause": "Task assignment failure confirmed by audit trail.",
            "root_cause_category": "TASK_ASSIGNMENT",
            "preventive_action": "Implement assignment verification control.",
            "impact_analysis": "Scope requires auditor assessment.",
        },
    })

    result = await _run_synthesis_and_verification(state, llm_response)
    # A VERIFIED evidence item exists (the audit trail claim), so the
    # validator has no basis to downgrade this ESTABLISHED-like claim.
    assert result["root_cause"].status == "VERIFIED"


@pytest.mark.asyncio
async def test_case_d_training_completed_deficiency_hypothesis_rejected():
    """Case D: 'Training was completed but the operator did not follow the
    procedure.' A hypothesis claiming training deficiency directly
    contradicts the VERIFIED fact that training was completed and must be
    dropped, even though it doesn't contradict the mechanism itself."""
    finding_text = "Training was completed but the operator did not follow the procedure."
    state = _build_state(
        finding_text,
        verified=["training was completed", "the operator did not follow the procedure"],
        reported=[],
    )

    llm_response = json.dumps({
        "root_cause": {
            "status": "NOT_ESTABLISHED",
            "category": "TO_BE_CONFIRMED",
            "candidate_hypotheses": [
                {
                    "id": "H1",
                    "name": "TRAINING_DEFICIENCY",
                    "statement": "A training deficiency may have caused the operator not to follow the procedure.",
                    "status": "POSSIBLE",
                    "evidence_needed": "Training records",
                    "supporting_claim_ids": ["C1"],
                    "contradicting_claim_ids": [],
                },
                {
                    "id": "H2",
                    "name": "PROCEDURE_CLARITY_GAP",
                    "statement": "The procedure steps may not have been clear to the operator despite training.",
                    "status": "POSSIBLE",
                    "evidence_needed": "Procedure clarity review",
                    "supporting_claim_ids": ["C2"],
                    "contradicting_claim_ids": [],
                },
            ],
            "narrative": "The operator did not follow the procedure despite completed training.",
        },
        "five_why": {"steps": [], "is_complete": False, "status_note": "Stopped at evidence boundary"},
        "impact_assessment": {"status": "IMPACT_REQUIRES_ASSESSMENT", "areas": []},
        "capa": {"status": "INVESTIGATION_REQUIRED", "potential_areas": [], "recommended_investigation": []},
        "contributing_factors": [],
        "ca_draft": {
            "immediate_action": "Review the affected procedure step.",
            "root_cause": "NOT_ESTABLISHED",
            "root_cause_category": "TO_BE_CONFIRMED",
            "preventive_action": "Pending root cause confirmation.",
            "impact_analysis": "Scope requires auditor assessment.",
        },
    })

    result = await _run_synthesis_and_verification(state, llm_response)

    hyp_names = {h.name for h in result["root_cause"].candidate_hypotheses}
    assert "TRAINING_DEFICIENCY" not in hyp_names
    assert "PROCEDURE_CLARITY_GAP" in hyp_names
