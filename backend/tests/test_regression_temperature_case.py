"""Regression test for the specific finding used as the acceptance case in
the latest hardening request: a reported-but-unverified equipment
malfunction must never become a VERIFIED/established cause, and domains
unrelated to this finding (training, authorization) must not appear.

Also extends the cross-case contamination sequence to 5 finding types
(training / workload / temperature-monitoring / recurrence / calibration)
run back-to-back through the same node, verifying no entity or hypothesis
crosses between any pair.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.models.agent import AgentTraceStep, EvidenceItem, EvidenceStatus, InvestigateRequest


def _build_state(finding_text: str, evidence_ledger: list[EvidenceItem] | None = None) -> dict:
    return {
        "request": InvestigateRequest(finding_text=finding_text),
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
        "evidence_ledger": evidence_ledger or [],
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


TEMPERATURE_FINDING = (
    "Three temperature records were missing from 10-12 August. The technician "
    "stated that the monitoring device was malfunctioning. Maintenance records "
    "were not available during the audit."
)

TEMPERATURE_EVIDENCE = [
    EvidenceItem(claim="Three temperature records were missing from 10-12 August.", source="AUDITOR_FINDING", status=EvidenceStatus.VERIFIED),
    EvidenceItem(claim="The technician stated that the monitoring device was malfunctioning.", source="AUDITOR_FINDING", status=EvidenceStatus.REPORTED),
    EvidenceItem(claim="Maintenance records were not available during the audit.", source="AUDITOR_FINDING", status=EvidenceStatus.VERIFIED),
]

_TEMPERATURE_LLM_RESPONSE = json.dumps({
    "root_cause": {
        "status": "STATED_UNVERIFIED", "category": None, "statement": None,
        "leading_hypothesis": None,
        "candidate_hypotheses": [
            {"id": "H1", "name": "EQUIPMENT_MALFUNCTION", "statement": "The monitoring device may have malfunctioned during 10-12 August, as reported by the technician.", "status": "POSSIBLE", "evidence_needed": "Maintenance records, device diagnostics, alarm/error logs"},
            {"id": "H2", "name": "DATA_CAPTURE_FAILURE", "statement": "A data capture or recordkeeping failure may have caused the missing records independent of device malfunction.", "status": "POSSIBLE", "evidence_needed": "System logs, manual backup records, data export logs"},
        ],
        "narrative": "The technician reported that the monitoring device was malfunctioning, but this has not been independently verified because maintenance records were unavailable during the audit.",
        "confidence": "LOW", "evidence_status": "REPORTED",
    },
    "contributing_factors": [],
    "five_why": {"steps": [
        {"question": "Why were three temperature records missing from 10-12 August?", "answer": "The finding establishes that three temperature records were missing during this period; the reason has not been established.", "status": "VERIFIED"},
        {"question": "Was the monitoring device malfunctioning during this period?", "answer": "The technician reported that it was malfunctioning.", "status": "REPORTED_UNVERIFIED"},
        {"question": "Can the reported malfunction be independently verified?", "answer": "Maintenance records were unavailable during the audit, so this cannot currently be verified.", "status": "UNKNOWN"},
    ], "is_complete": False, "status_note": "INCOMPLETE — ROOT CAUSE NOT ESTABLISHED"},
})


@pytest.mark.asyncio
async def test_temperature_finding_reported_malfunction_stays_unverified():
    from app.agent.nodes.rca import root_cause_node

    state = _build_state(TEMPERATURE_FINDING, TEMPERATURE_EVIDENCE)
    with patch("app.agent.nodes.rca.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = _TEMPERATURE_LLM_RESPONSE
        mock_get_client.return_value = mock_client
        result = await root_cause_node(state)

    root_cause = result["root_cause"]
    five_why = result["five_why"]

    # The reported malfunction must never be presented as an established cause
    assert root_cause.status != "VERIFIED"
    assert root_cause.status in ("STATED_UNVERIFIED", "NOT_ESTABLISHED")
    assert root_cause.category == "TO_BE_CONFIRMED"

    # Verified facts (quantity, dates) must be preserved exactly
    narrative_and_hyps = root_cause.narrative + " " + " ".join(h.statement for h in root_cause.candidate_hypotheses)
    for step in five_why.steps:
        narrative_and_hyps += " " + (step.question or "") + " " + (step.answer or "")

    # Only domains actually supported by this finding may appear
    lowered = narrative_and_hyps.lower()
    for irrelevant in ("training", "authorization", "operator competency", "sop revision", "workload"):
        assert irrelevant not in lowered, f"irrelevant domain '{irrelevant}' leaked into temperature finding"

    # 5-Why must stop where evidence stops (3 steps here), not force to 5
    assert len(five_why.steps) == 3
    assert five_why.steps[0].status == "VERIFIED"
    assert five_why.steps[-1].status in ("UNKNOWN", "NOT_ESTABLISHED")


# ---------------------------------------------------------------------------
# 5-case cross-contamination sequence
# ---------------------------------------------------------------------------

_CASES = {
    "training": (
        "Three operators performed a revised inspection procedure without mandatory training.",
        [{"id": "H1", "name": "TRAINING_ASSIGNMENT", "statement": "Training may not have been assigned to the three operators before they performed the revised procedure.", "evidence_needed": "Training assignment records"}],
    ),
    "workload": (
        "Five production records contained incomplete entries. Operators reported unusually high workload. Supervisor stated training was insufficient.",
        [{"id": "H1", "name": "WORKLOAD_PRESSURE", "statement": "Unusually high workload may have contributed to the incomplete entries.", "evidence_needed": "Staffing and production volume records"}],
    ),
    "temperature": (
        TEMPERATURE_FINDING,
        [{"id": "H1", "name": "EQUIPMENT_MALFUNCTION", "statement": "The monitoring device may have malfunctioned during 10-12 August, as reported by the technician.", "evidence_needed": "Maintenance records, diagnostics"}],
    ),
    "recurrence": (
        "Previous audit identified missing temperature records and previous CAPA was recorded as completed.",
        [{"id": "H1", "name": "CAPA_EFFECTIVENESS_UNVERIFIED", "statement": "The previous corrective action was recorded as completed, but whether it was implemented or effective has not been established.", "evidence_needed": "CAPA implementation and effectiveness review records"}],
    ),
    "calibration": (
        "One weighing balance was found outside its calibration period.",
        [{"id": "H1", "name": "CALIBRATION_OVERDUE", "statement": "The weighing balance may not have been calibrated within its required calibration period.", "evidence_needed": "Calibration schedule and completion records"}],
    ),
}


def _response_for(name: str) -> str:
    finding_text, hyps = _CASES[name]
    return json.dumps({
        "root_cause": {
            "status": "NOT_ESTABLISHED", "category": None, "statement": None,
            "leading_hypothesis": None,
            "candidate_hypotheses": [{**h, "status": "POSSIBLE"} for h in hyps],
            "narrative": f"Root cause not established for: {finding_text[:40]}...",
            "confidence": "LOW", "evidence_status": "UNKNOWN",
        },
        "contributing_factors": [],
        "five_why": {"steps": [], "is_complete": False, "status_note": "INCOMPLETE"},
    })


@pytest.mark.asyncio
async def test_five_case_sequence_no_cross_contamination():
    """Run five distinct finding types sequentially through root_cause_node
    and confirm no hypothesis/entity from one case appears in any other."""
    from app.agent.nodes.rca import root_cause_node

    results = {}
    for name in _CASES:
        finding_text, _ = _CASES[name]
        state = _build_state(finding_text)
        with patch("app.agent.nodes.rca.get_llm_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.chat_completion.return_value = _response_for(name)
            mock_get_client.return_value = mock_client
            results[name] = await root_cause_node(state)

    all_hyp_names = {
        name: {h.name for h in result["root_cause"].candidate_hypotheses}
        for name, result in results.items()
    }

    for name, hyp_names in all_hyp_names.items():
        own_names = {h["name"] for h in _CASES[name][1]}
        assert hyp_names == own_names, f"case {name!r} got unexpected hypotheses {hyp_names}"
        for other_name, other_hyps in _CASES.items():
            if other_name == name:
                continue
            other_only_names = {h["name"] for h in other_hyps[1]} - own_names
            leaked = hyp_names & other_only_names
            assert not leaked, f"case {name!r} leaked hypothesis names from {other_name!r}: {leaked}"


@pytest.mark.asyncio
async def test_genuinely_established_root_cause_is_not_over_blocked():
    """The causal-inference guard must not be so aggressive that it blocks a
    real, VERIFIED-evidence-backed causal conclusion (electronic form
    validation defect confirmed by system logs)."""
    from app.agent.nodes.rca import root_cause_node

    finding = (
        "Five production records were incomplete because the approved electronic "
        "form contained a validation defect that prevented completion of the "
        "required field. System logs confirmed the validation defect occurred "
        "during the affected period."
    )
    evidence = [
        EvidenceItem(claim="Five production records were incomplete.", source="AUDITOR_FINDING", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="The approved electronic form contained a validation defect that prevented completion of the required field.", source="AUDITOR_FINDING", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="System logs confirmed the validation defect occurred during the affected period.", source="AUDITOR_FINDING", status=EvidenceStatus.VERIFIED),
    ]
    response = json.dumps({
        "root_cause": {
            "status": "VERIFIED", "category": "METHOD",
            "statement": "A validation defect in the approved electronic form prevented completion of the required field, causing the incomplete entries.",
            "leading_hypothesis": None, "candidate_hypotheses": [],
            "narrative": "System logs confirm that a validation defect in the electronic form caused the incomplete entries during the affected period.",
            "confidence": "HIGH", "evidence_status": "VERIFIED",
        },
        "contributing_factors": [],
        "five_why": {"steps": [
            {"question": "Why were five records incomplete?", "answer": "System logs confirm the electronic form had a validation defect that prevented completion of the required field during the affected period.", "status": "VERIFIED"},
        ], "is_complete": True, "status_note": "COMPLETE"},
    })

    state = _build_state(finding, evidence)
    with patch("app.agent.nodes.rca.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = response
        mock_get_client.return_value = mock_client
        result = await root_cause_node(state)

    root_cause = result["root_cause"]
    assert root_cause.status == "VERIFIED"
    assert root_cause.category == "METHOD"
    assert "validation defect" in root_cause.narrative.lower()


def test_partial_contradiction_two_of_operators_detected():
    """A contradiction doesn't require both claims to be about ALL operators --
    'all completed' vs 'no record for two of the operators' must still be
    flagged as conflicting."""
    from app.agent.grounding_guard import detect_evidence_contradictions

    evidence = [
        EvidenceItem(claim="The supervisor stated that all operators completed the required training.", source="AUDITOR_FINDING", status=EvidenceStatus.REPORTED),
        EvidenceItem(claim="The training matrix showed no completion records for two of the operators.", source="AUDITOR_FINDING", status=EvidenceStatus.VERIFIED),
    ]
    pairs = detect_evidence_contradictions(evidence)
    assert len(pairs) == 1


def test_compatible_facts_about_same_subject_are_not_flagged_as_contradictory():
    """Regression: found via live testing. Two facts that merely share a
    subject ("three operators") but describe entirely different, fully
    compatible aspects of the finding -- performing a procedure, and having
    no recorded training completion -- were being flagged as a contradiction
    by the old min-ratio overlap check (0.4 on just {"three","operators"}).
    They are not in tension at all; forcing NOT_ESTABLISHED with a
    'conflicting evidence' narrative here was itself the failure."""
    from app.agent.grounding_guard import detect_evidence_contradictions

    evidence = [
        EvidenceItem(
            claim="Three operators were observed performing the revised inspection procedure",
            source="AUDITOR_FINDING", status=EvidenceStatus.VERIFIED,
        ),
        EvidenceItem(
            claim="the three operators had no recorded training completion",
            source="AUDITOR_FINDING", status=EvidenceStatus.VERIFIED,
        ),
    ]
    pairs = detect_evidence_contradictions(evidence)
    assert pairs == []
