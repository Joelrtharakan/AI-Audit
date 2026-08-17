"""Regression tests for three bugs found via a REAL (unmocked) LLM call
against the pure-equipment finding used in production validation:

    "Two temperature readings were missing from the cold-room monitoring
    system on 14 August. The technician stated that the sensor stopped
    responding. No maintenance report was available during the audit."

The live model (qwen2.5:3b via Ollama) generated:
  1. An investigation question about "mandatory training" and an evidence
     item "Training records for Cold Room Operations" -- despite zero
     mention of training anywhere in the finding. Entity/number grounding
     didn't catch this because "training" contains no invented entity or
     number, just an off-topic (but individually well-formed) domain.
  2. The CA draft's root_cause_category came back "MEASUREMENT" even though
     the RCA step's own root_cause.category was "TO_BE_CONFIRMED" (root
     cause not established) -- the CA draft is a separate LLM call and can
     independently be more confident than the RCA step that produced it.
  3. (Not reproduced deterministically here, but guarded against): the CA
     draft's impact_analysis field can independently invent recall/
     regulatory/product-safety language the finding never supports, the
     same failure mode impact.py already firewalls for its own output.

These tests mock the LLM to reproduce that exact observed output and verify
the fix (mentions_unsupported_domain guard + category discipline +
impact-severity firewall) actually catches it -- fast and deterministic,
unlike the live-model run that found the bug.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.models.agent import AgentTraceStep, InvestigateRequest


def _build_state(finding_text: str) -> dict:
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


COLD_ROOM_FINDING = (
    "Two temperature readings were missing from the cold-room monitoring "
    "system on 14 August. The technician stated that the sensor stopped "
    "responding. No maintenance report was available during the audit."
)


@pytest.mark.asyncio
async def test_investigation_planner_drops_unsupported_training_question():
    """Reproduces bug 1: the live model injected a training question/evidence
    item into a pure equipment finding. The planner must drop them."""
    from app.agent.nodes.investigation_planner import plan_investigation_node

    state = _build_state(COLD_ROOM_FINDING)
    llm_response = json.dumps({
        "needs_investigation": False,
        "investigation_rationale": "No LQMS tools needed.",
        "planned_tools": [],
        "investigation_plan": {
            "areas": ["Cold-room monitoring system maintenance process"],
            "questions": [
                {"question": "What specific step in the maintenance procedure did not occur?", "purpose": "To identify the process gap", "evidence": "Maintenance records"},
                {"question": "Was mandatory training conducted and completed for the department involved with the cold-room monitoring system on 14 August?", "purpose": "Training gap hypothesis", "evidence": "Training records for the department involved with the cold-room monitoring system"},
            ],
            "evidence_to_collect": [
                "Maintenance records for the cold-room monitoring system",
                "Training records for the department involved with the cold-room monitoring system",
            ],
        },
    })

    with patch("app.agent.nodes.investigation_planner.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client
        result = await plan_investigation_node(state)

    plan = result["investigation_plan"]
    # questions are now InvestigationQuestion objects; extract .question text for string checks
    question_texts = [iq.question for iq in plan.questions]
    joined = " ".join(question_texts + plan.evidence_to_collect).lower()
    assert "training" not in joined
    # The exact question count is an implementation detail of whichever
    # deterministic branch this finding lands in (it has no reported
    # explanation or conflict, so it now correctly produces a fuller set
    # of foundational investigation questions rather than a single generic
    # one) -- the test's actual point (per its docstring) is that the
    # unsupported training question/evidence never survives, which the
    # "training" not in joined check above already establishes.
    assert len(plan.questions) >= 1
    assert len(plan.evidence_to_collect) >= 1
    assert any("dropped" in t.model_dump()["message"] for t in result["trace"])


@pytest.mark.asyncio
async def test_rca_drops_unsupported_training_hypothesis():
    """Same domain guard applied to RCA's own candidate hypotheses."""
    from app.agent.nodes.rca import root_cause_node

    state = _build_state(COLD_ROOM_FINDING)
    llm_response = json.dumps({
        "root_cause": {
            "status": "NOT_ESTABLISHED", "category": None, "statement": None,
            "leading_hypothesis": None,
            "candidate_hypotheses": [
                {"id": "H1", "name": "EQUIPMENT_MALFUNCTION", "statement": "The sensor may have malfunctioned, as reported by the technician.", "status": "POSSIBLE", "evidence_needed": "Maintenance and diagnostic records"},
                {"id": "H2", "name": "TRAINING_GAP", "statement": "Staff may not have been adequately trained on monitoring system checks.", "status": "POSSIBLE", "evidence_needed": "Training records"},
            ],
            "narrative": "Root cause not established.",
            "confidence": "LOW", "evidence_status": "UNKNOWN",
        },
        "contributing_factors": [],
        "five_why": {"steps": [], "is_complete": False, "status_note": "INCOMPLETE"},
    })

    with patch("app.agent.nodes.rca.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client
        result = await root_cause_node(state)

    names = [h.name for h in result["root_cause"].candidate_hypotheses]
    assert "TRAINING_GAP" not in names
    assert "EQUIPMENT_MALFUNCTION" in names


@pytest.mark.asyncio
async def test_ca_draft_category_cannot_exceed_root_cause_confidence():
    """Reproduces bug 2: the CA draft LLM call independently guessed a
    definitive category (MEASUREMENT) even though the RCA step already
    determined the cause isn't established (TO_BE_CONFIRMED)."""
    from app.agent.nodes.ca_draft_generator import ca_draft_generator_node
    from app.models.agent import RootCauseAnalysis

    state = _build_state(COLD_ROOM_FINDING)
    state["root_cause"] = RootCauseAnalysis(
        status="STATED_UNVERIFIED",
        category="TO_BE_CONFIRMED",
        narrative="The technician reported the sensor malfunctioned; this has not been independently verified.",
    )

    llm_response = json.dumps({
        "immediate_action": "Review the cold-room monitoring system maintenance schedule.",
        "root_cause": "Root cause not established.",
        "root_cause_category": "MEASUREMENT",
        "preventive_action": "Preventive action pending investigation.",
        "impact_analysis": "Impact requires assessment.",
    })

    with patch("app.agent.nodes.ca_draft_generator.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client
        result = await ca_draft_generator_node(state)

    ca_draft = result["ca_draft"]
    assert ca_draft.root_cause_category == "TO_BE_CONFIRMED"
    assert any("overridden to TO_BE_CONFIRMED" in t.model_dump()["message"] for t in result["trace"])


@pytest.mark.asyncio
async def test_ca_draft_impact_cannot_invent_recall_or_regulatory_language():
    """Reproduces bug 3: the CA draft's impact_analysis is a separate LLM
    call from impact.py and can independently invent recall/regulatory/
    product-safety language the finding gives no basis for."""
    from app.agent.nodes.ca_draft_generator import ca_draft_generator_node
    from app.models.agent import ImpactAssessment, ImpactStatus, RootCauseAnalysis

    state = _build_state(COLD_ROOM_FINDING)
    state["root_cause"] = RootCauseAnalysis(status="NOT_ESTABLISHED", category="TO_BE_CONFIRMED", narrative="Not established.")
    state["impact_assessment"] = ImpactAssessment(status=ImpactStatus.IMPACT_REQUIRES_ASSESSMENT, areas=[], narrative=None)

    llm_response = json.dumps({
        "immediate_action": "Review the monitoring system.",
        "root_cause": "Root cause not established.",
        "root_cause_category": "TO_BE_CONFIRMED",
        "preventive_action": "Pending investigation.",
        "impact_analysis": (
            "The missing readings could lead to product recalls or non-compliance "
            "with regulatory requirements affecting product safety."
        ),
    })

    with patch("app.agent.nodes.ca_draft_generator.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client
        result = await ca_draft_generator_node(state)

    ca_draft = result["ca_draft"]
    lowered = ca_draft.impact_analysis.lower()
    assert "recall" not in lowered
    assert "regulatory" not in lowered
    assert "product safety" not in lowered


@pytest.mark.asyncio
async def test_rca_drops_fabricated_communication_mechanism():
    """Reproduces a bug found in further live testing: a pure calibration-
    label finding with zero mention of any auditor interaction still
    produced a hypothesis that the operator "miscommunicated the calibration
    status to the auditor" -- a fabricated mechanism, same failure class as
    the training-injection bug but via the "communication" trope instead."""
    from app.agent.nodes.rca import root_cause_node

    finding = (
        "Calibration status label missing from equipment EQ-104. Operator says "
        "equipment was recently calibrated. Certificate unavailable."
    )
    state = _build_state(finding)

    llm_response = json.dumps({
        "root_cause": {
            "status": "STATED_UNVERIFIED", "category": None, "statement": None,
            "leading_hypothesis": None,
            "candidate_hypotheses": [
                {"id": "H1", "name": "OPERATOR_MISCOMMUNICATION", "statement": "The operator may have miscommunicated the calibration status to the auditor, leading to a missing label.", "status": "POSSIBLE", "evidence_needed": "Communication records"},
                {"id": "H2", "name": "LABEL_NOT_UPDATED", "statement": "The calibration status label may not have been updated or reattached after calibration was performed.", "status": "POSSIBLE", "evidence_needed": "Calibration certificate and label issuance record for EQ-104"},
            ],
            "narrative": "Root cause not established.",
            "confidence": "LOW", "evidence_status": "UNKNOWN",
        },
        "contributing_factors": [],
        "five_why": {"steps": [], "is_complete": False, "status_note": "INCOMPLETE"},
    })

    with patch("app.agent.nodes.rca.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client
        result = await root_cause_node(state)

    names = [h.name for h in result["root_cause"].candidate_hypotheses]
    assert "OPERATOR_MISCOMMUNICATION" not in names
    assert "LABEL_NOT_UPDATED" in names
