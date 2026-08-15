"""End-to-end tests for the causal-reasoning fix in core_synthesis_node —
the LIVE node in the graph (app/agent/graph.py only registers core_synthesis,
not the legacy rca.py/impact.py/capa.py per-step nodes).

These mock the LLM to return exactly the kind of output the bug report
describes (a competing "performed but not documented" hypothesis alongside
an established "activity was missed" mechanism, and a circular 5-Why step)
to verify the code-level guards catch it even when the LLM itself doesn't
follow the prompt's instruction — the guard must not depend on LLM
compliance. No finding-specific hardcoding: the finding text below uses a
generic "monthly verification" scenario, not any example from prior work.
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


def _build_state(finding_text: str, mechanism: str | None, mechanism_status: str = "REPORTED") -> dict:
    ledger = [
        EvidenceItem(
            claim="the required entry was not completed",
            source="AUDITOR_FINDING",
            status=EvidenceStatus.VERIFIED,
        ),
    ]
    if mechanism:
        ledger.append(EvidenceItem(
            claim=mechanism,
            source="REPORTED_STATEMENT",
            status=EvidenceStatus.REPORTED,
        ))
    canonical = CanonicalFindingState(
        raw_finding=finding_text,
        observed_deviation="the required entry — not completed",
        deviation_condition="not completed",
        facts=["the required entry was not completed"],
        reported_statements=[mechanism] if mechanism else [],
        affected_objects=["the required entry"],
        immediate_mechanism=mechanism,
        immediate_mechanism_status=mechanism_status if mechanism else "UNKNOWN",
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


@pytest.mark.asyncio
async def test_core_synthesis_rejects_hypothesis_contradicting_established_mechanism():
    """The LLM (mocked here to simulate a weak/non-compliant model) returns
    both the correct SUPPORTED mechanism hypothesis AND a competing
    "performed but not documented" hypothesis despite the prompt's explicit
    instruction not to. The code-level guard must reject the contradicting
    one regardless."""
    from app.agent.nodes.core_synthesis import core_synthesis_node

    finding_text = (
        "The required monthly verification entry was not completed for the reporting period. "
        "The responsible reviewer confirmed that the verification was missed."
    )
    mechanism = "the responsible reviewer confirmed the verification was missed"
    state = _build_state(finding_text, mechanism)

    llm_response = json.dumps({
        "root_cause": {
            "status": "NOT_ESTABLISHED",
            "category": "TO_BE_CONFIRMED",
            "statement": None,
            "leading_hypothesis": None,
            "candidate_hypotheses": [
                {
                    "id": "H1",
                    "name": "VERIFICATION_NOT_PERFORMED",
                    "statement": "The required verification was not performed as scheduled.",
                    "status": "SUPPORTED",
                    "evidence_needed": "Shift/assignment records",
                },
                {
                    "id": "H2",
                    "name": "DOCUMENTATION_GAP",
                    "statement": "The verification may have been performed but not documented in the record.",
                    "status": "POSSIBLE",
                    "evidence_needed": "Contemporaneous records",
                },
            ],
            "narrative": "The verification was confirmed missed; the reviewer's account establishes the mechanism.",
        },
        "five_why": {"steps": [], "is_complete": False, "status_note": "Stopped at evidence boundary"},
        "impact_assessment": {"status": "IMPACT_REQUIRES_ASSESSMENT", "areas": []},
        "capa": {"status": "INVESTIGATION_REQUIRED", "potential_areas": [], "recommended_investigation": []},
        "contributing_factors": [],
        "ca_draft": {
            "immediate_action": "Assess the affected verification record.",
            "root_cause": "NOT_ESTABLISHED — verification was reportedly missed.",
            "root_cause_category": "TO_BE_CONFIRMED",
            "preventive_action": "Strengthen verification scheduling controls.",
            "impact_analysis": "Scope requires auditor assessment.",
        },
    })

    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client
        result = await core_synthesis_node(state)

    root_cause = result["root_cause"]
    hypothesis_ids = {h.id for h in root_cause.candidate_hypotheses}
    hypothesis_statements = " ".join(h.statement.lower() for h in root_cause.candidate_hypotheses)

    assert "H2" not in hypothesis_ids, "contradicting hypothesis must be dropped"
    assert "but not documented" not in hypothesis_statements
    assert "H1" in hypothesis_ids
    assert any(h.status == "SUPPORTED" for h in root_cause.candidate_hypotheses)

    # Trace must record WHY it was dropped, not silently vanish.
    warn_messages = " ".join(t.message for t in result["trace"] if t.icon == "⚠")
    assert "contradicts the established mechanism" in warn_messages


@pytest.mark.asyncio
async def test_core_synthesis_truncates_circular_five_why_step():
    from app.agent.nodes.core_synthesis import core_synthesis_node

    finding_text = "The required monthly verification entry was not completed for the reporting period."
    state = _build_state(finding_text, mechanism=None)

    llm_response = json.dumps({
        "root_cause": {
            "status": "NOT_ESTABLISHED",
            "category": "TO_BE_CONFIRMED",
            "candidate_hypotheses": [],
            "narrative": "The entry was not completed; the underlying cause is not established.",
        },
        "five_why": {
            "steps": [
                {
                    "level": 1,
                    "question": "Why was the required verification entry incomplete?",
                    "answer": "The required verification entry was incomplete.",
                    "status": "VERIFIED",
                },
                {
                    "level": 2,
                    "question": "Why did that happen?",
                    "answer": "Control effectiveness was not demonstrated.",
                    "status": "NOT_ESTABLISHED",
                },
            ],
            "is_complete": True,
            "status_note": "Complete",
        },
        "impact_assessment": {"status": "IMPACT_REQUIRES_ASSESSMENT", "areas": []},
        "capa": {"status": "INVESTIGATION_REQUIRED", "potential_areas": [], "recommended_investigation": []},
        "contributing_factors": [],
        "ca_draft": {
            "immediate_action": "Assess the affected verification record.",
            "root_cause": "NOT_ESTABLISHED",
            "root_cause_category": "TO_BE_CONFIRMED",
            "preventive_action": "Strengthen verification controls.",
            "impact_analysis": "Scope requires auditor assessment.",
        },
    })

    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client
        result = await core_synthesis_node(state)

    five_why = result["five_why"]
    assert len(five_why.steps) == 1, "chain must truncate at the circular step, not continue past it"
    assert five_why.steps[0].status == "UNKNOWN"
    assert five_why.steps[0].answer != "The required verification entry was incomplete."


@pytest.mark.asyncio
async def test_core_synthesis_populates_contributing_factors_from_llm():
    """Regression for the information-loss bug: core_synthesis previously
    never requested or parsed contributing_factors at all, so the field was
    always empty regardless of what the finding supported."""
    from app.agent.nodes.core_synthesis import core_synthesis_node

    finding_text = "The required monthly verification entry was not completed for the reporting period."
    state = _build_state(finding_text, mechanism=None)

    llm_response = json.dumps({
        "root_cause": {
            "status": "NOT_ESTABLISHED",
            "category": "TO_BE_CONFIRMED",
            "candidate_hypotheses": [],
            "narrative": "The entry was not completed; the underlying cause is not established.",
        },
        "five_why": {"steps": [], "is_complete": False, "status_note": "Stopped at evidence boundary"},
        "impact_assessment": {"status": "IMPACT_REQUIRES_ASSESSMENT", "areas": []},
        "capa": {"status": "INVESTIGATION_REQUIRED", "potential_areas": [], "recommended_investigation": []},
        "contributing_factors": [
            {
                "description": "The verification relies on a single reviewer completing a manual step.",
                "rationale": "The process as described has no secondary check.",
                "evidence_status": "INFERRED",
                "status": "POSSIBLE_UNCONFIRMED",
                "evidence_required": "Process design documentation",
            }
        ],
        "ca_draft": {
            "immediate_action": "Assess the affected verification record.",
            "root_cause": "NOT_ESTABLISHED",
            "root_cause_category": "TO_BE_CONFIRMED",
            "preventive_action": "Strengthen verification controls.",
            "impact_analysis": "Scope requires auditor assessment.",
        },
    })

    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client
        result = await core_synthesis_node(state)

    assert "contributing_factors" in result
    factors = result["contributing_factors"]
    assert len(factors) == 1
    assert "single reviewer" in factors[0].description.lower()
    assert factors[0].status == "POSSIBLE_UNCONFIRMED"


@pytest.mark.asyncio
async def test_core_synthesis_drops_generic_filler_contributing_factor():
    """A contributing factor whose description is pure boilerplate ("not
    established") carries zero analytical content and must be dropped —
    an empty list is the correct representation, not a filler sentence."""
    from app.agent.nodes.core_synthesis import core_synthesis_node

    finding_text = "The required monthly verification entry was not completed for the reporting period."
    state = _build_state(finding_text, mechanism=None)

    llm_response = json.dumps({
        "root_cause": {
            "status": "NOT_ESTABLISHED",
            "category": "TO_BE_CONFIRMED",
            "candidate_hypotheses": [],
            "narrative": "The entry was not completed; the underlying cause is not established.",
        },
        "five_why": {"steps": [], "is_complete": False, "status_note": "Stopped at evidence boundary"},
        "impact_assessment": {"status": "IMPACT_REQUIRES_ASSESSMENT", "areas": []},
        "capa": {"status": "INVESTIGATION_REQUIRED", "potential_areas": [], "recommended_investigation": []},
        "contributing_factors": [
            {
                "description": "Additional contributing factors are not established.",
                "evidence_status": "INFERRED",
                "status": "POSSIBLE_UNCONFIRMED",
            }
        ],
        "ca_draft": {
            "immediate_action": "Assess the affected verification record.",
            "root_cause": "NOT_ESTABLISHED",
            "root_cause_category": "TO_BE_CONFIRMED",
            "preventive_action": "Strengthen verification controls.",
            "impact_analysis": "Scope requires auditor assessment.",
        },
    })

    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client
        result = await core_synthesis_node(state)

    assert result["contributing_factors"] == []


@pytest.mark.asyncio
async def test_core_synthesis_sets_leading_hypothesis_when_root_cause_not_established():
    """NOT_ESTABLISHED must not mean the report goes silent about which
    candidate is best-evidenced — leading_hypothesis should be populated
    deterministically even if the LLM left it null."""
    from app.agent.nodes.core_synthesis import core_synthesis_node

    finding_text = "The required monthly verification entry was not completed for the reporting period."
    mechanism = "the responsible reviewer confirmed the verification was missed"
    state = _build_state(finding_text, mechanism)

    llm_response = json.dumps({
        "root_cause": {
            "status": "NOT_ESTABLISHED",
            "category": "TO_BE_CONFIRMED",
            "leading_hypothesis": None,
            "candidate_hypotheses": [
                {
                    "id": "H1",
                    "name": "TASK_ASSIGNMENT_GAP",
                    "statement": "Responsibility for the verification was not clearly assigned for the affected period.",
                    "status": "SUPPORTED",
                    "evidence_needed": "Assignment/roster records",
                },
                {
                    "id": "H2",
                    "name": "SCHEDULING_GAP",
                    "statement": "The verification schedule did not account for reviewer absence.",
                    "status": "POSSIBLE",
                    "evidence_needed": "Scheduling records",
                    "relevance_rank": "LOW",
                },
            ],
            "narrative": "The verification was reportedly missed; the deeper cause is not yet established.",
        },
        "five_why": {"steps": [], "is_complete": False, "status_note": "Stopped at evidence boundary"},
        "impact_assessment": {"status": "IMPACT_REQUIRES_ASSESSMENT", "areas": []},
        "capa": {"status": "INVESTIGATION_REQUIRED", "potential_areas": [], "recommended_investigation": []},
        "contributing_factors": [],
        "ca_draft": {
            "immediate_action": "Assess the affected verification record.",
            "root_cause": "NOT_ESTABLISHED",
            "root_cause_category": "TO_BE_CONFIRMED",
            "preventive_action": "Strengthen verification controls.",
            "impact_analysis": "Scope requires auditor assessment.",
        },
    })

    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client
        result = await core_synthesis_node(state)

    root_cause = result["root_cause"]
    assert root_cause.status == "NOT_ESTABLISHED"
    assert root_cause.leading_hypothesis is not None
    assert "H1" in root_cause.leading_hypothesis  # SUPPORTED hypothesis preferred over LOW-rank POSSIBLE one


@pytest.mark.asyncio
async def test_core_synthesis_truncates_five_why_question_reopening_mechanism():
    """A 5-Why QUESTION that re-litigates whether the established mechanism
    occurred (opposite polarity) must be truncated even if the answer text
    itself looks fine in isolation — the guard inspects the question, not
    just the answer."""
    from app.agent.nodes.core_synthesis import core_synthesis_node

    finding_text = (
        "The required monthly verification entry was not completed for the reporting period. "
        "The responsible reviewer confirmed that the verification was missed."
    )
    mechanism = "the responsible reviewer confirmed the verification was missed"
    state = _build_state(finding_text, mechanism)

    llm_response = json.dumps({
        "root_cause": {
            "status": "NOT_ESTABLISHED",
            "category": "TO_BE_CONFIRMED",
            "candidate_hypotheses": [],
            "narrative": "The verification was reportedly missed; the deeper cause is not established.",
        },
        "five_why": {
            "steps": [
                {
                    "level": 1,
                    "question": "Why was the required verification entry incomplete?",
                    "answer": "The reviewer confirmed the required verification was missed.",
                    "status": "SUPPORTED",
                },
                {
                    "level": 2,
                    "question": "Was the verification actually performed but not documented in the record?",
                    "answer": "It may have been performed but simply not logged.",
                    "status": "POSSIBLE",
                },
            ],
            "is_complete": False,
            "status_note": "In progress",
        },
        "impact_assessment": {"status": "IMPACT_REQUIRES_ASSESSMENT", "areas": []},
        "capa": {"status": "INVESTIGATION_REQUIRED", "potential_areas": [], "recommended_investigation": []},
        "contributing_factors": [],
        "ca_draft": {
            "immediate_action": "Assess the affected verification record.",
            "root_cause": "NOT_ESTABLISHED",
            "root_cause_category": "TO_BE_CONFIRMED",
            "preventive_action": "Strengthen verification controls.",
            "impact_analysis": "Scope requires auditor assessment.",
        },
    })

    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client
        result = await core_synthesis_node(state)

    five_why = result["five_why"]
    assert len(five_why.steps) == 1, "step 2 reopened the established mechanism and must be truncated"
    assert five_why.steps[0].question == "Why was the required verification entry incomplete?"
