"""Regression test for the "too generic" bug: the agent's own JSON-schema
prompt examples used fluent English sentences to describe what to write
(e.g. "A specific assessment pathway grounded in this finding's own affected
items/records."), and a weak model sometimes just echoed that instruction
text back verbatim as if it were real analysis instead of replacing it.

This is the actual mechanism behind outputs that felt generic/defensive: it
wasn't (only) the model being overly conservative, it was the model literally
copying meta-instructions because they were the closest thing at hand.

Fix: prompts now use unmistakable <<bracketed>> placeholders instead of
natural-language example sentences, plus a code-level `is_placeholder_leak()`
safety net wired into every generation node, tested here directly against
that mocked failure mode.
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


def test_is_placeholder_leak_catches_the_actual_observed_output():
    from app.agent.grounding_guard import is_placeholder_leak

    # This exact sentence was observed as literal live-model output.
    assert is_placeholder_leak(
        "A specific assessment pathway grounded in this finding's own affected items/records."
    )
    assert is_placeholder_leak("A second pathway addressing what changed and whether it could affect execution.")
    assert not is_placeholder_leak(
        "The cold-room monitoring system's sensor readings for 10-12 August are missing and their "
        "recoverability has not been established."
    )


def test_is_placeholder_leak_catches_literal_angle_bracket_syntax():
    """A second observed live-model variant: the model filled in the <<...>>
    placeholder's content but kept the angle brackets themselves, e.g.
    "<<the weighing balance was used for production measurements...>>"
    landing directly in the impact areas list."""
    from app.agent.grounding_guard import is_placeholder_leak

    assert is_placeholder_leak(
        "<<the weighing balance was used for production measurements despite its calibration "
        "certificate having expired>>"
    )
    assert not is_placeholder_leak(
        "The weighing balance was used for production measurements despite its calibration "
        "certificate having expired."
    )


@pytest.mark.asyncio
async def test_impact_node_strips_leaked_placeholder_text():
    from app.agent.nodes.impact import impact_assessment_node

    state = _build_state("Two temperature readings were missing from the cold-room monitoring system on 14 August.")
    llm_response = json.dumps({
        "impact_assessment": {
            "status": "IMPACT_REQUIRES_ASSESSMENT",
            "areas": [
                "A specific assessment pathway grounded in this finding's own affected items/records.",
                "A second pathway addressing what changed and whether it could affect execution or interpretation.",
            ],
            "narrative": "A specific assessment pathway grounded in this finding's own affected items/records.",
        }
    })

    with patch("app.agent.nodes.impact.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client
        result = await impact_assessment_node(state)

    impact = result["impact_assessment"]
    assert impact.areas == []
    assert impact.narrative is None
    assert any("echoed prompt instruction" in t.model_dump()["message"] for t in result["trace"])


@pytest.mark.asyncio
async def test_rca_forces_not_established_on_leaked_placeholder_narrative():
    from app.agent.nodes.rca import root_cause_node

    state = _build_state("Two temperature readings were missing from the cold-room monitoring system on 14 August.")
    llm_response = json.dumps({
        "root_cause": {
            "status": "STATED_UNVERIFIED", "category": "MACHINE", "statement": None,
            "leading_hypothesis": None, "candidate_hypotheses": [],
            "narrative": "clear description of root cause or leading hypothesis",
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

    root_cause = result["root_cause"]
    assert root_cause.status == "NOT_ESTABLISHED"
    assert "clear description of root cause" not in root_cause.narrative.lower()


def test_clean_structured_leak_strips_python_dict_reprs():
    """Reproduces a bug found via live-model testing: a weak model asked for
    a plain-string list item returned a nested dict instead, and the naive
    str(x) call put a Python dict repr directly into auditor-facing text --
    "{'affected_object': 'cold-room monitoring system', ...}"."""
    from app.agent.grounding_guard import clean_structured_leak

    leaked = "{'affected_object': 'cold-room monitoring system', 'retrospective_review_needed': True}"
    cleaned = clean_structured_leak(leaked)
    assert "{" not in cleaned
    assert "}" not in cleaned
    assert "'" not in cleaned
    assert "cold-room monitoring system" in cleaned

    assert clean_structured_leak({"retrospective_review_needed": True}) == "yes"
    assert clean_structured_leak("a normal plain sentence.") == "a normal plain sentence."
    assert clean_structured_leak(["item one", "item two"]) == "item one; item two"


@pytest.mark.asyncio
async def test_impact_node_cleans_dict_shaped_areas_from_llm():
    """End-to-end reproduction through impact_assessment_node: the LLM
    returns dict objects for 'areas' instead of plain strings."""
    from app.agent.nodes.impact import impact_assessment_node

    state = _build_state("Two temperature readings were missing from the cold-room monitoring system on 14 August.")
    llm_response = json.dumps({
        "impact_assessment": {
            "status": "IMPACT_REQUIRES_ASSESSMENT",
            "areas": [
                {"affected_object": "cold-room monitoring system", "period": "14 August"},
                {"retrospective_review_needed": True},
            ],
            "narrative": "The cold-room monitoring system had readings missing on 14 August.",
        }
    })

    with patch("app.agent.nodes.impact.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client
        result = await impact_assessment_node(state)

    impact = result["impact_assessment"]
    for area in impact.areas:
        assert "{" not in area and "}" not in area and "'" not in area
