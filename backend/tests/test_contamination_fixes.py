"""Regression tests for the cross-case contamination bug.

Root cause (found by inspection, not guessed): several prompt templates and
Python fallback defaults in the /investigate pipeline hardcoded a fixed
training-scenario narrative -- "three operators", "revised inspection
procedure", "SOP-OPS-014", "training matrix", "30 days" -- either as literal
instructed output (rca.txt, investigation_planner.txt, impact.txt, ca_draft.txt)
or as the value returned whenever an LLM call failed or an LLM field came back
empty (rca.py, report_generator.py, ca_draft_generator.py). There was no actual
shared/global request state -- each /investigate call builds a fresh AgentState
-- but every failure path funneled a *different* finding's output back into the
training scenario, which looked identical to cross-case memory leakage.

These tests mock the LLM client so they run without network access and are not
marked `integration`.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.models.agent import AgentTraceStep, InvestigateRequest
from app.services.llm_client import LLMError

CONTAMINATION_MARKERS = (
    "three operators",
    "revised inspection procedure",
    "sop-ops-014",
    "training matrix",
    "30 days",
    "30-day",
)


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


@pytest.mark.asyncio
async def test_rca_fallback_on_llm_error_has_no_training_entities():
    """RCA LLM failure must not inject the fixed training narrative into an
    unrelated finding's fallback output."""
    from app.agent.nodes.rca import root_cause_node

    state = _build_state(
        "Five production records contained incomplete entries. Operators "
        "reported unusually high workload during the affected shifts."
    )

    with patch("app.agent.nodes.rca.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.side_effect = LLMError("simulated failure")
        mock_get_client.return_value = mock_client

        result = await root_cause_node(state)

    root_cause = result["root_cause"]
    narrative = (root_cause.narrative or "").lower()
    for marker in CONTAMINATION_MARKERS:
        assert marker not in narrative, f"fallback narrative leaked '{marker}'"
    assert root_cause.candidate_hypotheses == []
    assert root_cause.leading_hypothesis is None
    assert root_cause.status == "NOT_ESTABLISHED"


@pytest.mark.asyncio
async def test_rca_empty_llm_hypotheses_are_not_replaced_with_training_defaults():
    """When the LLM returns no candidate_hypotheses, the node must not
    backfill the old hardcoded H1-H4 training hypothesis set."""
    from app.agent.nodes.rca import root_cause_node

    state = _build_state("Two batch records contained unsigned entries.")

    llm_response = json.dumps(
        {
            "root_cause": {
                "status": "NOT_ESTABLISHED",
                "category": None,
                "statement": None,
                "leading_hypothesis": None,
                "candidate_hypotheses": [],
                "narrative": "No leading hypothesis established from the available evidence.",
                "confidence": "LOW",
                "evidence_status": "UNKNOWN",
            },
            "contributing_factors": [],
            "five_why": {"steps": [], "is_complete": False, "status_note": "INCOMPLETE"},
        }
    )

    with patch("app.agent.nodes.rca.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client

        result = await root_cause_node(state)

    root_cause = result["root_cause"]
    assert root_cause.candidate_hypotheses == []


@pytest.mark.asyncio
async def test_five_why_stops_at_first_unknown_step():
    """5-Why must not be forced to 5 steps -- it must stop right after the
    first step whose evidence status is UNKNOWN/NOT_ESTABLISHED."""
    from app.agent.nodes.rca import root_cause_node

    state = _build_state("Four records contained incomplete entries.")

    llm_response = json.dumps(
        {
            "root_cause": {
                "status": "NOT_ESTABLISHED",
                "category": None,
                "statement": None,
                "leading_hypothesis": None,
                "candidate_hypotheses": [],
                "narrative": "Root cause not established.",
                "confidence": "LOW",
                "evidence_status": "UNKNOWN",
            },
            "contributing_factors": [],
            "five_why": {
                "steps": [
                    {
                        "question": "Why were four records incomplete?",
                        "answer": "The finding establishes four records had incomplete entries.",
                        "status": "VERIFIED",
                    },
                    {
                        "question": "Why were the entries incomplete?",
                        "answer": "The available evidence does not establish the cause.",
                        "status": "UNKNOWN",
                    },
                    {
                        "question": "Why did that cause occur?",
                        "answer": "Speculative continuation that should never be reached.",
                        "status": "NOT_ESTABLISHED",
                    },
                ],
                "is_complete": False,
                "status_note": "INCOMPLETE",
            },
        }
    )

    with patch("app.agent.nodes.rca.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client

        result = await root_cause_node(state)

    five_why = result["five_why"]
    assert len(five_why.steps) == 2
    assert five_why.steps[0].status == "VERIFIED"
    assert five_why.steps[1].status == "UNKNOWN"


def test_prompts_contain_no_hardcoded_training_scenario():
    """Prompt templates must not bake a fixed training-scenario example into
    their literal instructions -- models tend to copy concrete instructed
    examples verbatim across unrelated findings, which is how a training
    example ends up in a supplier or equipment finding's output."""
    from app.config import get_settings

    settings = get_settings()
    markers = (
        "three operators",
        "revised inspection procedure",
        "sop-ops-014",
        "30-day period",
        "30 days before",
    )
    for name in ("rca.txt", "investigation_planner.txt", "impact.txt", "ca_draft.txt"):
        text = (settings.agent_prompts_dir / name).read_text(encoding="utf-8").lower()
        for marker in markers:
            assert marker not in text, f"{name} still hardcodes '{marker}'"


@pytest.mark.asyncio
async def test_report_generator_fallback_has_no_training_entities():
    """If upstream nodes never populated root_cause/capa (e.g. a hard failure
    before RCA even ran), generate_report_node's fallbacks must stay generic
    rather than defaulting to the training scenario."""
    from app.agent.nodes.report_generator import generate_report_node

    state = _build_state("An issue was observed in production.")
    result = await generate_report_node(state)
    report = result["report"]

    narrative = (report.root_cause.narrative or "").lower()
    for marker in CONTAMINATION_MARKERS:
        assert marker not in narrative
    assert report.capa.potential_areas == []
