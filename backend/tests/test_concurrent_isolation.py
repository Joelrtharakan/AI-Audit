"""Concurrency isolation tests.

Runs multiple root_cause_node invocations for different findings concurrently
against a single shared (mocked) LLM client, to verify that AgentState objects
built per-request never cross-talk under concurrent execution — the async
equivalent of the sequential cross-case tests in test_grounding_guard.py.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from unittest.mock import AsyncMock, patch

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


_CASES = {
    "training": (
        "Three operators were observed performing the revised inspection procedure "
        "without having completed mandatory training. The SOP revision was issued "
        "30 days ago.",
        {
            "id": "H1", "name": "TRAINING_ASSIGNMENT",
            "statement": "Training may not have been assigned to the three operators after the SOP revision.",
        },
    ),
    "workload": (
        "Five production records contained incomplete entries. Operators reported "
        "unusually high workload during the affected shifts.",
        {
            "id": "H1", "name": "WORKLOAD_PRESSURE",
            "statement": "Unusually high workload may have contributed to the incomplete entries.",
        },
    ),
    "equipment": (
        "A temperature monitoring device displayed readings outside the expected "
        "range during one production shift.",
        {
            "id": "H1", "name": "EQUIPMENT_CONDITION",
            "statement": "The device may have degraded or malfunctioned during the shift.",
        },
    ),
    "supplier": (
        "Incoming material from supplier ABC failed dimensional inspection on four samples.",
        {
            "id": "H1", "name": "SUPPLIER_PROCESS_CONTROL",
            "statement": "Supplier ABC's process may not consistently meet the specified dimension.",
        },
    ),
}


def _response_for(finding_text: str) -> str:
    for text, hyp in _CASES.values():
        if text == finding_text:
            return json.dumps({
                "root_cause": {
                    "status": "NOT_ESTABLISHED", "category": None, "statement": None,
                    "leading_hypothesis": None,
                    "candidate_hypotheses": [{**hyp, "status": "POSSIBLE", "evidence_needed": "Relevant records"}],
                    "narrative": f"Root cause not established for: {finding_text[:30]}...",
                    "confidence": "LOW", "evidence_status": "UNKNOWN",
                },
                "contributing_factors": [],
                "five_why": {"steps": [], "is_complete": False, "status_note": "INCOMPLETE"},
            })
    raise AssertionError(f"no fixture response for {finding_text!r}")


@pytest.mark.asyncio
async def test_concurrent_requests_stay_isolated():
    """Four different findings analyzed concurrently must each get back only
    their own hypothesis — no cross-talk between concurrently running
    coroutines sharing the same (mocked) LLM client."""
    from app.agent.nodes.rca import root_cause_node

    async def fake_chat_completion(messages, **kwargs):
        # The finding text is embedded in the user prompt; recover it well
        # enough to route to the right fixture response.
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        for text, _ in _CASES.values():
            if text in user_content:
                return _response_for(text)
        raise AssertionError("could not identify finding in prompt")

    mock_client = AsyncMock()
    mock_client.chat_completion.side_effect = fake_chat_completion

    with patch("app.agent.nodes.rca.get_llm_client", return_value=mock_client):
        states = {name: _build_state(text) for name, (text, _) in _CASES.items()}
        results = await asyncio.gather(*(root_cause_node(s) for s in states.values()))

    results_by_name = dict(zip(_CASES.keys(), results))

    for name, (finding_text, expected_hyp) in _CASES.items():
        result = results_by_name[name]
        root_cause = result["root_cause"]
        hyp_names = [h.name for h in root_cause.candidate_hypotheses]
        assert hyp_names == [expected_hyp["name"]], (
            f"case {name!r} got hypotheses {hyp_names}, expected only {[expected_hyp['name']]} "
            f"— possible cross-talk between concurrent requests"
        )
        # No other case's hypothesis name leaked in
        other_names = {hyp["name"] for k, (_, hyp) in _CASES.items() if k != name}
        for other in other_names:
            assert other not in hyp_names
