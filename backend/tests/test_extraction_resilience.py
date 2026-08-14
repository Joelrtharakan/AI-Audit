"""Regression test for a real bug found via live testing: the Step-1
extraction LLM call hallucinated an ungrounded identifier ("SOP-LAB-001",
traced to a repeated example in extraction_prompt.txt) for a finding that
never mentioned any SOP at all. Extraction's own grounding check correctly
rejected it and retried, but after the retry also failed, extraction raised
LLMError and understand_finding_node fell back to a completely EMPTY
ExtractionResult() -- meaning the evidence ledger started empty and every
downstream node (RCA, investigation, impact, CAPA) had nothing but the raw
request object to reason from. That's the actual mechanism behind
"weak, generic, low-value analysis": not over-aggressive guards, but an
upstream extraction failure silently starving the whole pipeline.

Fix: extraction_prompt.txt no longer repeats one concrete, memorable example
identifier three times (the exact pattern that caused prior prompt-leak
bugs), and understand_finding_node now falls back to sentence-level facts
split directly from the finding text -- deterministic, no LLM needed --
instead of an empty extraction, whenever the extraction LLM call fails.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.models.agent import AgentTraceStep, InvestigateRequest
from app.services.llm_client import LLMError


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


def test_extraction_prompt_does_not_repeat_a_single_memorable_example():
    """The exact pattern that caused this bug: one concrete identifier
    ("SOP-LAB-001") repeated across multiple field descriptions in the same
    prompt, forming a single canonical example story a weak model latches
    onto and reproduces regardless of the actual observation text."""
    from app.config import get_settings

    settings = get_settings()
    text = (settings.prompts_dir / "extraction_prompt.txt").read_text(encoding="utf-8")
    assert text.count("SOP-LAB-001") == 0
    assert text.count("refrigerator R-12") == 0


@pytest.mark.asyncio
async def test_understanding_node_falls_back_to_sentence_facts_not_empty_extraction():
    """When the extraction LLM call fails entirely, the evidence ledger must
    still be populated from the finding text itself rather than left empty."""
    from app.agent.nodes.understanding import understand_finding_node

    finding_text = (
        "Three operators were observed performing the revised inspection procedure. "
        "Training records confirmed no completion was recorded for the three operators."
    )
    state = _build_state(finding_text)

    async def fake_extract_finding(text, client):
        raise LLMError("simulated: extraction hallucinated an ungrounded entity on every retry")

    with patch("app.agent.nodes.understanding.get_llm_client") as mock_get_client, \
         patch("app.agent.nodes.understanding.extract_finding", side_effect=fake_extract_finding), \
         patch("app.agent.nodes.understanding.check_observation_quality") as mock_quality:
        from app.models.analysis import ObservationQualityResult, ObservationQualityStatus
        mock_quality.return_value = ObservationQualityResult(status=ObservationQualityStatus.SUFFICIENT, missing_information=[])
        mock_get_client.return_value = AsyncMock()

        result = await understand_finding_node(state)

    ledger = result["evidence_ledger"]
    assert len(ledger) >= 2, "evidence ledger must not be left empty when extraction fails"
    joined = " ".join(e.claim for e in ledger)
    assert "three operators" in joined.lower()
    assert "training records" in joined.lower()
    assert all(e.status.value == "VERIFIED" for e in ledger)
