"""A finding expressed purely as an epistemic STANCE (belief / suspicion /
assumption / opinion) about the world asserts a mental state, not an
observed affected object. The stance must be recorded as BELIEF evidence,
but it must never become the canonical affected subject or an established
root cause.

Structural / modality-aware: the SAME `classify_epistemic_stance` classifier
that the evidence-ledger loop uses is now also consulted when selecting the
segments that feed subject resolution -- the two passes agree. No stance-verb
blacklist beyond the existing open-ended classifier; domain-agnostic.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from app.agent.nodes.understanding import understand_finding_node
from app.models.agent import InvestigateRequest


async def _understand(finding_text: str):
    state = {
        "request": InvestigateRequest(finding_text=finding_text),
        "evidence_ledger": [], "iteration_count": 0, "tool_call_count": 0,
        "critic_iteration": 0, "trace": [], "errors": [],
    }
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None):
        return await understand_finding_node(state)


def _belief_count(state) -> int:
    return sum(
        1 for e in state["evidence_ledger"]
        if str(getattr(e.status, "value", e.status)) == "BELIEF"
    )


PURE_STANCE_FINDINGS = [
    "The supervisor believed the deviation was probably a data-entry error.",
    "The technician suspected that gauge G-3 was reading high.",
    "The reviewer assumed the discrepancy stemmed from a rounding difference.",
    "In the auditor's opinion, the delay was likely caused by staffing shortages.",
    "The manager was of the view that the omission was an oversight.",
]


@pytest.mark.parametrize("finding", PURE_STANCE_FINDINGS)
def test_pure_stance_finding_has_no_concrete_subject(finding):
    state = asyncio.run(_understand(finding))
    cf = state["canonical_finding_state"]
    subj = (cf.finding_subject or "")
    # the stance is preserved as evidence...
    assert _belief_count(state) >= 1
    # ...but it never becomes a concrete affected object or a real root cause
    assert subj.upper().startswith(("UNRESOLVED", "UNKNOWN")) or cf.semantic_type == "NON_ACTIONABLE"
    assert cf.semantic_type == "NON_ACTIONABLE"
    # the hypothesised cause must never leak in as the subject
    for leaked in ("data-entry error", "rounding difference", "staffing shortages",
                   "oversight", "reading high"):
        assert leaked.lower() not in subj.lower()


def test_stance_clause_does_not_corrupt_an_observed_subject():
    # A real observed deviation + a belief about its cause -> the OBSERVED
    # entity is the subject; the belief is still recorded, separately.
    state = asyncio.run(_understand(
        "Pump P-9 failed twice during the month. The supervisor believed it was a data-entry error."
    ))
    cf = state["canonical_finding_state"]
    assert "p-9" in (cf.finding_subject or "").lower()
    assert "data-entry error" not in (cf.finding_subject or "").lower()
    assert _belief_count(state) >= 1


def test_reported_speech_is_not_treated_as_stance():
    # "reported / stated" are evidence verbs, not stance verbs -> the finding
    # stays actionable and resolves the observed entity.
    state = asyncio.run(_understand(
        "The operator reported that the interlock on press PR-4 was bypassed."
    ))
    cf = state["canonical_finding_state"]
    assert "pr-4" in (cf.finding_subject or "").lower()
    assert cf.semantic_type != "NON_ACTIONABLE"
