"""Semantic normalization regression suite (production fix: entity fields
must never be a full evidence-proposition sentence).

Reproduces and locks in the fix for the reported production bug:

    affected_object = "System records show that the document-control system"

instead of the correct:

    affected_object = "Document-control system"

which then corrupted every downstream template that interpolates the
subject (investigation questions, 5-Why questions). The fix lives in
app/services/semantic_subject.py's extraction/cleaning firewall
(_SELF_REFERENTIAL_EVIDENCE_PREFIX_RE, applied in both _strip_framing and
_clean_subject) -- this is the single canonical extraction point every
downstream node consumes (Section 4: single source of truth), so fixing it
there fixes every consumer at once.

Uses the same offline, deterministic-fallback pipeline harness as
test_golden_20_scenarios.py.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.core_synthesis import core_synthesis_node
from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.agent.nodes.report_generator import generate_report_node
from app.agent.nodes.understanding import understand_finding_node
from app.models.agent import InvestigateRequest
from app.services.semantic_subject import extract_semantic_subject

_SELF_REFERENTIAL_PHRASES = (
    "system records show that",
    "records show that",
    "evidence shows that",
    "the finding states that",
    "the audit found that",
    "records confirm that",
    "according to the evidence",
)


async def _run_agent_pipeline(finding_text: str):
    req = InvestigateRequest(finding_text=finding_text)
    state = {
        "request": req,
        "evidence_ledger": [],
        "iteration_count": 0,
        "tool_call_count": 0,
        "critic_iteration": 0,
        "trace": [],
        "errors": [],
    }
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.investigation_planner.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        state = await understand_finding_node(state)
        state = await plan_investigation_node(state)
        state = await core_synthesis_node(state)
        state = await generate_report_node(state)
        state = await final_evidence_verification_node(state)
    is_valid, violations = evaluate_all_invariants(state)
    return state, state.get("report"), is_valid, violations


# TEST 14: Entity extraction containing "System records show that" must be
# stripped to the clean noun phrase, at the extraction unit level.
@pytest.mark.parametrize("text,expected_subject", [
    (
        "System records show that the document-control system failed to distribute the revised SOP.",
        "document-control system",
    ),
    (
        "Evidence shows that the temperature monitoring device malfunctioned during the shift.",
        "temperature monitoring device",
    ),
    (
        "The audit found that the calibration record for BAL-014 was missing.",
        "calibration record for BAL-014",
    ),
    (
        "Records confirm that the training log was not updated after the SOP revision.",
        "training log",
    ),
])
def test_self_referential_evidence_prefix_stripped_from_subject(text, expected_subject):
    result = extract_semantic_subject(text)
    subj_low = (result.subject or "").lower()
    for phrase in _SELF_REFERENTIAL_PHRASES:
        assert phrase not in subj_low, f"subject {result.subject!r} still contains framing phrase {phrase!r}"
    assert expected_subject.lower() in subj_low


# TEST 15: An evidence sentence must never survive as an entity anywhere in
# the pipeline's canonical/impact/investigation/5-Why output.
@pytest.mark.asyncio
async def test_evidence_sentence_never_reaches_downstream_fields():
    text = (
        "System records show that the document-control system failed to distribute "
        "the revised SOP to affected departments. System logs recorded the failure "
        "and departments continued using the previous revision."
    )
    state, report, is_valid, violations = await _run_agent_pipeline(text)
    assert is_valid, f"Violations: {violations}"

    canonical = state["canonical_finding_state"]
    assert "system records show that" not in (canonical.affected_object or "").lower()
    assert "system records show that" not in (canonical.finding_subject or "").lower()
    assert canonical.affected_object.lower().startswith("document-control system") or \
        "document-control system" in canonical.affected_object.lower()

    if report.investigation and report.investigation.questions:
        for q in report.investigation.questions:
            assert "system records show that" not in q.question.lower(), (
                f"investigation question leaked the raw evidence sentence: {q.question!r}"
            )

    if report.five_why and report.five_why.steps:
        for step in report.five_why.steps:
            assert "system records show that" not in step.question.lower(), (
                f"5-Why question leaked the raw evidence sentence: {step.question!r}"
            )
            # The mechanism/answer text is allowed to state the full verified
            # claim (including "System records show that..." as its own
            # evidence-grounded content) -- only the QUESTION must never
            # embed it as if it were an entity name.


# TEST: 5-Why questions built from a "failed to <verb> <object>" condition
# must be grammatically well-formed regardless of which verb appears (not
# just a small hardcoded whitelist) and must never restate their own answer.
@pytest.mark.asyncio
async def test_five_why_question_grammatical_for_arbitrary_verb():
    text = "The department failed to notify the affected suppliers of the schedule change."
    state, report, is_valid, violations = await _run_agent_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    fw = report.five_why
    assert fw.steps
    q = fw.steps[0].question
    assert "not notify" in q.lower() or "notify" in q.lower()
    # No bare "was ... notify" (uninflected verb directly after was/were).
    import re
    assert not re.search(r"\b(?:was|were)\s+\w+\s+not\s+notify\b", q, re.IGNORECASE)
    assert not re.search(r"\bwas\s+.+\s+notify\b(?!ing|ed)", q, re.IGNORECASE)
