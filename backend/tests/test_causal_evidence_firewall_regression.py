"""Causal-evidence firewall regression suite (production fix: evidence-
ABSENCE claims must never be used as causal support, and 5-Why depth must
be evidence-driven, not padded to a fixed length).

Reproduces and locks in the fix for the reported production bug:

    C1: "One checklist was incomplete."
    C2: "No information is available about when the omission occurred, who
         performed the activity, whether the requirement was understood,
         whether training was provided, or why the checklist was not
         completed."

The generated 5-Why incorrectly cited C2 (a statement that causal
information is UNAVAILABLE) as VERIFIED causal support for a mechanism
answer, and produced a grammatically corrupted second question ("Why did
the checklist not incomplete?"). Both defects are fixed at the source:

  - app/agent/causal_guard.py: classify_mechanism_polarity()/
    extract_immediate_mechanism() now reject any claim matching
    is_evidence_absence_claim() before it can ever be treated as a
    mechanism, regardless of embedded action-verb vocabulary.
  - app/services/semantic_subject.py: format_deviation_why_question() no
    longer misapplies "did not <verb>" active-voice phrasing to adjective/
    participle conditions ("incomplete", "not completed") or to bare
    intransitive state verbs ("failed") -- only to a small whitelist of
    known transitive "failed to <verb> <object>" verbs.

Uses the same offline, deterministic-fallback pipeline harness as
test_golden_20_scenarios.py.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from app.agent.causal_guard import is_evidence_absence_claim
from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.core_synthesis import core_synthesis_node
from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.agent.nodes.report_generator import generate_report_node
from app.agent.nodes.understanding import understand_finding_node
from app.models.agent import InvestigateRequest, RootCauseStatus


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


# Unit-level: the evidence-absence detector itself.
@pytest.mark.parametrize("text,expected", [
    ("No information is available about when the omission occurred.", True),
    ("No information is available about why the checklist was not completed.", True),
    ("Insufficient evidence exists to determine the cause.", True),
    ("The reason is not established from the available evidence.", True),
    ("The operator did not perform the required inspection.", False),
    ("Server logs show the notification service crashed at 08:00.", False),
    ("The document-control system failed to distribute the revised SOP.", False),
])
def test_is_evidence_absence_claim(text, expected):
    assert is_evidence_absence_claim(text) is expected


# The exact reported production scenario, single-sentence variant: a bare
# observation with no evidence at all must stop at ONE Why step.
@pytest.mark.asyncio
async def test_observation_only_stops_at_one_why():
    text = "One checklist was incomplete."
    state, report, is_valid, violations = await _run_agent_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED
    fw = report.five_why
    assert len(fw.steps) == 1, f"expected exactly 1 Why step, got {len(fw.steps)}: {[s.question for s in fw.steps]}"
    assert fw.steps[0].status == "UNKNOWN"
    assert "why was the checklist incomplete" in fw.steps[0].question.lower()


# The exact reported production scenario, full two-claim variant: the
# evidence-absence claim (C2) must NOT be cited as VERIFIED causal support.
@pytest.mark.asyncio
async def test_evidence_absence_claim_never_cited_as_causal_support():
    text = (
        "One checklist was incomplete. No information is available about when the "
        "omission occurred, who performed the activity, whether the requirement was "
        "understood, whether training was provided, or why the checklist was not completed."
    )
    state, report, is_valid, violations = await _run_agent_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED

    fw = report.five_why
    for step in fw.steps:
        if step.status in ("VERIFIED", "SUPPORTED"):
            assert not is_evidence_absence_claim(step.answer), (
                f"step {step.question!r} is {step.status} but its answer is an "
                f"evidence-absence statement: {step.answer!r}"
            )
        # No malformed double-negative/garbled question survives.
        assert "not incomplete" not in step.question.lower()
        assert "not missing" not in step.question.lower()


# A finding with a genuinely SUPPORTED immediate mechanism must still only
# generate as many Why steps as the evidence supports (2, not 5) -- the
# deeper "why did the mechanism occur" layer correctly stays UNKNOWN.
@pytest.mark.asyncio
async def test_supported_mechanism_variable_length_not_padded_to_five():
    text = "Audit trail confirms the mandatory checklist review step was disabled in the workflow configuration."
    state, report, is_valid, violations = await _run_agent_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    fw = report.five_why
    assert len(fw.steps) < 5, f"chain padded to {len(fw.steps)} steps instead of stopping at the evidence boundary"


# Fully established root cause (control bypass, audit-terminal in QMS
# terms) is still allowed to reach ESTABLISHED/SUPPORTED -- variable length
# does not mean "never resolve."
@pytest.mark.asyncio
async def test_fully_established_root_cause_still_reachable():
    text = (
        "Audit trail logs confirm the mandatory dual-approval validation control "
        "was disabled by admin before the duplicate payment was released."
    )
    state, report, is_valid, violations = await _run_agent_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status in (RootCauseStatus.SUPPORTED, RootCauseStatus.ESTABLISHED)


# Grammar regression coverage for the specific corrupted-question shape
# reported ("Why did the checklist not incomplete?") across a range of
# condition shapes, ensuring no double-negative or dropped-object defect.
@pytest.mark.parametrize("subject,condition,forbidden_substrings", [
    ("checklist", "incomplete", ["not incomplete"]),
    ("checklist", "not completed", ["not not"]),
    ("equipment", "not calibrated", ["not not"]),
    ("the check", "failed", ["not failed"]),
])
def test_format_deviation_why_question_no_double_negative(subject, condition, forbidden_substrings):
    from app.services.semantic_subject import format_deviation_why_question
    q = format_deviation_why_question(subject, condition, None).lower()
    for bad in forbidden_substrings:
        assert bad not in q, f"question {q!r} contains forbidden fragment {bad!r}"
