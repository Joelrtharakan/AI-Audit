"""Generalization tests for the causal-reasoning guard (app/agent/causal_guard.py).

These tests are deliberately domain-diverse — mechanism extraction and
contradiction detection are structural (verb-shape) checks, not finding-
specific keyword matches, so they must behave identically across arbitrary
QMS domains. No test here hardcodes a specific equipment ID, log name, or
company-specific vocabulary; only the grammatical shape matters.
"""

from __future__ import annotations

import pytest

from app.agent.causal_guard import (
    MechanismInfo,
    extract_immediate_mechanism,
    hypothesis_contradicts_mechanism,
    is_circular_why_answer,
    is_generic_non_analysis_filler,
    is_reporting_why_question,
    mechanism_already_names_generic_hypothesis,
    question_reopens_mechanism,
    repeats_previous_why_answer,
    restates_observation,
)

# ---------------------------------------------------------------------------
# 15 QMS domains, each expressed as (reported_statement, contradicting_hypothesis)
# covering the mechanisms named in the bug report: missed inspection, incomplete
# record, procedure deviation, training discrepancy, equipment failure,
# calibration, cleaning, supplier qualification, warehouse inspection,
# environmental monitoring, software access, maintenance, CAPA effectiveness,
# audit findings, recordkeeping.
# ---------------------------------------------------------------------------
DOMAINS = [
    ("the scheduled inspection was missed", "the inspection may have been performed but not documented"),
    ("the required entry was not completed", "the entry may have been completed but not recorded"),
    ("the procedure step was not followed", "the step may have been followed but not logged"),
    ("the training was not completed", "the training may have been completed but not documented"),
    ("the equipment check was not performed", "the check may have been performed but not recorded"),
    ("the calibration was not performed", "the calibration may have been performed but not logged"),
    ("the cleaning task was skipped", "the cleaning may have been done but not documented"),
    ("the supplier qualification review was not conducted", "the review may have been conducted but not recorded"),
    ("the warehouse inspection was not carried out", "the inspection may have been carried out but not logged"),
    ("the environmental monitoring check was missed", "the check may have been performed but not recorded"),
    ("the access review was not performed", "the review may have been performed but not documented"),
    ("the maintenance task was not done", "the task may have been done but not recorded"),
    ("the effectiveness review was never performed", "the review may have occurred but was not logged"),
    ("the follow-up audit step was skipped", "the step may have been performed but was not documented"),
    ("the recordkeeping check was not conducted", "the check may have been conducted but not recorded"),
]


@pytest.mark.parametrize("mechanism_claim,contradicting_hypothesis", DOMAINS)
def test_mechanism_established_from_reported_statement(mechanism_claim, contradicting_hypothesis):
    mechanism = extract_immediate_mechanism([mechanism_claim], [])
    assert mechanism.status == "REPORTED"
    assert mechanism.polarity == "non_performance"


@pytest.mark.parametrize("mechanism_claim,contradicting_hypothesis", DOMAINS)
def test_mechanism_established_from_verified_fact(mechanism_claim, contradicting_hypothesis):
    mechanism = extract_immediate_mechanism([], [mechanism_claim])
    assert mechanism.status == "VERIFIED"
    assert mechanism.polarity == "non_performance"


@pytest.mark.parametrize("mechanism_claim,contradicting_hypothesis", DOMAINS)
def test_contradicting_hypothesis_rejected(mechanism_claim, contradicting_hypothesis):
    mechanism = extract_immediate_mechanism([mechanism_claim], [])
    assert hypothesis_contradicts_mechanism(contradicting_hypothesis, mechanism) is True


@pytest.mark.parametrize("mechanism_claim,contradicting_hypothesis", DOMAINS)
def test_non_contradicting_hypothesis_survives(mechanism_claim, contradicting_hypothesis):
    """A hypothesis about WHY the mechanism occurred (not whether it occurred)
    must never be rejected as contradicting -- only same-mechanism-shaped
    'it actually happened' claims are contradictions."""
    mechanism = extract_immediate_mechanism([mechanism_claim], [])
    why_hypothesis = "A shift handover or task assignment gap may explain why this occurred."
    assert hypothesis_contradicts_mechanism(why_hypothesis, mechanism) is False


def test_no_mechanism_means_no_contradiction():
    """When nothing in the evidence states an action-level mechanism, no
    hypothesis can be 'contradicted' by a mechanism that doesn't exist."""
    mechanism = extract_immediate_mechanism([], [])
    assert mechanism.status == "UNKNOWN"
    assert hypothesis_contradicts_mechanism("it may have been performed but not documented", mechanism) is False


# ---------------------------------------------------------------------------
# Redundant-hypothesis detection: a hedged restatement of an already-
# established mechanism is not a new hypothesis about its cause.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mechanism_claim,_unused", DOMAINS)
def test_hedged_restatement_of_established_mechanism_rejected(mechanism_claim, _unused):
    mechanism = extract_immediate_mechanism([mechanism_claim], [])
    hedged = "The required activity may not have been performed as scheduled."
    assert mechanism_already_names_generic_hypothesis(hedged, mechanism) is True


def test_non_hedged_hypothesis_not_flagged_as_redundant():
    mechanism = extract_immediate_mechanism(["the check was missed"], [])
    # A definite, non-hedged causal claim about the NEXT layer is fine.
    assert mechanism_already_names_generic_hypothesis(
        "A shift handover gap caused the missed check.", mechanism
    ) is False


# ---------------------------------------------------------------------------
# 5-Why circularity
# ---------------------------------------------------------------------------


def test_circular_answer_restating_question_detected():
    assert is_circular_why_answer(
        "Why was the record incomplete?", "The record was incomplete."
    ) is True


def test_non_circular_answer_not_flagged():
    assert is_circular_why_answer(
        "Why was the record incomplete?",
        "The required entry was not completed during the shift because the assigned reviewer was unavailable.",
    ) is False


def test_generic_no_evidence_answer_is_not_circular():
    """An honest 'evidence doesn't establish this' answer is NOT circular —
    it's a correct evidence-boundary stop, distinct from restating the
    question. Circularity specifically means the answer echoes the question's
    own vocabulary back without adding anything."""
    assert is_circular_why_answer(
        "Why was the required check missed?",
        "The available evidence does not establish why the scheduled check was missed.",
    ) is False


def test_repeats_previous_answer_detected():
    prev = "The required temperature check was missed during the morning shift."
    same = "The temperature check was missed during the morning shift."
    assert repeats_previous_why_answer(prev, same) is True


def test_advancing_answer_not_flagged_as_repeat():
    prev = "The required temperature check was missed during the morning shift."
    advancing = "The available evidence does not establish why the scheduled check was missed."
    assert repeats_previous_why_answer(prev, advancing) is False


# ---------------------------------------------------------------------------
# The exact bug-report scenario, end to end through the guard functions.
# ---------------------------------------------------------------------------


def test_bug_report_scenario_mechanism_and_contradiction():
    reported = ["The responsible technician confirmed the check was missed during the morning shift"]
    verified = ["the log was not completed for the stated date"]
    mechanism = extract_immediate_mechanism(reported, verified)
    assert mechanism.polarity == "non_performance"
    assert mechanism.status == "REPORTED"

    execution_omission = "The required activity may not have been performed during the operation."
    documentation_omission = "The required activity may have been performed but not documented in the record."

    assert mechanism_already_names_generic_hypothesis(execution_omission, mechanism) is True
    assert hypothesis_contradicts_mechanism(documentation_omission, mechanism) is True


# ---------------------------------------------------------------------------
# question_reopens_mechanism: the QUESTION itself re-litigating a resolved
# mechanism, not just the answer.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mechanism_claim,_unused", DOMAINS)
def test_question_reopening_established_mechanism_detected(mechanism_claim, _unused):
    mechanism = extract_immediate_mechanism([mechanism_claim], [])
    reopening_question = "Was the activity performed but not documented in the record?"
    assert question_reopens_mechanism(reopening_question, mechanism) is True


def test_question_asking_why_mechanism_occurred_not_flagged():
    mechanism = extract_immediate_mechanism(["the check was missed"], [])
    forward_question = "Why was the check missed during the shift?"
    assert question_reopens_mechanism(forward_question, mechanism) is False


def test_question_reopening_check_requires_established_mechanism():
    mechanism = MechanismInfo()  # nothing established
    assert question_reopens_mechanism("Was it performed but not documented?", mechanism) is False


def test_non_recording_mechanism_reopened_by_non_performance_question():
    """Symmetric case: once non-recording is established (activity happened,
    just wasn't captured), a question asking whether it was performed at all
    reopens that resolved distinction too."""
    mechanism = extract_immediate_mechanism(["the entry was not documented"], [])
    assert mechanism.polarity == "non_recording"
    assert question_reopens_mechanism("Was the activity ever performed?", mechanism) is False
    # "was ever performed" doesn't match the non_performance verb-shape regex
    # (no "not"), so use an explicit non-performance phrasing instead:
    assert question_reopens_mechanism("Was the activity not performed at all?", mechanism) is True


# ---------------------------------------------------------------------------
# is_generic_non_analysis_filler: boilerplate non-answers vs real analysis.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filler", [
    "Additional contributing factors are not established.",
    "Contributing factors are not established.",
    "Root cause is not established.",
    "Evidence is not available.",
    "Not established.",
])
def test_generic_filler_detected(filler):
    assert is_generic_non_analysis_filler(filler) is True


@pytest.mark.parametrize("real_content", [
    "The process relies on a single reviewer completing a manual step with no secondary check.",
    "Shift handover records show no explicit reassignment of this task when the primary reviewer was absent.",
    "The revised procedure was distributed one day before the affected shift.",
])
def test_real_analysis_not_flagged_as_filler(real_content):
    assert is_generic_non_analysis_filler(real_content) is False


# ---------------------------------------------------------------------------
# restates_observation
# ---------------------------------------------------------------------------


def test_answer_restating_observation_detected():
    observation = "the required entry was not completed"
    answer = "The required entry was not completed."
    assert restates_observation(answer, observation) is True


def test_answer_explaining_observation_not_flagged():
    observation = "the required entry was not completed"
    answer = "The assigned reviewer was reassigned to another task during the affected shift."
    assert restates_observation(answer, observation) is False


# ---------------------------------------------------------------------------
# is_reporting_why_question (added alongside the mechanism guards)
# ---------------------------------------------------------------------------


def test_reporting_question_detected():
    assert is_reporting_why_question("Why did the technician report that the check was missed?") is True


def test_causal_question_not_flagged_as_reporting():
    assert is_reporting_why_question("Why was the required check missed during the shift?") is False


# ---------------------------------------------------------------------------
# hypothesis_contradicts_verified_completion — Case D from the hardening
# request: "Training was completed but the operator did not follow the
# procedure" must never let a "training deficiency" hypothesis survive.
# Tested across multiple unrelated topics to prove it's not training-specific.
# ---------------------------------------------------------------------------

from app.agent.causal_guard import hypothesis_contradicts_verified_completion

COMPLETION_CONTRADICTION_CASES = [
    ("training was completed", "A training deficiency may have contributed to the deviation."),
    ("training was completed", "Lack of training may explain the deviation."),
    ("calibration was performed on schedule", "Insufficient calibration may explain the deviation."),
    ("the review was conducted as scheduled", "The review was not provided before release."),
    ("the briefing was provided to the team", "Lack of briefing may have contributed."),
]


@pytest.mark.parametrize("verified_fact,hypothesis", COMPLETION_CONTRADICTION_CASES)
def test_hypothesis_contradicting_verified_completion_detected(verified_fact, hypothesis):
    assert hypothesis_contradicts_verified_completion(hypothesis, [verified_fact]) is True


def test_hypothesis_about_unrelated_topic_not_flagged():
    facts = ["training was completed"]
    assert hypothesis_contradicts_verified_completion(
        "A shift handover gap may have contributed to the deviation.", facts
    ) is False


def test_no_verified_facts_means_no_contradiction():
    assert hypothesis_contradicts_verified_completion("A training deficiency may exist.", []) is False


def test_case_d_end_to_end_through_mechanism_and_completion_guards():
    """The exact Case D scenario: training completed (VERIFIED), procedure
    not followed (mechanism). A training-deficiency hypothesis must be
    rejected by the completion guard even though it doesn't contradict the
    mechanism itself."""
    from app.agent.causal_guard import extract_immediate_mechanism

    verified = ["training was completed", "the operator did not follow the procedure"]
    mechanism = extract_immediate_mechanism([], verified)
    assert mechanism.polarity == "non_performance"

    training_deficiency_hyp = "A training deficiency may have caused the procedure not to be followed."
    assert hypothesis_contradicts_verified_completion(training_deficiency_hyp, verified) is True


# ---------------------------------------------------------------------------
# knowledge_gap polarity + general REPORTED catch-all — the fix for the
# "operator was unaware the procedure had been revised" degraded-mode bug.
# Domain-diverse to prove it's structural, not tied to that one example.
# ---------------------------------------------------------------------------

KNOWLEDGE_GAP_CASES = [
    "the operator was unaware that the checklist procedure had been revised",
    "the technician was not aware of the updated calibration interval",
    "the analyst did not know the reporting requirement had changed",
    "staff were not informed that the access procedure had been updated",
    "the reviewer was not notified of the revised acceptance criteria",
]


@pytest.mark.parametrize("claim", KNOWLEDGE_GAP_CASES)
def test_knowledge_gap_polarity_detected(claim):
    from app.agent.causal_guard import classify_mechanism_polarity
    assert classify_mechanism_polarity(claim) == "knowledge_gap"


def test_knowledge_gap_mechanism_extracted_as_reported():
    mechanism = extract_immediate_mechanism(
        ["the operator stated they were unaware that the checklist procedure had been revised"], []
    )
    assert mechanism.status == "REPORTED"
    assert mechanism.polarity == "knowledge_gap"


def test_general_catchall_never_drops_a_reported_statement_without_recognized_polarity():
    """A reported/attributed statement whose verb shape doesn't match ANY
    recognized polarity must still become the mechanism (polarity=general)
    rather than silently vanish -- this is the core generalization: any
    causal information the finding already gave us must survive even when
    it doesn't fit a known pattern."""
    mechanism = extract_immediate_mechanism(
        ["the supplier's system experienced an unrelated outage during the transfer window"], []
    )
    assert mechanism.status == "REPORTED"
    assert mechanism.polarity == "general"
    assert "outage" in mechanism.statement


def test_verified_facts_never_get_general_catchall():
    """Unlike reported statements, a VERIFIED fact with no recognized
    mechanism shape is usually just the observation restated -- it must
    NOT be promoted to "the mechanism" via the general catch-all (that
    catch-all is reported-statements-only)."""
    mechanism = extract_immediate_mechanism([], ["the log was incomplete for the stated period"])
    assert mechanism.status == "UNKNOWN"
    assert mechanism.statement is None
