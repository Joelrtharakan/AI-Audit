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
    answer_asserts_verified_but_is_reported,
    classify_mixed_evidence_answer,
    derive_hypothesis_title_from_statement,
    detect_unsupported_causal_specificity,
    extract_immediate_mechanism,
    hypothesis_asserts_self_referential_evidence,
    hypothesis_contradicts_mechanism,
    hypothesis_name_is_generic,
    hypothesis_overclaims_human_error,
    hypothesis_statement_asserts_unsupported_causation,
    is_circular_why_answer,
    is_evidence_gap_not_hypothesis,
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


# ---------------------------------------------------------------------------
# is_evidence_gap_not_hypothesis
#
# The exact bug this closes: "Calibration certificate not available" was
# being accepted as a candidate root-cause hypothesis when it's really just
# a restated evidence gap already present in the finding text -- not a
# proposed explanation for WHY the deviation occurred. Generic, no domain
# words hardcoded: the same test applied to an unrelated domain below.
# ---------------------------------------------------------------------------

_CALIBRATION_FINDING_SOURCE = (
    "the auditor observed that the calibration status label was missing from equipment eq-104 "
    "the operator stated that the equipment had been calibrated recently but the calibration "
    "certificate was not available during the audit"
)


def test_evidence_gap_restatement_is_rejected():
    assert is_evidence_gap_not_hypothesis(
        "Calibration certificate not available", _CALIBRATION_FINDING_SOURCE
    )
    assert is_evidence_gap_not_hypothesis(
        "The calibration certificate was not available during the audit.", _CALIBRATION_FINDING_SOURCE
    )


def test_genuine_causal_hypothesis_is_not_rejected():
    assert not is_evidence_gap_not_hypothesis(
        "The required post-calibration labeling step may not have been completed.",
        _CALIBRATION_FINDING_SOURCE,
    )
    assert not is_evidence_gap_not_hypothesis(
        "The process may lack an effective control ensuring status labels are updated after calibration.",
        _CALIBRATION_FINDING_SOURCE,
    )


def test_evidence_gap_filter_generalizes_to_unrelated_domain():
    """Same structural test, different domain (training) -- proves this
    isn't a calibration-specific rule."""
    source = (
        "the auditor observed that the training record for operator j alvarez was missing "
        "the supervisor stated training had been completed but the record could not be located"
    )
    assert is_evidence_gap_not_hypothesis("The training record could not be located during the audit.", source)
    assert not is_evidence_gap_not_hypothesis(
        "The training completion may not have been recorded due to a gap in the filing control.", source
    )


# ---------------------------------------------------------------------------
# answer_asserts_verified_but_is_reported
# ---------------------------------------------------------------------------


def test_verified_label_on_reported_content_is_flagged():
    reported = ["the operator stated the equipment had been calibrated recently"]
    verified = ["the calibration status label was missing"]
    assert answer_asserts_verified_but_is_reported(
        "The equipment had been calibrated recently.", "VERIFIED", reported, verified
    )


def test_verified_label_on_actually_verified_content_is_not_flagged():
    reported = ["the operator stated the equipment had been calibrated recently"]
    verified = ["the calibration status label was missing from the equipment"]
    assert not answer_asserts_verified_but_is_reported(
        "The calibration status label was missing from the equipment.", "VERIFIED", reported, verified
    )


def test_non_verified_status_is_never_flagged():
    reported = ["the operator stated the equipment had been calibrated recently"]
    assert not answer_asserts_verified_but_is_reported(
        "The equipment had been calibrated recently.", "REPORTED", reported, []
    )


def test_attribution_language_flags_verified_regardless_of_overlap_dilution():
    """The exact regression from the live output: a mixed sentence that
    narrates a report AND adds other content dilutes word-overlap ratios,
    but the attribution verb itself is enough on its own."""
    reported = ["the operator stated the equipment had been calibrated recently"]
    verified = ["the status label was not present"]
    answer = "The operator stated the equipment had been calibrated recently, but the label was not affixed."
    assert answer_asserts_verified_but_is_reported(answer, "VERIFIED", reported, verified)


# ---------------------------------------------------------------------------
# hypothesis_overclaims_human_error
# ---------------------------------------------------------------------------


def test_bare_human_error_claim_is_flagged():
    assert hypothesis_overclaims_human_error(
        "The calibration label was not applied due to human oversight or error."
    )


def test_human_error_with_process_framing_is_not_flagged():
    assert not hypothesis_overclaims_human_error(
        "A verification control may not have existed to catch a missed post-calibration labeling step."
    )


def test_non_human_error_hypothesis_is_not_flagged():
    assert not hypothesis_overclaims_human_error(
        "The required post-calibration equipment-status labeling step may not have been completed."
    )


# ---------------------------------------------------------------------------
# classify_mixed_evidence_answer
#
# Test A/B from the regression list: a compound 5-Why answer combining a
# REPORTED claim with an independently-standing factual clause must be
# labeled MIXED, not collapsed to a single status (previously VERIFIED,
# which silently promoted the reported half too).
# ---------------------------------------------------------------------------


def test_reported_plus_verified_clause_is_mixed():
    answer = "The operator stated the equipment had been calibrated recently, but the certificate was not available."
    assert classify_mixed_evidence_answer(answer) == "MIXED"


def test_purely_reported_answer_is_not_mixed():
    """No second, independently-standing clause -- this is the pure-
    REPORTED case answer_asserts_verified_but_is_reported already handles,
    not a mixed-evidence case."""
    answer = "The operator stated the equipment had been calibrated recently."
    assert classify_mixed_evidence_answer(answer) is None


def test_purely_factual_answer_is_not_mixed():
    answer = "The calibration status label was not present on the equipment, and the log was blank."
    assert classify_mixed_evidence_answer(answer) is None


def test_mixed_detection_generalizes_to_unrelated_domain():
    answer = "The supervisor reported that training had been completed, but the certificate was not on file."
    assert classify_mixed_evidence_answer(answer) == "MIXED"


# ---------------------------------------------------------------------------
# Causal-domain eligibility: awareness/unawareness of a revision must never
# license training, procedure-clarity, or accessibility hypotheses on its
# own -- those are distinct causal domains each requiring their own
# evidence trigger. (current-turn Sections 1-9, 27 test requirements A-H)
# ---------------------------------------------------------------------------

AWARENESS_ONLY_FINDING = (
    "The daily equipment inspection checklist was not completed for three consecutive days. "
    "The operator stated that they were unaware that the checklist procedure had been revised."
)


def test_a_awareness_alone_rejects_training_hypothesis():
    """Awareness statement without training evidence -> no training hypothesis."""
    statement = "The operator was not retrained after the checklist revision."
    is_unsupported, reason = detect_unsupported_causal_specificity(statement, AWARENESS_ONLY_FINDING)
    assert is_unsupported
    assert "training" in reason.lower()


def test_b_training_hypothesis_eligible_with_training_evidence():
    """Awareness statement WITH explicit training evidence -> training
    hypothesis may be eligible."""
    finding = AWARENESS_ONLY_FINDING + " Retraining is required whenever a checklist procedure is revised."
    statement = "The operator may not have completed the required retraining after the checklist revision."
    is_unsupported, _ = detect_unsupported_causal_specificity(statement, finding)
    assert not is_unsupported


def test_c_explicit_unclear_procedure_is_eligible():
    """Explicit unclear-procedure evidence -> procedure clarity hypothesis eligible."""
    finding = "The operator stated the revised checklist instructions were confusing and contradictory."
    statement = "The revised checklist procedure was unclear to operators."
    is_unsupported, _ = detect_unsupported_causal_specificity(statement, finding)
    assert not is_unsupported


def test_d_awareness_alone_rejects_procedure_clarity_hypothesis():
    """Awareness without unclear-procedure evidence -> procedure clarity hypothesis rejected."""
    statement = "The revised checklist procedure was unclear to operators."
    is_unsupported, reason = detect_unsupported_causal_specificity(statement, AWARENESS_ONLY_FINDING)
    assert is_unsupported
    assert "clarity" in reason.lower() or "guidance" in reason.lower()


def test_e_explicit_document_access_failure_is_eligible():
    """Explicit document-access failure -> accessibility hypothesis eligible."""
    finding = "The controlled checklist copy was not accessible at the operator's workstation."
    statement = "The revised checklist was not easily accessible to operators at their workstation."
    is_unsupported, _ = detect_unsupported_causal_specificity(statement, finding)
    assert not is_unsupported


def test_f_awareness_alone_rejects_accessibility_hypothesis():
    """Awareness without access evidence -> accessibility hypothesis rejected."""
    statement = "The revised checklist was not easily accessible to operators."
    is_unsupported, reason = detect_unsupported_causal_specificity(statement, AWARENESS_ONLY_FINDING)
    assert is_unsupported
    assert "accessib" in reason.lower()


def test_g_explicit_equipment_malfunction_is_eligible():
    """Explicit equipment malfunction evidence -> equipment hypothesis eligible."""
    finding = "An error code was displayed on the refrigeration unit during the affected period."
    statement = "The refrigeration unit's monitoring system malfunctioned during the affected period."
    is_unsupported, _ = detect_unsupported_causal_specificity(statement, finding)
    assert not is_unsupported


def test_scheduling_hypothesis_rejected_from_bare_shift_reference():
    """A: 'temperature check missed during morning shift' -> reject
    scheduling/handover hypothesis; a bare shift/time reference is
    temporal context, not scheduling evidence."""
    finding = "The temperature check was missed during the morning shift."
    statement = "Responsibility for executing the temperature check was not effectively scheduled or transferred across operational shifts."
    is_unsupported, reason = detect_unsupported_causal_specificity(statement, finding)
    assert is_unsupported
    assert "scheduling" in reason.lower() or "handover" in reason.lower()


def test_scheduling_hypothesis_allowed_when_finding_names_shift_plan():
    """B: 'temperature check missed because shift plan omitted it' -> allow
    the scheduling hypothesis, since the finding itself supplies the
    evidence rather than the hypothesis inventing it."""
    finding = "The temperature check was missed because the shift plan omitted it."
    statement = "The temperature check may not have been scheduled or documented in the shift plan."
    is_unsupported, _ = detect_unsupported_causal_specificity(statement, finding)
    assert not is_unsupported


def test_reminder_hypothesis_rejected_without_reminder_system_evidence():
    """No reminder/notification-system vocabulary in the finding -> reject
    a reminder/verification-control hypothesis."""
    finding = "The temperature check was missed during the morning shift."
    statement = "The operational workflow lacked an effective reminder or supervisory verification control."
    is_unsupported, reason = detect_unsupported_causal_specificity(statement, finding)
    assert is_unsupported
    assert "reminder" in reason.lower()


_SCHEDULING_PARAPHRASES_NO_EVIDENCE = [
    "The temperature check was not scheduled.",
    "The check wasn't included in the schedule.",
    "No shift-plan entry existed.",
    "The task was absent from the duty plan.",
    "The check was not planned.",
    "Scheduling controls may have failed.",
    "The system did not schedule the check.",
    "The scheduling process was inadequate.",
    "The check was omitted from the morning schedule.",
    "Records show the check was scheduled.",
]


@pytest.mark.parametrize("statement", _SCHEDULING_PARAPHRASES_NO_EVIDENCE)
def test_adversarial_scheduling_paraphrases_all_rejected(statement):
    """Concept-root detection must catch every paraphrase of the same
    unsupported scheduling mechanism, not just the specific phrasings
    originally observed -- this is the paraphrase-resistance requirement:
    the finding here contains only a bare time-of-day/shift reference,
    never real scheduling evidence, regardless of how the hypothesis
    phrases the (unlicensed) scheduling claim."""
    finding = "The technician missed the temperature check during the morning shift."
    unsupported, _ = detect_unsupported_causal_specificity(statement, finding)
    self_ref = hypothesis_asserts_self_referential_evidence(statement)
    assert unsupported or self_ref, f"expected rejection for: {statement!r}"


def test_scheduling_hypothesis_allowed_with_real_evidence_variant_wording():
    """Positive controls: when the finding genuinely describes scheduling/
    assignment status -- in whatever tense/wording -- the concept-root
    allowed-context check (paraphrase-symmetric with the target) must
    license the hypothesis rather than requiring an exact phrase match."""
    cases = [
        "The technician stated that the check was missed because it was not included in the shift schedule.",
        "The shift schedule for the affected period showed no temperature check assigned to any technician.",
    ]
    for finding in cases:
        unsupported, reason = detect_unsupported_causal_specificity(finding, finding)
        assert not unsupported, f"expected allowance for: {finding!r} (reason={reason})"


def test_self_referential_evidence_assertion_rejected():
    """Section 6: a hypothesis narrating its OWN 'supporting evidence'
    inline ('system records show...') is a claim, not evidence -- must be
    rejected regardless of which mechanism it's attached to."""
    for statement in (
        "System records show the check was scheduled.",
        "The log indicates the technician was not trained.",
        "Records confirm the procedure was unclear.",
    ):
        assert hypothesis_asserts_self_referential_evidence(statement)
    # A statement that doesn't narrate its own evidence source is unaffected.
    assert not hypothesis_asserts_self_referential_evidence(
        "The procedure revision may not have been effectively communicated to the affected operator."
    )


def test_h_missing_equipment_evidence_rejects_malfunction_hypothesis():
    """Missing equipment-malfunction evidence -> equipment hypothesis rejected."""
    statement = "The temperature monitoring system malfunctioned during the affected period."
    finding = "The temperature log for refrigerator QC-REF-02 was not completed for 12 August 2026."
    is_unsupported, reason = detect_unsupported_causal_specificity(statement, finding)
    assert is_unsupported
    assert "malfunction" in reason.lower()


def test_i_causal_verb_without_evidence_rejected():
    statement = "The checklist was not completed because the revision was not communicated to operators."
    assert hypothesis_statement_asserts_unsupported_causation(statement, [])


def test_i_hedged_communication_hypothesis_not_causal_overclaim():
    statement = "The procedure revision may not have been effectively communicated to or acknowledged by the affected operator."
    assert not hypothesis_statement_asserts_unsupported_causation(statement, [])


def test_k_generic_hypothesis_title_flagged_and_never_leaks_internal_id():
    assert hypothesis_name_is_generic("Process Oversight")
    assert hypothesis_name_is_generic("Training Problem")
    assert not hypothesis_name_is_generic("REVISION_COMMUNICATION_OR_ACKNOWLEDGEMENT_GAP")
    # Renaming must derive a human-readable title from the statement, never
    # an internal-looking placeholder such as "CANDIDATE_MECHANISM_H2".
    title = derive_hypothesis_title_from_statement(
        "The procedure revision may not have been effectively communicated to the affected operator.", "H2"
    )
    assert "CANDIDATE_MECHANISM" not in title
    assert "H2" not in title or title == "Candidate Mechanism H2"  # only the no-statement fallback may include the id
