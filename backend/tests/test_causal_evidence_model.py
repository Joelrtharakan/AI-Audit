"""Tests for the canonical causal evidence model across Cases A through J."""

import pytest

from app.agent.analytical_validator import (
    score_hypothesis,
    select_leading_hypothesis,
    validate_causal_graph,
    validate_root_cause_establishment,
    validate_root_cause_state,
)
from app.agent.causal_guard import (
    MechanismInfo,
    hypothesis_contradicts_mechanism,
    hypothesis_contradicts_verified_completion,
    mechanism_from_conflicts,
    validate_hypothesis_quality,
    validate_why_question,
)
from app.agent.claim_extractor import detect_evidence_conflicts, extract_claims
from app.models.agent import (
    CandidateHypothesis,
    CapaAnalysis,
    ConditionalCapaAction,
    EvidenceClaim,
    EvidenceConflict,
    EvidenceItem,
    EvidenceStatus,
    FiveWhyAnalysis,
    FiveWhyStep,
    RootCauseAnalysis,
    RootCauseStatus,
)


def test_case_a_verified_observation_only():
    """Case A: Verified observation only -> Root Cause MUST be NOT_ESTABLISHED."""
    finding = "During audit of Room 102, the daily cleaning log was missing entries for March 10-12."
    claims = extract_claims(finding)
    mechanism = MechanismInfo(statement=None, status="UNKNOWN")
    
    rc = RootCauseAnalysis(
        status=RootCauseStatus.NOT_ESTABLISHED,
        category="TO_BE_CONFIRMED",
        candidate_hypotheses=[
            CandidateHypothesis(
                id="H1",
                name="TASK_ASSIGNMENT_OMISSION",
                statement="Cleaning responsibility was not assigned during shift change.",
                status="POSSIBLE",
                relevance_rank="HIGH",
                evidence_needed="Duty roster and shift handover log",
                confirms_if="Duty roster shows no cleaner assigned for March 10-12",
                refutes_if="Duty roster shows designated cleaner signed for March 10-12",
            )
        ]
    )
    can_est, reasons = validate_root_cause_establishment(rc, mechanism, claims)
    assert not can_est
    assert any("mechanism" in r.lower() for r in reasons)


def test_case_b_verified_observation_with_reported_mechanism():
    """Case B: Verified observation + reported mechanism -> Root Cause NOT_ESTABLISHED (stays hypothesis)."""
    finding = "The autoclave cycle log lacked temperature charts. The operator stated the printer was jammed."
    claims = extract_claims(finding)
    mechanism = MechanismInfo(statement="the printer was jammed", status="REPORTED", polarity="non_recording")
    
    rc = RootCauseAnalysis(
        status=RootCauseStatus.VERIFIED, # Model mistakenly tries to claim VERIFIED
        statement="Printer jam caused missing log",
        supporting_evidence=["Operator statement"]
    )
    warnings = validate_root_cause_state(rc, mechanism)
    assert len(warnings) >= 1
    assert rc.status == "STATED_UNVERIFIED"


def test_case_c_verified_mechanism():
    """Case C: Verified mechanism (audit trail confirms assignment missing) -> May be ESTABLISHED."""
    mechanism = MechanismInfo(
        statement="The audit trail confirms the user was never assigned the task in the system",
        status="VERIFIED",
        polarity="non_performance",
    )
    rc = RootCauseAnalysis(
        status=RootCauseStatus.VERIFIED,
        statement="Task was never assigned in LIMS workflow",
        supporting_evidence=["LIMS audit trail export 2026-03-15"],
        candidate_hypotheses=[
            CandidateHypothesis(
                id="H1",
                name="LIMS_WORKFLOW_ASSIGNMENT_GAP",
                statement="User was never assigned the task in LIMS",
                status="SUPPORTED",
                relevance_rank="HIGH",
                evidence_needed="LIMS audit trail",
            )
        ]
    )
    can_est, reasons = validate_root_cause_establishment(rc, mechanism)
    assert can_est
    assert len(reasons) == 0


def test_case_d_verified_mechanism_contradicts_hypothesis():
    """Case D: Mechanism confirms non-performance -> performed-but-not-recorded hypothesis is REJECTED."""
    mechanism = MechanismInfo(statement="The balance check was omitted", status="VERIFIED", polarity="non_performance")
    hyp_text = "The balance check was performed as required but not recorded in the log."
    assert hypothesis_contradicts_mechanism(hyp_text, mechanism)


def test_case_e_two_conflicting_reports():
    """Case E: Conflicting reports -> Conflict detected, mechanism UNKNOWN, root cause NOT_ESTABLISHED."""
    finding = "The operator stated they were unaware of SOP revision 3, but the trainer claimed the briefing was given to all operators."
    claims = extract_claims(finding)
    conflicts = detect_evidence_conflicts(claims)
    assert len(conflicts) >= 1
    
    mechanism = mechanism_from_conflicts(conflicts)
    assert mechanism.status == "UNKNOWN"

    rc = RootCauseAnalysis(
        status=RootCauseStatus.NOT_ESTABLISHED,
        candidate_hypotheses=[
            CandidateHypothesis(
                id="H1",
                name="TRAINING_ACKNOWLEDGMENT_GAP",
                statement="Training briefing occurred but operator acknowledgment was not documented",
                status="POSSIBLE",
                relevance_rank="HIGH",
                evidence_needed="Training attendance sheet",
                confirms_if="Training sign-off sheet is missing operator signature",
                refutes_if="Operator signature is present on training sheet",
            ),
            CandidateHypothesis(
                id="H2",
                name="BRIEFING_COMMUNICATION_OMISSION",
                statement="Training briefing did not reach the operator due to shift absence",
                status="POSSIBLE",
                relevance_rank="HIGH",
                evidence_needed="Shift attendance records on training date",
                confirms_if="Shift roster shows operator was on leave during briefing",
                refutes_if="Shift roster shows operator present during briefing",
            )
        ]
    )
    can_est, reasons = validate_root_cause_establishment(rc, mechanism, claims, conflicts)
    assert not can_est
    assert any("conflict" in r.lower() for r in reasons)


def test_case_f_verified_completion_contradiction():
    """Case F: Verified completion fact contradicts deficiency hypothesis."""
    verified_facts = ["Training on SOP-LAB-014 was completed on 2026-01-10."]
    hypothesis = "Lack of training on SOP-LAB-014 led to incorrect reagent preparation."
    assert hypothesis_contradicts_verified_completion(hypothesis, verified_facts)


def test_case_g_training_conflict_comprehensive():
    """Case G: The main example finding.
    'The operator stated they had not received training, but the supervisor claimed the training was completed.'
    """
    finding = "The operator stated they had not received training, but the supervisor claimed the training was completed."
    claims = extract_claims(finding)
    conflicts = detect_evidence_conflicts(claims)
    
    # 1. Claims decomposed with attribution
    assert any(c.attribution.value == "PERSON_REPORTED" and c.polarity == "negative" for c in claims)
    assert any(c.attribution.value == "SUPERVISOR_REPORTED" and c.polarity == "positive" for c in claims)
    
    # 2. Conflict detected
    assert len(conflicts) >= 1
    assert conflicts[0].status == "UNRESOLVED"
    
    # 3. Leading hypothesis selection with scoring
    h1 = CandidateHypothesis(
        id="H1",
        name="TRAINING_RECORD_RECONCILIATION_GAP",
        statement="Training was delivered but records were not reconciled or filed in the training matrix.",
        status="POSSIBLE",
        relevance_rank="HIGH",
        evidence_needed="Training sign-in sheet, LMS completion log",
        confirms_if="Physical sign-in exists but LMS status is pending",
        refutes_if="Neither physical sign-in nor LMS record exists",
        discrimination_evidence="Distinguishes documentation omission from non-delivery",
    )
    h2 = CandidateHypothesis(
        id="H2",
        name="TRAINING_DELIVERY_OMISSION",
        statement="Training was scheduled but not delivered to this specific operator.",
        status="POSSIBLE",
        relevance_rank="HIGH",
        evidence_needed="Operator attendance record",
        confirms_if="Attendance record confirms operator absence during session",
        refutes_if="Attendance record contains verified operator signature",
        discrimination_evidence="Distinguishes non-delivery from documentation gap",
    )
    
    score1 = score_hypothesis(h1, claims, conflicts)
    score2 = score_hypothesis(h2, claims, conflicts)
    assert score1 > 0.5
    assert score2 > 0.5


def test_causal_graph_validation_rules():
    """Test consolidated causal graph edge validator."""
    mechanism = MechanismInfo(statement="Operator report only", status="REPORTED", polarity="non_performance")
    rc = RootCauseAnalysis(
        status=RootCauseStatus.VERIFIED,
        statement="Operator was not trained",
        supporting_evidence=[],
    )
    capa = CapaAnalysis(
        status="INVESTIGATION_REQUIRED",
        conditional_actions=[
            ConditionalCapaAction(
                if_cause_confirmed="If unlinked cause is confirmed",
                recommended_action="Do something",
                action_type="SYSTEMIC_ACTION",
                verification_method=None,
            )
        ]
    )
    violations = validate_causal_graph(rc, None, capa, mechanism)
    assert len(violations) >= 2
    assert any("certainty" in v.lower() or "reported" in v.lower() for v in violations)
    assert any("linkage" in v.lower() or "verification" in v.lower() for v in violations)


def test_case_h_unsupported_supervisory_control_rejected():
    """Case H: Unsupported supervisory control hypothesis is rejected."""
    source_text = "The daily equipment inspection checklist was not completed."
    hyp = "The supervisory check or automated verification for daily equipment inspection checklist failed."
    valid, reason = validate_hypothesis_quality(hyp, source_text)
    assert not valid
    assert "supervisory" in reason.lower() or "automated" in reason.lower()


def test_case_i_unsupported_procedure_accessibility_rejected():
    """Case I: Unsupported procedure accessibility hypothesis is rejected."""
    source_text = "The daily equipment inspection checklist was not completed."
    hyp = "The active version of the procedure was not accessible at the point of use."
    valid, reason = validate_hypothesis_quality(hyp, source_text)
    assert not valid
    assert "accessibility" in reason.lower() or "point-of-use" in reason.lower()


def test_case_j_missing_training_record():
    """Case J: Missing training record finding."""
    from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
    finding = "The training record for the operator was missing from the personnel file."
    hyps, plan = build_deterministic_investigation_plan(finding, [])
    assert len(hyps) >= 2
    assert any("not executed" in h.statement.lower() or "controls" in h.statement.lower() or "omitted" in h.statement.lower() for h in hyps)
    assert len(plan.questions) >= 1


def test_case_k_completed_training_missing_documentation():
    """Case K: Training was completed but documentation missing."""
    verified_facts = ["Training was completed on 2026-02-15."]
    hyp = "Lack of training caused the operator deviation."
    assert hypothesis_contradicts_verified_completion(hyp, verified_facts)


def test_case_l_confirmed_root_cause_requires_verified_mechanism():
    """Case L: Confirmed root cause requires verified mechanism and supporting evidence."""
    rc = RootCauseAnalysis(
        status=RootCauseStatus.VERIFIED,
        statement="Power outage caused refrigerator temperature rise",
        supporting_evidence=["Facility UPS event log 2026-08-12"],
    )
    mechanism = MechanismInfo(
        statement="Facility UPS event log confirms power outage",
        status="VERIFIED",
        polarity="non_performance",
    )
    can_est, reasons = validate_root_cause_establishment(rc, mechanism)
    assert can_est
    assert reasons == []


def test_case_m_multiple_equally_strong_hypotheses_tied():
    """Case M: Multiple hypotheses equally ranked return TIED leading hypothesis."""
    from app.agent.analytical_validator import leading_hypothesis_status
    h1 = CandidateHypothesis(
        id="H1",
        name="CAUSE_ONE",
        statement="First plausible cause",
        evidence_needed="Evidence for cause 1",
        status="POSSIBLE",
        relevance_rank="HIGH",
    )
    h2 = CandidateHypothesis(
        id="H2",
        name="CAUSE_TWO",
        statement="Second plausible cause",
        evidence_needed="Evidence for cause 2",
        status="POSSIBLE",
        relevance_rank="HIGH",
    )
    status = leading_hypothesis_status([h1, h2])
    assert status == "TIED"



def test_case_n_no_causal_evidence():
    """Case N: No causal evidence -> 5-Why stops cleanly at the evidence
    boundary, root cause NOT_ESTABLISHED. With a completely empty evidence
    ledger there is nothing to put in a second WHY step -- the chain
    correctly stops at 1 step (Section 14: never force steps to fill five
    rows) and signals the boundary via status_note rather than fabricating
    an UNKNOWN step with no content."""
    from app.agent.nodes.five_why_fallback import build_deterministic_five_why
    finding = "The cleaning checklist was missing."
    five_why = build_deterministic_five_why(finding, [])
    assert len(five_why.steps) == 1
    assert "EVIDENCE BOUNDARY" in five_why.status_note


# ---------------------------------------------------------------------------
# Bare non-performance report vs. genuine reported causal explanation
# (current-turn hardening: a "missed" report with no causal clause must
# produce zero hypotheses but a real, non-presupposing investigation plan,
# not a generic causal bucket, and evidence must not be lost between the
# investigation plan and the final evidence_needed field).
# ---------------------------------------------------------------------------


def _reported_ledger(text: str):
    from app.models.agent import EvidenceItem, EvidenceStatus
    return [EvidenceItem(claim=text, source="REPORTED_STATEMENT", source_reference="x", status=EvidenceStatus.REPORTED, relevance="HIGH")]


def _verified_ledger(text: str):
    from app.models.agent import EvidenceItem, EvidenceStatus
    return [EvidenceItem(claim=text, source="AUDITOR_FINDING", source_reference="x", status=EvidenceStatus.VERIFIED, relevance="HIGH")]


def test_case_o_bare_missed_report_produces_zero_hypotheses():
    """A: bare 'missed during shift' report (no causal clause) -> 0 hypotheses."""
    from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
    finding = "The temperature log for refrigerator QC-REF-02 was not completed for 12 August 2026. The responsible technician confirmed that the temperature check was missed during the morning shift."
    ledger = (
        _verified_ledger("The temperature log for refrigerator QC-REF-02 was not completed for 12 August 2026.")
        + _reported_ledger("technician: the temperature check was missed during the morning shift")
    )
    hyps, plan = build_deterministic_investigation_plan(finding, ledger)
    assert hyps == []
    # Zero hypotheses must still yield a real, non-presupposing investigation
    # plan -- never "no questions generated".
    assert len(plan.questions) >= 3
    assert plan.evidence_to_collect


def test_case_p_reported_causal_explanation_allowed():
    """B: 'missed because not trained' -> one hypothesis reflecting the
    reported reason, status POSSIBLE (never SUPPORTED/VERIFIED from a
    reported statement alone)."""
    from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
    finding = "The temperature log was not completed. The technician stated the check was missed because they had not received retraining."
    ledger = _reported_ledger("technician: the check was missed because they had not received retraining")
    hyps, _ = build_deterministic_investigation_plan(finding, ledger)
    assert len(hyps) == 1
    assert hyps[0].status == "POSSIBLE"
    assert "retraining" in hyps[0].statement.lower()


def test_case_q_no_generic_causal_bucket_hypothesis():
    """E: bare 'missed' report -> no generic execution/task-control/human/
    process/management-factor hypothesis substituted in."""
    from app.agent.causal_guard import hypothesis_statement_is_generic_causal_bucket
    for statement in (
        "Execution or task-control factors may have contributed to the missed check.",
        "Human factors may have contributed to the missed activity.",
        "Process factors may have contributed to the deviation.",
        "Management oversight may have contributed to the missed check.",
    ):
        assert hypothesis_statement_is_generic_causal_bucket(statement)


# ---------------------------------------------------------------------------
# LOW-specificity generic-allegation findings: zero hypotheses, but a real
# foundational investigation plan (never "no questions generated"), and no
# hallucinated 5-Why mechanism, affected object, or evidence-needed field.
# ---------------------------------------------------------------------------

_LOW_SPECIFICITY_FINDINGS = [
    "The department is not following the required procedure correctly.",
    "The team is not following the procedure.",
    "The department is not complying with the required process.",
    "Staff are not following procedures correctly.",
    "The process is not being followed as required.",
    "Required procedures are not being followed.",
    "The department is failing to comply with the procedure.",
]


@pytest.mark.parametrize("finding", _LOW_SPECIFICITY_FINDINGS)
def test_low_specificity_generic_allegation_classified_low(finding):
    from app.services.semantic_subject import classify_finding_specificity
    assert classify_finding_specificity(finding, [], None) == "LOW"


def test_low_specificity_five_why_answer_never_hallucinates_mechanism():
    """The core-reported bug: a 5-Why answer like 'Because staff may lack
    training or awareness' must never survive final verification, even
    when the LLM labeled the step MIXED rather than UNKNOWN."""
    from app.agent.causal_guard import detect_unsupported_causal_specificity, hypothesis_statement_asserts_unsupported_causation
    finding = "The department is not following the required procedure correctly."
    bad_answers = [
        "Because staff may lack training or awareness.",
        "Because training records are not available for verification.",
        "Because documentation may be outdated or incomplete.",
    ]
    for answer in bad_answers:
        unsupported, _ = detect_unsupported_causal_specificity(answer, finding)
        unsupported_causation = hypothesis_statement_asserts_unsupported_causation(answer, [])
        assert unsupported or unsupported_causation, f"expected rejection for 5-Why answer: {answer!r}"


# ---------------------------------------------------------------------------
# Systemic-cause escalation (Level 1 -> Level 4 jump): a hypothesis pairing
# a systemic noun (process/system/control/mechanism/management) with a
# failure verb must be rejected unless the finding independently cites a
# review/audit/documentation/record finding about that process -- checked
# with ONE domain-agnostic pattern, verified across multiple mechanism
# families (training, scheduling, maintenance, calibration, document
# control) rather than one regex per domain.
# ---------------------------------------------------------------------------

_SYSTEMIC_ESCALATION_CASES = [
    ("The daily equipment inspection checklist was not completed for three consecutive days. The operator stated that they were unaware that the checklist procedure had been revised.",
     "The revision process lacked a mechanism to ensure operator awareness."),
    ("The temperature log was incomplete.", "The scheduling system failed to assign the check."),
    ("The equipment maintenance was overdue.", "The maintenance management process failed."),
    ("The training record was unavailable.", "The training system lacked a verification mechanism."),
    ("The calibration certificate was not available.", "The calibration control was not established."),
    ("The document was not communicated.", "The document distribution process was defective."),
]


@pytest.mark.parametrize("finding,hypothesis_statement", _SYSTEMIC_ESCALATION_CASES)
def test_systemic_escalation_rejected_without_process_evidence(finding, hypothesis_statement):
    from app.agent.causal_guard import hypothesis_asserts_systemic_cause_without_process_evidence
    assert hypothesis_asserts_systemic_cause_without_process_evidence(hypothesis_statement, finding)


def test_systemic_hypothesis_allowed_with_genuine_process_evidence():
    from app.agent.causal_guard import hypothesis_asserts_systemic_cause_without_process_evidence
    finding = "Change-control review showed no acknowledgement step was defined for procedure revisions."
    statement = "The revision-control process may have lacked an acknowledgement control."
    assert not hypothesis_asserts_systemic_cause_without_process_evidence(statement, finding)


# ---------------------------------------------------------------------------
# Causal-event inversion: a REPORTED downstream state (e.g. "operator was
# unaware") cannot establish that an upstream event (notification/
# communication/distribution/acknowledgement) did NOT occur. An unhedged
# assertion of that upstream event's absence must be rejected; the SAME
# mechanism, hedged, is a legitimate candidate hypothesis and must survive.
# ---------------------------------------------------------------------------

_EVENT_INVERSION_FINDING = (
    "The daily equipment inspection checklist was not completed for three consecutive days. "
    "The operator stated that they were unaware that the checklist procedure had been revised."
)


def test_reported_awareness_cannot_establish_notice_failure_as_fact():
    """'notice' is a synonym of 'notification' not covered by the notif\\w*
    root alone -- confirms the concept pattern catches the synonym, and
    that a compound hypothesis (notice + documentation clauses) is
    rejected via either clause independently."""
    from app.agent.causal_guard import hypothesis_asserts_unhedged_notification_failure
    statement = "The checklist revision was implemented without prior notice or documentation."
    assert hypothesis_asserts_unhedged_notification_failure(statement, _EVENT_INVERSION_FINDING)


def test_reported_awareness_cannot_establish_notification_failure_as_fact():
    """The exact reported bug: 'occurred without proper notification' is
    an unhedged, unlicensed causal-event-inversion claim."""
    from app.agent.causal_guard import hypothesis_asserts_unhedged_notification_failure
    statement = "The checklist revision occurred without proper notification to the operator."
    assert hypothesis_asserts_unhedged_notification_failure(statement, _EVENT_INVERSION_FINDING)


def test_hedged_communication_hypothesis_survives_notification_guard():
    """The legitimate Level-2 candidate must NOT be caught by the same
    guard that rejects the unhedged Level-3 claim -- hedging distinguishes
    a candidate hypothesis from an asserted fact."""
    from app.agent.causal_guard import hypothesis_asserts_unhedged_notification_failure
    statement = (
        "The revision affecting daily equipment inspection checklist may not have been "
        "effectively communicated to or acknowledged by the affected personnel."
    )
    assert not hypothesis_asserts_unhedged_notification_failure(statement, _EVENT_INVERSION_FINDING)


# ---------------------------------------------------------------------------
# Structural change-event-defect guard: catches paraphrases the concept-root
# notification guard misses entirely ("documented", "disseminated",
# "accessible" are not notification/communication/distribution/
# acknowledgement/announcement roots) by matching SENTENCE SHAPE (change-
# event subject + unhedged negative-outcome clause) instead of vocabulary.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("statement", [
    "The checklist revision was not properly documented or accessible to the operator.",
    "The checklist revision was not properly documented or disseminated.",
    "The checklist revision was implemented without prior notice or documentation.",
    "The procedure change lacked a formal review before rollout.",
])
def test_unhedged_change_event_defect_claims_are_rejected_regardless_of_vocabulary(statement):
    from app.agent.causal_guard import hypothesis_asserts_unlicensed_change_event_defect
    assert hypothesis_asserts_unlicensed_change_event_defect(statement, _EVENT_INVERSION_FINDING)


def test_hedged_change_event_defect_claim_survives_structural_guard():
    from app.agent.causal_guard import hypothesis_asserts_unlicensed_change_event_defect
    statement = (
        "The revision affecting daily equipment inspection checklist may not have been "
        "effectively communicated to or acknowledged by the affected personnel."
    )
    assert not hypothesis_asserts_unlicensed_change_event_defect(statement, _EVENT_INVERSION_FINDING)


def test_change_event_defect_claim_licensed_when_finding_independently_establishes_it():
    from app.agent.causal_guard import hypothesis_asserts_unlicensed_change_event_defect
    finding = (
        "The checklist revision was not properly documented in the change control log. "
        "The operator stated they were unaware the procedure had been revised."
    )
    statement = "The checklist revision was not properly documented."
    assert not hypothesis_asserts_unlicensed_change_event_defect(statement, finding)


@pytest.mark.parametrize("finding,statement", [
    ("The calibration procedure for scale SC-04 was updated. "
     "The technician stated they were not aware of the update.",
     "The calibration procedure update was not properly rolled out without any prior communication."),
    ("The supplier qualification policy was amended in March. "
     "The reviewer said they had not seen the amended version.",
     "The policy amendment was not adequately disseminated to reviewers."),
])
def test_change_event_defect_guard_generalizes_across_domains(finding, statement):
    from app.agent.causal_guard import hypothesis_asserts_unlicensed_change_event_defect
    assert hypothesis_asserts_unlicensed_change_event_defect(statement, finding)


@pytest.mark.parametrize("finding,statement", [
    ("The document-control record shows the checklist revision was not approved prior to issue. "
     "The operator stated they were unaware the procedure had been revised.",
     "The checklist revision was not approved through the document-control process before issue."),
    ("The change-control log shows the checklist revision update was not reviewed by quality "
     "before release. The operator stated they were unaware of the revision.",
     "The checklist revision update was not reviewed before release."),
])
def test_change_event_defect_hypothesis_allowed_when_finding_names_the_control_record(finding, statement):
    """Item 9/22: a hedged or unhedged change-event defect claim grounded in
    a VERIFIED record the finding itself names (document-control / change-
    control log) must NOT be rejected -- the guard targets UNLICENSED
    inference from a downstream awareness report, not every statement about
    the change event. Over-correcting to reject evidence-grounded claims
    would violate Section 22 (evidence-grounded specificity, not maximum
    conservatism)."""
    from app.agent.causal_guard import hypothesis_asserts_unlicensed_change_event_defect
    assert not hypothesis_asserts_unlicensed_change_event_defect(statement, finding)


@pytest.mark.parametrize("finding,statement", [
    ("The operator stated they were unaware of the retraining.",
     "The technician was not notified of the retraining requirement."),
    ("The technician confirmed they did not know about the revised procedure.",
     "Management failed to communicate the change."),
    ("The operator stated they had not been informed.",
     "The department did not distribute the updated checklist."),
])
def test_causal_event_inversion_rejected_across_domains(finding, statement):
    from app.agent.causal_guard import hypothesis_asserts_unhedged_notification_failure
    assert hypothesis_asserts_unhedged_notification_failure(statement, finding)


@pytest.mark.parametrize("finding,statement", [
    ("Distribution log shows the operator was omitted from the checklist distribution.",
     "The revision may not have been distributed to the operator."),
    ("Change-control procedure requires acknowledgement and no acknowledgement record exists.",
     "The required acknowledgement may not have been obtained."),
])
def test_notification_hypothesis_allowed_with_explicit_evidence(finding, statement):
    from app.agent.causal_guard import hypothesis_asserts_unhedged_notification_failure
    assert not hypothesis_asserts_unhedged_notification_failure(statement, finding)


def test_investigation_question_split_into_independent_branches():
    """Section 5: the unrecorded-performance and documented-event branches
    must be two separate questions, not one combined either/or question."""
    from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
    finding = "The temperature log for refrigerator QC-REF-02 was not completed for 12 August 2026. The responsible technician confirmed that the temperature check was missed during the morning shift."
    ledger = (
        _verified_ledger("The temperature log for refrigerator QC-REF-02 was not completed for 12 August 2026.")
        + _reported_ledger("technician: the temperature check was missed during the morning shift")
    )
    _, plan = build_deterministic_investigation_plan(finding, ledger)
    question_texts = [q.question.lower() for q in plan.questions]
    assert any("performed but not recorded" in q for q in question_texts)
    assert any("documented event" in q for q in question_texts)
    # Never combined into one question via "or"
    assert not any("performed but not recorded" in q and "documented event" in q for q in question_texts)

