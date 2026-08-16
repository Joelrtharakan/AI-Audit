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
    """Case N: No causal evidence -> 5-Why stops cleanly, root cause NOT_ESTABLISHED."""
    from app.agent.nodes.five_why_fallback import build_deterministic_five_why
    finding = "The cleaning checklist was missing."
    five_why = build_deterministic_five_why(finding, [])
    assert len(five_why.steps) == 2
    assert five_why.steps[-1].status == "UNKNOWN"

