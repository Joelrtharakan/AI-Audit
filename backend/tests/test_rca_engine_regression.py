"""Comprehensive regression tests for the LQMS RCA engine across all 10 critical cases (A-J)."""

import pytest

from app.agent.analytical_validator import (
    score_hypothesis,
    select_leading_hypothesis,
    validate_causal_graph,
    validate_root_cause_establishment,
)
from app.agent.causal_guard import (
    MechanismInfo,
    extract_immediate_mechanism,
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
    EvidenceStatus,
    FiveWhyAnalysis,
    FiveWhyStep,
    RootCauseAnalysis,
    RootCauseStatus,
)


def test_regression_case_a_verified_observation_only():
    """Case A: Verified observation only.
    Input: 'During the audit of Room 102, the daily cleaning log was missing entries for March 10-12.'
    Expected: Root cause = NOT_ESTABLISHED, mechanism = None/UNKNOWN, hypotheses allowed.
    """
    finding = "During the audit of Room 102, the daily cleaning log was missing entries for March 10-12."
    claims = extract_claims(finding)
    assert len(claims) >= 1
    assert all(c.status == EvidenceStatus.VERIFIED for c in claims)
    
    mechanism = extract_immediate_mechanism([], [c.text for c in claims])
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, category="TO_BE_CONFIRMED")
    can_est, reasons = validate_root_cause_establishment(rc, mechanism, claims)
    assert not can_est


def test_regression_case_b_verified_obs_and_reported_mechanism():
    """Case B: Verified observation + reported mechanism.
    Input: 'The autoclave cycle log lacked temperature charts. The operator stated the printer was jammed.'
    Expected: Observation = VERIFIED, Mechanism = REPORTED, Root cause = NOT_ESTABLISHED.
    """
    finding = "The autoclave cycle log lacked temperature charts. The operator stated the printer was jammed."
    claims = extract_claims(finding)
    assert any(c.status == EvidenceStatus.VERIFIED for c in claims)
    assert any(c.status == EvidenceStatus.REPORTED for c in claims)
    
    reported_texts = [c.text for c in claims if c.status == EvidenceStatus.REPORTED]
    fact_texts = [c.text for c in claims if c.status == EvidenceStatus.VERIFIED]
    mechanism = extract_immediate_mechanism(reported_texts, fact_texts)
    assert mechanism.status == "REPORTED"
    
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, category="TO_BE_CONFIRMED")
    can_est, _ = validate_root_cause_establishment(rc, mechanism, claims)
    assert not can_est


def test_regression_case_c_verified_mechanism():
    """Case C: Verified mechanism.
    Input: 'The calibration record for pH meter PH-02 was overdue. The audit trail confirmed calibration was not scheduled.'
    Expected: Mechanism = VERIFIED, Root cause may be ESTABLISHED if evidence confirms why.
    """
    mechanism = MechanismInfo(
        statement="Audit trail confirmed calibration was not scheduled in maintenance module",
        status="VERIFIED",
        polarity="non_performance",
    )
    rc = RootCauseAnalysis(
        status=RootCauseStatus.VERIFIED,
        statement="Maintenance module automatic scheduling was disabled for PH-02",
        supporting_evidence=["System audit trail export"],
    )
    can_est, reasons = validate_root_cause_establishment(rc, mechanism)
    assert can_est


def test_regression_case_d_verified_mechanism_contradicts_hypothesis():
    """Case D: Verified mechanism contradicts hypothesis.
    Mechanism: non-performance confirmed.
    Hypothesis: 'performed but not recorded' -> REJECTED.
    """
    mechanism = MechanismInfo(statement="Activity was omitted", status="VERIFIED", polarity="non_performance")
    hyp = "Activity was performed but documentation was omitted."
    assert hypothesis_contradicts_mechanism(hyp, mechanism)


def test_regression_case_e_two_conflicting_reported_statements():
    """Case E: Two conflicting reported statements.
    'The operator stated they had not received training, but the supervisor claimed the training was completed.'
    Expected: Both claims REPORTED, Conflict detected, Root cause NOT_ESTABLISHED.
    """
    finding = "The operator stated they had not received training, but the supervisor claimed the training was completed."
    claims = extract_claims(finding)
    conflicts = detect_evidence_conflicts(claims)
    assert len(conflicts) >= 1
    assert conflicts[0].status == "UNRESOLVED"
    
    mechanism = mechanism_from_conflicts(conflicts)
    assert mechanism.status == "UNKNOWN"


def test_regression_case_f_verified_completion_contradiction():
    """Case F: Verified completion fact.
    Verified: 'Training on SOP-001 was completed.'
    Hypothesis: 'Lack of training on SOP-001' -> REJECTED.
    """
    verified = ["Training on SOP-001 was completed on January 15, 2026."]
    hyp = "Lack of training on SOP-001 caused the error."
    assert hypothesis_contradicts_verified_completion(hyp, verified)


def test_regression_case_g_training_conflict():
    """Case G: Compound training conflict.
    Hypotheses must answer WHY (e.g. record reconciliation gap vs delivery omission).
    """
    h1 = CandidateHypothesis(
        id="H1",
        name="TRAINING_RECORD_RECONCILIATION_GAP",
        statement="Training was delivered but records were not reconciled in LMS.",
        status="POSSIBLE",
        relevance_rank="HIGH",
        evidence_needed="Training attendance sheet vs LMS database",
        confirms_if="Attendance sheet is signed but LMS record is missing",
        refutes_if="Neither attendance sheet nor LMS record exists",
    )
    valid, reason = validate_hypothesis_quality(h1.statement, "The operator stated...")
    assert valid


def test_regression_case_h_calibration_label_missing():
    """Case H: Calibration label missing on in-calibration balance.
    'Balance BAL-04 was observed without a calibration sticker; the calibration certificate confirmed active calibration.'
    Expected: Physical tagging gap hypothesis, NOT fabricated calibration failure.
    """
    finding = "Balance BAL-04 was observed without a calibration sticker; the calibration certificate confirmed active calibration."
    verified = ["The calibration certificate confirmed active calibration."]
    hyp_bad = "Lack of calibration on Balance BAL-04 led to inaccurate measurements."
    assert hypothesis_contradicts_verified_completion(hyp_bad, verified)


def test_regression_case_i_checklist_missed_revision():
    """Case I: Checklist missed + unknown revision.
    'Operator used Rev 2 checklist instead of Rev 3.'
    Hypothesis must focus on document distribution/point-of-use control.
    """
    hyp = "Outdated document copy remained accessible at the workstation due to document retrieval gap."
    valid, _ = validate_hypothesis_quality(hyp, "Operator used Rev 2 checklist instead of Rev 3.")
    assert valid


def test_regression_case_j_previous_capa_ineffective():
    """Case J: Previous CAPA recurrence.
    'Recurrence of nonconformity NC-2025-042 observed; prior CAPA-012 actions were completed.'
    Expected: Prior CAPA effectiveness gap hypothesis.
    """
    hyp = "Prior corrective action verification did not evaluate execution controls across all operating shifts."
    valid, _ = validate_hypothesis_quality(hyp, "Recurrence of NC-2025-042 after CAPA-012.")
    assert valid


def test_why_question_quality_gate():
    """Ensure validate_why_question rejects invalid patterns."""
    # Rejects reporting behavior question
    valid, reason = validate_why_question("Why did personnel report that the check was missed?")
    assert not valid
    assert "reporting behavior" in reason.lower()

    # Rejects question restating previous answer
    valid, reason = validate_why_question(
        "Why was the log incomplete?",
        previous_answer="The log was incomplete due to missing entries.",
    )
    assert not valid

    # Accepts valid causal progression question
    valid, reason = validate_why_question(
        "Why was the calibration not scheduled in the maintenance module?",
        previous_answer="The automated reminder was disabled during software migration.",
        observation="The calibration log was overdue.",
    )
    assert valid
    assert reason is None
