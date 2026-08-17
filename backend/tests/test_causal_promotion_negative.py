"""Negative Causal Promotion Invariant Tests (Mandatory Cases 1 to 5).

Verifies that when evidence is incomplete, merely temporal, reported, conflicted,
or consists of a missing record, the engine strictly outputs Root Cause: NOT_ESTABLISHED
and Leading Hypothesis: NONE.
"""

from __future__ import annotations

import pytest

from app.agent.causal_graph import (
    evaluate_root_cause_eligibility,
    select_authoritative_leading_hypothesis,
)
from app.models.agent import (
    CandidateHypothesis,
    CausalLevel,
    EvidenceConflict,
    EvidenceItem,
    EvidenceStatus,
    RootCauseStatus,
    SupportLevel,
)


def test_negative_case_1_calibration_expiry_and_use_only():
    """Case 1: Expired calibration + equipment used without evidence explaining why.
    Must NOT infer renewal failure or calibration-control failure as established cause.
    """
    evidence = [
        EvidenceItem(claim="calibration certificate for balance BAL-014 expired on 10 August 2026", source="cert", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="the balance was used on 12 August 2026", source="log", status=EvidenceStatus.VERIFIED),
    ]
    hyp = CandidateHypothesis(
        id="H1",
        name="RENEWAL_FAILURE",
        statement="The external calibration vendor failed to renew the certificate.",
        evidence_needed="Vendor communications",
    )
    eligible, supp_lvl, reason, missing, lvl, promo_allowed = evaluate_root_cause_eligibility(
        hyp, evidence_items=evidence
    )
    assert eligible
    assert supp_lvl == SupportLevel.POSSIBLE
    assert not promo_allowed

    lead_id, lead_stat, rc_stat, rationale = select_authoritative_leading_hypothesis([hyp], evidence_ledger=evidence)
    assert lead_id is None
    assert lead_stat == "NONE"
    assert rc_stat == RootCauseStatus.NOT_ESTABLISHED


def test_negative_case_2_missed_temperature_check_reported():
    """Case 2: Missed temperature check + technician says it was missed.
    Must NOT infer scheduling failure or lack of supervision as established cause.
    """
    evidence = [
        EvidenceItem(claim="temperature log for refrigerator QC-REF-02 was incomplete", source="log", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="technician confirmed check was missed during morning shift", source="statement", status=EvidenceStatus.REPORTED),
    ]
    hyp = CandidateHypothesis(
        id="H1",
        name="SCHEDULING_FAILURE",
        statement="Shift scheduling procedures failed to allocate monitoring time.",
        evidence_needed="Roster records",
    )
    eligible, supp_lvl, reason, missing, lvl, promo_allowed = evaluate_root_cause_eligibility(
        hyp, evidence_items=evidence
    )
    assert eligible
    assert supp_lvl == SupportLevel.POSSIBLE
    assert not promo_allowed

    lead_id, lead_stat, rc_stat, rationale = select_authoritative_leading_hypothesis([hyp], evidence_ledger=evidence)
    assert lead_id is None
    assert lead_stat == "NONE"
    assert rc_stat == RootCauseStatus.NOT_ESTABLISHED


def test_negative_case_3_operator_unaware_of_revision():
    """Case 3: Operator unaware of procedure revision.
    Must NOT infer systemic communication failure without independent proof.
    """
    evidence = [
        EvidenceItem(claim="operator stated they were unaware that the procedure was revised", source="statement", status=EvidenceStatus.REPORTED)
    ]
    hyp = CandidateHypothesis(
        id="H1",
        name="COMMUNICATION_SYSTEM_BREAKDOWN",
        statement="Document control communication systems failed across the facility.",
        evidence_needed="Facility distribution logs",
    )
    eligible, supp_lvl, reason, missing, lvl, promo_allowed = evaluate_root_cause_eligibility(
        hyp, evidence_items=evidence
    )
    assert eligible
    assert supp_lvl == SupportLevel.POSSIBLE
    assert not promo_allowed

    lead_id, lead_stat, rc_stat, rationale = select_authoritative_leading_hypothesis([hyp], evidence_ledger=evidence)
    assert lead_id is None
    assert lead_stat == "NONE"
    assert rc_stat == RootCauseStatus.NOT_ESTABLISHED


def test_negative_case_4_notification_delivery_vs_receipt_conflict():
    """Case 4: Notification delivery log says delivered + operators say not received.
    Must NOT infer technical failure or operator fault.
    """
    conflicts = [
        EvidenceConflict(
            conflict_id="CONF1",
            conflict_type="SYSTEM_RECORD_VS_HUMAN_REPORT",
            proposition="notification delivery vs operator receipt",
        )
    ]
    hyp = CandidateHypothesis(
        id="H1",
        name="NOTIFICATION_GATEWAY_CRASH",
        statement="The notification gateway crashed during transmission.",
        evidence_needed="Gateway server logs",
    )
    lead_id, lead_stat, rc_stat, rationale = select_authoritative_leading_hypothesis([hyp], conflicts=conflicts)
    assert lead_id is None
    assert lead_stat == "NONE"
    assert rc_stat == RootCauseStatus.NOT_ESTABLISHED


def test_negative_case_5_training_record_missing():
    """Case 5: Training record missing.
    Must NOT infer training non-completion. Missing evidence = EVIDENCE_NOT_AVAILABLE.
    """
    evidence = [
        EvidenceItem(claim="training certificate was missing from the physical binder", source="audit_check", status=EvidenceStatus.VERIFIED)
    ]
    hyp = CandidateHypothesis(
        id="H1",
        name="TRAINING_OMISSION",
        statement="The operator never received training.",
        evidence_needed="LMS training records",
    )
    eligible, supp_lvl, reason, missing, lvl, promo_allowed = evaluate_root_cause_eligibility(
        hyp, evidence_items=evidence
    )
    assert eligible
    assert supp_lvl == SupportLevel.POSSIBLE
    assert not promo_allowed

    lead_id, lead_stat, rc_stat, rationale = select_authoritative_leading_hypothesis([hyp], evidence_ledger=evidence)
    assert lead_id is None
    assert lead_stat == "NONE"
    assert rc_stat == RootCauseStatus.NOT_ESTABLISHED
