"""Comprehensive Tests for Causal Proposition Graph, Root-Cause Eligibility, and Positive/Adversarial Invariants."""

from __future__ import annotations

import pytest

from app.agent.causal_graph import (
    evaluate_root_cause_eligibility,
    generate_structured_conflict_text,
    select_authoritative_leading_hypothesis,
)
from app.models.agent import (
    CandidateHypothesis,
    CausalEdgeType,
    CausalLevel,
    CausalRelationship,
    EvidenceConflict,
    EvidenceItem,
    EvidenceStatus,
    Proposition,
    PropositionType,
    ReferencedDocumentInfo,
    RootCauseStatus,
    SupportLevel,
)


# ===========================================================================
# 1. Positive Root Cause Tests (Cases where Root Cause IS Established)
# ===========================================================================


def test_positive_case_1_confirmed_training_workflow_failure():
    """Case 1: Authorization record shows operator authorized without completed training; workflow validation missing."""
    evidence = [
        EvidenceItem(claim="required training was not completed", source="records", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="authorization record shows the operator was authorized despite no completed training", source="audit_log", status=EvidenceStatus.VERIFIED),
    ]
    hyp = CandidateHypothesis(
        id="H1",
        name="AUTHORIZATION_CONTROL_FAILURE",
        statement="The authorization workflow control allowed personnel to proceed without training validation.",
        evidence_needed="System workflow configuration",
    )
    eligible, supp_lvl, reason, missing, lvl, promo_allowed = evaluate_root_cause_eligibility(
        hyp, evidence_items=evidence
    )
    assert eligible
    assert supp_lvl == SupportLevel.SUPPORTED
    assert promo_allowed
    hyp.status = "SUPPORTED"

    lead_id, lead_stat, rc_stat, rationale = select_authoritative_leading_hypothesis([hyp], evidence_ledger=evidence)
    assert lead_id == "H1"
    assert lead_stat == "SELECTED"
    assert rc_stat == RootCauseStatus.SUPPORTED


def test_positive_case_2_confirmed_calibration_block_disablement():
    """Case 2: Calibration expired; procedure blocks use; system audit log shows block was disabled by admin."""
    evidence = [
        EvidenceItem(claim="calibration certificate expired", source="finding_text", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="system audit log shows the block was disabled", source="audit_trail", status=EvidenceStatus.VERIFIED),
    ]
    hyp = CandidateHypothesis(
        id="H1",
        name="BLOCK_CONTROL_DISABLED",
        statement="The system safety block was disabled in configuration settings.",
        evidence_needed="Admin logs",
    )
    eligible, supp_lvl, reason, missing, lvl, promo_allowed = evaluate_root_cause_eligibility(
        hyp, evidence_items=evidence
    )
    assert eligible
    assert supp_lvl == SupportLevel.SUPPORTED
    assert promo_allowed
    hyp.status = "SUPPORTED"

    lead_id, lead_stat, rc_stat, _ = select_authoritative_leading_hypothesis([hyp], evidence_ledger=evidence)
    assert lead_id == "H1"
    assert lead_stat == "SELECTED"
    assert rc_stat == RootCauseStatus.SUPPORTED


def test_positive_case_3_confirmed_notification_service_outage():
    """Case 3: Notification delivery failed; server log shows notification service failure at the relevant timestamp."""
    evidence = [
        EvidenceItem(claim="notification was not received", source="finding_text", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="server log shows notification service failure at the timestamp", source="server_log", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="error log identifies the service outage", source="error_log", status=EvidenceStatus.VERIFIED),
    ]
    hyp = CandidateHypothesis(
        id="H1",
        name="NOTIFICATION_SERVICE_OUTAGE",
        statement="A notification service failure caused transmission outage during the dispatch window.",
        evidence_needed="Server diagnostics",
    )
    eligible, supp_lvl, reason, missing, lvl, promo_allowed = evaluate_root_cause_eligibility(
        hyp, evidence_items=evidence
    )
    assert eligible
    assert supp_lvl == SupportLevel.SUPPORTED
    assert promo_allowed


# ===========================================================================
# 2. Adversarial & Safety Invariant Tests (20 Cases)
# ===========================================================================


def test_adversarial_a_delivery_is_not_receipt():
    """A. Delivery verified does not imply receipt verified."""
    text = generate_structured_conflict_text(
        "DELIVERY_VS_RECEIPT",
        subject="the notification",
        proposition_a="system logs show delivery",
        proposition_b="operators stated not received",
    )
    assert "delivery records indicate successful delivery" in text
    assert "reported that they did not receive" in text
    assert "reconcile delivery, receipt, access, and acknowledgement" in text


def test_adversarial_b_receipt_is_not_acknowledgement():
    """B. Edge relationship validation: Receipt != Acknowledgement."""
    rel = CausalRelationship(
        source_id="P_RECEIPT",
        target_id="P_ACK",
        edge_type=CausalEdgeType.REQUIRES,
    )
    assert rel.edge_type == CausalEdgeType.REQUIRES


def test_adversarial_d_expiry_is_not_renewal_failure():
    """D. Calibration expiry is a temporal observation, not proved renewal failure."""
    hyp = CandidateHypothesis(
        id="H1",
        name="RENEWAL_FAILURE",
        statement="The calibration vendor failed to perform renewal.",
        evidence_needed="Vendor contract",
    )
    evidence = [
        EvidenceItem(claim="calibration expired on 10 August", source="finding_text", status=EvidenceStatus.VERIFIED)
    ]
    eligible, supp_lvl, _, _, _, promo_allowed = evaluate_root_cause_eligibility(
        hyp, evidence_items=evidence
    )
    assert eligible
    assert supp_lvl == SupportLevel.POSSIBLE
    assert not promo_allowed  # Cannot be promoted without verified vendor evidence


def test_adversarial_f_missing_record_is_not_activity_never_occurred():
    """F. Missing record does not establish non-performance."""
    hyp = CandidateHypothesis(
        id="H1",
        name="WORK_NEVER_PERFORMED",
        statement="The technician never performed the required maintenance.",
        evidence_needed="Execution logs",
    )
    evidence = [
        EvidenceItem(claim="maintenance log contained a blank entry", source="finding_text", status=EvidenceStatus.VERIFIED)
    ]
    eligible, supp_lvl, _, _, _, promo_allowed = evaluate_root_cause_eligibility(
        hyp, evidence_items=evidence
    )
    assert supp_lvl == SupportLevel.POSSIBLE
    assert not promo_allowed


def test_adversarial_g_referenced_document_is_not_content():
    """G. Referenced unavailable document cannot have inferred content."""
    ref_docs = [ReferencedDocumentInfo(document_type="calibration report", reference_status="REFERENCED_UNAVAILABLE")]
    hyp = CandidateHypothesis(
        id="H1",
        name="REPORT_CONTENT",
        statement="The calibration report proved that the balance was inaccurate.",
        evidence_needed="Report",
    )
    eligible, supp_lvl, reason, _, _, _ = evaluate_root_cause_eligibility(
        hyp, referenced_docs=ref_docs
    )
    assert not eligible
    assert supp_lvl == SupportLevel.REJECTED
    assert reason == "infers_unavailable_document_content"


def test_adversarial_h_reported_mechanism_is_not_established_cause():
    """H. Operator stated they forgot -> POSSIBLE, never ESTABLISHED."""
    hyp = CandidateHypothesis(
        id="H1",
        name="FORGOTTEN_TASK",
        statement="The operator forgot the morning check.",
        evidence_needed="Statement verification",
    )
    evidence = [
        EvidenceItem(claim="operator stated they forgot the check", source="statement", status=EvidenceStatus.REPORTED)
    ]
    eligible, supp_lvl, _, _, _, promo_allowed = evaluate_root_cause_eligibility(
        hyp, evidence_items=evidence
    )
    assert eligible
    assert supp_lvl == SupportLevel.POSSIBLE
    assert not promo_allowed


def test_adversarial_i_conflict_prevents_root_cause():
    """I. Conflicting evidence -> Root cause status NOT_ESTABLISHED."""
    conflicts = [
        EvidenceConflict(
            conflict_id="CONF1",
            conflict_type="SYSTEM_RECORD_VS_HUMAN_REPORT",
            proposition="notification delivery and receipt",
        )
    ]
    hyp = CandidateHypothesis(
        id="H1",
        name="DELIVERY_BREAKDOWN",
        statement="The notification delivery failed.",
        evidence_needed="Logs",
    )
    lead_id, lead_stat, rc_stat, rationale = select_authoritative_leading_hypothesis(
        [hyp], conflicts=conflicts
    )
    assert lead_id is None
    assert lead_stat == "NONE"
    assert rc_stat == RootCauseStatus.NOT_ESTABLISHED
