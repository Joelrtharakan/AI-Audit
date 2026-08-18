"""Positive Root Cause Promotion Tests (Mandatory Cases A, B, C).

Verifies that the engine successfully promotes root causes to ESTABLISHED/SUPPORTED
WHEN objective evidence actually establishes the causal control failure.
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
    EvidenceItem,
    EvidenceStatus,
    RootCauseStatus,
    SupportLevel,
)


def test_case_a_confirmed_training_control_failure():
    """CASE A — CONFIRMED TRAINING CONTROL FAILURE
    C1: Training was mandatory before authorization. (VERIFIED)
    C2: Training completion record shows operator did not complete training. (VERIFIED)
    C3: Authorization record shows operator was nevertheless authorized. (VERIFIED)
    C4: Authorization workflow requires training verification. (VERIFIED)
    C5: Workflow audit log shows training verification was bypassed. (VERIFIED)

    Expected: Root Cause Status = SUPPORTED/ESTABLISHED, Leading = H1 (bypass of required training verification).
    """
    evidence = [
        EvidenceItem(claim="C1: Training was mandatory before authorization", source="procedure", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C2: Training completion record shows operator did not complete training", source="lms", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C3: Authorization record shows operator was authorized without completed training", source="access_system", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C4: The authorization workflow requires training verification", source="workflow_spec", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C5: The workflow audit log shows training verification was bypassed", source="audit_trail", status=EvidenceStatus.VERIFIED),
    ]

    hyp = CandidateHypothesis(
        id="H1",
        name="TRAINING_VERIFICATION_BYPASS",
        statement="Required training verification was bypassed in the authorization process.",
        evidence_needed="Workflow audit trail",
        supporting_evidence=[e.claim for e in evidence],
        causal_level=CausalLevel.L5_SYSTEMIC_CAUSE,
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
    assert rc_stat in (RootCauseStatus.SUPPORTED, RootCauseStatus.ESTABLISHED)


def test_case_b_confirmed_calibration_control_failure():
    """CASE B — CONFIRMED CALIBRATION CONTROL FAILURE
    C1: Calibration expired on 10 August. (VERIFIED)
    C2: Equipment-use procedure prohibits use after expiry. (VERIFIED)
    C3: Equipment was used on 12 August. (VERIFIED)
    C4: System records show expiry control was disabled on 11 August. (VERIFIED)
    C5: Administrator audit record confirms the control was intentionally disabled. (VERIFIED)

    Expected: Root Cause Status = SUPPORTED/ESTABLISHED, Leading = H1 (expiry-use control disabled).
    """
    evidence = [
        EvidenceItem(claim="C1: Calibration expired on 10 August", source="cert", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C2: Equipment-use procedure prohibits use after expiry", source="sop", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C3: Equipment was used on 12 August", source="log", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C4: System audit log shows the block was disabled on 11 August", source="system_log", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C5: Administrator audit record confirms the control was disabled", source="admin_trail", status=EvidenceStatus.VERIFIED),
    ]

    hyp = CandidateHypothesis(
        id="H1",
        name="EXPIRY_CONTROL_DISABLED",
        statement="The expiry-use control block was disabled, permitting equipment use after calibration expiry.",
        evidence_needed="Admin audit log",
        supporting_evidence=[e.claim for e in evidence],
        causal_level=CausalLevel.L5_SYSTEMIC_CAUSE,
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
    assert rc_stat in (RootCauseStatus.SUPPORTED, RootCauseStatus.ESTABLISHED)


def test_case_c_confirmed_notification_service_failure():
    """CASE C — CONFIRMED NOTIFICATION SERVICE FAILURE
    C1: Notification was required. (VERIFIED)
    C2: Notification was not delivered. (VERIFIED)
    C3: Server logs show notification service failure at the exact required time. (VERIFIED)
    C4: Service error prevented delivery. (VERIFIED)
    C5: Affected recipients did not receive notification during outage. (VERIFIED)

    Expected: Root Cause Status = SUPPORTED/ESTABLISHED, Leading = H1 (notification service failure).
    """
    evidence = [
        EvidenceItem(claim="C1: Notification was required", source="sop", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C2: Notification was not delivered", source="dispatch_log", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C3: Server log shows notification service failure at the timestamp", source="server_log", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C4: Error log identifies the service outage", source="error_log", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C5: Recipients did not receive notification during the outage", source="delivery_report", status=EvidenceStatus.VERIFIED),
    ]

    hyp = CandidateHypothesis(
        id="H1",
        name="NOTIFICATION_SERVICE_OUTAGE",
        statement="Notification service failure during dispatch outage prevented required delivery.",
        evidence_needed="Server diagnostics",
        supporting_evidence=[e.claim for e in evidence],
        causal_level=CausalLevel.L5_SYSTEMIC_CAUSE,
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
    assert rc_stat in (RootCauseStatus.SUPPORTED, RootCauseStatus.ESTABLISHED)


def test_case_d_confirmed_duplicate_payment_detection_failure():
    """CASE D — CONFIRMED DUPLICATE PAYMENT DETECTION CONTROL FAILURE
    C1: Duplicate payment rule was active and mandatory. (VERIFIED)
    C2: Second invoice with identical vendor ID and invoice number was submitted. (VERIFIED)
    C3: System audit log confirms duplicate detection rule was disabled at 14:32. (VERIFIED)
    C4: Transaction log confirms duplicate payment of ₹1,25,000 was executed at 14:47 without validation. (VERIFIED)
    
    Expected: Root Cause Status = SUPPORTED/ESTABLISHED, Leading = H1 (duplicate detection rule failure/disabled).
    """
    evidence = [
        EvidenceItem(claim="C1: Duplicate payment rule was mandatory before payment execution", source="financial_policy", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C2: Second invoice with identical vendor ID and invoice number was submitted", source="invoice_registry", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C3: System audit log confirms duplicate detection rule was disabled at 14:32", source="erp_audit_trail", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C4: Transaction log confirms payment of ₹125,000 was executed at 14:47 without validation", source="disbursement_log", status=EvidenceStatus.VERIFIED),
    ]

    hyp = CandidateHypothesis(
        id="H1",
        name="DUPLICATE_DETECTION_DISABLED",
        statement="Duplicate payment validation rule was disabled in the ERP system, allowing the duplicate transaction to process.",
        evidence_needed="ERP audit logs",
        supporting_evidence=[e.claim for e in evidence],
        causal_level=CausalLevel.L5_SYSTEMIC_CAUSE,
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
    assert rc_stat in (RootCauseStatus.SUPPORTED, RootCauseStatus.ESTABLISHED)


def test_case_e_confirmed_approval_workflow_admin_override():
    """CASE E — CONFIRMED APPROVAL WORKFLOW ADMIN OVERRIDE
    C1: Dual payment approval was required for amounts exceeding ₹100,000. (VERIFIED)
    C2: Second payment was executed without second approver review. (VERIFIED)
    C3: Security audit trail proves administrator override token was used to bypass approval workflow. (VERIFIED)
    C4: Payment disbursement executed immediately following override. (VERIFIED)

    Expected: Root Cause Status = SUPPORTED/ESTABLISHED, Leading = H1 (administrator workflow override).
    """
    evidence = [
        EvidenceItem(claim="C1: Dual payment authorization was required for amounts exceeding ₹100,000", source="doa_matrix", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C2: Second payment of ₹125,000 executed without second approver sign-off", source="ap_workflow", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C3: Security audit trail proves administrator override token was used to bypass approval", source="security_audit_log", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C4: Payment disbursement executed immediately following override", source="banking_gateway", status=EvidenceStatus.VERIFIED),
    ]

    hyp = CandidateHypothesis(
        id="H1",
        name="ADMIN_WORKFLOW_OVERRIDE_BYPASS",
        statement="Approval workflow was bypassed using administrator override token, executing payment without dual authorization.",
        evidence_needed="Security audit logs",
        supporting_evidence=[e.claim for e in evidence],
        causal_level=CausalLevel.L5_SYSTEMIC_CAUSE,
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
    assert rc_stat in (RootCauseStatus.SUPPORTED, RootCauseStatus.ESTABLISHED)
