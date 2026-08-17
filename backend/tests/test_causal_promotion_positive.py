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
