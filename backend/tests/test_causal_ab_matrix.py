"""Causal A/B Verification Matrix (Phase 10 & 32 Mandatory Tests).

Proves that the engine reasons over causal evidence rather than domain keywords:
  RUN A: Complete causal evidence chain -> ESTABLISHED
  RUN B: Missing one essential causal link -> NOT_ESTABLISHED
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


def test_ab_case_1_training_workflow_bypass():
    """Case 1 A/B: Training workflow bypass."""
    # RUN A: Complete causal chain
    evidence_a = [
        EvidenceItem(claim="C1: Training mandatory before authorization", source="sop", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C2: LMS shows training uncompleted", source="lms", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C3: Operator was authorized without completed training", source="access", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C4: Workflow audit log shows training verification was bypassed", source="audit_log", status=EvidenceStatus.VERIFIED),
    ]
    hyp_a = CandidateHypothesis(
        id="H1",
        name="WORKFLOW_BYPASS",
        statement="Training verification was bypassed in the authorization process.",
        evidence_needed="Workflow audit trail",
        supporting_evidence=[e.claim for e in evidence_a],
        causal_level=CausalLevel.L4_ROOT_CAUSE,
    )
    eligible_a, supp_a, _, _, _, promo_a = evaluate_root_cause_eligibility(hyp_a, evidence_items=evidence_a)
    assert eligible_a and supp_a == SupportLevel.SUPPORTED and promo_a
    hyp_a.status = "SUPPORTED"
    lead_a, _, stat_a, _ = select_authoritative_leading_hypothesis([hyp_a], evidence_ledger=evidence_a)
    assert lead_a == "H1"
    assert stat_a == RootCauseStatus.SUPPORTED

    # RUN B: Remove the essential bypass proof link
    evidence_b = [
        EvidenceItem(claim="C1: Training mandatory before authorization", source="sop", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C2: LMS shows training uncompleted", source="lms", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C3: Operator was authorized", source="access", status=EvidenceStatus.VERIFIED),
    ]
    hyp_b = CandidateHypothesis(
        id="H1",
        name="WORKFLOW_BYPASS",
        statement="Training verification was bypassed in the authorization process.",
        evidence_needed="Workflow audit trail",
        supporting_evidence=[e.claim for e in evidence_b],
        causal_level=CausalLevel.L4_ROOT_CAUSE,
    )
    eligible_b, supp_b, _, _, _, promo_b = evaluate_root_cause_eligibility(hyp_b, evidence_items=evidence_b)
    assert eligible_b and supp_b == SupportLevel.POSSIBLE and not promo_b
    hyp_b.status = "POSSIBLE"
    lead_b, _, stat_b, _ = select_authoritative_leading_hypothesis([hyp_b], evidence_ledger=evidence_b)
    assert lead_b is None
    assert stat_b == RootCauseStatus.NOT_ESTABLISHED


def test_ab_case_2_calibration_block_disablement():
    """Case 2 A/B: Calibration safety block disablement."""
    # RUN A: Complete causal chain
    evidence_a = [
        EvidenceItem(claim="C1: Calibration expired August 10", source="cert", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C2: Balance used August 12", source="log", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C3: System audit log shows the block was disabled on August 11", source="audit_trail", status=EvidenceStatus.VERIFIED),
    ]
    hyp_a = CandidateHypothesis(
        id="H1",
        name="BLOCK_DISABLED",
        statement="The expiry-use control block was disabled, permitting use.",
        evidence_needed="System audit logs",
        supporting_evidence=[e.claim for e in evidence_a],
        causal_level=CausalLevel.L4_ROOT_CAUSE,
    )
    eligible_a, supp_a, _, _, _, promo_a = evaluate_root_cause_eligibility(hyp_a, evidence_items=evidence_a)
    assert eligible_a and supp_a == SupportLevel.SUPPORTED and promo_a
    hyp_a.status = "SUPPORTED"
    lead_a, _, stat_a, _ = select_authoritative_leading_hypothesis([hyp_a], evidence_ledger=evidence_a)
    assert lead_a == "H1"
    assert stat_a == RootCauseStatus.SUPPORTED

    # RUN B: Remove block disablement proof
    evidence_b = [
        EvidenceItem(claim="C1: Calibration expired August 10", source="cert", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C2: Balance used August 12", source="log", status=EvidenceStatus.VERIFIED),
    ]
    hyp_b = CandidateHypothesis(
        id="H1",
        name="BLOCK_DISABLED",
        statement="The expiry-use control block was disabled, permitting use.",
        evidence_needed="System audit logs",
        supporting_evidence=[e.claim for e in evidence_b],
        causal_level=CausalLevel.L4_ROOT_CAUSE,
    )
    eligible_b, supp_b, _, _, _, promo_b = evaluate_root_cause_eligibility(hyp_b, evidence_items=evidence_b)
    assert eligible_b and supp_b == SupportLevel.POSSIBLE and not promo_b
    hyp_b.status = "POSSIBLE"
    lead_b, _, stat_b, _ = select_authoritative_leading_hypothesis([hyp_b], evidence_ledger=evidence_b)
    assert lead_b is None
    assert stat_b == RootCauseStatus.NOT_ESTABLISHED


def test_ab_case_3_notification_service_outage():
    """Case 3 A/B: Notification service outage."""
    # RUN A: Complete causal chain
    evidence_a = [
        EvidenceItem(claim="C1: Notification was required", source="sop", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C2: Notification was not delivered", source="log", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C3: Server log shows notification service failure at timestamp", source="server_log", status=EvidenceStatus.VERIFIED),
    ]
    hyp_a = CandidateHypothesis(
        id="H1",
        name="SERVICE_OUTAGE",
        statement="Notification service failure during dispatch outage prevented delivery.",
        evidence_needed="Server diagnostics",
        supporting_evidence=[e.claim for e in evidence_a],
        causal_level=CausalLevel.L4_ROOT_CAUSE,
    )
    eligible_a, supp_a, _, _, _, promo_a = evaluate_root_cause_eligibility(hyp_a, evidence_items=evidence_a)
    assert eligible_a and supp_a == SupportLevel.SUPPORTED and promo_a
    hyp_a.status = "SUPPORTED"
    lead_a, _, stat_a, _ = select_authoritative_leading_hypothesis([hyp_a], evidence_ledger=evidence_a)
    assert lead_a == "H1"
    assert stat_a == RootCauseStatus.SUPPORTED

    # RUN B: Remove server outage evidence
    evidence_b = [
        EvidenceItem(claim="C1: Notification was required", source="sop", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C2: Notification was not delivered", source="log", status=EvidenceStatus.VERIFIED),
    ]
    hyp_b = CandidateHypothesis(
        id="H1",
        name="SERVICE_OUTAGE",
        statement="Notification service failure during dispatch outage prevented delivery.",
        evidence_needed="Server diagnostics",
        supporting_evidence=[e.claim for e in evidence_b],
        causal_level=CausalLevel.L4_ROOT_CAUSE,
    )
    eligible_b, supp_b, _, _, _, promo_b = evaluate_root_cause_eligibility(hyp_b, evidence_items=evidence_b)
    assert eligible_b and supp_b == SupportLevel.POSSIBLE and not promo_b
    hyp_b.status = "POSSIBLE"
    lead_b, _, stat_b, _ = select_authoritative_leading_hypothesis([hyp_b], evidence_ledger=evidence_b)
    assert lead_b is None
    assert stat_b == RootCauseStatus.NOT_ESTABLISHED


def test_ab_case_4_validated_range_interlock_disablement():
    """Case 4 A/B: Validated operating range with interlock disablement vs without."""
    # RUN A: Complete causal chain establishing root cause
    evidence_a = [
        EvidenceItem(claim="C1: Equipment was operated outside validated range.", source="finding", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C2: The approved validation range was 10-20°C.", source="spec", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C3: Operating logs show 28°C.", source="log", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C4: The range interlock was intentionally disabled.", source="audit_trail", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C5: The equipment was operated while the interlock was disabled.", source="run_log", status=EvidenceStatus.VERIFIED),
    ]
    hyp_a = CandidateHypothesis(
        id="H1",
        name="INTERLOCK_DISABLED",
        statement="Operation occurred while the validated-range interlock was intentionally disabled.",
        evidence_needed="Audit logs and run records",
        supporting_evidence=[e.claim for e in evidence_a],
        causal_level=CausalLevel.L4_ROOT_CAUSE,
    )
    eligible_a, supp_a, _, _, _, promo_a = evaluate_root_cause_eligibility(hyp_a, evidence_items=evidence_a)
    assert eligible_a and supp_a == SupportLevel.SUPPORTED and promo_a
    hyp_a.status = "SUPPORTED"
    lead_a, _, stat_a, _ = select_authoritative_leading_hypothesis([hyp_a], evidence_ledger=evidence_a)
    assert lead_a == "H1"
    assert stat_a == RootCauseStatus.SUPPORTED

    # RUN B: Remove the essential interlock disablement proof (C4 and C5)
    evidence_b = [
        EvidenceItem(claim="C1: Equipment was operated outside validated range.", source="finding", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C2: The approved validation range was 10-20°C.", source="spec", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="C3: Operating logs show 28°C.", source="log", status=EvidenceStatus.VERIFIED),
    ]
    hyp_b = CandidateHypothesis(
        id="H1",
        name="INTERLOCK_DISABLED",
        statement="Operation occurred while the validated-range interlock was intentionally disabled.",
        evidence_needed="Audit logs and run records",
        supporting_evidence=[e.claim for e in evidence_b],
        causal_level=CausalLevel.L4_ROOT_CAUSE,
    )
    eligible_b, supp_b, _, _, _, promo_b = evaluate_root_cause_eligibility(hyp_b, evidence_items=evidence_b)
    assert eligible_b and supp_b == SupportLevel.POSSIBLE and not promo_b
    hyp_b.status = "POSSIBLE"
    lead_b, _, stat_b, _ = select_authoritative_leading_hypothesis([hyp_b], evidence_ledger=evidence_b)
    assert lead_b is None
    assert stat_b == RootCauseStatus.NOT_ESTABLISHED


def test_ab_case_5_missing_attachment_vs_available_attachment():
    """Case 5 A/B: Observation + missing attachment vs available attachment."""
    from app.agent.claim_extractor import extract_claims
    from app.agent.proposition_engine import classify_evidence_completeness
    from app.models.agent import EvidenceCompleteness, ReferencedDocumentInfo

    finding_text = (
        "During the audit observation, the equipment was operated outside its validated range. "
        "The auditor referenced an attached calibration report, but the report was not available to the AI agent."
    )
    claims = extract_claims(finding_text)
    obs_claim = next(c for c in claims if "outside its validated range" in c.text)

    # RUN A (Missing attachment): Observation is VERIFIED, completeness is PARTIAL, root cause NOT_ESTABLISHED
    ref_docs_a = [ReferencedDocumentInfo(document_type="calibration report", reference_status="REFERENCED_UNAVAILABLE")]
    comp_a = classify_evidence_completeness(finding_text, claims, referenced_docs=ref_docs_a)
    assert obs_claim.status == EvidenceStatus.VERIFIED
    assert comp_a == EvidenceCompleteness.PARTIAL

    # RUN B (Attachment available): Observation remains VERIFIED, completeness is COMPLETE
    ref_docs_b = [ReferencedDocumentInfo(document_type="calibration report", reference_status="REFERENCED_AVAILABLE")]
    comp_b = classify_evidence_completeness(finding_text.replace("was not available", "was available"), claims, referenced_docs=ref_docs_b)
    assert obs_claim.status == EvidenceStatus.VERIFIED
    assert comp_b == EvidenceCompleteness.COMPLETE
