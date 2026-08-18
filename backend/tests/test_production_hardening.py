"""Production Hardening Test Suite for LQMS Finding Investigation & Root Cause Agent.

Tests cover:
1. Evidence Status != Causal Support: verified event does not promote causal mechanism.
2. Leading hypothesis consistency: no contradictory status combinations.
3. Positive root cause cases: genuine evidence-supported mechanisms are promoted.
4. Negative causal boundary cases: unverified mechanisms stay NOT_ESTABLISHED / POSSIBLE.
5. Universal Validation Firewall: rejects unsupported causal leaps, normalizes questions, deduplicates areas and evidence.
6. Cross-path semantic consistency: Primary, Recovery, and Fallback paths maintain identical causal truth.
"""

from __future__ import annotations

import pytest

from app.agent.causal_graph import (
    CausalLevel,
    SupportLevel,
    evaluate_root_cause_eligibility,
    select_authoritative_leading_hypothesis,
)
from app.agent.causal_guard import determine_hypothesis_status
from app.agent.analytical_validator import (
    normalize_investigation_decision_tree,
    deduplicate_investigation_questions,
    derive_investigation_areas,
    normalize_and_dedupe_evidence_items,
    is_compound_decision_question,
)
from app.models.agent import (
    CandidateHypothesis,
    EvidenceConflict,
    EvidenceItem,
    EvidenceStatus,
    InvestigationQuestion,
    RootCauseAnalysis,
    RootCauseStatus,
)


# ===========================================================================
# 1. Evidence Status != Causal Support Tests
# ===========================================================================

def test_verified_event_does_not_promote_unverified_causal_mechanism():
    """An event fact (e.g. notification was sent) must NOT promote a failure mechanism to SUPPORTED."""
    statement = "The automated notification system experienced a technical queue delivery failure."
    verified_facts = [
        "System records confirm SOP revision notification was dispatched to 14 operators on August 1",
    ]
    status, strength = determine_hypothesis_status(
        statement=statement,
        verified_facts=verified_facts,
        reported_claims=[],
        conflicts=[],
        allow_verified_promotion=True,
    )
    assert status == "POSSIBLE", f"Expected POSSIBLE, got {status}"
    assert strength == "NONE", f"Expected NONE, got {strength}"


def test_temperature_event_does_not_promote_cooling_unit_failure():
    """An out-of-range temperature log does not prove the refrigeration unit had a compressor failure."""
    statement = "The refrigeration unit cooling compressor experienced mechanical failure."
    verified_facts = [
        "Temperature log recorded 9.2°C at 04:00 on July 15 against standard of 2-8°C",
    ]
    status, strength = determine_hypothesis_status(
        statement=statement,
        verified_facts=verified_facts,
        reported_claims=[],
        conflicts=[],
        allow_verified_promotion=True,
    )
    assert status == "POSSIBLE"
    assert strength == "NONE"


# ===========================================================================
# 2. Leading Hypothesis Consistency Tests
# ===========================================================================

def test_when_multiple_hypotheses_are_possible_leading_hypothesis_is_tied():
    """When multiple candidate hypotheses are POSSIBLE, leading_hypothesis is None / TIED."""
    h1 = CandidateHypothesis(
        id="H1",
        name="TECHNICAL_FAILURE",
        statement="Automated notification had a technical delivery failure",
        evidence_needed="System transmission and error logs",
        status="POSSIBLE",
        evidence_strength="NONE",
    )
    h2 = CandidateHypothesis(
        id="H2",
        name="ACCOUNT_CONFIG_GAP",
        statement="User email account was incorrectly configured",
        evidence_needed="Account mapping and configuration logs",
        status="POSSIBLE",
        evidence_strength="NONE",
    )
    lead_id, lead_mode, rc_status, rationale = select_authoritative_leading_hypothesis([h1, h2])
    assert lead_id is None
    assert lead_mode == "TIED"
    assert rc_status == RootCauseStatus.NOT_ESTABLISHED


def test_when_single_hypothesis_is_possible_leading_hypothesis_is_selected_as_possible():
    """When exactly one candidate hypothesis is POSSIBLE, leading_hypothesis is H1 with POSSIBLE status (not SUPPORTED)."""
    h1 = CandidateHypothesis(
        id="H1",
        name="TECHNICAL_FAILURE",
        statement="Automated notification had a technical delivery failure",
        evidence_needed="System transmission and error logs",
        status="POSSIBLE",
        evidence_strength="NONE",
    )
    lead_id, lead_mode, rc_status, rationale = select_authoritative_leading_hypothesis([h1])
    assert lead_id == "H1"
    assert lead_mode == "POSSIBLE"
    assert rc_status == RootCauseStatus.NOT_ESTABLISHED


def test_unresolved_conflict_forces_leading_hypothesis_none():
    """When evidence conflicts exist, leading_hypothesis must be None / NONE."""
    h1 = CandidateHypothesis(
        id="H1",
        name="TECHNICAL_FAILURE",
        statement="Automated notification had a technical delivery failure",
        evidence_needed="System transmission and error logs",
        status="SUPPORTED",
    )
    conflict = EvidenceConflict(
        conflict_id="CONF-01",
        claim_ids=["C1", "C2"],
        proposition="Notification was received by the operator",
        conflict_type="DELIVERY_VS_RECEIPT",
    )
    lead_id, lead_mode, rc_status, rationale = select_authoritative_leading_hypothesis(
        [h1],
        conflicts=[conflict],
    )
    assert lead_id is None
    assert lead_mode == "NONE"
    assert rc_status == RootCauseStatus.NOT_ESTABLISHED


# ===========================================================================
# 3. Positive Root-Cause Promotion Cases
# ===========================================================================

def test_positive_workflow_bypass_promotion():
    """When an audit trail directly establishes a workflow bypass, the hypothesis is SUPPORTED."""
    h = CandidateHypothesis(
        id="H1",
        name="WORKFLOW_BYPASS",
        statement="Production batch was released through an unauthorized workflow bypass of QA review",
        evidence_needed="System audit logs",
    )
    evidence = [
        EvidenceItem(
            id="E1",
            claim="System audit log confirms user bypassed QA verification step at 14:22",
            status=EvidenceStatus.VERIFIED,
            source="audit_trail",
        )
    ]
    eligible, supp, reason, missing, c_lvl, promo = evaluate_root_cause_eligibility(
        h, evidence_items=evidence, conflicts=[]
    )
    assert eligible is True
    assert supp == SupportLevel.SUPPORTED
    assert promo is True


def test_positive_disabled_interlock_promotion():
    """When maintenance logs confirm a safety interlock was disabled, the hypothesis is SUPPORTED."""
    h = CandidateHypothesis(
        id="H1",
        name="INTERLOCK_DISABLED",
        statement="Autoclave door safety interlock was defeated during operation",
        evidence_needed="Maintenance inspection report",
    )
    evidence = [
        EvidenceItem(
            id="E1",
            claim="Maintenance inspection report confirms door safety interlock switch was physically bypassed and disabled",
            status=EvidenceStatus.VERIFIED,
            source="maintenance_log",
        )
    ]
    eligible, supp, reason, missing, c_lvl, promo = evaluate_root_cause_eligibility(
        h, evidence_items=evidence, conflicts=[]
    )
    assert eligible is True
    assert supp == SupportLevel.SUPPORTED
    assert promo is True


def test_positive_service_outage_promotion():
    """When server logs confirm a service outage during processing, the hypothesis is SUPPORTED."""
    h = CandidateHypothesis(
        id="H1",
        name="SERVER_OUTAGE",
        statement="Notification delivery failed due to network server service outage",
        evidence_needed="Server infrastructure logs",
    )
    evidence = [
        EvidenceItem(
            id="E1",
            claim="Server log confirms notification service outage and network connection failure from 08:00 to 12:00",
            status=EvidenceStatus.VERIFIED,
            source="server_log",
        )
    ]
    eligible, supp, reason, missing, c_lvl, promo = evaluate_root_cause_eligibility(
        h, evidence_items=evidence, conflicts=[]
    )
    assert eligible is True
    assert supp == SupportLevel.SUPPORTED
    assert promo is True


# ===========================================================================
# 4. Negative Causal Boundary Cases
# ===========================================================================

def test_missing_record_does_not_promote_non_performance():
    """A missing training log does not prove training was never performed."""
    h = CandidateHypothesis(
        id="H1",
        name="TRAINING_NOT_CONDUCTED",
        statement="Operator was never trained on the revised procedure",
        evidence_needed="Training records",
    )
    evidence = [
        EvidenceItem(
            id="E1",
            claim="Training sign-off sheet was not located in the binder during the audit",
            status=EvidenceStatus.VERIFIED,
            source="audit_observation",
        )
    ]
    eligible, supp, reason, missing, c_lvl, promo = evaluate_root_cause_eligibility(
        h, evidence_items=evidence, conflicts=[]
    )
    assert promo is False
    assert supp != SupportLevel.SUPPORTED


def test_subjective_supervisor_statement_does_not_promote_root_cause():
    """A reported opinion without objective verification cannot promote a root cause."""
    h = CandidateHypothesis(
        id="H1",
        name="OPERATOR_INATTENTION",
        statement="Operator failed to notice the warning alarm due to distraction",
        evidence_needed="Alarm logs",
    )
    evidence = [
        EvidenceItem(
            id="E1",
            claim="Supervisor stated that operators were probably distracted during shift change",
            status=EvidenceStatus.REPORTED,
            source="interview",
        )
    ]
    eligible, supp, reason, missing, c_lvl, promo = evaluate_root_cause_eligibility(
        h, evidence_items=evidence, conflicts=[]
    )
    assert promo is False
    assert supp == SupportLevel.POSSIBLE


# ===========================================================================
# 5. Universal Validation Firewall & Deduplication Tests
# ===========================================================================

def test_compound_questions_are_rejected_by_firewall():
    """Questions containing multiple compound decisions must be detected."""
    assert is_compound_decision_question(
        "Was notification acknowledgement mandatory and was it completed by the operators?"
    ) is True

    assert is_compound_decision_question(
        "Do system records confirm technical delivery of the notification to the user mailbox?"
    ) is False


def test_investigation_areas_are_never_rendered_as_root_causes():
    """Investigation areas must represent control domains, distinct from causal hypotheses."""
    questions = [
        InvestigationQuestion(
            id="Q1",
            question="Do system logs confirm delivery?",
            objective="Confirm delivery",
            area="Notification Delivery and Transmission Control",
            priority="P1",
        ),
        InvestigationQuestion(
            id="Q2",
            question="Do access logs confirm receipt?",
            objective="Confirm receipt",
            area="Operator Receipt and Access Verification",
            priority="P2",
        ),
    ]
    areas = derive_investigation_areas(questions)
    for a in areas:
        assert not any(bad in a.lower() for bad in ("because", "failed due to", "caused by"))


def test_evidence_needed_deduplication():
    """Overlapping evidence needed items must be normalized and deduplicated."""
    raw = [
        "System transmission logs",
        "system transmission logs",
        "System Transmission and Delivery Logs",
        "Mail server access logs",
    ]
    deduped = normalize_and_dedupe_evidence_items(raw)
    assert len(deduped) <= 3
    assert len(deduped) == len(set(d.lower() for d in deduped))


def test_internal_metadata_and_short_code_filtered_from_areas():
    """Internal tokens like SHORT_CODE, H1, etc. must never appear as investigation areas."""
    h1 = CandidateHypothesis(
        id="H1",
        name="SHORT_CODE",
        statement="The automated notification system had a technical failure on 1 August",
        evidence_needed="System logs",
    )
    areas = derive_investigation_areas(
        candidate_hypotheses=[h1],
        existing_areas=["Short Code", "Notification workflow and event-processing verification"],
        canonical_subject="automated notification",
    )
    for a in areas:
        assert "short code" not in a.lower()
        assert not a.startswith("H1")
        assert len(a) > 3


def test_notification_failure_fallback_generates_mechanism_questions():
    """When a notification failure finding occurs, fallback questions must target the unresolved mechanism."""
    from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
    finding = "SOP uploaded on 1 August. Automated notification failed. No affected operators received notification. Operators continued using previous revision until 10 August."
    evidence = [
        EvidenceItem(id="C1", claim="SOP uploaded on 1 August", status=EvidenceStatus.VERIFIED, source="audit_trail"),
        EvidenceItem(id="C2", claim="Automated notification failed", status=EvidenceStatus.VERIFIED, source="audit_trail"),
        EvidenceItem(id="C3", claim="No affected operators received notification", status=EvidenceStatus.VERIFIED, source="audit_trail"),
        EvidenceItem(id="C4", claim="Operators continued using previous revision until 10 August", status=EvidenceStatus.VERIFIED, source="audit_trail"),
    ]
    hyps, plan = build_deterministic_investigation_plan(
        finding_text=finding,
        evidence_ledger=evidence,
        canonical_subject="automated notification",
    )
    assert len(plan.questions) >= 4
    q_texts = " ".join(q.question for q in plan.questions)
    assert any(k in q_texts.lower() for k in ("trigger", "event", "queue", "dispatch", "error", "recipient", "downstream"))
    for a in plan.areas:
        assert "short code" not in a.lower()
        assert len(a) > 4


def test_five_why_evidence_boundary_identifies_unresolved_mechanisms():
    """5-Why chain must state what is established and identify the unresolved mechanism at the evidence boundary."""
    from app.agent.nodes.five_why_fallback import build_deterministic_five_why
    finding = "SOP uploaded on 1 August. Automated notification failed. No affected operators received notification. Operators continued using previous revision until 10 August."
    evidence = [
        EvidenceItem(id="C1", claim="SOP uploaded on 1 August", status=EvidenceStatus.VERIFIED, source="audit_trail"),
        EvidenceItem(id="C2", claim="Automated notification failed", status=EvidenceStatus.VERIFIED, source="audit_trail"),
        EvidenceItem(id="C3", claim="No affected operators received notification", status=EvidenceStatus.VERIFIED, source="audit_trail"),
    ]
    fw = build_deterministic_five_why(
        finding_text=finding,
        evidence_ledger=evidence,
        canonical_subject="automated notification",
    )
    assert fw.is_complete is False
    assert len(fw.steps) >= 1
    last_step = fw.steps[-1]
    assert last_step.status == "UNKNOWN"
    assert "delivery did not occur" in last_step.answer or "does not establish" in last_step.answer

