"""Conflict-Center hardening: CONFLICT MUST BECOME THE INVESTIGATION CENTER.

A conflict between a system/record claim and a REPORTED statement (e.g.
"system logs show delivered" vs. "operators say they never received it")
must not be discarded in favor of a generic "process compliance" fallback,
and must not be collapsed into a fabricated completion/failure hypothesis.
DELIVERY, RECEIPT, ACCESS, and ACKNOWLEDGEMENT are distinct propositions --
see app.agent.claim_extractor.classify_conflict_proposition_type and the
DELIVERY_VS_RECEIPT branch in app.agent.nodes.plan_investigation_fallback /
app.agent.nodes.core_synthesis._derive_deterministic_impact.
"""

from __future__ import annotations

import pytest

from app.models.agent import CanonicalFindingState, CapaAnalysis, CapaStatus

NOTIFICATION_FINDING = (
    "System logs show that the revised SOP notification was successfully delivered to all "
    "operators. However, three operators stated that they never received the notification."
)

TRAINING_CONFLICT_FINDING = (
    "The operator stated they had not received training on the revised procedure, but the "
    "supervisor claimed the training was completed."
)


# ---------------------------------------------------------------------------
# Claim extraction / conflict detection
# ---------------------------------------------------------------------------

def test_discourse_connector_does_not_break_attribution():
    """"However, three operators stated..." must be attributed as a REPORTED
    statement, not silently misclassified as a direct VERIFIED observation
    (the root cause of the conflict going undetected entirely)."""
    from app.agent.claim_extractor import extract_claims
    from app.models.agent import ClaimAttribution, EvidenceStatus

    claims = extract_claims(NOTIFICATION_FINDING, [])
    reported = [c for c in claims if c.status == EvidenceStatus.REPORTED]
    assert reported, "the operators' statement must be classified REPORTED, not VERIFIED"
    assert reported[0].attribution == ClaimAttribution.PERSON_REPORTED


def test_notification_finding_produces_one_conflict():
    from app.agent.claim_extractor import extract_claims, detect_evidence_conflicts

    claims = extract_claims(NOTIFICATION_FINDING, [])
    conflicts = detect_evidence_conflicts(claims)
    assert len(conflicts) == 1
    assert conflicts[0].proposition_type == "DELIVERY_VS_RECEIPT"


def test_training_conflict_still_classified_completion_not_delivery():
    """"had not received TRAINING" shares the verb "received" with the
    notification case but is a completion dispute, not a transmission one --
    must not be misclassified as DELIVERY_VS_RECEIPT (that would silently
    strip its existing, already-validated NOT_COMPLETED hypothesis
    behavior -- see test_causal_model.test_training_conflict_end_to_end)."""
    from app.agent.claim_extractor import extract_claims, detect_evidence_conflicts

    claims = extract_claims(TRAINING_CONFLICT_FINDING, [])
    conflicts = detect_evidence_conflicts(claims)
    assert len(conflicts) == 1
    assert conflicts[0].proposition_type == "COMPLETION_VS_MISSING_RECORD"


@pytest.mark.parametrize("label,finding,expected_type", [
    ("A_email", "System shows email delivered; employee says it was never received.", "DELIVERY_VS_RECEIPT"),
    ("B_training", "Training system records completion; employee says training was never completed.", "COMPLETION_VS_MISSING_RECORD"),
    ("C_maintenance", "Maintenance log shows maintenance completed; technician says maintenance was not performed.", "COMPLETION_VS_MISSING_RECORD"),
    ("E_access", "Access log shows document opened; user states the document was never accessible.", "DELIVERY_VS_RECEIPT"),
])
def test_adversarial_conflict_classification(label, finding, expected_type):
    from app.agent.claim_extractor import extract_claims, detect_evidence_conflicts

    claims = extract_claims(finding, [])
    conflicts = detect_evidence_conflicts(claims)
    assert conflicts, f"case {label}: expected a conflict to be detected"
    assert conflicts[0].proposition_type == expected_type


def test_positive_control_no_spurious_conflict_when_evidence_agrees():
    from app.agent.claim_extractor import extract_claims, detect_evidence_conflicts

    finding = (
        "System logs show delivery and authenticated access logs confirm the operator opened the "
        "revised SOP."
    )
    claims = extract_claims(finding, [])
    conflicts = detect_evidence_conflicts(claims)
    assert conflicts == []


# ---------------------------------------------------------------------------
# Investigation plan: DELIVERY_VS_RECEIPT must not fall back to generic
# process-compliance questions, and must generate zero hypotheses.
# ---------------------------------------------------------------------------

def test_delivery_vs_receipt_yields_zero_hypotheses_and_conflict_specific_questions():
    from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan

    hyps, plan = build_deterministic_investigation_plan(
        NOTIFICATION_FINDING, [], canonical_subject="revised SOP notification",
    )
    assert hyps == []
    assert 5 <= len(plan.questions) <= 6
    joined = " ".join(q.question for q in plan.questions).lower()
    assert "process compliance" not in joined
    for forbidden in ("failed", "did not receive", "were not aware"):
        assert forbidden not in joined, f"question presupposes a side: {forbidden!r}"
    assert "delivery" in joined and "receipt" in joined and "acknowledgement" in joined


def test_delivery_vs_receipt_subject_not_degraded_when_canonical_missing():
    """Even without a canonical_subject, the DELIVERY_VS_RECEIPT branch must
    not surface the internal "process compliance" placeholder in questions
    -- it should recover a real object noun structurally."""
    from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan

    finding = "System shows email delivered; employee says it was never received."
    hyps, plan = build_deterministic_investigation_plan(finding, [])
    assert hyps == []
    joined = " ".join(q.question for q in plan.questions).lower()
    assert "process compliance" not in joined
    assert "email" in joined


def test_training_conflict_hypothesis_behavior_unchanged():
    """Regression guard: COMPLETION_VS_MISSING_RECORD conflicts (training,
    maintenance) must keep their existing, previously-validated single
    NOT_COMPLETED hypothesis -- the Conflict-Center hardening only changes
    behavior for DELIVERY_VS_RECEIPT-shaped conflicts."""
    from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan

    hyps, plan = build_deterministic_investigation_plan(TRAINING_CONFLICT_FINDING, [])
    assert len(hyps) == 1
    assert hyps[0].name == "TRAINING_NOT_COMPLETED"


# ---------------------------------------------------------------------------
# Impact derivation: process_at_risk/potential_effect must be domain-local,
# never inheriting equipment/calibration/validated-use vocabulary for a
# notification finding, and conversely must not leak into unrelated
# non-conflict findings either.
# ---------------------------------------------------------------------------

def test_delivery_vs_receipt_impact_is_domain_local_not_equipment_template():
    from app.agent.nodes.core_synthesis import _derive_deterministic_impact

    canonical = CanonicalFindingState(
        raw_finding=NOTIFICATION_FINDING, observed_deviation="x",
        finding_subject="revised SOP notification",
    )
    from app.agent.claim_extractor import extract_claims, detect_evidence_conflicts
    canonical.evidence_conflicts = detect_evidence_conflicts(extract_claims(NOTIFICATION_FINDING, []))

    impact, clean_noun, topic, actor = _derive_deterministic_impact(NOTIFICATION_FINDING, canonical, "x")
    assert impact.affected_object == "Revised SOP notification receipt"
    forbidden = ("validated-use", "validated range", "calibration", "equipment")
    for word in forbidden:
        assert word not in impact.process_at_risk.lower()
        assert word not in impact.potential_effect.lower()
    assert "delivery" in impact.process_at_risk.lower() or "receipt" in impact.process_at_risk.lower()


def test_generic_entity_finding_does_not_inherit_validated_use_template():
    """Regression guard for the "workload"/"missed inspections" case found
    during live verification: the equipment/calibration
    "operation and validated-use control" template must only apply when the
    deviation condition actually describes something operated/used outside
    a range/limit -- never as the default for every plain-entity subject."""
    from app.agent.nodes.core_synthesis import _derive_deterministic_impact

    finding = "The supervisor stated that the missed inspections were due to high workload during the audit period."
    canonical = CanonicalFindingState(
        raw_finding=finding, observed_deviation="x", finding_subject="missed inspections",
        deviation_condition="UNKNOWN",
    )
    impact, *_ = _derive_deterministic_impact(finding, canonical, "x")
    assert "validated-use" not in impact.process_at_risk.lower()
    assert "validated-use" not in impact.potential_effect.lower()


def test_equipment_outside_range_still_uses_validated_use_template():
    """Regression guard: the legitimate equipment/validated-range case
    (Referenced-Evidence Boundary hardening) must still get the
    operation/validated-use framing -- only the OVER-GENERALIZATION to
    unrelated findings was the defect, not the template itself."""
    from app.agent.nodes.core_synthesis import _derive_deterministic_impact

    finding = (
        "The audit observation states that the equipment was operated outside its validated range. "
        "The auditor referenced an attached calibration report, but the report was not available to "
        "the AI agent."
    )
    canonical = CanonicalFindingState(
        raw_finding=finding, observed_deviation="x", finding_subject="equipment",
        deviation_condition="operated outside its validated range",
    )
    impact, *_ = _derive_deterministic_impact(finding, canonical, "x")
    assert "validated-use" in impact.process_at_risk.lower()


# ---------------------------------------------------------------------------
# End-to-end (mocked LLM boundary): final_evidence_verification_node must
# not promote either side of an unresolved DELIVERY_VS_RECEIPT conflict to
# root cause / leading hypothesis.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_final_verification_keeps_root_cause_unestablished_for_delivery_conflict():
    from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
    from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
    from app.agent.claim_extractor import extract_claims, detect_evidence_conflicts
    from app.models.agent import EvidenceItem, EvidenceStatus

    ledger = [
        EvidenceItem(claim="System logs show that the revised SOP notification was successfully delivered to all operators.",
                     source="finding_text", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="Three operators stated that they never received the notification.",
                     source="finding_text", status=EvidenceStatus.REPORTED),
    ]
    conflicts = detect_evidence_conflicts(extract_claims(NOTIFICATION_FINDING, []))
    canonical = CanonicalFindingState(
        raw_finding=NOTIFICATION_FINDING, observed_deviation="x",
        finding_subject="revised SOP notification", evidence_conflicts=conflicts,
    )
    hyps, plan = build_deterministic_investigation_plan(
        NOTIFICATION_FINDING, ledger, canonical_subject="revised SOP notification",
    )
    mock_state = {
        "request": type("Request", (), {"finding_text": NOTIFICATION_FINDING})(),
        "evidence_ledger": ledger,
        "canonical_finding_state": canonical,
        "root_cause": type("RC", (), {
            "narrative": None, "statement": None, "status": "NOT_ESTABLISHED",
            "category": "TO_BE_CONFIRMED", "candidate_hypotheses": hyps,
            "leading_hypothesis": None, "leading_hypothesis_status": "NONE",
        })(),
        "investigation_plan": plan,
        "capa_analysis": CapaAnalysis(status=CapaStatus.INVESTIGATION_REQUIRED, conditional_actions=[]),
        "ca_draft": None,
        "trace": [],
        "errors": [],
    }
    result = await final_evidence_verification_node(mock_state)
    rc = result["root_cause"]
    assert rc.status == "NOT_ESTABLISHED"
    assert rc.candidate_hypotheses == []
    assert result["capa_analysis"].conditional_actions == []
    assert result["investigation_plan"].questions
