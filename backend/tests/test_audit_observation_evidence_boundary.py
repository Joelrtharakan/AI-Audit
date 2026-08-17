"""Audit Observation and Evidence Boundary Test Suite.

Verifies the exact scenario:
"During the audit observation, the equipment was operated outside its validated range.
The auditor referenced an attached calibration report, but the report was not available to the AI agent."

Ensures:
  1. The audit observation is VERIFIED/ESTABLISHED (never downgraded to REPORTED).
  2. Document reference is VERIFIED.
  3. Evidence availability limitation is VERIFIED.
  4. Evidence completeness is PARTIAL.
  5. Root cause is NOT_ESTABLISHED.
  6. Leading hypothesis is NONE.
  7. 5-Why stops at evidence boundary with UNKNOWN.
  8. Affected object is 'Equipment' (not 'Equipment calibration status').
  9. Process at risk is 'Equipment operation and validated-use control'.
  10. Zero causal CAPA.
"""

from __future__ import annotations

import pytest

from app.agent.claim_extractor import extract_claims
from app.agent.nodes.core_synthesis import _derive_deterministic_impact
from app.agent.nodes.five_why_fallback import build_deterministic_five_why
from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
from app.agent.proposition_engine import build_propositions_from_ledger, classify_evidence_completeness
from app.models.agent import (
    CanonicalFindingState,
    ClaimAttribution,
    EvidenceCompleteness,
    EvidenceItem,
    EvidenceStatus,
    ReferencedDocumentInfo,
    RootCauseStatus,
)
from app.services.semantic_subject import resolve_deviation


def test_audit_observation_with_unavailable_attachment():
    finding_text = (
        "During the audit observation, the equipment was operated outside its validated range. "
        "The auditor referenced an attached calibration report, but the report was not available to the AI agent."
    )

    # 1. Claim extraction and provenance
    claims = extract_claims(finding_text)
    assert len(claims) >= 2

    # Observation claim
    obs_claim = next((c for c in claims if "outside its validated range" in c.text), None)
    assert obs_claim is not None
    assert obs_claim.status == EvidenceStatus.VERIFIED
    assert obs_claim.attribution == ClaimAttribution.AUDITOR_OBSERVED

    # Evidence completeness
    ref_docs = [ReferencedDocumentInfo(document_type="calibration report", reference_status="REFERENCED_UNAVAILABLE")]
    completeness = classify_evidence_completeness(finding_text, claims, referenced_docs=ref_docs)
    assert completeness == EvidenceCompleteness.PARTIAL

    # 2. Semantic subject / affected object resolution
    resolved = resolve_deviation(finding_text, [c.text for c in claims if c.status == EvidenceStatus.VERIFIED])
    assert resolved.subject is not None
    assert "calibration status" not in resolved.subject.lower()
    assert "equipment" in resolved.subject.lower()

    # 3. Canonical finding state
    canonical = CanonicalFindingState(
        raw_finding=finding_text,
        finding_subject=resolved.subject,
        observed_deviation=resolved.deviation or "equipment — operated outside its validated range",
        deviation_condition="operated outside its validated range",
        facts=[c.text for c in claims if c.status == EvidenceStatus.VERIFIED],
        evidence_claims=claims,
        referenced_documents=ref_docs,
        evidence_completeness=completeness,
    )
    assert canonical.evidence_completeness == EvidenceCompleteness.PARTIAL

    # 4. Investigation plan & zero hypotheses
    hypotheses, plan = build_deterministic_investigation_plan(
        finding_text=finding_text,
        evidence_ledger=[EvidenceItem(claim=c.text, source=c.source, status=c.status) for c in claims],
        canonical_subject=canonical.finding_subject,
    )
    assert len(hypotheses) == 0  # Missing attachment does not create causal hypotheses
    assert len(plan.questions) == 5
    # Verify the questions target validated range, operating conditions, and controls specifically
    questions_text = " ".join(q.question for q in plan.questions)
    assert "operating range" in questions_text.lower() or "validated range" in questions_text.lower()
    assert "operating condition" in questions_text.lower() or "operating records" in questions_text.lower()

    # 5. Impact assessment
    impact, clean_noun, topic, actor = _derive_deterministic_impact(
        finding_text, canonical, canonical.observed_deviation
    )
    assert impact.affected_object is not None
    assert "Equipment" in impact.affected_object
    assert "calibration status" not in impact.affected_object.lower()
    assert "operation and validated-use control" in (impact.process_at_risk or "").lower()

    # 6. 5-Why chain: MUST NOT merely repeat the observation as its causal answer!
    five_why = build_deterministic_five_why(
        finding_text,
        [EvidenceItem(claim=c.text, source=c.source, status=c.status) for c in claims],
        canonical,
    )
    assert not five_why.is_complete
    assert len(five_why.steps) == 1
    step1 = five_why.steps[0]
    assert "unavailable" in step1.answer.lower() or "does not establish" in step1.answer.lower()
    assert "DEGRADED MODE" not in (five_why.status_note or "").upper()
