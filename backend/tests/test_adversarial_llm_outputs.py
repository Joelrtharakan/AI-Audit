"""Adversarial LLM Output Rejection Tests.

Injects deliberately malformed, speculative, or ungrounded LLM outputs and verifies
that validate_final_analysis() and evaluate_root_cause_eligibility() deterministically
reject or downgrade them.
"""

from __future__ import annotations

import pytest

from app.agent.causal_graph import evaluate_root_cause_eligibility, validate_final_analysis
from app.models.agent import (
    CandidateHypothesis,
    CanonicalFindingState,
    CapaAnalysis,
    ConditionalCapaAction,
    EvidenceConflict,
    EvidenceItem,
    EvidenceStatus,
    FiveWhyAnalysis,
    FiveWhyStep,
    ImpactAssessment,
    InvestigationPlan,
    InvestigationQuestion,
    ReferencedDocumentInfo,
    RootCauseAnalysis,
    RootCauseStatus,
    SupportLevel,
)


def test_adversarial_unsupported_systemic_cause_rejected():
    """Adversarial: LLM generates a grand systemic failure hypothesis with 0 supporting evidence."""
    hyp = CandidateHypothesis(
        id="H1",
        name="GLOBAL_GOVERNANCE_COLLAPSE",
        statement="Global corporate governance failed across the entire organization.",
        evidence_needed="Executive records",
        supporting_evidence=[],
        supporting_claim_ids=[],
    )
    eligible, supp_lvl, reason, _, _, promo = evaluate_root_cause_eligibility(
        hyp, evidence_items=[]
    )
    assert not promo
    assert supp_lvl in (SupportLevel.POSSIBLE, SupportLevel.REJECTED)

    state = {
        "canonical_finding_state": CanonicalFindingState(raw_finding="Log incomplete", observed_deviation="Log incomplete"),
        "root_cause": RootCauseAnalysis(status=RootCauseStatus.ESTABLISHED, candidate_hypotheses=[hyp]),
    }
    is_valid, violations = validate_final_analysis(state)
    assert not is_valid
    assert any("lacks supporting evidence provenance" in v for v in violations)


def test_adversarial_reported_cause_promoted_to_established_rejected():
    """Adversarial: LLM tries to promote a single reported statement to ESTABLISHED."""
    state = {
        "canonical_finding_state": CanonicalFindingState(
            raw_finding="Operator stated they were not trained",
            observed_deviation="Operator stated they were not trained",
            evidence_conflicts=[EvidenceConflict(conflict_id="CONF1", conflict_type="CONFLICTING_REPORTS", proposition="training")],
        ),
        "root_cause": RootCauseAnalysis(
            status=RootCauseStatus.ESTABLISHED,
            leading_hypothesis_status="SELECTED",
            candidate_hypotheses=[
                CandidateHypothesis(
                    id="H1",
                    name="TRAINING_FAILURE",
                    statement="The operator did not receive training.",
                    evidence_needed="LMS record",
                    supporting_evidence=["Operator stated they were not trained"],
                    supporting_claim_ids=["C1"],
                )
            ],
        ),
    }
    is_valid, violations = validate_final_analysis(state)
    assert not is_valid
    assert any("unresolved evidence conflicts" in v for v in violations)


def test_adversarial_missing_record_treated_as_nonperformance():
    """Adversarial: LLM asserts non-performance solely from missing binder entry."""
    hyp = CandidateHypothesis(
        id="H1",
        name="MAINTENANCE_NEVER_DONE",
        statement="The technician never performed the required maintenance.",
        evidence_needed="Logs",
    )
    evidence = [EvidenceItem(claim="maintenance record was not in the binder", source="audit", status=EvidenceStatus.VERIFIED)]
    eligible, supp_lvl, _, _, _, promo = evaluate_root_cause_eligibility(hyp, evidence_items=evidence)
    assert not promo
    assert supp_lvl == SupportLevel.POSSIBLE


def test_adversarial_unavailable_doc_content_inferred_rejected():
    """Adversarial: LLM asserts report contents when report is unavailable."""
    ref_docs = [ReferencedDocumentInfo(document_type="calibration report", reference_status="REFERENCED_UNAVAILABLE")]
    state = {
        "canonical_finding_state": CanonicalFindingState(
            raw_finding="Calibration report was referenced but unavailable.",
            observed_deviation="Calibration report unavailable",
            referenced_documents=ref_docs,
        ),
        "root_cause": RootCauseAnalysis(
            status=RootCauseStatus.NOT_ESTABLISHED,
            candidate_hypotheses=[
                CandidateHypothesis(
                    id="H1",
                    name="REPORT_SHOWED_OUT_OF_TOLERANCE",
                    statement="The calibration report showed that the equipment was out of tolerance.",
                    evidence_needed="Report",
                    supporting_evidence=["Calibration report was referenced"],
                    supporting_claim_ids=["C1"],
                )
            ],
        ),
    }
    is_valid, violations = validate_final_analysis(state)
    assert not is_valid
    assert any("infers contents of unavailable document" in v for v in violations)


def test_adversarial_zero_hypotheses_with_causal_capa_rejected():
    """Adversarial: LLM generates corrective actions when 0 candidate hypotheses exist."""
    state = {
        "canonical_finding_state": CanonicalFindingState(raw_finding="Log missing", observed_deviation="Log missing"),
        "root_cause": RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[]),
        "capa_analysis": CapaAnalysis(
            status="CAPA_RECOMMENDED",
            conditional_actions=[
                ConditionalCapaAction(
                    if_cause_confirmed="IF training failed",
                    recommended_action="Retrain all staff on procedure.",
                    action_type="CORRECTIVE_ACTION",
                )
            ],
        ),
    }
    is_valid, violations = validate_final_analysis(state)
    assert not is_valid
    assert any("zero candidate hypotheses" in v for v in violations)
