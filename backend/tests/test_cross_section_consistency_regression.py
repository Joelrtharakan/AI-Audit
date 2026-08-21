"""Regression suite for the centralized cross-section consistency invariant.

Locks in INV-CONSISTENCY-001 (app.agent.invariants.
_check_cross_section_consistency_graph): root_cause, CAPA, impact,
cost_impact, and five_why are written by different code paths (LLM
synthesis, deterministic fallback, critic repair, financial analysis) and
must not disagree about the same underlying fact -- root cause vs CAPA
certainty, financial recovery status between cost_impact and the impact
narrative, and 5-Why completion vs root cause establishment. This is
additive: it does not replace any of the ~90 existing narrower point-checks,
it catches pairs those don't happen to cover.

Scenarios are built from hand-constructed state dicts (structural, not tied
to any one finding's wording) so the cross-section logic is verified in
isolation from LLM/deterministic-fallback synthesis behavior.
"""

from __future__ import annotations

from app.agent.invariants import evaluate_all_invariants
from app.models.agent import (
    CapaAnalysis,
    CapaStatus,
    ConditionalCapaAction,
    CostImpact,
    FiveWhyAnalysis,
    FiveWhyStep,
    ImpactAssessment,
    ImpactStatus,
    RootCauseAnalysis,
    RootCauseStatus,
)


def _rc(status: RootCauseStatus) -> RootCauseAnalysis:
    return RootCauseAnalysis(status=status)


def test_unconditional_corrective_capa_with_unestablished_cause_flagged():
    rc = _rc(RootCauseStatus.NOT_ESTABLISHED)
    capa = CapaAnalysis(
        status=CapaStatus.INVESTIGATION_REQUIRED,
        conditional_actions=[
            ConditionalCapaAction(
                if_cause_confirmed="", recommended_action="Retrain the operator",
                action_type="CORRECTIVE_ACTION",
            )
        ],
    )
    state = {"root_cause": rc, "capa_analysis": capa}
    is_valid, violations = evaluate_all_invariants(state)
    assert any("INV-CONSISTENCY-001" in v and "unconditional" in v for v in violations), violations


def test_properly_conditional_capa_with_unestablished_cause_not_flagged():
    rc = _rc(RootCauseStatus.NOT_ESTABLISHED)
    capa = CapaAnalysis(
        status=CapaStatus.INVESTIGATION_REQUIRED,
        conditional_actions=[
            ConditionalCapaAction(
                if_cause_confirmed="If training was never assigned",
                recommended_action="Retrain the operator",
                action_type="CORRECTIVE_ACTION",
            )
        ],
    )
    state = {"root_cause": rc, "capa_analysis": capa}
    is_valid, violations = evaluate_all_invariants(state)
    assert not any("INV-CONSISTENCY-001" in v for v in violations), violations


def test_financial_recovery_disagreement_flagged():
    cost = CostImpact(cost_factor_detected=True, recoverability_status="RECOVERED")
    impact = ImpactAssessment(
        status=ImpactStatus.IMPACT_VERIFIED,
        impact_observed="The unrecovered balance remains outstanding pending reconciliation.",
    )
    state = {"cost_impact": cost, "impact_assessment": impact}
    is_valid, violations = evaluate_all_invariants(state)
    assert any("INV-CONSISTENCY-001" in v and "Financial recovery" in v for v in violations), violations


def test_financial_recovery_agreement_not_flagged():
    cost = CostImpact(cost_factor_detected=True, recoverability_status="RECOVERED")
    impact = ImpactAssessment(
        status=ImpactStatus.IMPACT_VERIFIED,
        impact_observed="The full amount was recovered via supplier credit note.",
    )
    state = {"cost_impact": cost, "impact_assessment": impact}
    is_valid, violations = evaluate_all_invariants(state)
    assert not any("INV-CONSISTENCY-001" in v for v in violations), violations


def test_five_why_complete_verified_step_with_unestablished_root_cause_flagged():
    rc = _rc(RootCauseStatus.NOT_ESTABLISHED)
    fw = FiveWhyAnalysis(
        steps=[FiveWhyStep(question="Why did X occur?", answer="Because Y.", status="VERIFIED")],
        is_complete=True,
    )
    state = {"root_cause": rc, "five_why": fw}
    is_valid, violations = evaluate_all_invariants(state)
    assert any("INV-CONSISTENCY-001" in v and "5-Why marked complete" in v for v in violations), violations


def test_five_why_complete_verified_step_with_established_root_cause_not_flagged():
    rc = _rc(RootCauseStatus.ESTABLISHED)
    fw = FiveWhyAnalysis(
        steps=[FiveWhyStep(question="Why did X occur?", answer="Because Y.", status="VERIFIED")],
        is_complete=True,
    )
    state = {"root_cause": rc, "five_why": fw}
    is_valid, violations = evaluate_all_invariants(state)
    assert not any("INV-CONSISTENCY-001" in v for v in violations), violations


def test_five_why_deferred_with_unestablished_root_cause_not_flagged():
    """5-Why deferred at the evidence boundary (not complete, UNKNOWN final
    step) is the expected outcome when root cause isn't established -- must
    never be treated as a contradiction."""
    rc = _rc(RootCauseStatus.NOT_ESTABLISHED)
    fw = FiveWhyAnalysis(
        steps=[FiveWhyStep(question="Why did X occur?", answer="Not established from available evidence.", status="UNKNOWN")],
        is_complete=False,
    )
    state = {"root_cause": rc, "five_why": fw}
    is_valid, violations = evaluate_all_invariants(state)
    assert not any("INV-CONSISTENCY-001" in v for v in violations), violations
