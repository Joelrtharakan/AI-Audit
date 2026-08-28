"""Deterministic arithmetic coverage for app.remediation.calculator."""

from __future__ import annotations

from app.remediation.calculator import assemble_estimate
from app.remediation.models import CostBasis
from app.remediation.semantic_models import RemediationCostComponent


def _c(**kw):
    base = dict(component_id="C?", description="x", cost_category="labor",
               unit_cost_basis="ESTIMATED", currency="INR", amount_type="TOTAL", recurrence="ONE_TIME")
    base.update(kw)
    return RemediationCostComponent(**base)


# --- Spec sections 4, 5, 23: additive components must be SUMMED, not ranged ---

def test_two_required_components_are_added_not_ranged():
    """Component cost = X, Installation cost = Y, both required -> Total = X + Y,
    with low == most_likely == high (no manufactured range). NOT low=Y, ml=X, high=X+Y."""
    comps = [
        _c(component_id="C0", description="equipment", unit_cost=120000, amount_type="COMPONENT"),
        _c(component_id="C1", description="installation", unit_cost=35000, amount_type="COMPONENT"),
    ]
    est = assemble_estimate(comps, [], [])
    assert est.one_time_cost == 155000.0
    assert est.low == est.most_likely == est.high == 155000.0
    assert "sum of the required implementation components" in est.estimation_method


def test_llm_produces_low_high_proposals_do_not_manufacture_a_range():
    """Even if the LLM proposes produces=LOW over [C1], produces=MOST_LIKELY over
    [C0], produces=HIGH over [C0,C1], the deterministic range still comes only
    from component structure: both are required -> single total."""
    from app.remediation.semantic_models import RemediationCalculationProposal
    comps = [
        _c(component_id="C0", description="a", unit_cost=120000, amount_type="COMPONENT"),
        _c(component_id="C1", description="b", unit_cost=35000, amount_type="COMPONENT"),
    ]
    props = [
        RemediationCalculationProposal(calculation_id="K0", operation="SUM", component_ids=["C1"], produces="LOW"),
        RemediationCalculationProposal(calculation_id="K1", operation="SUM", component_ids=["C0"], produces="MOST_LIKELY"),
        RemediationCalculationProposal(calculation_id="K2", operation="SUM", component_ids=["C0", "C1"], produces="HIGH"),
    ]
    est = assemble_estimate(comps, props, [])
    assert est.low == est.most_likely == est.high == 155000.0


def test_alternative_options_are_bracketed_not_summed():
    comps = [
        _c(component_id="C0", description="option A", unit_cost=40000, amount_type="ALTERNATIVE", alternative_group="g1"),
        _c(component_id="C1", description="option B", unit_cost=90000, amount_type="ALTERNATIVE", alternative_group="g1"),
    ]
    est = assemble_estimate(comps, [], [])
    assert est.low == 40000.0 and est.high == 90000.0
    assert est.one_time_cost == 40000.0  # conservative most-likely (no primary flagged)
    assert "alternative implementation option" in est.estimation_method


def test_alternative_plus_required_component_combine_correctly():
    comps = [
        _c(component_id="C0", description="mandatory base work", unit_cost=20000, amount_type="COMPONENT"),
        _c(component_id="C1", description="option A", unit_cost=40000, amount_type="ALTERNATIVE", alternative_group="g1", is_primary_option=True),
        _c(component_id="C2", description="option B", unit_cost=90000, amount_type="ALTERNATIVE", alternative_group="g1"),
    ]
    est = assemble_estimate(comps, [], [])
    assert est.low == 60000.0        # 20000 + min(40000, 90000)
    assert est.most_likely == 60000.0  # 20000 + primary(40000)
    assert est.high == 110000.0       # 20000 + max(40000, 90000)


def test_grand_total_consistent_with_parts_is_authoritative():
    comps = [
        _c(component_id="C0", description="a", unit_cost=120000, amount_type="COMPONENT"),
        _c(component_id="C1", description="b", unit_cost=35000, amount_type="COMPONENT"),
        _c(component_id="C2", description="quoted total", unit_cost=155000, amount_type="TOTAL", unit_cost_basis="REPORTED"),
    ]
    est = assemble_estimate(comps, [], [])
    assert est.most_likely == 155000.0
    assert est.low == est.high == 155000.0
    assert "reconciled" in est.estimation_method


def test_grand_total_conflicting_with_parts_preserves_both():
    comps = [
        _c(component_id="C0", description="a", unit_cost=120000, amount_type="COMPONENT"),
        _c(component_id="C1", description="b", unit_cost=35000, amount_type="COMPONENT"),
        _c(component_id="C2", description="quoted total", unit_cost=200000, amount_type="TOTAL"),
    ]
    est = assemble_estimate(comps, [], [])
    assert any("does not reconcile" in u for u in est.uncertainty_reasons)
    assert est.low == 155000.0 and est.high == 200000.0


def test_per_unit_multiplies_total_never_does():
    comps = [
        _c(component_id="C0", quantity=10, unit_cost=500, amount_type="PER_UNIT"),
        _c(component_id="C1", quantity=3, unit_cost=9999, amount_type="COMPONENT"),  # flat fee; qty ignored (not PER_*)
    ]
    est = assemble_estimate(comps, [], [])
    # C0 = 10 x 500 = 5000 ; C1 = 9999 flat ; both required -> total 14999, no range
    assert est.one_time_cost == 14999.0
    assert est.low == est.most_likely == est.high == 14999.0


def test_stated_total_matching_components_not_double_counted():
    comps = [
        _c(component_id="C0", unit_cost=4000, amount_type="COMPONENT"),
        _c(component_id="C1", unit_cost=6000, amount_type="COMPONENT"),
        _c(component_id="C2", unit_cost=10000, amount_type="SUBTOTAL"),
    ]
    est = assemble_estimate(comps, [], [])
    assert est.most_likely == 10000.0
    assert any(r.is_derived for r in est.component_results)


def test_stated_total_disagreeing_uses_components_and_flags():
    comps = [
        _c(component_id="C0", unit_cost=4000, amount_type="COMPONENT"),
        _c(component_id="C1", unit_cost=6000, amount_type="COMPONENT"),
        _c(component_id="C2", unit_cost=25000, amount_type="SUBTOTAL"),
    ]
    est = assemble_estimate(comps, [], [])
    assert est.most_likely == 10000.0
    assert any("reconcile" in u for u in est.uncertainty_reasons)


def test_range_from_component_unit_cost_bounds():
    comps = [_c(component_id="C0", quantity=100, unit_cost=50, unit_cost_low=40, unit_cost_high=70,
                amount_type="PER_UNIT")]
    est = assemble_estimate(comps, [], [])
    assert est.low == 4000.0 and est.most_likely == 5000.0 and est.high == 7000.0


def test_single_verified_cost_all_three_equal():
    comps = [_c(component_id="C0", unit_cost=85000, unit_cost_basis="VERIFIED", amount_type="TOTAL")]
    est = assemble_estimate(comps, [], [])
    assert est.low == est.most_likely == est.high == 85000.0
    assert est.estimate_classification == CostBasis.VERIFIED


def test_one_time_and_recurring_kept_separate():
    comps = [
        _c(component_id="C0", unit_cost=100000, amount_type="TOTAL", recurrence="ONE_TIME"),
        _c(component_id="C1", unit_cost=12000, amount_type="TOTAL", recurrence="RECURRING", recurring_period="year"),
    ]
    est = assemble_estimate(comps, [], [])
    assert est.one_time_cost == 100000.0
    assert est.recurring_cost == 12000.0
    assert est.recurring_period == "year"
    assert est.most_likely == 100000.0  # range is the one-time implementation cost


def test_nothing_calculable_leaves_all_none():
    comps = [_c(component_id="C0", unit_cost=None, amount_type="COMPONENT")]
    est = assemble_estimate(comps, [], [])
    assert est.low is None and est.most_likely is None and est.high is None
    assert est.estimate_classification == CostBasis.NOT_ESTABLISHED


def test_no_invented_spread_when_no_uncertainty():
    comps = [_c(component_id="C0", unit_cost=50000, amount_type="TOTAL", unit_cost_basis="REPORTED")]
    est = assemble_estimate(comps, [], [])
    assert est.low == est.most_likely == est.high == 50000.0
