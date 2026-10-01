"""Deterministic arithmetic coverage for app.remediation.calculator."""

from __future__ import annotations

import pytest

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
    # Phase 9.9 (deliberate semantic change, reviewed): with no option the LLM
    # DECLARED primary, no most-likely figure exists. The former behaviour
    # reported the first/lowest option as "conservative most-likely", i.e. it
    # chose a bound as a point estimate. The range stays; the point is absent.
    assert est.one_time_cost is None and est.most_likely is None
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


# --------------------------------------------------------------------------- #
# Phase 9.6 §6: quantity-unit pluralization in the rendered formula string --
# pure English count-agreement, not a domain-specific formatting rule. "45
# hour x INR 1100 = INR 49500" must read "45 hours x ...".
# --------------------------------------------------------------------------- #

from app.remediation.calculator import _pluralize_unit


@pytest.mark.parametrize("unit,count,expected", [
    ("hour", 45, "hours"), ("hour", 1, "hour"),
    ("machine", 9, "machines"), ("machine", 1, "machine"),
    ("box", 3, "boxes"), ("activity", 4, "activities"),
    ("visit", 1, "visit"), ("visit", 2, "visits"),
    ("day", 30, "days"),
])
def test_pluralize_unit_general_english_count_agreement(unit, count, expected):
    assert _pluralize_unit(unit, count) == expected


def test_formula_uses_pluralized_unit_for_a_multiplied_component():
    from app.remediation.semantic_models import RemediationCostComponent
    from app.remediation.calculator import _formula
    c = RemediationCostComponent(
        component_id="C0", description="technician labor", cost_category="labor",
        amount_type="PER_HOUR", unit_cost_basis="ESTIMATED", unit_cost=1100,
        quantity=45, quantity_unit="hour", currency="INR",
    )
    assert _formula(c) == "45 hours x INR 1100 = INR 49500"


def test_formula_keeps_singular_unit_for_a_quantity_of_one():
    from app.remediation.semantic_models import RemediationCostComponent
    from app.remediation.calculator import _formula
    c = RemediationCostComponent(
        component_id="C0", description="single inspection visit", cost_category="labor",
        amount_type="PER_EVENT", unit_cost_basis="ESTIMATED", unit_cost=2000,
        quantity=1, quantity_unit="visit", currency="INR",
    )
    assert _formula(c) == "1 visit x INR 2000 = INR 2000"


# --------------------------------------------------------------------------- #
# Phase 9.7 §2/§4/§16: general range-aware cost representation. Ranges may
# live on the QUANTITY (fixed rate), the RATE (fixed quantity), a flat
# (non-multiplying) amount, or be absent entirely -- the aggregate low/high
# must propagate correctly through addition and multiplication in every
# combination, without collapsing to a bound, a midpoint, or a fabricated
# quantity/horizon. The 70-110h/₹1,900 + ₹55,000-85,000 fixture is the
# REGRESSION CASE from the reported defect -- used here only as one of
# several generalized cases, never hardcoded into production code.
# --------------------------------------------------------------------------- #

def _range_labor(low=70, high=110, rate=1900, **over):
    d = dict(component_id="C1", description="labor", cost_category="labor",
              amount_type="PER_HOUR", unit_cost_basis="ESTIMATED",
              quantity_low=low, quantity_high=high, unit_cost=rate,
              quantity_unit="hour", currency="INR")
    d.update(over)
    return RemediationCostComponent(**d)


def _range_materials(low=55000, high=85000, **over):
    d = dict(component_id="C2", description="materials", cost_category="materials",
              amount_type="COMPONENT", unit_cost_basis="ESTIMATED",
              unit_cost_low=low, unit_cost_high=high, currency="INR")
    d.update(over)
    return RemediationCostComponent(**d)


def test_regression_fixture_range_plus_range_produces_correct_bounds():
    # THE reported defect: this must be 188000-294000, never 188000-218000.
    est = assemble_estimate([_range_labor(), _range_materials()], [], [])
    assert est.low == 188000.0
    assert est.high == 294000.0
    assert est.most_likely is None  # no manufactured midpoint/most-likely


def test_fixed_times_fixed_is_unaffected():
    c = RemediationCostComponent(
        component_id="C0", description="labor", cost_category="labor",
        amount_type="PER_HOUR", unit_cost_basis="VERIFIED",
        quantity=10, unit_cost=500, currency="INR",
    )
    est = assemble_estimate([c], [], [])
    assert est.low == est.most_likely == est.high == 5000.0


def test_range_quantity_times_fixed_rate():
    est = assemble_estimate([_range_labor(low=10, high=20, rate=100)], [], [])
    assert est.low == 1000.0
    assert est.high == 2000.0
    assert est.most_likely is None


def test_fixed_quantity_times_range_rate():
    c = RemediationCostComponent(
        component_id="C0", description="labor", cost_category="labor",
        amount_type="PER_HOUR", unit_cost_basis="ESTIMATED",
        quantity=10, unit_cost_low=80, unit_cost_high=120, currency="INR",
    )
    est = assemble_estimate([c], [], [])
    assert est.low == 800.0
    assert est.high == 1200.0
    assert est.most_likely is None


def test_range_plus_fixed_component():
    fixed = RemediationCostComponent(
        component_id="C0", description="fee", cost_category="services",
        amount_type="COMPONENT", unit_cost_basis="VERIFIED", unit_cost=20000, currency="INR",
    )
    est = assemble_estimate([_range_materials(), fixed], [], [])
    assert est.low == 75000.0   # 55000 + 20000
    assert est.high == 105000.0  # 85000 + 20000
    assert est.most_likely is None  # materials has no point -> ml unavailable


def test_multiple_range_components_sum_bound_by_bound():
    est = assemble_estimate([_range_labor(), _range_materials(low=10000, high=20000)], [], [])
    assert est.low == 143000.0   # 133000 + 10000
    assert est.high == 229000.0  # 209000 + 20000


def test_missing_upper_bound_does_not_fabricate_one():
    c = RemediationCostComponent(
        component_id="C0", description="materials", cost_category="materials",
        amount_type="COMPONENT", unit_cost_basis="ESTIMATED", unit_cost_low=55000, currency="INR",
    )
    # only a lower bound stated -- must not silently invent an upper bound
    est = assemble_estimate([c], [], [])
    assert est.high != 55000.0 or est.low is None  # never collapse to a single point silently
    assert est.most_likely is None


def test_missing_lower_bound_does_not_fabricate_one():
    c = RemediationCostComponent(
        component_id="C0", description="materials", cost_category="materials",
        amount_type="COMPONENT", unit_cost_basis="ESTIMATED", unit_cost_high=85000, currency="INR",
    )
    est = assemble_estimate([c], [], [])
    assert est.most_likely is None


def test_invalid_inverted_quantity_range_is_rejected_by_validator():
    from app.remediation.validator import validate_and_plan
    from app.remediation.semantic_models import RemediationInterpretation
    interp = RemediationInterpretation(cost_components=[RemediationCostComponent(
        component_id="C0", description="labor", cost_category="labor",
        amount_type="PER_HOUR", unit_cost_basis="ESTIMATED",
        quantity_low=110, quantity_high=70, unit_cost=1900, currency="INR",
        source_reference_ids=["E0"],
    )])
    comps, _, outcome = validate_and_plan(interp, {"E0", "E1"})
    assert comps[0].quantity_low is None and comps[0].quantity_high is None
    assert any("inverted range" in d for d in outcome.llm_disagreements)


def test_no_fabricated_horizon_for_a_range_only_recurring_component():
    c = RemediationCostComponent(
        component_id="C0", description="monthly service", cost_category="services",
        amount_type="COMPONENT", unit_cost_basis="REPORTED",
        unit_cost_low=12000, unit_cost_high=16000, recurrence="RECURRING",
        recurring_period="month", currency="INR",
    )
    est = assemble_estimate([c], [], [])
    assert est.recurring_cost is None  # no point -> no single recurring figure manufactured
    assert est.recurring_horizon_total is None
