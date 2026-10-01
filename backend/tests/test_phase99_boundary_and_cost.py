"""Phase 9.9 -- deterministic tests: range/recurring arithmetic, epistemic
fidelity of the 5-Why status sync, review-state structural reasons, and the
heuristic-inventory guard. Structure only; no wording, no live LLM."""

from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path

import pytest

from app.agent.analytical_validator import sync_five_why_status_with_causal_state
from app.agent.causal_guard import MechanismInfo
from app.models.agent import FiveWhyStep
from app.remediation.calculator import assemble_estimate
from app.remediation.semantic_models import RemediationCalculationProposal, RemediationCostComponent

_BACKEND = Path(__file__).resolve().parent.parent


def _c(**kw):
    base = dict(component_id="C?", description="x", cost_category="other", unit_cost_basis="ESTIMATED",
                currency="USD", amount_type="COMPONENT", recurrence="ONE_TIME")
    base.update(kw)
    return RemediationCostComponent(**base)


# --------------------------------------------------------------------------- #
# range arithmetic (deterministic, dimensionally valid, never collapsed)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("a,b", list(itertools.product([(100, 150), (0, 40), (7, 7)], [(10, 25), (300, 301)])))
def test_range_plus_range_adds_endpoints(a, b):
    comps = [_c(component_id="A", unit_cost_low=a[0], unit_cost_high=a[1]),
             _c(component_id="B", unit_cost_low=b[0], unit_cost_high=b[1])]
    est = assemble_estimate(comps, [], [])
    assert (est.low, est.high) == (a[0] + b[0], a[1] + b[1])
    assert est.low <= est.high
    if a[0] != a[1] or b[0] != b[1]:
        assert est.most_likely is None  # no invented midpoint / most-likely


def test_fixed_plus_range_keeps_range():
    est = assemble_estimate([_c(component_id="F", unit_cost=500),
                             _c(component_id="R", unit_cost_low=100, unit_cost_high=300)], [], [])
    assert (est.low, est.high) == (600, 800) and est.most_likely is None


def test_quantity_times_rate_range_propagates_both_sides():
    comp = _c(component_id="L", amount_type="PER_HOUR", quantity_low=10, quantity_high=20, quantity_unit="hour",
              unit_cost_low=50, unit_cost_high=80, quantity_basis="EVIDENCED")
    est = assemble_estimate([comp], [], [])
    assert (est.low, est.high) == (500, 1600) and est.most_likely is None


def test_ranged_quantity_fixed_rate_not_collapsed():
    comp = _c(component_id="L", amount_type="PER_HOUR", quantity_low=70, quantity_high=110, quantity_unit="hour",
              unit_cost=1900, quantity_basis="EVIDENCED")
    est = assemble_estimate([comp], [], [])
    assert (est.low, est.high) == (133000, 209000) and est.most_likely is None


def test_alternatives_without_declared_primary_have_no_point_estimate():
    comps = [_c(component_id="A", amount_type="ALTERNATIVE", alternative_group="g", unit_cost=40),
             _c(component_id="B", amount_type="ALTERNATIVE", alternative_group="g", unit_cost=90)]
    est = assemble_estimate(comps, [], [])
    assert (est.low, est.high) == (40, 90) and est.most_likely is None


def test_alternatives_with_declared_primary_have_point_estimate():
    comps = [_c(component_id="A", amount_type="ALTERNATIVE", alternative_group="g", unit_cost=40, is_primary_option=True),
             _c(component_id="B", amount_type="ALTERNATIVE", alternative_group="g", unit_cost=90)]
    est = assemble_estimate(comps, [], [])
    assert est.most_likely == 40


def test_multiple_alternative_groups_one_without_primary_has_no_point():
    comps = [_c(component_id="A", amount_type="ALTERNATIVE", alternative_group="g1", unit_cost=10, is_primary_option=True),
             _c(component_id="B", amount_type="ALTERNATIVE", alternative_group="g1", unit_cost=20),
             _c(component_id="C", amount_type="ALTERNATIVE", alternative_group="g2", unit_cost=100),
             _c(component_id="D", amount_type="ALTERNATIVE", alternative_group="g2", unit_cost=200)]
    est = assemble_estimate(comps, [], [])
    assert (est.low, est.high) == (110, 220) and est.most_likely is None


# --------------------------------------------------------------------------- #
# one-time vs recurring: kept separate, no invented horizon
# --------------------------------------------------------------------------- #

def test_recurring_without_horizon_is_not_totalled_or_annualised():
    comps = [_c(component_id="O", unit_cost=1000),
             _c(component_id="R", unit_cost=200, recurrence="RECURRING", recurring_period="month")]
    est = assemble_estimate(comps, [], [])
    assert est.one_time_cost == 1000
    assert est.recurring_cost == 200 and est.recurring_period == "month"
    assert est.recurring_horizon_total is None and est.recurring_horizon is None


def test_recurring_only_does_not_pollute_one_time():
    est = assemble_estimate([_c(component_id="R", unit_cost=300, recurrence="RECURRING", recurring_period="quarter")], [], [])
    assert est.recurring_cost == 300 and not est.one_time_cost


def test_horizon_total_only_from_explicit_llm_horizon():
    comp = _c(component_id="R", unit_cost=100, recurrence="RECURRING", recurring_period="month")
    prop = RemediationCalculationProposal(calculation_id="K", operation="SUM", component_ids=["R"],
                                          produces="MOST_LIKELY", horizon=6, horizon_unit="month", horizon_basis="EXPLICIT")
    assert assemble_estimate([comp], [prop], []).recurring_horizon_total == 600
    inferred = prop.model_copy(update={"horizon_basis": "INFERRED"})
    assert assemble_estimate([comp], [inferred], []).recurring_horizon_total is None


# --------------------------------------------------------------------------- #
# 5-Why status sync: a REPORTED mechanism is never promoted to SUPPORTED
# --------------------------------------------------------------------------- #

def _step(answer, status="UNKNOWN"):
    return FiveWhyStep(question="why?", answer=answer, status=status)


@pytest.mark.parametrize("statement", [
    "the valve seat was worn through", "the interface dropped messages", "the courier delayed the shipment",
])
def test_reported_mechanism_syncs_to_reported_status_not_supported(statement):
    mech = MechanismInfo(statement=statement, status="REPORTED")
    out = sync_five_why_status_with_causal_state([_step(statement)], mech, None)
    assert out[0].status == "REPORTED_UNVERIFIED"


def test_verified_mechanism_still_syncs_to_verified():
    mech = MechanismInfo(statement="the valve seat was worn through", status="VERIFIED")
    out = sync_five_why_status_with_causal_state([_step("the valve seat was worn through")], mech, None)
    assert out[0].status == "VERIFIED"


def test_unrelated_step_is_untouched():
    mech = MechanismInfo(statement="the valve seat was worn through", status="REPORTED")
    out = sync_five_why_status_with_causal_state([_step("an entirely different deeper question", "UNKNOWN")], mech, None)
    assert out[0].status == "UNKNOWN"


# --------------------------------------------------------------------------- #
# review-state structural reasons
# --------------------------------------------------------------------------- #

def _report_stub(**kw):
    from types import SimpleNamespace
    base = dict(analysis_mode="LLM", semantic_mode="CANONICAL_LLM", canonical_semantic_status="SUCCESS",
                fallback_used=False, observation_quality="SUFFICIENT",
                evidence_completeness=SimpleNamespace(value="COMPLETE"),
                root_cause=SimpleNamespace(status=SimpleNamespace(value="NOT_ESTABLISHED")),
                remediation_cost=None, financial_analysis=None, semantic_consistency_issues=[])
    base.update(kw)
    return SimpleNamespace(**base)


def test_review_state_surfaces_consistency_issues_and_regex_financials():
    from types import SimpleNamespace
    from app.agent.review_state import derive_review_state
    st = derive_review_state(_report_stub(
        semantic_consistency_issues=["LEADING_HYPOTHESIS_WITHOUT_ESTABLISHED_CAUSE"],
        financial_analysis=SimpleNamespace(reasoning_source="DETERMINISTIC_REGEX", financial_semantic_status="OK"),
    ))
    codes = {r.code for r in st.reasons}
    assert st.required is True
    assert {"SEMANTIC_CONSISTENCY_ISSUES", "FINANCIAL_SEMANTICS_NOT_LLM_VERIFIED"} <= codes


def test_review_state_clean_has_neither_but_is_still_required():
    from types import SimpleNamespace
    from app.agent.review_state import derive_review_state
    st = derive_review_state(_report_stub(financial_analysis=SimpleNamespace(reasoning_source="LLM_SEMANTIC",
                                                                              financial_semantic_status="OK")))
    codes = {r.code for r in st.reasons}
    assert st.required is True and st.status == "PENDING_HUMAN_REVIEW"
    assert not ({"SEMANTIC_CONSISTENCY_ISSUES", "FINANCIAL_SEMANTICS_NOT_LLM_VERIFIED"} & codes)


# --------------------------------------------------------------------------- #
# heuristic inventory guard: no undocumented heuristic-bearing module
# --------------------------------------------------------------------------- #

def test_every_heuristic_bearing_module_is_in_the_inventory():
    sys.path.insert(0, str(_BACKEND / "scripts"))
    import audit_semantic_heuristics as audit
    doc = (_BACKEND / "docs" / "PHASE_9_9_SEMANTIC_HEURISTIC_INVENTORY.md").read_text()
    files = {h["file"] for p in sorted(audit.APP.rglob("*.py")) for h in audit.scan_file(p)}
    missing = sorted(f for f in files if f not in doc)
    assert not missing, "Heuristic-bearing modules missing from the Phase 9.9 inventory:\n  " + "\n  ".join(missing)


def test_autonomy_gate_and_human_review_remain_fail_closed():
    from app.agent.autonomy import get_certificate_store  # noqa: F401  (import must succeed)
    from app.agent.review_state import derive_review_state
    assert derive_review_state(_report_stub()).required is True
