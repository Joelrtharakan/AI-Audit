"""Deterministic structural-validation coverage for app.remediation.validator."""

from __future__ import annotations

import pytest

from app.remediation.semantic_models import (
    RemediationCalculationProposal,
    RemediationCostComponent,
    RemediationInterpretation,
)
from app.remediation.validator import validate_and_plan

EV = {"E0", "E1"}


def _interp(components, proposals=None):
    return RemediationInterpretation(
        cost_components=[RemediationCostComponent(**c) for c in components],
        calculation_proposals=[RemediationCalculationProposal(**p) for p in (proposals or [])],
    )


def test_unsupported_verified_claim_is_downgraded():
    interp = _interp([{
        "component_id": "C0", "description": "labor", "cost_category": "labor",
        "unit_cost": 5000, "unit_cost_basis": "VERIFIED", "currency": "INR",
        "amount_type": "TOTAL", "source_reference_ids": [],  # nothing cited
    }])
    comps, _, outcome = validate_and_plan(interp, EV)
    assert comps[0].unit_cost_basis in ("ESTIMATED", "ASSUMED")
    assert any("downgraded" in d for d in outcome.llm_disagreements)


def test_verified_claim_with_real_evidence_ref_survives():
    interp = _interp([{
        "component_id": "C0", "description": "part", "cost_category": "replacement",
        "unit_cost": 5000, "unit_cost_basis": "VERIFIED", "currency": "INR",
        "amount_type": "TOTAL", "source_reference_ids": ["E1"],
    }])
    comps, _, _ = validate_and_plan(interp, EV)
    assert comps[0].unit_cost_basis == "VERIFIED"


def test_unknown_reference_is_ignored_and_recorded():
    interp = _interp([{
        "component_id": "C0", "description": "x", "cost_category": "labor",
        "unit_cost": 10, "unit_cost_basis": "ESTIMATED", "currency": "INR",
        "amount_type": "TOTAL", "source_reference_ids": ["E9", "HYP-does-not-exist"],
    }])
    comps, _, outcome = validate_and_plan(interp, EV)
    assert comps[0].source_reference_ids == []
    assert any("do not exist" in d for d in outcome.llm_disagreements)


def test_invented_pricing_without_basis_is_stripped():
    interp = _interp([{
        "component_id": "C0", "description": "software licence", "cost_category": "licensing",
        "unit_cost": 250000, "unit_cost_basis": "NOT_ESTABLISHED", "currency": "USD",
        "amount_type": "TOTAL", "assumptions": [],
    }])
    comps, _, outcome = validate_and_plan(interp, EV)
    assert comps[0].unit_cost is None
    assert comps[0].unit_cost_basis == "NOT_ESTABLISHED"
    assert any("removed" in d for d in outcome.llm_disagreements)


def test_total_amount_type_never_keeps_a_quantity():
    interp = _interp([{
        "component_id": "C0", "description": "program", "cost_category": "implementation effort",
        "quantity": 3, "unit_cost": 10000, "unit_cost_basis": "REPORTED", "currency": "INR",
        "amount_type": "TOTAL", "source_reference_ids": ["E0"],
    }])
    comps, _, outcome = validate_and_plan(interp, EV)
    assert comps[0].quantity is None
    assert any("double counting" in d for d in outcome.llm_disagreements)


def test_multiply_proposal_needs_qty_and_unit_cost():
    interp = _interp(
        [
            {"component_id": "C0", "description": "a", "cost_category": "labor",
             "unit_cost": 100, "unit_cost_basis": "ESTIMATED", "currency": "INR", "amount_type": "TOTAL"},
            {"component_id": "C1", "description": "b", "cost_category": "labor",
             "unit_cost": 200, "unit_cost_basis": "ESTIMATED", "currency": "INR", "amount_type": "TOTAL"},
        ],
        [{"calculation_id": "K0", "operation": "MULTIPLY", "component_ids": ["C0", "C1"], "produces": "MOST_LIKELY"}],
    )
    _, accepted, outcome = validate_and_plan(interp, EV)
    assert accepted == []
    assert outcome.rejected and outcome.rejected[0].reason_code == "AMBIGUOUS_OPERANDS"


def test_mixed_currency_sum_proposal_rejected():
    interp = _interp(
        [
            {"component_id": "C0", "description": "a", "cost_category": "labor",
             "unit_cost": 100, "unit_cost_basis": "ESTIMATED", "currency": "INR", "amount_type": "TOTAL"},
            {"component_id": "C1", "description": "b", "cost_category": "labor",
             "unit_cost": 200, "unit_cost_basis": "ESTIMATED", "currency": "USD", "amount_type": "TOTAL"},
        ],
        [{"calculation_id": "K0", "operation": "SUM", "component_ids": ["C0", "C1"], "produces": "MOST_LIKELY"}],
    )
    _, accepted, outcome = validate_and_plan(interp, EV)
    assert accepted == []
    assert any(r.reason_code == "INCOMPATIBLE_CURRENCY" for r in outcome.rejected)
