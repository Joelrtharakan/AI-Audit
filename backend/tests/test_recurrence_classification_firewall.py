"""Pass 57 — RECURRENCE-CLASSIFICATION FIREWALL (Strategy A).

When qwen3:8b mis-prices a recurring remediation cost it leaves a STRUCTURED
contradiction in its own output: it puts the calendar period into
`quantity_unit` and the horizon count into `quantity`, then either tags the
component ONE_TIME (R1) or RECURRING-with-the-horizon-folded-in (R2).

The firewall detects the contradiction between the model's OWN structured
fields (`amount_type`, `quantity_unit`, `quantity`, `recurrence`,
`recurring_period`) -- it reads NO finding text, matches NO monthly/weekly
keyword in the finding, and NEVER converts ONE_TIME->RECURRING. It only strips
the un-trustworthy number and forces review.

It must NOT fire on legitimate one-time work measured in hours, or on a
correctly-formed recurring cost whose per-occurrence quantity is in hours.
"""
from __future__ import annotations

from app.remediation.semantic_models import (
    RemediationCalculationProposal,
    RemediationCostComponent,
    RemediationInterpretation,
)
from app.remediation.validator import validate_and_plan

EV = {"E0", "E1", "FINDING"}


def _interp(components, proposals=None):
    return RemediationInterpretation(
        cost_components=[RemediationCostComponent(**c) for c in components],
        calculation_proposals=[RemediationCalculationProposal(**p) for p in (proposals or [])],
    )


def _rej_codes(outcome):
    return {r.reason_code for r in outcome.rejected}


# ---- R1: calendar-period quantity + ONE_TIME  (the G7 failure) --------------

def test_r1_monthly_quantity_tagged_one_time_is_stripped():
    # "monthly for six months, Rs 2,000" -> qwen3:8b: PER_QUANTITY, quantity 6
    # month, ONE_TIME.
    interp = _interp([{
        "component_id": "C0", "description": "Monthly supplementary verification",
        "cost_category": "verification", "value_kind": "REMEDIATION_COST",
        "quantity": 6.0, "quantity_unit": "month", "quantity_basis": "EVIDENCED",
        "unit_cost": 2000.0, "unit_cost_basis": "VERIFIED", "currency": "INR",
        "amount_type": "PER_QUANTITY", "recurrence": "ONE_TIME",
        "source_reference_ids": ["FINDING"],
    }])
    comps, _, outcome = validate_and_plan(interp, EV)
    assert comps[0].unit_cost is None
    assert comps[0].unit_cost_basis == "NOT_ESTABLISHED"
    assert "RECURRENCE_INCONSISTENT" in _rej_codes(outcome)
    # NOT reinterpreted -- recurrence field untouched
    assert comps[0].recurrence == "ONE_TIME"


def test_r1_variants_week_quarter_year():
    for unit in ("week", "quarter", "year", "annum"):
        interp = _interp([{
            "component_id": "C0", "description": "periodic check", "cost_category": "x",
            "value_kind": "REMEDIATION_COST", "quantity": 4.0, "quantity_unit": unit, "quantity_basis": "EVIDENCED",
            "unit_cost": 500.0, "unit_cost_basis": "VERIFIED", "currency": "INR",
            "amount_type": "PER_QUANTITY", "recurrence": "ONE_TIME",
            "source_reference_ids": ["FINDING"],
        }])
        comps, _, outcome = validate_and_plan(interp, EV)
        assert comps[0].unit_cost is None, unit
        assert "RECURRENCE_INCONSISTENT" in _rej_codes(outcome), unit


# ---- R2: RECURRING but quantity is a horizon count (the G7b failure) --------

def test_r2_recurring_per_month_with_month_quantity_is_stripped():
    interp = _interp([{
        "component_id": "C0", "description": "Monthly verification of analysers",
        "cost_category": "verification", "value_kind": "REMEDIATION_COST",
        "quantity": 12.0, "quantity_unit": "month", "quantity_basis": "DERIVED",
        "quantity_derivation": "1 year x 12 months/year = 12 months",
        "unit_cost": 1500.0, "unit_cost_basis": "VERIFIED", "currency": "INR",
        "amount_type": "PER_QUANTITY", "recurrence": "RECURRING", "recurring_period": "month",
        "source_reference_ids": ["FINDING"],
    }])
    comps, _, outcome = validate_and_plan(interp, EV)
    assert comps[0].unit_cost is None
    assert "RECURRENCE_INCONSISTENT" in _rej_codes(outcome)


# ---- MUST NOT FIRE ---------------------------------------------------------

def test_one_time_job_measured_in_hours_is_untouched():
    # "5 balances, 2 hours each at Rs 800/hr" labour -> ONE_TIME, quantity in
    # HOURS -- legitimate, must survive.
    interp = _interp([{
        "component_id": "C0", "description": "Recalibration labour", "cost_category": "labor",
        "value_kind": "REMEDIATION_COST", "quantity": 10.0, "quantity_unit": "hour",
        "quantity_basis": "EVIDENCED", "unit_cost": 800.0, "unit_cost_basis": "VERIFIED",
        "currency": "INR", "amount_type": "PER_HOUR", "recurrence": "ONE_TIME",
        "source_reference_ids": ["FINDING"],
    }])
    comps, _, outcome = validate_and_plan(interp, EV)
    assert comps[0].unit_cost == 800.0
    assert "RECURRENCE_INCONSISTENT" not in _rej_codes(outcome)


def test_two_day_audit_duration_is_untouched():
    interp = _interp([{
        "component_id": "C0", "description": "Quality audit labour", "cost_category": "labor",
        "value_kind": "REMEDIATION_COST", "quantity": 16.0, "quantity_unit": "hour", "quantity_basis": "EVIDENCED",
        "unit_cost": 2000.0, "unit_cost_basis": "VERIFIED", "currency": "INR",
        "amount_type": "PER_HOUR", "recurrence": "ONE_TIME", "source_reference_ids": ["FINDING"],
    }])
    comps, _, outcome = validate_and_plan(interp, EV)
    assert comps[0].unit_cost == 2000.0
    assert "RECURRENCE_INCONSISTENT" not in _rej_codes(outcome)


def test_correct_recurring_per_hour_quantity_is_untouched():
    # "verify monthly, each verification 2 hours at Rs 800/hr" -> RECURRING,
    # recurring_period month, quantity 2 HOURS (per occurrence) -- correct.
    interp = _interp([{
        "component_id": "C0", "description": "Monthly verification labour", "cost_category": "labor",
        "value_kind": "REMEDIATION_COST", "quantity": 2.0, "quantity_unit": "hour", "quantity_basis": "EVIDENCED",
        "unit_cost": 800.0, "unit_cost_basis": "VERIFIED", "currency": "INR",
        "amount_type": "PER_HOUR", "recurrence": "RECURRING", "recurring_period": "month",
        "source_reference_ids": ["FINDING"],
    }])
    comps, _, outcome = validate_and_plan(interp, EV)
    assert comps[0].unit_cost == 800.0
    assert "RECURRENCE_INCONSISTENT" not in _rej_codes(outcome)


def test_population_quantity_per_unit_is_untouched():
    # "8 panels x Rs 350" -> PER_UNIT, quantity 8 PANEL, ONE_TIME -- fine.
    interp = _interp([{
        "component_id": "C0", "description": "Labels", "cost_category": "materials",
        "value_kind": "REMEDIATION_COST", "quantity": 8.0, "quantity_unit": "panel", "quantity_basis": "EVIDENCED",
        "unit_cost": 350.0, "unit_cost_basis": "VERIFIED", "currency": "INR",
        "amount_type": "PER_UNIT", "recurrence": "ONE_TIME", "source_reference_ids": ["FINDING"],
    }])
    comps, _, outcome = validate_and_plan(interp, EV)
    assert comps[0].unit_cost == 350.0
    assert "RECURRENCE_INCONSISTENT" not in _rej_codes(outcome)


def test_correct_recurring_no_quantity_is_untouched():
    interp = _interp([{
        "component_id": "C0", "description": "Monthly manual verification", "cost_category": "x",
        "value_kind": "REMEDIATION_COST", "unit_cost": 2000.0, "unit_cost_basis": "VERIFIED",
        "currency": "INR", "amount_type": "PER_EVENT", "recurrence": "RECURRING",
        "recurring_period": "month", "quantity": 1.0, "quantity_basis": "EVIDENCED",
        "source_reference_ids": ["FINDING"],
    }])
    comps, _, outcome = validate_and_plan(interp, EV)
    assert comps[0].unit_cost == 2000.0
    assert "RECURRENCE_INCONSISTENT" not in _rej_codes(outcome)
