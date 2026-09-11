"""Pass 62 (spec §28/§35/§37): domain-agnostic property invariants of the
SEMANTIC + deterministic pipeline. Structured fixtures only -- no live model,
no finding-text parsing. These prove the ARCHITECTURE generalises; they are
never production logic.
"""
from __future__ import annotations

import json

import pytest

from app.remediation.engine import estimate_remediation_cost
from app.models.agent import EvidenceItem, EvidenceStatus
from app.services.canonical_semantic_models import (
    CanonicalFindingContext, SemComparison, comparison_is_active,
)
from app.services.canonical_context_validator import validate_canonical_context


class _FakeLLM:
    def __init__(self, p): self._p = p if isinstance(p, str) else json.dumps(p)
    async def chat_completion(self, m, **k): return self._p


def _ev(c): return EvidenceItem(claim=c, status=EvidenceStatus.VERIFIED, source="t")


def _interp(desc_noun: str, qty=3, unit="unit", rate=4000, atype="PER_UNIT",
            recurrence="ONE_TIME", period=None, price=True):
    comp = {
        "component_id": "C0", "description": f"Replace the {desc_noun}", "activity_ids": ["A0"],
        "cost_category": "parts", "value_kind": "REMEDIATION_COST",
        "quantity": qty, "quantity_unit": unit, "quantity_basis": "EVIDENCED",
        "unit_cost": rate if price else None,
        "unit_cost_basis": "VERIFIED" if price else "NOT_ESTABLISHED",
        "currency": "INR", "amount_type": atype, "recurrence": recurrence,
        "recurring_period": period, "source_reference_ids": ["E0"],
    }
    return {
        "strategy": {"remediation_summary": "x", "remediation_type": "corrective",
                     "interpretation_confidence": "HIGH"},
        "activities": [{"activity_id": "A0", "description": f"Replace the {desc_noun}",
                        "derived_from": "FINDING"}],
        "cost_components": [comp], "calculation_proposals": [],
        "overall_status": "EVIDENCE_BACKED", "estimability": "ESTIMABLE",
    }


async def _run(interp):
    return await estimate_remediation_cost(
        finding_text="A corrective action is required.", evidence_ledger=[_ev("x")],
        client=_FakeLLM(interp),
    )


# --- P1: domain-noun swap must not change the cost arithmetic structure ------
@pytest.mark.asyncio
@pytest.mark.parametrize("noun", [
    "pressure sensor", "patient wristband printer", "software licence",
    "brake caliper", "reinforcement bar", "firewall appliance",
    "cold-store compressor", "runway edge light", "ballast tank valve",
])
async def test_domain_noun_swap_is_cost_invariant(noun):
    res = await _run(_interp(noun))
    assert res.one_time_cost == 12000.0        # 3 x 4000, regardless of the noun
    assert res.pricing_status == "EXACT_ESTIMATE"


# --- P2: removing the price can never produce zero --------------------------
@pytest.mark.asyncio
async def test_missing_price_is_never_zero():
    res = await _run(_interp("widget", price=False))
    c0 = res.cost_components[0]
    assert c0.calculated_amount is None       # not 0.0
    assert res.one_time_cost in (None, 0.0) and (res.one_time_cost or 0) == 0
    assert res.pricing_status in ("NOT_ASSESSABLE", "PARTIAL_ESTIMATE")


# --- P3: recurring with no horizon can never become a finite total ---------
@pytest.mark.asyncio
async def test_recurring_no_horizon_never_finite_total():
    res = await _run(_interp("filter", atype="PER_EVENT", recurrence="RECURRING", period="month"))
    assert res.recurring_cost == 12000.0 or res.recurring_cost == 4000.0
    assert res.recurring_horizon_total is None
    assert res.one_time_cost is None


# --- P4: uncertain recurrence never silently becomes ONE_TIME --------------
@pytest.mark.asyncio
async def test_unknown_recurrence_never_silently_one_time():
    res = await _run(_interp("actuator", atype="PER_EVENT", recurrence="UNKNOWN"))
    assert res.one_time_cost is None
    assert res.review_required is True


# --- P5: adding unrelated numeric values does NOT activate comparison ------
@pytest.mark.parametrize("kw", [
    dict(left="Rs 4000 per part", right="Rs 800 per hour",
         status="ACTUAL_CONFLICT", why_comparable="both are remediation costs",
         comparison_basis="cost", direction="UNKNOWN"),
    dict(left="12 units", right="Rs 90000 total", status="UNRESOLVED_COMPARISON",
         why_comparable="the finding mentions both", direction="UNKNOWN"),
    dict(left="3 machines", right="6 months", status="ACTUAL_CONFLICT",
         why_comparable="numbers in the finding", direction="UNKNOWN"),
])
def test_unrelated_numbers_do_not_activate_comparison(kw):
    ctx = validate_canonical_context(
        CanonicalFindingContext(comparison=SemComparison(**kw)), [], "")
    assert not comparison_is_active(getattr(ctx, "comparison", None))


# --- P6: a comparison against a stated standard STAYS active --------------
@pytest.mark.parametrize("kw", [
    dict(left="actual Rs 220000", right="approved budget Rs 150000",
         status="ACTUAL_CONFLICT", why_comparable="cost must be within the approved budget",
         direction="ABOVE"),
    dict(left="measured 7 bar", right="required 5 bar", status="ACTUAL_CONFLICT",
         why_comparable="measured value must meet the requirement", direction="UNKNOWN"),
])
def test_genuine_comparison_against_a_standard_stays_active(kw):
    ctx = validate_canonical_context(
        CanonicalFindingContext(comparison=SemComparison(**kw)), [], "")
    assert comparison_is_active(getattr(ctx, "comparison", None))
