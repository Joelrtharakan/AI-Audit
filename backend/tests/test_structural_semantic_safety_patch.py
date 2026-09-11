"""Phase 4 structural semantic safety patch.

Three classes of silent material error, fixed by validating the MODEL'S OWN
structured output (never the finding text):

  A. comparison vs composition -- a pricing/rate basis with no baseline
     relationship, or a bare direction=MISMATCH, does not make a comparison.
  B. folded recurring horizon -- a RECURRING cost whose quantity is a count of
     occurrences (a folded horizon) is de-escalated, not priced.
  C. multi-currency composition -- components in different currencies with no
     conversion basis are not summed into one headline.

CRITICAL: all decisions read ONLY model-authored structured fields. Part E
proves that changing the raw finding text while holding the structure constant
does not change the decision.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.models.agent import EvidenceItem, EvidenceStatus
from app.remediation.engine import estimate_remediation_cost
from app.services.canonical_context_validator import validate_canonical_context
from app.services.canonical_semantic_models import CanonicalFindingContext, comparison_is_active


def _cmp_active(cmp: dict, finding_text: str = "irrelevant") -> bool:
    ctx = CanonicalFindingContext.model_validate({"comparison": cmp})
    v = validate_canonical_context(
        ctx, [EvidenceItem(claim="x", status=EvidenceStatus.VERIFIED, source="t")], finding_text)
    return comparison_is_active(v.comparison)


class _FakeLLM:
    def __init__(self, payload):
        self._p = payload

    async def chat_completion(self, messages, **kw):
        return json.dumps(self._p)


def _price(components, finding="A required control needs remediation.", evidence=("a quote",)):
    interp = {"strategy": {"remediation_summary": "x"},
              "activities": [{"activity_id": "A0", "description": "corrective work", "derived_from": "FINDING"}],
              "cost_components": components, "overall_status": "EVIDENCE_BACKED"}
    return asyncio.run(estimate_remediation_cost(
        finding_text=finding,
        evidence_ledger=[EvidenceItem(claim=c, status=EvidenceStatus.REPORTED, source="t") for c in evidence],
        client=_FakeLLM(interp)))


def _comp(cid, desc, **over):
    c = {"component_id": cid, "description": desc, "activity_ids": ["A0"],
         "cost_category": "x", "value_kind": "REMEDIATION_COST", "amount_type": "COMPONENT",
         "recurrence": "ONE_TIME", "unit_cost_basis": "REPORTED", "currency": "INR",
         "source_reference_ids": ["E0"]}
    c.update(over)
    return c


# --------------------------------------------------------------------------- #
# PART A -- comparison vs composition (10 finding-agnostic cases)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name, cmp, expected", [
    ("actual vs approved budget", dict(left="actual cost Rs 220,000", right="approved budget Rs 150,000",
        reference="approved budget", status="ACTUAL_CONFLICT", why_comparable="actual against approved budget",
        comparison_basis="reconciliation cost", direction="ABOVE", magnitude=70000), True),
    ("actual vs target", dict(left="actual output 850 units", right="target 1000 units", reference="target",
        status="ACTUAL_CONFLICT", why_comparable="actual vs target", comparison_basis="output", direction="BELOW"), True),
    ("measured vs limit", dict(left="measured 92 C", right="limit 80 C", reference="limit",
        status="ACTUAL_CONFLICT", why_comparable="measured against limit", comparison_basis="temperature",
        direction="ABOVE"), True),
    ("before vs after", dict(left="before correction 12%", right="after correction 3%",
        status="ACTUAL_CONFLICT", why_comparable="before vs after", comparison_basis="defect rate",
        direction="BELOW"), True),
    ("two independent estimates", dict(left="engineering estimate Rs 3 lakh", right="supplier quotation Rs 4.2 lakh",
        reference="engineering estimate", status="UNRESOLVED_COMPARISON", why_comparable="two independent estimates",
        comparison_basis="replacement cost", direction="MISMATCH"), True),
    ("independent cost components", dict(left="replacement Rs 145,000", right="installation Rs 12,000",
        status="ACTUAL_CONFLICT", why_comparable="listed separately", comparison_basis="cost components",
        direction="MISMATCH"), False),
    ("labour rate vs material price", dict(left="material Rs 350 per panel", right="labour Rs 900 per hour",
        status="ACTUAL_CONFLICT", why_comparable="different rates", comparison_basis="pricing basis",
        direction="MISMATCH"), False),
    ("pricing basis emitted as comparison_basis", dict(left="labour cost per cabinet", right="Rs 30",
        reference="Rs 900 per hour", status="ACTUAL_CONFLICT", why_comparable="cost per cabinet vs hourly rate",
        comparison_basis="labour cost per cabinet", direction="MISMATCH"), False),
    ("bare MISMATCH no baseline", dict(left="Rs 6,000", right="Rs 350", status="ACTUAL_CONFLICT",
        why_comparable="two figures", comparison_basis="cost", direction="MISMATCH"), False),
    ("comparison with magnitude and real reference", dict(left="actual 4%", right="specification 1%",
        reference="specification", status="ACTUAL_CONFLICT", why_comparable="actual vs spec",
        comparison_basis="defect rate", direction="ABOVE", magnitude=3), True),
])
def test_comparison_coherence(name, cmp, expected):
    assert _cmp_active(cmp) is expected, name


def test_malformed_comparison_object_is_safe():
    for bad in (dict(status="ACTUAL_CONFLICT"), dict(left="", right="", status="ACTUAL_CONFLICT"),
                dict(left=None, right=None, status="UNRESOLVED_COMPARISON", comparison_basis=None)):
        assert _cmp_active(bad) is False


# --------------------------------------------------------------------------- #
# PART B -- folded recurring horizon
# --------------------------------------------------------------------------- #

def test_folded_horizon_occurrence_count_is_de_escalated():
    r = _price([_comp("C0", "monitoring round", quantity=6, quantity_unit="round",
                      quantity_basis="NOT_ESTABLISHED", unit_cost=2000, unit_cost_basis="VERIFIED",
                      amount_type="PER_UNIT", recurrence="RECURRING", recurring_period="month")],
               finding="A monitoring round must run monthly for six months. Each round Rs 2,000.")
    assert r.pricing_status == "NOT_ASSESSABLE"
    assert r.recurring_cost is None
    assert r.review_required is True


def test_recurring_with_resource_quantity_is_preserved():
    r = _price([_comp("C0", "monitoring labour", quantity=2, quantity_unit="hour",
                      quantity_basis="EVIDENCED", unit_cost=900, unit_cost_basis="VERIFIED",
                      amount_type="PER_HOUR", recurrence="RECURRING", recurring_period="month")],
               finding="Monthly monitoring needs 2 hours at Rs 900/hour.")
    assert r.recurring_cost == 1800.0
    assert r.pricing_status == "EXACT_ESTIMATE"


def test_flat_recurring_no_quantity_is_preserved():
    r = _price([_comp("C0", "inspection", unit_cost=3500, unit_cost_basis="VERIFIED",
                      recurrence="RECURRING", recurring_period="month")],
               finding="Recurring inspection every month, Rs 3,500 each. No end date.")
    assert r.recurring_cost == 3500.0
    assert r.pricing_status == "EXACT_ESTIMATE"


# --------------------------------------------------------------------------- #
# PART C -- multi-currency composition
# --------------------------------------------------------------------------- #

def test_multi_currency_components_are_not_summed():
    r = _price([_comp("C0", "customs consultancy", unit_cost=4000, currency="EUR"),
                _comp("C1", "warehouse re-racking", unit_cost=90000, currency="INR")],
               finding="EUR 4,000 consultancy and Rs 90,000 re-racking. No conversion rate.",
               evidence=("EUR 4,000 consultancy", "Rs 90,000 re-racking"))
    assert r.one_time_cost is None
    assert r.most_likely_estimate is None
    assert r.pricing_status in ("NOT_ASSESSABLE", "PARTIAL_ESTIMATE")
    assert r.review_required is True
    assert any("currenc" in u.lower() for u in r.uncertainty_reasons)


def test_single_currency_composition_still_sums():
    r = _price([_comp("C0", "part", unit_cost=145000, currency="INR"),
                _comp("C1", "install", unit_cost=12000, currency="INR")],
               finding="Part Rs 145,000 and installation Rs 12,000.",
               evidence=("Rs 145,000 part", "Rs 12,000 install"))
    assert r.one_time_cost == 157000.0


def test_missing_currency_adopts_single_working_currency():
    r = _price([_comp("C0", "part", unit_cost=40000, currency="INR"),
                _comp("C1", "service", unit_cost=10000, currency=None)],
               finding="Rs 40,000 part and a Rs 10,000 service fee.",
               evidence=("Rs 40,000 part", "Rs 10,000 fee"))
    assert r.one_time_cost == 50000.0
    assert r.currency == "INR"


# --------------------------------------------------------------------------- #
# PART E -- raw finding text has NO authority over these decisions
# --------------------------------------------------------------------------- #

def test_comparison_decision_independent_of_finding_text():
    cmp = dict(left="labour cost per cabinet", right="Rs 30", reference="Rs 900 per hour",
               status="ACTUAL_CONFLICT", why_comparable="rate vs cost", comparison_basis="labour cost per cabinet",
               direction="MISMATCH")
    for ft in ("", "actual against the approved budget baseline specification limit target",
               "twelve cabinets require earthing labels", "compared versus exceeds threshold"):
        assert _cmp_active(cmp, ft) is False, ft


def test_recurrence_decision_independent_of_finding_text():
    comp = lambda: _comp("C0", "round", quantity=6, quantity_unit="round", quantity_basis="NOT_ESTABLISHED",
                         unit_cost=2000, unit_cost_basis="VERIFIED", amount_type="PER_UNIT",
                         recurrence="RECURRING", recurring_period="month")
    for ft in ("", "monthly for six months, Rs 2,000 each",
               "one-time single round costing Rs 12,000 total forever"):
        r = _price([comp()], finding=ft)
        assert r.pricing_status == "NOT_ASSESSABLE", ft


def test_currency_decision_independent_of_finding_text():
    for ft in ("", "all in rupees", "EUR 4000 plus INR 90000 total is EUR 94000"):
        r = _price([_comp("C0", "a", unit_cost=4000, currency="EUR"),
                    _comp("C1", "b", unit_cost=90000, currency="INR")], finding=ft,
                   evidence=("a", "b"))
        assert r.one_time_cost is None, ft


# --------------------------------------------------------------------------- #
# PART F -- structured-output mutation fuzzing
# --------------------------------------------------------------------------- #

import math
import random

_JUNK = [None, "", "abc", math.nan, math.inf, -math.inf, -1, 0, "1,000", [], {}, True, 1e309]


import pydantic

_STR_JUNK = [None, "", "  ", "abc", "Rs 5,000", "approved budget", "per hour", "MISMATCH",
             "several numbers", "\x00", "A" * 400]


@pytest.mark.parametrize("seed", range(80))
def test_comparison_mutation_fuzz_is_safe(seed):
    rng = random.Random(seed)
    str_keys = ["left", "right", "reference", "comparison_basis", "why_comparable", "unit"]
    cmp = {"status": rng.choice(["ACTUAL_CONFLICT", "UNRESOLVED_COMPARISON", "NOT_ESTABLISHED"])}
    for k in rng.sample(str_keys, rng.randint(0, len(str_keys))):
        cmp[k] = rng.choice(_STR_JUNK)
    if rng.random() < 0.6:
        cmp["direction"] = rng.choice(["ABOVE", "BELOW", "MISMATCH", "UNKNOWN"])
    if rng.random() < 0.4:
        cmp["magnitude"] = rng.choice([None, 0, -5, 1e12, 3.5])
    try:
        active = _cmp_active(cmp)
    except pydantic.ValidationError:
        return  # schema rejected the object -- that is a safe outcome
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"comparison validation raised: {type(exc).__name__}: {exc}")
    assert active in (True, False)
    # a comparison with no baseline/reference relationship is never active
    blob = " ".join(str(v).lower() for v in cmp.values())
    if not any(w in blob for w in ("budget", "approved", "limit", "target", "spec", "before ", "after ",
                                   "estimate", "prior", "previous", "threshold", "standard", "actual ",
                                   "quotation", "quoted", "requirement", "required", "benchmark",
                                   "allowable", "tolerance", "nominal", "planned", "forecast", "baseline")):
        assert active is False


@pytest.mark.parametrize("seed", range(60))
def test_recurrence_mutation_fuzz_is_safe(seed):
    rng = random.Random(seed)
    over = {}
    for k, opts in {
        "recurrence": ["ONE_TIME", "RECURRING", "UNKNOWN", "banana", None],
        "recurring_period": ["month", "week", "year", "round", "", None, 7],
        "quantity": _JUNK + [1, 2, 6, 12],
        "quantity_unit": ["round", "hour", "month", "occurrence", "unit", "person", "", None],
        "quantity_basis": ["EVIDENCED", "DERIVED", "ASSUMED", "NOT_ESTABLISHED", "banana", None],
        "unit_cost": _JUNK + [2000, 3500],
        "unit_cost_basis": ["VERIFIED", "REPORTED", "ASSUMED", "NOT_ESTABLISHED", None],
        "amount_type": ["COMPONENT", "PER_UNIT", "PER_HOUR", "TOTAL", "banana", None],
    }.items():
        if rng.random() < 0.75:
            over[k] = rng.choice(opts)
    try:
        r = _price([_comp("C0", "activity", **over)])
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"remediation pricing raised: {type(exc).__name__}: {exc}")
    for v in (r.one_time_cost, r.recurring_cost, r.recurring_horizon_total, r.most_likely_estimate):
        assert v is None or (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0)
    assert r.review_required is True
    assert isinstance(r.ai_provenance, dict) and r.ai_provenance
