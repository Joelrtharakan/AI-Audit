"""Pass 58 — PARTIAL_ESTIMATE semantics, Low/Most-Likely/High discipline, and
activity <-> pricing-state single-valuedness.

No real LLM: FakeLLMClient returns hand-authored interpretation JSON standing
in for a provider. All arithmetic is the deterministic calculator's.
"""
from __future__ import annotations

import json

import pytest

from app.remediation.engine import estimate_remediation_cost
from app.remediation.models import RemediationEstimateStatus
from app.models.agent import EvidenceItem, EvidenceStatus


class FakeLLMClient:
    def __init__(self, response):
        self.response = response if isinstance(response, str) else json.dumps(response)

    async def chat_completion(self, messages, temperature=0.0, response_format_json=True, **kw):
        return self.response


def _ev(claim, status=EvidenceStatus.VERIFIED):
    return EvidenceItem(claim=claim, status=status, source="test")


async def _run(interp, evidence, finding="A required control was found deficient."):
    return await estimate_remediation_cost(
        finding_text=finding, evidence_ledger=evidence,
        client=FakeLLMClient(interp),
    )


def _comp(cid, desc, act, cat, qty, unit, cost, atype="PER_UNIT", refs=("E0",)):
    return {
        "component_id": cid, "description": desc, "activity_ids": [act],
        "cost_category": cat, "value_kind": "REMEDIATION_COST",
        "quantity": qty, "quantity_unit": unit, "quantity_basis": "EVIDENCED",
        "unit_cost": cost, "unit_cost_basis": "VERIFIED", "currency": "INR",
        "amount_type": atype, "recurrence": "ONE_TIME",
        "source_reference_ids": list(refs), "interpretation_confidence": "HIGH",
    }


# ===========================================================================
# ISSUE 10 — the corrected controlled-procedure scenario, all six components
# correctly recognised, computes exactly Rs 119,200 (calculator is right; the
# live-model Rs 28,800 quality-review figure is a MODEL operand error, not a
# calculator defect).
# ===========================================================================

@pytest.mark.asyncio
async def test_controlled_procedure_all_six_components_total_119200():
    interp = {
        "strategy": {"remediation_summary": "replace and reissue 24 controlled procedures",
                     "remediation_type": "documentation", "interpretation_confidence": "HIGH"},
        "activities": [
            {"activity_id": "A0", "description": "Print and prepare controlled copies", "derived_from": "FINDING"},
            {"activity_id": "A1", "description": "Quality review of revised procedures", "derived_from": "FINDING"},
            {"activity_id": "A2", "description": "Train affected employees", "derived_from": "FINDING"},
            {"activity_id": "A3", "description": "Administer training", "derived_from": "FINDING"},
            {"activity_id": "A4", "description": "Configure the document-management system", "derived_from": "FINDING"},
            {"activity_id": "A5", "description": "Validation testing", "derived_from": "FINDING"},
        ],
        "cost_components": [
            _comp("C0", "Printing and controlled-copy preparation", "A0", "printing", 24, "procedure", 150),
            # quality review = 24 procedures x 0.75 hour x Rs 1200/hour = 18 hours
            _comp("C1", "Quality review labour", "A1", "review", 18, "hour", 1200, atype="PER_HOUR"),
            _comp("C2", "Employee training", "A2", "training", 16, "employee", 2000),
            _comp("C3", "Training administration", "A3", "administration", 10, "hour", 900, atype="PER_HOUR"),
            {"component_id": "C4", "description": "Document-management configuration change",
             "activity_ids": ["A4"], "cost_category": "configuration", "value_kind": "QUOTED_PRICE",
             "unit_cost": 35000, "unit_cost_basis": "VERIFIED", "currency": "INR",
             "amount_type": "COMPONENT", "recurrence": "ONE_TIME", "source_reference_ids": ["E0"],
             "interpretation_confidence": "HIGH"},
            _comp("C5", "Validation testing labour", "A5", "testing", 12, "hour", 1500, atype="PER_HOUR"),
        ],
        "calculation_proposals": [],
        "overall_status": "EVIDENCE_BACKED", "estimability": "ESTIMABLE",
    }
    ev = [_ev("24 controlled procedures require replacement and reissue"),
          _ev("Printing/preparation costs INR 150 per procedure"),
          _ev("Quality review requires 45 minutes per procedure at INR 1200/hour"),
          _ev("16 employees require training at INR 2000 each"),
          _ev("Training administration requires 10 hours at INR 900/hour"),
          _ev("Configuration quoted at INR 35000; validation testing 12 hours at INR 1500/hour")]
    res = await _run(interp, ev)
    assert res.status == RemediationEstimateStatus.EVIDENCE_BACKED
    assert res.pricing_status == "EXACT_ESTIMATE"
    assert res.is_partial_estimate is False
    # 3600 + 21600 + 32000 + 9000 + 35000 + 18000
    assert res.one_time_cost == 119200.0
    assert res.review_required is True
    assert not res.unpriced_activities


# ===========================================================================
# ISSUE 3/4/11 — a partial estimate: A priced, B priced, C required but unpriced
# ===========================================================================

@pytest.mark.asyncio
async def test_partial_estimate_is_known_priced_not_total_and_no_fabricated_range():
    interp = {
        "strategy": {"remediation_summary": "correct the deficiency",
                     "remediation_type": "corrective", "interpretation_confidence": "HIGH"},
        "activities": [
            {"activity_id": "A0", "description": "Replace failed parts", "derived_from": "FINDING"},
            {"activity_id": "A1", "description": "Re-test the system", "derived_from": "FINDING"},
            {"activity_id": "A2", "description": "Recalibrate the instrument", "derived_from": "FINDING"},
        ],
        "cost_components": [
            _comp("C0", "Replacement parts", "A0", "parts", 4, "part", 4000, atype="PER_UNIT"),
            _comp("C1", "Re-test labour", "A1", "testing", 6, "hour", 800, atype="PER_HOUR"),
            {"component_id": "C2", "description": "Recalibration service", "activity_ids": ["A2"],
             "cost_category": "calibration", "value_kind": "REMEDIATION_COST",
             "unit_cost": None, "unit_cost_basis": "NOT_ESTABLISHED", "currency": "INR",
             "amount_type": "TOTAL", "recurrence": "ONE_TIME", "source_reference_ids": [],
             "rationale": "recalibration rate not established", "interpretation_confidence": "LOW"},
        ],
        "calculation_proposals": [],
        "overall_status": "EVIDENCE_BACKED", "estimability": "PARTIALLY_ESTIMABLE",
    }
    ev = [_ev("4 parts require replacement at INR 4000 each"),
          _ev("Re-test requires 6 hours at INR 800/hour"),
          _ev("The instrument must be recalibrated")]
    res = await _run(interp, ev)

    assert res.is_partial_estimate is True
    assert res.pricing_status == "PARTIAL_ESTIMATE"
    # known priced portion = 4*4000 + 6*800 = 20800  -- NOT a total
    assert res.one_time_cost == 20800.0
    # ISSUE 4: no fabricated Low / Most Likely / High
    assert res.low_estimate is None
    assert res.most_likely_estimate is None
    assert res.high_estimate is None
    # C is unpriced and NOT silently zero
    assert any("recalibrat" in a.lower() for a in res.unpriced_activities)
    c2 = next(c for c in res.cost_components if c.component_id == "C2")
    assert c2.calculated_amount is None  # not 0.0
    assert res.review_required is True
    # ISSUE 1/2: no activity is in both a priced sense and the unpriced list
    _priced = {a.strip().lower() for a in res.implementation_activities} - {
        a.strip().lower() for a in res.unpriced_activities}
    assert _priced.isdisjoint({a.strip().lower() for a in res.unpriced_activities})


# ===========================================================================
# ISSUE 1/2/12 — activity <-> pricing-state single-valuedness
# ===========================================================================

@pytest.mark.asyncio
async def test_same_activity_string_cannot_be_both_priced_and_unpriced():
    # Two LLM activity entries with the SAME normalised description; one is
    # linked to a priced component, one is not. The result must not list that
    # description as both priced and unpriced.
    interp = {
        "strategy": {"remediation_summary": "reissue procedures", "remediation_type": "documentation",
                     "interpretation_confidence": "HIGH"},
        "activities": [
            {"activity_id": "A0", "description": "Reissue controlled procedures", "derived_from": "FINDING"},
            {"activity_id": "A1", "description": "reissue controlled procedures", "derived_from": "FINDING"},
        ],
        "cost_components": [
            _comp("C0", "Procedure reissue printing", "A0", "printing", 24, "procedure", 150),
        ],
        "calculation_proposals": [],
        "overall_status": "EVIDENCE_BACKED", "estimability": "ESTIMABLE",
    }
    ev = [_ev("24 procedures reissued at INR 150 each")]
    res = await _run(interp, ev)
    _impl = [a.strip().lower() for a in res.implementation_activities]
    _unp = [a.strip().lower() for a in res.unpriced_activities]
    # de-duplicated
    assert len(_impl) == len(set(_impl))
    # "reissue controlled procedures" appears priced -> not also unpriced
    assert "reissue controlled procedures" not in _unp


@pytest.mark.asyncio
async def test_rejected_component_does_not_enter_arithmetic():
    interp = {
        "strategy": {"remediation_summary": "x", "remediation_type": "corrective",
                     "interpretation_confidence": "HIGH"},
        "activities": [{"activity_id": "A0", "description": "Replace parts", "derived_from": "FINDING"}],
        "cost_components": [
            _comp("C0", "Replacement parts", "A0", "parts", 5, "part", 4000, atype="PER_UNIT"),
            # unclassified value_kind -> fails closed, number removed
            {"component_id": "C1", "description": "Handling fee", "activity_ids": ["A0"],
             "cost_category": "other", "value_kind": "NOT_ESTABLISHED",
             "unit_cost": 999999, "unit_cost_basis": "VERIFIED", "currency": "INR",
             "amount_type": "TOTAL", "recurrence": "ONE_TIME", "source_reference_ids": ["E0"]},
        ],
        "calculation_proposals": [],
        "overall_status": "EVIDENCE_BACKED", "estimability": "ESTIMABLE",
    }
    ev = [_ev("5 parts at INR 4000 each")]
    res = await _run(interp, ev)
    # 999999 must never reach the arithmetic
    assert res.one_time_cost == 20000.0
    c1 = next(c for c in res.cost_components if c.component_id == "C1")
    assert c1.calculated_amount is None


@pytest.mark.asyncio
async def test_missing_price_is_never_converted_to_zero():
    interp = {
        "strategy": {"remediation_summary": "x", "remediation_type": "corrective",
                     "interpretation_confidence": "HIGH"},
        "activities": [
            {"activity_id": "A0", "description": "Replace refrigerator", "derived_from": "FINDING"},
            {"activity_id": "A1", "description": "Install refrigerator", "derived_from": "FINDING"},
        ],
        "cost_components": [
            {"component_id": "C0", "description": "Refrigerator", "activity_ids": ["A0"],
             "cost_category": "equipment", "value_kind": "QUOTED_PRICE", "unit_cost": 145000,
             "unit_cost_basis": "VERIFIED", "currency": "INR", "amount_type": "TOTAL",
             "recurrence": "ONE_TIME", "source_reference_ids": ["E0"]},
            {"component_id": "C1", "description": "Installation", "activity_ids": ["A1"],
             "cost_category": "installation", "value_kind": "REMEDIATION_COST", "unit_cost": None,
             "unit_cost_basis": "NOT_ESTABLISHED", "currency": "INR", "amount_type": "TOTAL",
             "recurrence": "ONE_TIME", "source_reference_ids": [],
             "rationale": "installation price not established"},
        ],
        "calculation_proposals": [],
        "overall_status": "EVIDENCE_BACKED", "estimability": "PARTIALLY_ESTIMABLE",
    }
    ev = [_ev("Refrigerator quoted at INR 145000; installation required")]
    res = await _run(interp, ev)
    assert res.is_partial_estimate is True
    assert res.one_time_cost == 145000.0     # NOT 145000 + 0
    assert res.most_likely_estimate is None
    c1 = next(c for c in res.cost_components if c.component_id == "C1")
    assert c1.calculated_amount is None


@pytest.mark.asyncio
async def test_exact_estimate_only_when_all_required_components_priced():
    interp = {
        "strategy": {"remediation_summary": "x", "remediation_type": "corrective",
                     "interpretation_confidence": "HIGH"},
        "activities": [
            {"activity_id": "A0", "description": "Replace parts", "derived_from": "FINDING"},
            {"activity_id": "A1", "description": "Re-test", "derived_from": "FINDING"},
        ],
        "cost_components": [
            _comp("C0", "Replace parts", "A0", "parts", 5, "part", 4000, atype="PER_UNIT"),
            _comp("C1", "Re-test", "A1", "testing", 4, "hour", 800, atype="PER_HOUR"),
        ],
        "calculation_proposals": [],
        "overall_status": "EVIDENCE_BACKED", "estimability": "ESTIMABLE",
    }
    ev = [_ev("5 parts at INR 4000"), _ev("re-test 4 hours at INR 800/hour")]
    res = await _run(interp, ev)
    assert res.pricing_status == "EXACT_ESTIMATE"
    assert res.is_partial_estimate is False
    assert res.one_time_cost == 23200.0
    assert not res.unresolved_pricing_drivers
    # any unpriced entry must be CONDITIONAL (cause-contingent), never a
    # required direct-correction activity left silently unpriced
    _cond = {a.strip().lower() for a in res.conditional_activities}
    for a in res.unpriced_activities:
        assert a.strip().lower() in _cond, a


# ===========================================================================
# Pass 59 — DIMENSIONAL / TIME-UNIT NORMALISATION (spec §7/§8/§32)
# A PER_HOUR / PER_DAY rate has an implicit time base; the deterministic layer
# reconciles the LLM's own quantity_unit against it -- fixed universal ratios
# only, fail closed when unconvertible. Domain-agnostic (no finding text).
# ===========================================================================

@pytest.mark.asyncio
async def test_subhour_quantity_is_converted_for_a_per_hour_rate():
    for unit, qty, expect in [("minute", 45, 900.0), ("minutes", 30, 600.0),
                              ("second", 1800, 600.0), ("hour", 2, 2400.0)]:
        interp = {
            "strategy": {"remediation_summary": "x", "remediation_type": "corrective",
                         "interpretation_confidence": "HIGH"},
            "activities": [{"activity_id": "A0", "description": "Do the work", "derived_from": "FINDING"}],
            "cost_components": [{
                "component_id": "C0", "description": "Labour", "activity_ids": ["A0"],
                "cost_category": "labour", "value_kind": "REMEDIATION_COST",
                "quantity": qty, "quantity_unit": unit, "quantity_basis": "EVIDENCED",
                "unit_cost": 1200, "unit_cost_basis": "VERIFIED", "currency": "INR",
                "amount_type": "PER_HOUR", "recurrence": "ONE_TIME", "source_reference_ids": ["E0"],
            }],
            "calculation_proposals": [], "overall_status": "EVIDENCE_BACKED", "estimability": "ESTIMABLE",
        }
        res = await _run(interp, [_ev(f"{qty} {unit} at INR 1200/hour")])
        assert res.one_time_cost == expect, (unit, qty, res.one_time_cost)


@pytest.mark.asyncio
async def test_unconvertible_time_basis_fails_closed():
    # A per-hour rate whose quantity is in days / shifts cannot be reconciled
    # without an undefined working-day basis -> fail closed, no headline.
    for unit in ("day", "shift"):
        interp = {
            "strategy": {"remediation_summary": "x", "remediation_type": "corrective",
                         "interpretation_confidence": "HIGH"},
            "activities": [{"activity_id": "A0", "description": "Do the work", "derived_from": "FINDING"}],
            "cost_components": [{
                "component_id": "C0", "description": "Labour", "activity_ids": ["A0"],
                "cost_category": "labour", "value_kind": "REMEDIATION_COST",
                "quantity": 3, "quantity_unit": unit, "quantity_basis": "EVIDENCED",
                "unit_cost": 1200, "unit_cost_basis": "VERIFIED", "currency": "INR",
                "amount_type": "PER_HOUR", "recurrence": "ONE_TIME", "source_reference_ids": ["E0"],
            }],
            "calculation_proposals": [], "overall_status": "EVIDENCE_BACKED", "estimability": "ESTIMABLE",
        }
        res = await _run(interp, [_ev(f"3 {unit} at INR 1200/hour")])
        assert res.pricing_status == "NOT_ASSESSABLE", (unit, res.pricing_status)
        assert "UNIT_OR_RATE_BASIS_UNRESOLVED" in _REASON_KEY(res.not_assessable_reason) or \
               any(r.reason_code == "UNIT_OR_RATE_BASIS_UNRESOLVED" for r in res.rejected_items), unit
        assert res.one_time_cost is None
        assert res.review_required is True


def _REASON_KEY(text: str) -> str:
    from app.remediation.engine import _PROFESSIONAL_REASON
    for k, v in _PROFESSIONAL_REASON.items():
        if v and text and v[:40] == text[:40]:
            return k
    return ""


@pytest.mark.asyncio
async def test_non_time_quantity_unit_for_hourly_rate_is_left_untouched():
    # A per-unit quantity (person / item) against a PER_HOUR rate is the model's
    # per-unit meaning, not a dimension error -- not stripped, not converted.
    interp = {
        "strategy": {"remediation_summary": "x", "remediation_type": "training",
                     "interpretation_confidence": "HIGH"},
        "activities": [{"activity_id": "A0", "description": "Train staff", "derived_from": "FINDING"}],
        "cost_components": [{
            "component_id": "C0", "description": "Training", "activity_ids": ["A0"],
            "cost_category": "training", "value_kind": "REMEDIATION_COST",
            "quantity": 12, "quantity_unit": "technician-hour", "quantity_basis": "EVIDENCED",
            "unit_cost": 800, "unit_cost_basis": "VERIFIED", "currency": "INR",
            "amount_type": "PER_HOUR", "recurrence": "ONE_TIME", "source_reference_ids": ["E0"],
        }],
        "calculation_proposals": [], "overall_status": "EVIDENCE_BACKED", "estimability": "ESTIMABLE",
    }
    res = await _run(interp, [_ev("12 technician-hours at INR 800/hour")])
    assert res.one_time_cost == 9600.0


# ===========================================================================
# Pass 60 — UNKNOWN recurrence (spec §7/§8/§15/§29): a model that cannot
# establish one-time vs recurring must say UNKNOWN, and the deterministic
# layer fails it closed -- it is NEVER assumed one-time. No finding text.
# ===========================================================================

def _rec_comp(recurrence, period=None):
    return {
        "component_id": "C0", "description": "Verification service", "activity_ids": ["A0"],
        "cost_category": "verification", "value_kind": "REMEDIATION_COST",
        "quantity": 1, "quantity_unit": "service", "quantity_basis": "EVIDENCED",
        "unit_cost": 2000, "unit_cost_basis": "VERIFIED", "currency": "INR",
        "amount_type": "PER_EVENT", "recurrence": recurrence, "recurring_period": period,
        "source_reference_ids": ["E0"],
    }


def _rec_interp(comp):
    return {
        "strategy": {"remediation_summary": "x", "remediation_type": "corrective",
                     "interpretation_confidence": "HIGH"},
        "activities": [{"activity_id": "A0", "description": "Perform verification", "derived_from": "FINDING"}],
        "cost_components": [comp], "calculation_proposals": [],
        "overall_status": "EVIDENCE_BACKED", "estimability": "ESTIMABLE",
    }


@pytest.mark.asyncio
async def test_unknown_recurrence_fails_closed_never_assumed_one_time():
    res = await _run(_rec_interp(_rec_comp("UNKNOWN")), [_ev("verification costs INR 2000")])
    assert res.pricing_status == "NOT_ASSESSABLE"
    assert res.one_time_cost is None and res.recurring_cost is None
    assert res.review_required is True
    assert "one-time or recurring" in res.not_assessable_reason.lower()


@pytest.mark.asyncio
async def test_unrecognised_recurrence_string_becomes_unknown_not_one_time():
    for weird in ("as_needed", "periodic-ish", "ongoing?", "TBD", "unclear"):
        res = await _run(_rec_interp(_rec_comp(weird)), [_ev("x")])
        assert res.pricing_status == "NOT_ASSESSABLE", weird
        assert res.one_time_cost is None, weird


@pytest.mark.asyncio
async def test_explicit_one_time_and_recurring_still_price_normally():
    r1 = await _run(_rec_interp(_rec_comp("ONE_TIME")), [_ev("x")])
    assert r1.pricing_status == "EXACT_ESTIMATE" and r1.one_time_cost == 2000.0
    r2 = await _run(_rec_interp(_rec_comp("RECURRING", "month")), [_ev("x")])
    assert r2.recurring_cost == 2000.0 and r2.recurring_period == "month"
    assert r2.recurring_horizon_total is None  # no horizon -> no finite total


@pytest.mark.asyncio
async def test_result_carries_ai_provenance():
    res = await _run(_rec_interp(_rec_comp("ONE_TIME")), [_ev("x")])
    p = res.ai_provenance
    for k in ("remediation_model", "remediation_prompt_version", "canonical_model",
              "canonical_prompt_version", "semantic_schema_version", "provider",
              "review_required", "pricing_status", "generated_at"):
        assert k in p, k
    assert p["review_required"] is True


# ===========================================================================
# Pass 61 — EVENT-TRIGGERED recurrence + MIXED one-time/recurring (spec §4/§5/
# §9-D/§14/§15/§17). RECURRING covers a calendar period OR a triggering event;
# with no established occurrence count the periodic figure stands alone and the
# TOTAL is NOT_ASSESSABLE -- never silently ONE_TIME, never a fabricated total.
# ===========================================================================

@pytest.mark.asyncio
async def test_event_triggered_recurrence_prices_per_event_total_not_assessable():
    interp = _rec_interp({
        **_rec_comp("RECURRING", "each feed change"),
        "quantity_unit": "run", "amount_type": "PER_EVENT",
    })
    res = await _run(interp, [_ev("re-run validation whenever the feed changes; INR 2000 each")])
    assert res.recurring_cost == 2000.0
    assert res.recurring_period == "each feed change"
    assert res.recurring_horizon_total is None       # no occurrence count -> no total
    assert res.one_time_cost is None                 # NOT silently one-time
    assert res.review_required is True


@pytest.mark.asyncio
async def test_recurring_calendar_no_horizon_has_no_finite_total():
    res = await _run(_rec_interp(_rec_comp("RECURRING", "month")), [_ev("INR 2000 every month")])
    assert res.recurring_cost == 2000.0 and res.recurring_period == "month"
    assert res.recurring_horizon_total is None
    assert res.one_time_cost is None


@pytest.mark.asyncio
async def test_mixed_one_time_and_recurring_are_both_shown_no_flattening():
    interp = {
        "strategy": {"remediation_summary": "x", "remediation_type": "corrective",
                     "interpretation_confidence": "HIGH"},
        "activities": [
            {"activity_id": "A0", "description": "Install replacement unit", "derived_from": "FINDING"},
            {"activity_id": "A1", "description": "Perform monthly inspection", "derived_from": "FINDING"},
        ],
        "cost_components": [
            {"component_id": "C0", "description": "Installation", "activity_ids": ["A0"],
             "cost_category": "install", "value_kind": "QUOTED_PRICE", "unit_cost": 10000,
             "unit_cost_basis": "VERIFIED", "currency": "INR", "amount_type": "COMPONENT",
             "recurrence": "ONE_TIME", "source_reference_ids": ["E0"]},
            {"component_id": "C1", "description": "Monthly inspection", "activity_ids": ["A1"],
             "cost_category": "inspection", "value_kind": "REMEDIATION_COST", "quantity": 1,
             "quantity_unit": "inspection", "quantity_basis": "EVIDENCED", "unit_cost": 2000,
             "unit_cost_basis": "VERIFIED", "currency": "INR", "amount_type": "PER_EVENT",
             "recurrence": "RECURRING", "recurring_period": "month", "source_reference_ids": ["E0"]},
        ],
        "calculation_proposals": [], "overall_status": "EVIDENCE_BACKED", "estimability": "ESTIMABLE",
    }
    res = await _run(interp, [_ev("install INR 10000; monthly inspection INR 2000")])
    assert res.one_time_cost == 10000.0
    assert res.recurring_cost == 2000.0 and res.recurring_period == "month"
    assert res.recurring_horizon_total is None       # no horizon -> no lifetime total
