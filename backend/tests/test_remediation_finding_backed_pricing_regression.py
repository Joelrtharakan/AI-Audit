"""Pass 51 regression: a remediation whose pricing facts are stated in the
FINDING TEXT (no separate evidence ledger) must be priced from the finding.

Before Pass 51 the remediation validator only credited a price cited against an
enumerated evidence item (E0, E1, ...). A finding that states its own rates
("labels cost Rs 350 per panel; electrician 1.5 h/panel at Rs 900/hour;
inspection Rs 6,000") therefore produced NOT_ASSESSABLE with the stated rates
requested back as auditor inputs. Pass 51 makes the finding a citable pricing
source ("FINDING"); a component the LLM anchors to the finding survives
validation and is priced.

These tests exercise the DETERMINISTIC pipeline (validate_and_plan ->
assemble_estimate -> estimate_remediation_cost) with a hand-built
interpretation that mirrors the real qwen3:8b structured output for the
Pass-50 "eight electrical panels" failure. No live model.
"""

from __future__ import annotations

import pytest

from app.remediation.engine import estimate_remediation_cost
from app.remediation.interpreter import normalize_to_canonical
from app.remediation.semantic_models import RemediationInterpretation
from app.remediation.validator import validate_and_plan
from app.remediation.calculator import assemble_estimate


PANELS_FINDING = (
    "Eight electrical panels require corrective labeling and inspection. New labels cost "
    "Rs 350 per panel. An electrician requires 1.5 hours per panel at Rs 900 per hour, "
    "followed by a safety inspection costing Rs 6,000 for the complete area."
)


def _panels_interp_dict(*, cite: str = "FINDING", basis: str = "VERIFIED") -> dict:
    """Mirrors the real qwen3:8b remediation output for the panels finding
    (3 evidence-backed components), parametrised on how the price is cited."""
    return {
        "strategy": {"remediation_summary": "Label and inspect the eight electrical panels.",
                     "interpretation_confidence": "HIGH"},
        "activities": [
            {"activity_id": "A0", "description": "Corrective labeling of the eight panels",
             "disposition": "IMMEDIATE_CORRECTION", "depends_on_root_cause": False},
            {"activity_id": "A1", "description": "Safety inspection of the complete area",
             "disposition": "IMMEDIATE_CORRECTION", "depends_on_root_cause": False},
        ],
        "cost_components": [
            {"component_id": "C0", "description": "Labeling materials", "activity_ids": ["A0"],
             "cost_category": "materials", "value_kind": "REMEDIATION_COST", "quantity": 8,
             "quantity_unit": "panel", "quantity_basis": "EVIDENCED", "unit_cost": 350,
             "unit_cost_basis": basis, "currency": "INR", "amount_type": "PER_UNIT",
             "source_reference_ids": [cite], "rationale": "8 panels x 350"},
            {"component_id": "C1", "description": "Electrician labour", "activity_ids": ["A0"],
             "cost_category": "labor", "value_kind": "REMEDIATION_COST", "quantity": 12,
             "quantity_unit": "hour", "quantity_basis": "DERIVED",
             "quantity_derivation": "8 panels x 1.5 h/panel = 12 h", "unit_cost": 900,
             "unit_cost_basis": basis, "currency": "INR", "amount_type": "PER_HOUR",
             "source_reference_ids": [cite], "rationale": "12 h x 900"},
            {"component_id": "C2", "description": "Safety inspection of the area",
             "activity_ids": ["A1"], "cost_category": "services", "value_kind": "REMEDIATION_COST",
             "quantity": 1, "quantity_unit": "inspection", "quantity_basis": "EVIDENCED",
             "unit_cost": 6000, "unit_cost_basis": basis, "currency": "INR",
             "amount_type": "PER_EVENT", "source_reference_ids": [cite],
             "rationale": "1 area inspection x 6,000"},
        ],
        "calculation_proposals": [
            {"calculation_id": "K0", "target_component_id": "C1", "operation": "MULTIPLY",
             "operands": [{"label": "panels", "value": 8, "evidence_refs": [cite]},
                          {"label": "hours per panel", "value": 1.5, "evidence_refs": [cite]}],
             "produces": "COMPONENT_AMOUNT", "frequency": "ONE_TIME",
             "result_represents": "electrician labour hours"},
            {"calculation_id": "K1", "operation": "SUM", "component_ids": ["C0", "C1", "C2"],
             "produces": "MOST_LIKELY", "frequency": "ONE_TIME",
             "result_represents": "total one-time remediation cost"},
        ],
        "estimability": "ESTIMABLE",
        "overall_status": "EVIDENCE_BACKED",
    }


def _interp(d: dict) -> RemediationInterpretation:
    return RemediationInterpretation.model_validate(normalize_to_canonical(d))


class _FakeClient:
    def __init__(self, response_json: str):
        self._r = response_json

    async def chat_completion(self, messages, **kw):
        return self._r


def _canonical_ctx():
    """Minimal canonical context: corrective obligation established, RCA not."""
    from app.services.canonical_semantic_models import CanonicalFindingContext

    return CanonicalFindingContext.model_validate(
        {
            "primary_deviation": "Eight electrical panels require corrective labeling and inspection.",
            "finding_subject": "eight electrical panels",
            "root_cause_status": "NOT_ESTABLISHED",
            "remediation_obligation": "ESTABLISHED_CORRECTIVE_OBLIGATION",
            "remediation_activities": [
                {"action_id": "labeling", "activity": "Corrective labeling of the eight panels",
                 "disposition": "IMMEDIATE_CORRECTION", "depends_on_root_cause": False},
                {"action_id": "inspection", "activity": "Safety inspection of the complete area",
                 "disposition": "IMMEDIATE_CORRECTION", "depends_on_root_cause": False},
            ],
        }
    )


# --------------------------------------------------------------------------
# 1. The deterministic pipeline prices a finding-cited interpretation.
# --------------------------------------------------------------------------
def test_finding_cited_components_are_priced_not_stripped():
    interp = _interp(_panels_interp_dict(cite="FINDING"))
    components, proposals, outcome = validate_and_plan(
        interp, valid_evidence_ids={"FINDING"}
    )
    assert len(components) == 3, outcome.rejected
    for c in components:
        assert c.unit_cost is not None, f"{c.component_id} unit_cost was stripped: {outcome.llm_disagreements}"
        assert c.unit_cost_basis in ("VERIFIED", "REPORTED"), c.unit_cost_basis

    est = assemble_estimate(components, proposals, outcome.traces)
    # 8*350 + 12*900 + 6000 = 2800 + 10800 + 6000 = 19600
    assert est.one_time_cost == pytest.approx(19600), est
    assert not est.recurring_cost


# --------------------------------------------------------------------------
# 2. Without the FINDING id being valid, the same interpretation is stripped
#    (proves the fix is what makes the difference, not something else).
# --------------------------------------------------------------------------
def test_without_finding_source_the_prices_are_still_stripped():
    interp = _interp(_panels_interp_dict(cite="FINDING"))
    # valid_evidence_ids WITHOUT "FINDING"
    components, proposals, outcome = validate_and_plan(interp, valid_evidence_ids=set())
    est = assemble_estimate(components, proposals, outcome.traces)
    assert not est.one_time_cost, "expected pre-fix behaviour when FINDING is not a valid source"


# --------------------------------------------------------------------------
# 3. End-to-end engine: empty evidence ledger, prices in finding text.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_estimate_remediation_cost_prices_from_finding_text():
    fake = _FakeClient(__import__("json").dumps(_panels_interp_dict(cite="FINDING")))
    rc = await estimate_remediation_cost(
        finding_text=PANELS_FINDING,
        evidence_ledger=[],
        semantic_context=_canonical_ctx(),
        client=fake,
    )
    assert str(rc.status) != "RemediationEstimateStatus.NOT_ASSESSABLE", rc.not_assessable_reason
    assert rc.pricing_status in ("EXACT_ESTIMATE", "PARTIAL_ESTIMATE"), rc.pricing_status
    headline = rc.one_time_cost or rc.most_likely_estimate
    assert headline == pytest.approx(19600), (headline, rc.cost_components)
    assert not rc.recurring_cost
    # the stated rates must NOT be requested back
    joined = " ".join(rc.auditor_inputs_required or []).lower()
    for token in ("350", "900", "6,000", "6000", "label cost", "inspection cost"):
        assert token not in joined, f"auditor input requests already-present pricing: {rc.auditor_inputs_required}"


# --------------------------------------------------------------------------
# 4. Safety invariant preserved: an incurred loss cited against FINDING is
#    still rejected as not-remediation (value_kind gate unchanged).
# --------------------------------------------------------------------------
def test_finding_source_does_not_bypass_observed_loss_rejection():
    d = _panels_interp_dict(cite="FINDING")
    d["cost_components"] = [
        {"component_id": "C0", "description": "incurred incident cost", "activity_ids": ["A0"],
         "cost_category": "loss", "value_kind": "OBSERVED_FINANCIAL_LOSS", "quantity": 1,
         "quantity_unit": "event", "quantity_basis": "EVIDENCED", "unit_cost": 20000,
         "unit_cost_basis": "VERIFIED", "currency": "INR", "amount_type": "COMPONENT",
         "source_reference_ids": ["FINDING"]},
    ]
    interp = _interp(d)
    components, proposals, outcome = validate_and_plan(interp, valid_evidence_ids={"FINDING"})
    assert components == [], "observed loss must still be rejected even when cited to FINDING"
    assert any(r.reason_code == "OBSERVED_VALUE_NOT_REMEDIATION" for r in outcome.rejected)


# ==========================================================================
# Pass 52 generalization matrix (deterministic pipeline; hand-built
# interpretations mirroring evidence-backed structured output). Prices cited
# to E-ids (a populated evidence ledger, as the full graph produces) OR to
# FINDING -- both must price identically.
# ==========================================================================
# (helpers below)


def _comp(cid, desc, qty, unit, rate, atype, cite, *, qbasis="EVIDENCED",
          recurrence="ONE_TIME", period=None, deriv="", vkind="REMEDIATION_COST"):
    c = {"component_id": cid, "description": desc, "activity_ids": ["A0"],
         "cost_category": "x", "value_kind": vkind, "quantity": qty, "quantity_unit": unit,
         "quantity_basis": qbasis, "unit_cost": rate, "unit_cost_basis": "VERIFIED",
         "currency": "INR", "amount_type": atype, "source_reference_ids": [cite],
         "recurrence": recurrence}
    if period:
        c["recurring_period"] = period
    if deriv:
        c["quantity_derivation"] = deriv
    return c


def _plan(interp_dict):
    return _interp(interp_dict)


def _run(interp_dict, valid_ids):
    interp = _interp(interp_dict)
    comps, props, outcome = validate_and_plan(interp, valid_evidence_ids=set(valid_ids))
    est = assemble_estimate(comps, props, outcome.traces)
    return comps, est, outcome


def _wrap(components, proposals=None):
    return {
        "strategy": {"remediation_summary": "x", "interpretation_confidence": "HIGH"},
        "activities": [{"activity_id": "A0", "description": "corrective work",
                        "disposition": "IMMEDIATE_CORRECTION", "depends_on_root_cause": False}],
        "cost_components": components,
        "calculation_proposals": proposals or [],
        "estimability": "ESTIMABLE", "overall_status": "EVIDENCE_BACKED",
    }


@pytest.mark.parametrize("cite,ids", [("FINDING", {"FINDING"}), ("E1", {"E0", "E1", "E2", "FINDING"})])
def test_g_equipment_unit_rate(cite, ids):
    # 5 machines x Rs 4,000/machine = 20,000
    _, est, _ = _run(_wrap([_comp("C0", "replacement part", 5, "machine", 4000, "PER_UNIT", cite)]), ids)
    assert est.one_time_cost == pytest.approx(20000)


@pytest.mark.parametrize("cite,ids", [("FINDING", {"FINDING"}), ("E1", {"E0", "E1", "FINDING"})])
def test_g_labor_derived_quantity(cite, ids):
    # 6 machines x 2 h x Rs 800/h = 9,600
    _, est, _ = _run(_wrap([_comp("C0", "inspection labour", 12, "hour", 800, "PER_HOUR", cite,
                                  qbasis="DERIVED", deriv="6 machines x 2 h = 12 h")]), ids)
    assert est.one_time_cost == pytest.approx(9600)


@pytest.mark.parametrize("cite,ids", [("FINDING", {"FINDING"}), ("E1", {"E0", "E1", "FINDING"})])
def test_g_fixed_scope_service(cite, ids):
    # Rs 6,000 for the complete area -> quantity 1, NOT x6
    _, est, _ = _run(_wrap([_comp("C0", "area inspection", 1, "complete area", 6000, "PER_EVENT", cite)]), ids)
    assert est.one_time_cost == pytest.approx(6000)


@pytest.mark.parametrize("cite,ids", [("FINDING", {"FINDING"}), ("E1", {"E0", "E1", "E2", "FINDING"})])
def test_g_multi_component_sum(cite, ids):
    # 6x4000 + 6x2x800 = 24000 + 9600 = 33600
    _, est, _ = _run(_wrap([
        _comp("C0", "module", 6, "machine", 4000, "PER_UNIT", cite),
        _comp("C1", "technician labour", 12, "hour", 800, "PER_HOUR", cite,
              qbasis="DERIVED", deriv="6 machines x 2 h = 12 h"),
    ]), ids)
    assert est.one_time_cost == pytest.approx(33600)


def test_g_recurring_periodic_only():
    # 6 machines x 2 h x 800 monthly -> 9,600/month, no one-time, no horizon
    _, est, _ = _run(_wrap([_comp("C0", "monthly inspection labour", 12, "hour", 800, "PER_HOUR",
                                  "FINDING", qbasis="DERIVED", deriv="6 x 2 h = 12 h",
                                  recurrence="RECURRING", period="month")]), {"FINDING"})
    assert est.recurring_cost == pytest.approx(9600)
    assert not est.one_time_cost
    assert not est.recurring_horizon_total


def test_g_recurring_with_explicit_horizon():
    # monthly for 4 months -> periodic 9,600, horizon_total 38,400
    comps = [_comp("C0", "monthly inspection labour", 12, "hour", 800, "PER_HOUR", "FINDING",
                   qbasis="DERIVED", deriv="6 x 2 h = 12 h", recurrence="RECURRING", period="month")]
    props = [{"calculation_id": "K0", "target_component_id": "C0", "operation": "MULTIPLY",
              "operands": [{"label": "months", "value": 4, "evidence_refs": ["FINDING"]}],
              "produces": "MOST_LIKELY", "frequency": "RECURRING", "recurring_period": "month",
              "horizon": 4, "horizon_unit": "month", "horizon_basis": "EXPLICIT",
              "result_represents": "4-month inspection total"}]
    _, est, _ = _run(_wrap(comps, props), {"FINDING"})
    assert est.recurring_cost == pytest.approx(9600)
    assert est.recurring_horizon_total == pytest.approx(38400)


def test_g_unknown_price_is_partial_when_others_priced():
    comps = [
        _comp("C0", "replacement part", 6, "machine", 4000, "PER_UNIT", "FINDING"),
        {"component_id": "C1", "description": "specialist recommissioning", "activity_ids": ["A0"],
         "cost_category": "x", "value_kind": "REMEDIATION_COST", "amount_type": "COMPONENT",
         "source_reference_ids": []},  # no price, no basis
    ]
    comps_out, est, outcome = _run(_wrap(comps), {"FINDING"})
    assert est.one_time_cost == pytest.approx(24000)  # priced part survives
    assert any(getattr(c, "unit_cost", None) is None for c in comps_out) or est.unpriced_component_ids


def test_g_unknown_amount_type_per_area_is_normalized_not_dropped():
    # Pass 52: qwen3:8b emitted amount_type="PER_AREA" for the fixed complete-area
    # inspection; strict validation dropped the whole Rs 6,000 component and the
    # total came out Rs 13,600 instead of Rs 19,600.
    comps = [
        _comp("C0", "labeling materials", 8, "panel", 350, "PER_UNIT", "FINDING"),
        _comp("C1", "electrician labour", 12, "hour", 900, "PER_HOUR", "FINDING",
              qbasis="DERIVED", deriv="8 x 1.5 h = 12 h"),
        {"component_id": "C2", "description": "safety inspection of the area", "activity_ids": ["A0"],
         "cost_category": "inspection", "value_kind": "REMEDIATION_COST", "quantity": 1,
         "quantity_unit": "area", "quantity_basis": "EVIDENCED", "unit_cost": 6000,
         "unit_cost_basis": "VERIFIED", "currency": "INR", "amount_type": "PER_AREA",
         "source_reference_ids": ["FINDING"]},
    ]
    comps_out, est, outcome = _run(_wrap(comps), {"FINDING"})
    assert len(comps_out) == 3, outcome.rejected
    assert est.one_time_cost == pytest.approx(19600), (est, [c.model_dump() for c in comps_out])


def test_g_packaging_audit_composition_56000():
    """Pass 54 golden: duration x rate + fixed travel + follow-up duration x rate.
    2d x 8h x 2000 + 18000 + 3h x 2000 = 32000 + 18000 + 6000 = 56000 ONE_TIME.
    A shared Rs 2,000/hour rate across two components is not a comparison."""
    comps = [
        _comp("C0", "quality audit labour", 16, "hour", 2000, "PER_HOUR", "FINDING",
              qbasis="DERIVED", deriv="2 days x 8 h/day = 16 h"),
        {"component_id": "C1", "description": "travel and accommodation", "activity_ids": ["A0"],
         "cost_category": "travel", "value_kind": "QUOTED_PRICE", "quantity": 1,
         "quantity_unit": "trip", "quantity_basis": "EVIDENCED", "unit_cost": 18000,
         "unit_cost_basis": "VERIFIED", "currency": "INR", "amount_type": "COMPONENT",
         "source_reference_ids": ["FINDING"]},
        _comp("C2", "post-audit follow-up review", 3, "hour", 2000, "PER_HOUR", "FINDING"),
    ]
    comps_out, est, outcome = _run(_wrap(comps), {"FINDING"})
    assert len(comps_out) == 3, outcome.rejected
    assert est.one_time_cost == pytest.approx(56000), (est, [c.model_dump() for c in comps_out])
    assert not est.recurring_cost and not est.recurring_horizon_total


def test_g_observed_loss_never_priced_even_with_valid_evidence():
    comps = [{"component_id": "C0", "description": "incident cost", "activity_ids": ["A0"],
              "cost_category": "loss", "value_kind": "OBSERVED_FINANCIAL_LOSS", "quantity": 1,
              "quantity_unit": "event", "quantity_basis": "EVIDENCED", "unit_cost": 25000,
              "unit_cost_basis": "VERIFIED", "currency": "INR", "amount_type": "COMPONENT",
              "source_reference_ids": ["E1"]}]
    comps_out, est, outcome = _run(_wrap(comps), {"E0", "E1", "FINDING"})
    assert comps_out == []
    assert not est.one_time_cost


# ==========================================================================
# Pass 52 §26 failure injection -- the engine must fail closed, never fabricate.
# ==========================================================================
@pytest.mark.asyncio
async def test_fi_remediation_llm_unparseable_json():
    rc = await estimate_remediation_cost(
        finding_text=PANELS_FINDING, evidence_ledger=[],
        semantic_context=_canonical_ctx(), client=_FakeClient("not json at all {{{"),
    )
    assert not rc.one_time_cost and not rc.recurring_cost
    assert rc.pricing_status == "NOT_ASSESSABLE"


@pytest.mark.asyncio
async def test_fi_remediation_llm_timeout_is_not_assessable_not_fabricated():
    class _Boom:
        async def chat_completion(self, messages, **kw):
            raise TimeoutError("simulated ollama timeout")

    rc = await estimate_remediation_cost(
        finding_text=PANELS_FINDING, evidence_ledger=[],
        semantic_context=_canonical_ctx(), client=_Boom(),
    )
    assert not rc.one_time_cost and not rc.recurring_cost
    assert rc.pricing_status == "NOT_ASSESSABLE"
    assert not (rc.cost_components and any(c.calculated_amount for c in rc.cost_components))


def test_fi_malformed_calculation_proposal_is_dropped_component_still_priced():
    from app.remediation.interpreter import _salvage

    d = _panels_interp_dict(cite="FINDING")
    d["calculation_proposals"] = [
        {"calculation_id": "K9", "operation": "WHARRGARBL", "component_ids": ["C0"],
         "produces": "MOST_LIKELY"},  # invalid operation -> salvage drops it
    ]
    interp = _salvage(normalize_to_canonical(d))
    assert interp is not None
    comps, props, outcome = validate_and_plan(interp, valid_evidence_ids={"FINDING"})
    est = assemble_estimate(comps, props, outcome.traces)
    # role-based (amount_type) assembly still yields the real total
    assert est.one_time_cost == pytest.approx(19600)


@pytest.mark.asyncio
async def test_concurrency_no_cross_request_pricing_contamination():
    """5 different findings priced concurrently -- each result must reflect ITS
    OWN inputs (no evidence / pricing / calculation / metadata bleed)."""
    import asyncio

    cases = [
        (5, 4000, 20000), (6, 800, 4800), (3, 1500, 4500), (8, 350, 2800), (10, 250, 2500),
    ]

    async def _one(qty, rate, expected):
        d = _wrap([_comp("C0", f"part for {qty} units", qty, "unit", rate, "PER_UNIT", "FINDING")])
        rc = await estimate_remediation_cost(
            finding_text=f"{qty} units require a part costing Rs {rate} per unit.",
            evidence_ledger=[], semantic_context=_canonical_ctx(),
            client=_FakeClient(__import__("json").dumps(d)),
        )
        return (rc.one_time_cost or rc.most_likely_estimate), expected

    results = await asyncio.gather(*[_one(q, r, e) for q, r, e in cases])
    for got, expected in results:
        assert got == pytest.approx(expected), results


def test_fi_invalid_evidence_ref_is_scrubbed_component_unpriced():
    d = _panels_interp_dict(cite="E7")  # E7 does not exist
    interp = _interp(d)
    comps, props, outcome = validate_and_plan(interp, valid_evidence_ids={"E0", "FINDING"})
    est = assemble_estimate(comps, props, outcome.traces)
    assert not est.one_time_cost, "a component citing a non-existent evidence id must not be priced"
