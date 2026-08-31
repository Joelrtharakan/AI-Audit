"""Pass 30: a multi-component remediation (equipment + installation + validation)
must sum EVERY priced component; a "per X" rate whose count is not evidenced
must NOT be assumed; a stated total that does not reconcile with the itemised
parts must not silently lower the estimate.
"""

from __future__ import annotations

import json

import pytest

from app.remediation.engine import estimate_remediation_cost
from app.remediation.models import RemediationEstimateStatus
from app.models.agent import EvidenceItem, EvidenceStatus


class _Fake:
    def __init__(self, r):
        self.r = r
    async def chat_completion(self, messages, **kw):
        return self.r


def _ev(claim):
    return EvidenceItem(claim=claim, status=EvidenceStatus.VERIFIED, source="t")


PHARM_FIND = (
    "A hospital pharmacy requires replacement of two temperature sensors and installation "
    "of an automated alarm system. Each sensor costs Rs.7,500. Installation of the sensors "
    "costs Rs.3,000 per refrigerator. The alarm system costs Rs.45,000, installation costs "
    "Rs.8,000, and validation requires 6 hours at Rs.1,500 per hour."
)


def _pharm_ev(refrigerator_count_stated: bool):
    ev = [
        _ev("A hospital pharmacy requires replacement of two temperature sensors and "
            "installation of an automated alarm system."),
        _ev("Each sensor costs Rs.7,500."),
        _ev("Installation of the sensors costs Rs.3,000 per refrigerator" +
            ("; the two sensors are installed in two refrigerators." if refrigerator_count_stated else ".")),
        _ev("The alarm system costs Rs.45,000, installation costs Rs.8,000, and validation "
            "requires 6 hours at Rs.1,500 per hour."),
    ]
    return ev


def _pharm_interp(refrigerator_qty, refrig_basis="EVIDENCED", refrig_auditor=False):
    comps = [
        {"component_id": "C0", "description": "replacement temperature sensors",
         "activity_ids": ["RA001"], "cost_category": "materials", "quantity": 2,
         "quantity_unit": "sensor", "quantity_basis": "EVIDENCED", "unit_cost": 7500,
         "unit_cost_basis": "VERIFIED", "currency": "INR", "amount_type": "PER_UNIT",
         "source_reference_ids": ["E1"], "rationale": "2 sensors x 7,500"},
        {"component_id": "C2", "description": "automated alarm system (equipment)",
         "activity_ids": ["RA002"], "cost_category": "equipment", "quantity": 1,
         "quantity_unit": "system", "quantity_basis": "EVIDENCED", "unit_cost": 45000,
         "unit_cost_basis": "VERIFIED", "currency": "INR", "amount_type": "COMPONENT",
         "source_reference_ids": ["E3"], "rationale": "1 alarm system x 45,000"},
        {"component_id": "C3", "description": "alarm system installation",
         "activity_ids": ["RA002"], "cost_category": "installation", "unit_cost": 8000,
         "unit_cost_basis": "VERIFIED", "currency": "INR", "amount_type": "COMPONENT",
         "source_reference_ids": ["E3"], "rationale": "installation 8,000"},
        {"component_id": "C4", "description": "alarm system validation",
         "activity_ids": ["RA003"], "cost_category": "labor", "quantity": 6,
         "quantity_unit": "hour", "quantity_basis": "EVIDENCED", "unit_cost": 1500,
         "unit_cost_basis": "VERIFIED", "currency": "INR", "amount_type": "PER_HOUR",
         "source_reference_ids": ["E3"], "rationale": "6 h x 1,500 = 9,000"},
    ]
    c1 = {"component_id": "C1", "description": "sensor installation",
          "activity_ids": ["RA001"], "cost_category": "installation", "unit_cost": 3000,
          "quantity_unit": "refrigerator", "unit_cost_basis": "VERIFIED", "currency": "INR",
          "amount_type": "PER_UNIT", "source_reference_ids": ["E2"]}
    if refrigerator_qty is not None:
        c1["quantity"] = refrigerator_qty
        c1["quantity_basis"] = refrig_basis
    else:
        c1["quantity_basis"] = "NOT_ESTABLISHED"
    comps.insert(1, c1)
    interp = {
        "activities": [
            {"activity_id": "RA001", "description": "Replace the two temperature sensors",
             "disposition": "IMMEDIATE_CORRECTION"},
            {"activity_id": "RA002", "description": "Install the automated alarm system",
             "disposition": "IMMEDIATE_CORRECTION"},
            {"activity_id": "RA003", "description": "Validate the automated alarm system",
             "disposition": "EFFECTIVENESS_CHECK"},
        ],
        "cost_components": comps,
        "estimability": "ESTIMABLE", "overall_status": "EVIDENCE_BACKED",
    }
    if refrig_auditor:
        interp["auditor_inputs_required"] = [{
            "remediation_activity": "Install the temperature sensors",
            "current_pricing_evidence": "Rs.3,000 per refrigerator",
            "missing_input": "the number of refrigerators requiring sensor installation",
            "why_required": "determines the sensor-installation quantity",
            "acceptable_evidence": "installation scope / equipment list / work order",
            "enables_estimate_type": "EXACT_ESTIMATE",
        }]
    return interp


@pytest.mark.asyncio
async def test_case_a_two_refrigerators_stated_all_components_summed_exact():
    res = await estimate_remediation_cost(
        finding_text=PHARM_FIND, evidence_ledger=_pharm_ev(True),
        client=_Fake(json.dumps(_pharm_interp(2))), semantic_context=None,
    )
    # 15,000 + 6,000 + 45,000 + 8,000 + 9,000
    assert res.most_likely_estimate == 83000.0
    assert res.pricing_status == "EXACT_ESTIMATE"
    assert res.auditor_inputs_required == []
    amounts = sorted(c.calculated_amount for c in res.cost_components if c.calculated_amount)
    assert 45000.0 in amounts        # the equipment line survived


@pytest.mark.asyncio
async def test_case_c_refrigerator_count_not_stated_is_partial_with_auditor_input():
    res = await estimate_remediation_cost(
        finding_text=PHARM_FIND, evidence_ledger=_pharm_ev(False),
        client=_Fake(json.dumps(_pharm_interp(None, refrig_auditor=True))),
        semantic_context=None,
    )
    # priced portion: 15,000 + 45,000 + 8,000 + 9,000
    assert res.most_likely_estimate == 77000.0
    assert res.pricing_status == "PARTIAL_ESTIMATE"
    assert len(res.auditor_inputs_required) == 1
    assert "refrigerator" in res.auditor_inputs_required[0].missing_input.lower()


@pytest.mark.asyncio
async def test_case_d_missing_alarm_price_partial_prices_the_rest():
    interp = _pharm_interp(2)
    for c in interp["cost_components"]:
        if c["component_id"] == "C2":
            c["unit_cost"] = None
            c["unit_cost_basis"] = "NOT_ESTABLISHED"
            c["source_reference_ids"] = []
    interp["auditor_inputs_required"] = [{
        "remediation_activity": "Install the automated alarm system",
        "current_pricing_evidence": "installation Rs.8,000; validation 6h x Rs.1,500",
        "missing_input": "the automated alarm system equipment purchase price",
        "why_required": "equipment cost is not stated in the evidence",
        "acceptable_evidence": "supplier quotation or approved internal price for the alarm system",
        "enables_estimate_type": "EXACT_ESTIMATE",
    }]
    res = await estimate_remediation_cost(
        finding_text=PHARM_FIND, evidence_ledger=_pharm_ev(True),
        client=_Fake(json.dumps(interp)), semantic_context=None,
    )
    # 15,000 + 6,000 + 8,000 + 9,000  (alarm equipment unpriced)
    assert res.most_likely_estimate == 38000.0
    assert res.pricing_status == "PARTIAL_ESTIMATE"
    assert any("alarm" in a.missing_input.lower() for a in res.auditor_inputs_required)


@pytest.mark.asyncio
async def test_pricing_input_the_llm_dropped_is_surfaced_not_silently_lost():
    """§8 NO SILENT DROP: the canonical pricing_information names Rs.45,000 for
    the alarm activity; the pricing LLM (weak model) omits that component.
    The engine must surface the unaccounted amount and mark PARTIAL, never a
    clean EXACT below the true figure."""
    from app.services.canonical_semantic_models import (
        CanonicalFindingContext, SemRemediationAction, SemPricingItem,
    )
    sc = CanonicalFindingContext(
        finding_subject="temperature sensors and alarm system",
        observed_condition="sensors require replacement; alarm system required",
        root_cause_status="NOT_ESTABLISHED",
        remediation_obligation="ESTABLISHED_CORRECTIVE_OBLIGATION",
        remediation_activities=[
            SemRemediationAction(action_id="RA1", activity="Replace the two temperature sensors",
                                 disposition="IMMEDIATE_CORRECTION", depends_on_root_cause=False),
            SemRemediationAction(action_id="RA2", activity="Install the automated alarm system",
                                 disposition="IMMEDIATE_CORRECTION", depends_on_root_cause=False),
        ],
        pricing_information=[
            SemPricingItem(action_id="RA1", pricing_basis="sensor unit price",
                           rationale="each sensor 7,500", evidence_available=True),
            SemPricingItem(action_id="RA2", pricing_basis="alarm equipment + installation",
                           observed_value_in_finding="45,000, 8,000", evidence_available=True),
        ],
    )
    interp = {
        "activities": [
            {"activity_id": "RA1", "description": "Replace the two temperature sensors",
             "disposition": "IMMEDIATE_CORRECTION"},
            {"activity_id": "RA2", "description": "Install the automated alarm system",
             "disposition": "IMMEDIATE_CORRECTION"},
        ],
        "cost_components": [
            {"component_id": "C0", "description": "sensor replacement", "activity_ids": ["RA1"],
             "quantity": 2, "quantity_basis": "EVIDENCED", "unit_cost": 7500,
             "unit_cost_basis": "VERIFIED", "currency": "INR", "amount_type": "PER_UNIT",
             "source_reference_ids": ["E1"]},
            # the weak model priced ONLY the alarm installation, dropping the 45,000 equipment
            {"component_id": "C1", "description": "alarm system installation", "activity_ids": ["RA2"],
             "unit_cost": 8000, "unit_cost_basis": "VERIFIED", "currency": "INR",
             "amount_type": "COMPONENT", "source_reference_ids": ["E3"]},
        ],
        "estimability": "ESTIMABLE", "overall_status": "EVIDENCE_BACKED",
    }
    res = await estimate_remediation_cost(
        finding_text="A pharmacy requires two temperature sensors replaced and an automated "
                     "alarm system installed. Each sensor 7,500. The alarm system costs 45,000, "
                     "installation 8,000.",
        evidence_ledger=[_ev("each sensor Rs.7,500"),
                         _ev("the alarm system costs Rs.45,000, installation costs Rs.8,000")],
        client=_Fake(json.dumps(interp)), semantic_context=sc,
    )
    assert res.pricing_status != "EXACT_ESTIMATE"
    assert res.pricing_status in ("PARTIAL_ESTIMATE", "RANGE_ESTIMATE")
    blob = " ".join(res.uncertainty_reasons) + " ".join(
        d.description for d in (res.unresolved_pricing_drivers or []))
    assert "45" in blob        # the dropped 45,000 is surfaced somewhere


@pytest.mark.asyncio
async def test_nonreconciling_subtotal_is_not_a_clean_exact():
    """A stated SUBTOTAL that exceeds the itemised parts -> PARTIAL, not EXACT,
    and the conflicting figure is not silently discarded (§8/§10)."""
    interp = _pharm_interp(2)
    interp["cost_components"].append({
        "component_id": "CS", "description": "vendor subtotal for the alarm package",
        "activity_ids": ["RA002"], "cost_category": "equipment", "unit_cost": 70000,
        "unit_cost_basis": "VERIFIED", "currency": "INR", "amount_type": "SUBTOTAL",
        "source_reference_ids": ["E3"],
    })
    res = await estimate_remediation_cost(
        finding_text=PHARM_FIND, evidence_ledger=_pharm_ev(True),
        client=_Fake(json.dumps(interp)), semantic_context=None,
    )
    assert res.pricing_status in ("PARTIAL_ESTIMATE", "RANGE_ESTIMATE")
    assert res.pricing_status != "EXACT_ESTIMATE"
    assert res.status != RemediationEstimateStatus.NOT_ASSESSABLE
