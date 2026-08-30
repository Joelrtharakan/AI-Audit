"""Core invariant: "Do not manufacture precision."

A remediation-cost figure enters the estimate ONLY when it is anchored to
evidence. An assumed effort quantity, an assumed / defaulted labour rate, or
a lump sum the LLM merely labelled ESTIMATED/ASSUMED with no cited evidence
must NOT become a monetary amount -- the cost driver survives, unpriced.

Evidence-backed costs are still preserved and calculated deterministically.
"""

from __future__ import annotations

import json

import pytest

from app.models.agent import EvidenceItem, EvidenceStatus
from app.remediation.engine import estimate_remediation_cost
from app.remediation.models import RemediationEstimateStatus


class _Fake:
    def __init__(self, payload):
        self.payload = payload

    async def chat_completion(self, messages, **kw):
        return json.dumps(self.payload)


def _ev(claim, status=EvidenceStatus.VERIFIED):
    return EvidenceItem(claim=claim, status=status, source="test")


async def _run(payload, evidence, finding="A required verification was not performed."):
    return await estimate_remediation_cost(
        finding_text=finding, evidence_ledger=evidence, client=_Fake(payload),
    )


def _c(**kw):
    base = {"component_id": "C0", "description": "d", "cost_category": "labor",
            "amount_type": "COMPONENT", "recurrence": "ONE_TIME"}
    base.update(kw)
    return base


@pytest.mark.asyncio
@pytest.mark.parametrize("comp", [
    # assumed hours x assumed rate
    _c(quantity=4, quantity_basis="ASSUMED", unit_cost=50, unit_cost_basis="ASSUMED",
       currency="USD", amount_type="PER_HOUR"),
    # assumed hours x evidenced-looking-but-uncited rate
    _c(quantity=10, quantity_basis="ASSUMED", unit_cost=2000, unit_cost_basis="ESTIMATED",
       currency="INR", amount_type="PER_HOUR"),
    # flat lump the LLM just called ESTIMATED, nothing cited
    _c(unit_cost=15000, unit_cost_basis="ESTIMATED", currency="USD", amount_type="TOTAL"),
    # flat lump the LLM just called ASSUMED, with a hand-wave assumption
    _c(unit_cost=15000, unit_cost_basis="ASSUMED", currency="USD", amount_type="TOTAL",
       assumptions=["ballpark figure"]),
    # invented low/high range with no evidence
    _c(unit_cost=100, unit_cost_low=50, unit_cost_high=200, unit_cost_basis="ASSUMED",
       currency="USD", amount_type="TOTAL"),
])
async def test_unanchored_figures_never_become_money(comp):
    res = await _run(
        {"strategy": {"remediation_summary": "define and perform the verification"},
         "cost_components": [comp], "overall_status": "ASSUMPTION_BASED"},
        [_ev("The verification step is undefined")],
    )
    assert res.most_likely_estimate is None
    assert res.one_time_cost is None
    assert res.low_estimate is None and res.high_estimate is None
    assert res.status == RemediationEstimateStatus.NOT_ASSESSABLE
    assert res.cost_components and res.cost_components[0].calculated_amount is None
    # semantic info still there
    assert res.remediation_strategy
    for bad in ("LLM", "schema", "parser", "rejected", "None", "NoneType"):
        assert bad not in (res.not_assessable_reason or "")


@pytest.mark.asyncio
async def test_evidenced_quantity_times_reported_rate_is_still_calculated():
    res = await _run(
        {"strategy": {"remediation_summary": "retrain the affected staff"},
         "cost_components": [_c(
             component_id="C0", description="refresher training",
             quantity=30, quantity_unit="person", quantity_basis="EVIDENCED",
             unit_cost=800, unit_cost_basis="REPORTED", currency="INR",
             amount_type="PER_UNIT", source_reference_ids=["E0", "E1"])],
         "overall_status": "EVIDENCE_BACKED"},
        [_ev("30 staff are affected"),
         _ev("Training provider quoted INR 800 per person", EvidenceStatus.REPORTED)],
    )
    assert res.most_likely_estimate == 24000.0
    assert res.one_time_cost == 24000.0
    assert res.currency == "INR"
    assert res.status == RemediationEstimateStatus.EVIDENCE_BACKED


@pytest.mark.asyncio
async def test_verified_total_is_preserved():
    res = await _run(
        {"strategy": {"remediation_summary": "replace the failed unit"},
         "cost_components": [_c(
             component_id="C0", description="replacement + install",
             unit_cost=120000, unit_cost_basis="VERIFIED", currency="INR",
             amount_type="TOTAL", source_reference_ids=["E0"])],
         "overall_status": "EVIDENCE_BACKED"},
        [_ev("Approved quotation totals INR 120,000 for replacement and installation")],
    )
    assert res.most_likely_estimate == 120000.0
    assert res.low_estimate == 120000.0 and res.high_estimate == 120000.0


@pytest.mark.asyncio
async def test_partial_estimate_keeps_priced_and_lists_unpriced():
    res = await _run(
        {"strategy": {"remediation_summary": "replace the part and revalidate"},
         "cost_components": [
             _c(component_id="C0", description="replacement part", unit_cost=40000,
                unit_cost_basis="REPORTED", currency="INR", amount_type="COMPONENT",
                source_reference_ids=["E0"]),
             _c(component_id="C1", description="revalidation effort", quantity=8,
                quantity_basis="ASSUMED", unit_cost=1500, unit_cost_basis="ASSUMED",
                currency="INR", amount_type="PER_HOUR"),
         ],
         "overall_status": "EVIDENCE_BACKED"},
        [_ev("Vendor quoted INR 40,000 for the replacement part", EvidenceStatus.REPORTED)],
    )
    assert res.most_likely_estimate == 40000.0          # only the anchored part
    assert res.is_partial_estimate is True
    assert any("revalidation" in a.lower() for a in res.unpriced_activities)
