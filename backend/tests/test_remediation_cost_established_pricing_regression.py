"""Regression: a finding that already carries every pricing input must not be
returned as NOT_ASSESSABLE, and the cost interpreter's own auditor-input entry
that states "nothing is missing" must never survive into the result (spec §25).

Demonstrated production failure: two machines with damaged safety guards; each
replacement guard 18,000; internal labour rate + effort stated. The pipeline
returned NOT_ASSESSABLE while forwarding an auditor-input entry whose text said
"No missing input; pricing is fully established" -- internally contradictory.
"""

from __future__ import annotations

import json

import pytest

from app.remediation.engine import estimate_remediation_cost
from app.remediation.models import RemediationEstimateStatus
from app.models.agent import EvidenceItem, EvidenceStatus


class FakeLLMClient:
    def __init__(self, response):
        self.response = response

    async def chat_completion(self, messages, **kwargs):
        return self.response


def _ev(claim, status=EvidenceStatus.VERIFIED):
    return EvidenceItem(claim=claim, status=status, source="test")


FINDING = (
    "During a shop-floor audit two machines were found with damaged safety guards. "
    "C1: Machines MC-11 and MC-12 both have cracked guard panels. "
    "C2: The internal maintenance labour rate is 1,200 per hour and each machine "
    "needs 6 technician-hours to refit a guard. "
    "C3: Each replacement guard panel costs 18,000 from the approved supplier."
)


def _evidence():
    return [
        _ev("C1: Machines MC-11 and MC-12 both have cracked guard panels."),
        _ev("C2: Internal maintenance labour rate 1,200/hour; 6 technician-hours per machine."),
        _ev("C3: Each replacement guard panel costs 18,000 from the approved supplier."),
    ]


# The cost interpreter recognised the pricing but (a) emitted a self-nullifying
# auditor-input entry and (b) cited the finding's own "C3" label, not "E2".
INTERP = {
    "strategy": {"remediation_summary": "Replace the two damaged safety guards."},
    "activities": [],
    "cost_components": [
        {
            "component_id": "K0", "description": "replacement guard panels",
            "cost_category": "materials", "quantity": 2, "unit_cost": 18000,
            "quantity_basis": "EVIDENCED", "unit_cost_basis": "VERIFIED",
            "currency": "INR", "amount_type": "PER_UNIT", "source_reference_ids": ["C3"],
        },
        {
            "component_id": "K1", "description": "guard refit labour",
            "cost_category": "labor", "quantity": 12, "unit_cost": 1200,
            "quantity_basis": "EVIDENCED", "unit_cost_basis": "VERIFIED",
            "currency": "INR", "amount_type": "PER_HOUR", "source_reference_ids": ["C2"],
        },
    ],
    "estimability": "ESTIMABLE",
    "overall_status": "EVIDENCE_BACKED",
    "auditor_inputs_required": [{
        "remediation_activity": "Replace the damaged safety guards on two machines",
        "current_pricing_evidence": "E2",
        "missing_input": "No missing input; pricing is fully established",
        "acceptable_evidence": "Fixed-price service quotation or internal labour rate + effort",
    }],
}


@pytest.mark.asyncio
async def test_established_pricing_is_not_reported_not_assessable():
    res = await estimate_remediation_cost(
        finding_text=FINDING,
        evidence_ledger=_evidence(),
        client=FakeLLMClient(json.dumps(INTERP)),
        semantic_context=None,  # fallback path: canonical interpretation unavailable
    )

    # 2*18000 + 12*1200 = 50400
    assert res.status != RemediationEstimateStatus.NOT_ASSESSABLE
    assert res.most_likely_estimate == 50400.0
    assert res.pricing_status == "EXACT_ESTIMATE"


@pytest.mark.asyncio
async def test_self_nullifying_auditor_input_never_survives():
    res = await estimate_remediation_cost(
        finding_text=FINDING,
        evidence_ledger=_evidence(),
        client=FakeLLMClient(json.dumps(INTERP)),
        semantic_context=None,
    )
    blob = " ".join(
        f"{a.missing_input} {a.why_required}" for a in res.auditor_inputs_required
    ).lower()
    assert "no missing input" not in blob
    assert "fully established" not in blob
