"""Pass 28 (§29): a weak model emits `calculation_proposals[].produces = "TOTAL"`
(or "SUM"/"GRAND_TOTAL"), which is not in the schema enum. Provider
normalization must map the malformed enum spelling to a valid value so
compositional salvage does not drop an otherwise-valid SUM proposal -- and the
calculator still computes the total.
"""

from __future__ import annotations

import json

import pytest

from app.remediation.provider_normalization import normalize_to_canonical
from app.remediation.semantic_models import RemediationInterpretation
from app.remediation.engine import estimate_remediation_cost
from app.models.agent import EvidenceItem, EvidenceStatus


@pytest.mark.parametrize("bad,expected", [
    ("TOTAL", "MOST_LIKELY"), ("sum", "MOST_LIKELY"), ("Grand Total", "MOST_LIKELY"),
    ("SUBTOTAL", "MOST_LIKELY"), ("component", "COMPONENT_AMOUNT"),
    ("MOST_LIKELY", "MOST_LIKELY"), ("LOW", "LOW"), ("wibble", "MOST_LIKELY"),
])
def test_produces_enum_synonyms_normalized(bad, expected):
    raw = {
        "cost_components": [{"component_id": "C0", "description": "x", "unit_cost": 10,
                             "unit_cost_basis": "VERIFIED", "currency": "INR", "amount_type": "COMPONENT"}],
        "calculation_proposals": [{"calculation_id": "K0", "operation": "SUM",
                                   "component_ids": ["C0"], "produces": bad}],
    }
    parsed = normalize_to_canonical(raw)
    interp = RemediationInterpretation.model_validate(parsed)
    assert interp.calculation_proposals[0].produces == expected


class _Fake:
    def __init__(self, r): self.r = r
    async def chat_completion(self, messages, **kw): return self.r


@pytest.mark.asyncio
async def test_total_produces_does_not_drop_the_sum_proposal():
    interp = {
        "activities": [{"activity_id": "R1", "description": "Replace two units",
                        "disposition": "IMMEDIATE_CORRECTION"}],
        "cost_components": [
            {"component_id": "C0", "description": "units", "activity_ids": ["R1"],
             "quantity": 2, "quantity_basis": "EVIDENCED", "unit_cost": 100,
             "unit_cost_basis": "VERIFIED", "currency": "INR", "amount_type": "PER_UNIT",
             "source_reference_ids": ["E0"]},
            {"component_id": "C1", "description": "labour", "activity_ids": ["R1"],
             "quantity": 3, "quantity_basis": "EVIDENCED", "unit_cost": 50,
             "unit_cost_basis": "VERIFIED", "currency": "INR", "amount_type": "PER_HOUR",
             "source_reference_ids": ["E0"]},
        ],
        "calculation_proposals": [{"calculation_id": "K0", "operation": "SUM",
                                   "component_ids": ["C0", "C1"], "produces": "TOTAL"}],
        "estimability": "ESTIMABLE", "overall_status": "EVIDENCE_BACKED",
    }
    res = await estimate_remediation_cost(
        finding_text="Two units are damaged and require replacement.",
        evidence_ledger=[EvidenceItem(claim="2 units at INR 100 each; 3 hours at INR 50/hour",
                                      status=EvidenceStatus.VERIFIED, source="t")],
        client=_Fake(json.dumps(interp)), semantic_context=None,
    )
    assert res.most_likely_estimate == 350.0            # 2*100 + 3*50
    assert not any(r.get("reason_code") == "SCHEMA_INVALID" for r in (res.rejected_items or [])
                   if isinstance(r, dict))
