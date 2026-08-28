"""Guardrail coverage: the LLM interpretation can never overwrite deterministic
values, invent verified pricing, or leak an internal diagnostic to the auditor.
"""

from __future__ import annotations

import json

import pytest

from app.remediation.engine import estimate_remediation_cost
from app.remediation.models import CostBasis, RemediationEstimateStatus
from app.models.agent import EvidenceItem, EvidenceStatus


class FakeLLMClient:
    def __init__(self, response):
        self.response = response

    async def chat_completion(self, messages, **kwargs):
        return self.response


def _ev(claim, status=EvidenceStatus.UNVERIFIED):
    return EvidenceItem(claim=claim, status=status, source="test")


async def _run(interp, evidence, **kw):
    return await estimate_remediation_cost(
        finding_text="A finding requiring correction.",
        evidence_ledger=evidence,
        client=FakeLLMClient(json.dumps(interp)),
        **kw,
    )


@pytest.mark.asyncio
async def test_llm_cannot_set_the_headline_numbers_directly():
    # LLM claims a MOST_LIKELY of 1,000,000 via proposed_result_value; the real
    # component math is 5 × 2,000 = 10,000.
    interp = {
        "cost_components": [{
            "component_id": "C0", "description": "install effort", "cost_category": "installation",
            "quantity": 5, "unit_cost": 2000, "quantity_basis": "EVIDENCED",
            "unit_cost_basis": "REPORTED", "currency": "INR", "amount_type": "PER_UNIT",
            "source_reference_ids": ["E0"],
        }],
        "calculation_proposals": [{
            "calculation_id": "K0", "operation": "MULTIPLY", "component_ids": ["C0"],
            "produces": "MOST_LIKELY", "proposed_result_value": 1000000,
        }],
        "overall_status": "EVIDENCE_BACKED",
    }
    res = await _run(interp, [_ev("5 units need installation at INR 2,000 each", EvidenceStatus.REPORTED)])
    assert res.most_likely_estimate == 10000.0
    assert res.low_estimate != 1000000 and res.high_estimate != 1000000


@pytest.mark.asyncio
async def test_llm_verified_claim_without_evidence_is_downgraded_not_trusted():
    interp = {
        "cost_components": [{
            "component_id": "C0", "description": "consultant fees", "cost_category": "professional services",
            "unit_cost": 500000, "unit_cost_basis": "VERIFIED", "currency": "INR",
            "amount_type": "TOTAL", "source_reference_ids": [],  # nothing backs it
        }],
        "overall_status": "EVIDENCE_BACKED",
    }
    res = await _run(interp, [_ev("A consultant may be engaged")])
    comp = res.cost_components[0]
    assert comp.unit_cost_basis != CostBasis.VERIFIED
    assert res.estimate_classification != CostBasis.VERIFIED
    assert res.status != RemediationEstimateStatus.EVIDENCE_BACKED


@pytest.mark.asyncio
async def test_fabricated_market_price_with_no_basis_is_stripped():
    interp = {
        "cost_components": [{
            "component_id": "C0", "description": "replacement equipment", "cost_category": "equipment",
            "unit_cost": 7500000, "unit_cost_basis": "NOT_ESTABLISHED", "currency": "INR",
            "amount_type": "TOTAL", "assumptions": [],
        }],
        "overall_status": "EVIDENCE_BACKED",
    }
    res = await _run(interp, [_ev("Equipment may require replacement")])
    assert res.cost_components[0].unit_cost is None
    assert res.status == RemediationEstimateStatus.NOT_ASSESSABLE
    assert "7500000" not in json.dumps(res.model_dump(), default=str)


@pytest.mark.asyncio
async def test_nonexistent_evidence_reference_is_ignored():
    interp = {
        "cost_components": [{
            "component_id": "C0", "description": "x", "cost_category": "labor",
            "quantity": 10, "unit_cost": 100, "quantity_basis": "EVIDENCED",
            "unit_cost_basis": "VERIFIED", "currency": "INR", "amount_type": "PER_UNIT",
            "source_reference_ids": ["E7", "E8"],
        }],
        "overall_status": "EVIDENCE_BACKED",
    }
    res = await _run(interp, [_ev("only one evidence item exists")])
    assert res.cost_components[0].source_reference_ids == []
    assert res.cost_components[0].unit_cost_basis != CostBasis.VERIFIED


@pytest.mark.asyncio
async def test_not_assessable_reason_never_leaks_internal_state():
    interp = {"cost_components": [], "overall_status": "NOT_ASSESSABLE",
              "not_assessable_reason": "REMEDIATION_NOT_DEFINED"}
    res = await _run(interp, [_ev("something")])
    blob = json.dumps(res.model_dump(), default=str)
    assert "cannot be reliably estimated" in res.not_assessable_reason
    for bad in ("LLM_INVALID", "LLM_UNAVAILABLE", "schema", "ValidationError", "parser", "traceback"):
        assert bad not in res.not_assessable_reason
    # the machine status still exists for logs, just not in the user-facing field
    assert res.remediation_semantic_status in ("OK", "LLM_INCOMPLETE")
