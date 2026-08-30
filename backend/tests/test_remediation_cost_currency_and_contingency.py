"""Remediation cost: currency is never invented (spec section 7), and a
remediation approach is presented as a candidate when the root cause is not
established (spec section 5).
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

    async def chat_completion(self, messages, temperature=0.0, response_format_json=True, **kw):
        return json.dumps(self.payload)


def _ev(claim, status=EvidenceStatus.VERIFIED):
    return EvidenceItem(claim=claim, status=status, source="test")


class _RC:
    def __init__(self, status):
        self.status = status
        self.candidate_hypotheses = []


async def _run(payload, evidence, root_cause=None):
    return await estimate_remediation_cost(
        finding_text="A required control was found missing.",
        evidence_ledger=evidence,
        root_cause=root_cause,
        client=_Fake(payload),
    )


@pytest.mark.asyncio
async def test_figure_with_no_currency_anywhere_is_carried_unpriced():
    # Evidence states an amount (so it survives the pricing gate) but gives
    # no currency -> the figure cannot be meaningfully expressed.
    payload = {
        "strategy": {"remediation_summary": "install the missing control"},
        "cost_components": [{
            "component_id": "C0", "description": "control installation",
            "cost_category": "implementation", "unit_cost": 15000,
            "unit_cost_basis": "REPORTED", "amount_type": "COMPONENT",
            "recurrence": "ONE_TIME", "source_reference_ids": ["E0"],
        }],
        "overall_status": "EVIDENCE_BACKED", "estimability": "ESTIMABLE",
    }
    res = await _run(payload, [_ev("A vendor note lists 15,000 for the control installation", EvidenceStatus.REPORTED)])
    # No currency -> the 15000 is NOT expressed as an amount.
    assert res.most_likely_estimate is None
    assert res.one_time_cost is None
    assert res.currency is None
    assert "control installation" in " ".join(res.unpriced_activities).lower()
    assert res.cost_components[0].calculated_amount is None
    assert any("no currency" in u.lower() for u in res.uncertainty_reasons)
    # never a bare number leaked as an amount
    assert res.low_estimate is None and res.high_estimate is None


@pytest.mark.asyncio
async def test_missing_currency_adopts_the_single_established_one():
    payload = {
        "strategy": {"remediation_summary": "replace and recertify"},
        "cost_components": [
            {"component_id": "C0", "description": "replacement part", "cost_category": "materials",
             "unit_cost": 40000, "unit_cost_basis": "REPORTED", "currency": "INR",
             "amount_type": "COMPONENT", "recurrence": "ONE_TIME", "source_reference_ids": ["E0"]},
            {"component_id": "C1", "description": "recertification service", "cost_category": "services",
             "unit_cost": 10000, "unit_cost_basis": "REPORTED", "amount_type": "COMPONENT",
             "recurrence": "ONE_TIME", "source_reference_ids": ["E1"]},
        ],
        "overall_status": "EVIDENCE_BACKED", "estimability": "ESTIMABLE",
    }
    res = await _run(payload, [
        _ev("Vendor quoted INR 40,000 for the replacement part", EvidenceStatus.REPORTED),
        _ev("Recertification body lists a 10,000 fee", EvidenceStatus.REPORTED),
    ])
    # C1 had no currency but exactly one currency (INR) is established -> adopt it.
    assert res.currency == "INR"
    assert res.most_likely_estimate == 50000.0
    assert all(c.currency == "INR" for c in res.cost_components)


@pytest.mark.asyncio
async def test_contingent_approach_is_framed_as_candidate_when_root_cause_not_established():
    payload = {
        "strategy": {"remediation_summary": "Replace the failed relay and update the PM schedule"},
        "cost_components": [{
            "component_id": "C0", "description": "relay replacement", "cost_category": "materials",
            "unit_cost": 8000, "unit_cost_basis": "REPORTED", "currency": "INR",
            "amount_type": "COMPONENT", "recurrence": "ONE_TIME", "source_reference_ids": ["E0"],
        }],
        "overall_status": "EVIDENCE_BACKED", "estimability": "ESTIMABLE",
    }
    res = await _run(payload, [_ev("Vendor quoted INR 8,000 for the relay", EvidenceStatus.REPORTED)],
                     root_cause=_RC("NOT_ESTABLISHED"))
    assert res.remediation_strategy.lower().startswith("potential implementation approach")
    assert any("contingent" in a.lower() or "not fully established" in a.lower() for a in res.assumptions)


@pytest.mark.asyncio
async def test_established_root_cause_keeps_direct_approach_wording():
    payload = {
        "strategy": {"remediation_summary": "Replace the failed relay"},
        "cost_components": [{
            "component_id": "C0", "description": "relay replacement", "cost_category": "materials",
            "unit_cost": 8000, "unit_cost_basis": "VERIFIED", "currency": "INR",
            "amount_type": "COMPONENT", "recurrence": "ONE_TIME", "source_reference_ids": ["E0"],
        }],
        "overall_status": "EVIDENCE_BACKED", "estimability": "ESTIMABLE",
    }
    res = await _run(payload, [_ev("Invoice shows INR 8,000 paid for the replacement relay")],
                     root_cause=_RC("ESTABLISHED"))
    assert res.remediation_strategy == "Replace the failed relay"
    assert res.status == RemediationEstimateStatus.EVIDENCE_BACKED
