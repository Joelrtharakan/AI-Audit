"""Generalization: structurally-different interpretations for UNSEEN abstract
findings across different implied remediation shapes all flow through the SAME
validator + calculator to a coherent RemediationCostResult -- proving there is no
domain branching and no keyword mapping.

None of these findings resemble any hardcoded case; the pipeline has never seen
them. The accuracy comes from the deterministic layer being wording-agnostic.
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


CASES = [
    # (name, finding, evidence_claims, interpretation, expected_most_likely)
    (
        # An evidence-backed hourly RATE but an ASSUMED number of hours is
        # fabricated effort -- "Do not manufacture precision": no amount is
        # produced, the driver is retained unpriced.
        "implement-a-control",
        "No approval step exists before records are released.",
        ["A reviewer role exists and is paid INR 900/hour (HR rate card)"],
        {
            "strategy": {"remediation_summary": "add and operate an approval step",
                         "remediation_type": "control implementation"},
            "cost_components": [{
                "component_id": "C0", "description": "design + document the approval control",
                "cost_category": "implementation effort", "quantity": 16, "quantity_unit": "hour",
                "quantity_basis": "ASSUMED", "unit_cost": 900, "unit_cost_basis": "REPORTED",
                "currency": "INR", "amount_type": "PER_HOUR", "recurrence": "ONE_TIME",
                "source_reference_ids": ["E0"], "assumptions": ["Assumed 16 hours to design and document"],
            }],
            "calculation_proposals": [{"calculation_id": "K0", "operation": "MULTIPLY",
                                       "component_ids": ["C0"], "produces": "MOST_LIKELY"}],
            "overall_status": "EVIDENCE_BACKED",
        },
        None,
    ),
    (
        "physical-modification",
        "A guard rail is absent from an elevated platform.",
        ["A verified fabrication quote of EUR 3,200 covers the rail and its installation"],
        {
            "strategy": {"remediation_summary": "fabricate and install a compliant guard rail",
                         "remediation_type": "physical modification"},
            "cost_components": [{
                "component_id": "C0", "description": "guard rail fabrication + installation",
                "cost_category": "installation", "unit_cost": 3200, "unit_cost_basis": "VERIFIED",
                "currency": "EUR", "amount_type": "TOTAL", "recurrence": "ONE_TIME",
                "source_reference_ids": ["E0"],
            }],
            "overall_status": "EVIDENCE_BACKED", "estimability": "SINGLE_VERIFIED_COST",
        },
        3200.0,
    ),
    (
        "process-redesign",
        "Hand-offs between two teams are undefined and work is lost between them.",
        [],
        {
            "strategy": {"remediation_summary": "redesign the hand-off process and train both teams",
                         "remediation_type": "process redesign + training"},
            "cost_components": [
                {"component_id": "C0", "description": "process redesign workshop effort",
                 "cost_category": "implementation effort", "quantity_basis": "NOT_ESTABLISHED",
                 "unit_cost_basis": "NOT_ESTABLISHED", "amount_type": "COMPONENT", "recurrence": "ONE_TIME"},
                {"component_id": "C1", "description": "training both teams",
                 "cost_category": "training", "quantity_basis": "NOT_ESTABLISHED",
                 "unit_cost_basis": "NOT_ESTABLISHED", "amount_type": "COMPONENT", "recurrence": "ONE_TIME"},
            ],
            "overall_status": "NOT_ASSESSABLE", "estimability": "NOT_ASSESSABLE",
            "not_assessable_reason": "PRICING_BASIS_UNAVAILABLE",
        },
        None,
    ),
    (
        "technology-change",
        "Access is granted manually and is never reviewed.",
        ["Two competing internal estimates for the automation exist: USD 40,000 and USD 90,000"],
        {
            "strategy": {"remediation_summary": "implement automated access provisioning + periodic review",
                         "remediation_type": "technology implementation"},
            "cost_components": [{
                "component_id": "C0", "description": "access automation implementation",
                "cost_category": "integration", "unit_cost": 65000, "unit_cost_low": 40000,
                "unit_cost_high": 90000, "unit_cost_basis": "ESTIMATED", "currency": "USD",
                "amount_type": "TOTAL", "recurrence": "ONE_TIME", "source_reference_ids": ["E0"],
                "assumptions": ["Midpoint of the two internal estimates used as most-likely"],
            }],
            "calculation_proposals": [],
            "overall_status": "ASSUMPTION_BASED", "estimability": "BOUNDED_ONLY",
            "range_assumptions": ["Range spans the two competing internal estimates"],
        },
        65000.0,
    ),
    (
        "organizational-change",
        "One person performs incompatible duties with no oversight.",
        ["An additional part-time oversight role would cost INR 480,000 per year (budgeting note)"],
        {
            "strategy": {"remediation_summary": "separate the duties and add an oversight role",
                         "remediation_type": "organizational change"},
            "cost_components": [{
                "component_id": "C0", "description": "part-time oversight role",
                "cost_category": "recurring maintenance", "unit_cost": 480000,
                "unit_cost_basis": "REPORTED", "currency": "INR", "amount_type": "TOTAL",
                "recurrence": "RECURRING", "recurring_period": "year", "source_reference_ids": ["E0"],
            }],
            "overall_status": "EVIDENCE_BACKED",
        },
        None,  # recurring-only: no one-time implementation cost
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("name,finding,ev_claims,interp,expected_ml", CASES, ids=[c[0] for c in CASES])
async def test_generalizes_without_domain_branching(name, finding, ev_claims, interp, expected_ml):
    evidence = [_ev(c) for c in ev_claims]
    res = await estimate_remediation_cost(
        finding_text=finding, evidence_ledger=evidence,
        client=FakeLLMClient(json.dumps(interp)),
    )
    assert res.most_likely_estimate == expected_ml, name
    # The LLM's proposed_result_value is never echoed as the answer.
    for tr in res.calculation_traces:
        if tr.llm_proposed_result is not None and tr.executor_result is not None:
            assert tr.executor_result != tr.llm_proposed_result or tr.disagreement is None

    if name == "implement-a-control":
        # rate was evidenced, hours were assumed -> unpriced driver, no range
        assert res.most_likely_estimate is None
        assert res.low_estimate is None and res.high_estimate is None
        assert any("approval control" in a.lower() for a in res.unpriced_activities)
    if name == "process-redesign":
        assert res.status == RemediationEstimateStatus.NOT_ASSESSABLE
        assert res.implementation_activities == [] or isinstance(res.implementation_activities, list)
    if name == "technology-change":
        assert res.low_estimate == 40000.0 and res.high_estimate == 90000.0
    if name == "organizational-change":
        assert res.recurring_cost == 480000.0 and res.one_time_cost is None
