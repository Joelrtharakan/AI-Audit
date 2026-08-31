"""Pass 27 -- PRIMARY BUSINESS REQUIREMENT: an unknown root cause must not
suppress an otherwise directly calculable remediation cost.

The LLM (canonical + remediation-cost) does the semantic + pricing reasoning;
these tests feed scripted LLM responses and assert the deterministic pipeline
neither discards nor overrides the established direct correction and its price.

CASE A/B/C/D/E/G/H/K/L from the brief's test matrix.
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


def _ev(claim):
    return EvidenceItem(claim=claim, status=EvidenceStatus.VERIFIED, source="test")


GUARD_FINDING = (
    "C1: During inspection of the production area, two machines were found with "
    "damaged safety guards. "
    "C2: Engineering determined that both guards require replacement. "
    "C3: Each replacement guard costs 18,000 and installation requires 6 "
    "technician-hours per machine at an internal labor rate of 1,200 per hour."
)


def _guard_evidence():
    return [
        _ev("C1: Two machines were found with damaged safety guards."),
        _ev("C2: Engineering determined that both guards require replacement."),
        _ev("C3: Each replacement guard costs 18,000; installation 6 technician-hours "
            "per machine at an internal labor rate of 1,200 per hour."),
    ]


# What a correct remediation-cost LLM returns for the guard finding: it identifies
# the direct correction itself, derives 12 technician-hours transparently, and
# cites the finding's own claim labels.
GUARD_INTERP = {
    "strategy": {
        "remediation_summary": "Replace the two damaged safety guards and install them.",
        "established_basis": "Engineering determined both guards require replacement (C2); "
                             "unit price, effort and labour rate are stated (C3).",
    },
    "activities": [{
        "activity_id": "RA-001",
        "description": "Replace the two damaged safety guards.",
        "disposition": "IMMEDIATE_CORRECTION",
        "depends_on_root_cause": False,
        "derived_from": "FINDING",
        "source_reference_ids": ["C1", "C2"],
    }],
    "cost_components": [
        {
            "component_id": "P0", "description": "replacement guards",
            "activity_ids": ["RA-001"], "cost_category": "materials",
            "quantity": 2, "quantity_unit": "guard", "quantity_basis": "EVIDENCED",
            "unit_cost": 18000, "unit_cost_basis": "VERIFIED", "currency": "INR",
            "amount_type": "PER_UNIT", "source_reference_ids": ["C3", "C1"],
            "rationale": "2 guards x 18,000 per guard = 36,000",
        },
        {
            "component_id": "P1", "description": "installation labour",
            "activity_ids": ["RA-001"], "cost_category": "labor",
            "quantity": 12, "quantity_unit": "technician-hour", "quantity_basis": "EVIDENCED",
            "unit_cost": 1200, "unit_cost_basis": "VERIFIED", "currency": "INR",
            "amount_type": "PER_HOUR", "source_reference_ids": ["C3", "C1"],
            "rationale": "2 machines x 6 technician-hours x 1,200 per hour = 14,400",
        },
    ],
    "estimability": "ESTIMABLE",
    "overall_status": "EVIDENCE_BACKED",
    "auditor_inputs_required": [],
}


@pytest.mark.asyncio
async def test_case_c_unknown_rca_known_remediation_known_cost_is_exact():
    """CASE C: unknown RCA + established direct correction + full pricing -> EXACT."""
    res = await estimate_remediation_cost(
        finding_text=GUARD_FINDING,
        evidence_ledger=_guard_evidence(),
        client=FakeLLMClient(json.dumps(GUARD_INTERP)),
        semantic_context=None,          # canonical layer unavailable in this scenario
        root_cause=None,                # -> contingent / NOT_ESTABLISHED
    )
    assert res.status != RemediationEstimateStatus.NOT_ASSESSABLE
    assert res.most_likely_estimate == 50400.0
    assert res.pricing_status == "EXACT_ESTIMATE"
    assert res.currency == "INR"
    assert res.auditor_inputs_required == []
    # the established direct correction, not a systemic "strengthen process" line
    impl = " ".join(res.implementation_activities).lower()
    assert "replace" in impl
    assert "operational process" not in impl


@pytest.mark.asyncio
async def test_case_c_provenance_and_calculation_preserved():
    res = await estimate_remediation_cost(
        finding_text=GUARD_FINDING,
        evidence_ledger=_guard_evidence(),
        client=FakeLLMClient(json.dumps(GUARD_INTERP)),
        semantic_context=None,
    )
    refs = {r for c in res.cost_components for r in c.source_reference_ids}
    assert refs and refs <= {"E0", "E1", "E2"}          # claim labels resolved to E-ids
    formulae = " ".join(c.calculation_formula for c in res.cost_components)
    assert "36000" in formulae.replace(",", "") or "36,000" in formulae
    assert "14400" in formulae.replace(",", "") or "14,400" in formulae


@pytest.mark.asyncio
async def test_case_d_partial_pricing_yields_partial_estimate():
    """CASE D/H: one activity priced, one lacks a basis -> PARTIAL, auditor input
    only for the unpriced one."""
    interp = json.loads(json.dumps(GUARD_INTERP))
    interp["cost_components"][1]["unit_cost"] = None
    interp["cost_components"][1]["unit_cost_basis"] = "NOT_ESTABLISHED"
    interp["cost_components"][1]["source_reference_ids"] = []
    interp["cost_components"][1]["rationale"] = "installation labour -- rate not established"
    interp["estimability"] = "BOUNDED_ONLY"
    interp["auditor_inputs_required"] = [{
        "remediation_activity": "Install the replacement guards",
        "current_pricing_evidence": "quantity of technician-hours",
        "missing_input": "the internal labour rate applicable to guard installation",
        "why_required": "labour cost cannot be computed without a rate",
        "acceptable_evidence": "approved internal labour rate schedule",
        "enables_estimate_type": "EXACT_ESTIMATE",
    }]
    res = await estimate_remediation_cost(
        finding_text=GUARD_FINDING,
        evidence_ledger=_guard_evidence(),
        client=FakeLLMClient(json.dumps(interp)),
        semantic_context=None,
    )
    assert res.most_likely_estimate == 36000.0
    assert res.is_partial_estimate
    assert res.pricing_status == "PARTIAL_ESTIMATE"
    assert len(res.auditor_inputs_required) == 1
    assert "rate" in res.auditor_inputs_required[0].missing_input.lower()


@pytest.mark.asyncio
async def test_case_e_known_remediation_no_pricing_is_not_assessable():
    interp = json.loads(json.dumps(GUARD_INTERP))
    interp["cost_components"] = []
    interp["estimability"] = "NOT_ASSESSABLE"
    interp["not_assessable_reason"] = "PRICING_BASIS_UNAVAILABLE"
    interp["auditor_inputs_required"] = [{
        "remediation_activity": "Replace the two damaged safety guards.",
        "missing_input": "the replacement guard unit price and installation effort",
        "why_required": "no pricing basis is stated in the evidence",
        "acceptable_evidence": "supplier quotation or internal rate + effort estimate",
    }]
    res = await estimate_remediation_cost(
        finding_text="Two machines were found with damaged safety guards; both require replacement.",
        evidence_ledger=[_ev("Two machines have damaged safety guards; both guards require replacement.")],
        client=FakeLLMClient(json.dumps(interp)),
        semantic_context=None,
    )
    assert res.status == RemediationEstimateStatus.NOT_ASSESSABLE
    assert res.pricing_status == "NOT_ASSESSABLE"
    assert res.most_likely_estimate is None
    assert len(res.auditor_inputs_required) == 1


@pytest.mark.asyncio
async def test_case_g_conditional_systemic_does_not_suppress_priced_direct_correction():
    """CASE G: the LLM returns BOTH a priced direct correction and a
    cause-dependent systemic action -> direct correction is priced; the
    systemic action is conditional and never priced."""
    interp = json.loads(json.dumps(GUARD_INTERP))
    interp["activities"].append({
        "activity_id": "RA-002",
        "description": "Subject to confirming the cause, determine whether strengthened "
                       "guarding controls are required and implement the change identified.",
        "disposition": "CONDITIONAL_SYSTEMIC",
        "depends_on_root_cause": True,
        "derived_from": "ROOT_CAUSE_HYPOTHESIS",
        "is_hypothetical": True,
    })
    res = await estimate_remediation_cost(
        finding_text=GUARD_FINDING,
        evidence_ledger=_guard_evidence(),
        client=FakeLLMClient(json.dumps(interp)),
        semantic_context=None,
    )
    assert res.most_likely_estimate == 50400.0
    assert res.pricing_status == "EXACT_ESTIMATE"
    impl = " ".join(res.implementation_activities).lower()
    cond = " ".join(res.conditional_activities).lower()
    assert "replace" in impl
    assert "strengthened guarding controls" in cond


@pytest.mark.asyncio
async def test_case_f_investigation_only_is_not_priced():
    """CASE F: the LLM identifies no remediation, only investigation -> no price."""
    interp = {
        "strategy": {"remediation_summary": "Determine why records are missing."},
        "activities": [],
        "cost_components": [],
        "estimability": "NOT_ASSESSABLE",
        "not_assessable_reason": "REMEDIATION_NOT_DEFINED",
    }
    res = await estimate_remediation_cost(
        finding_text="Three calibration records for gauge G-1 could not be located during the audit.",
        evidence_ledger=[_ev("Three calibration records for gauge G-1 were not located.")],
        client=FakeLLMClient(json.dumps(interp)),
        semantic_context=None,
    )
    assert res.status == RemediationEstimateStatus.NOT_ASSESSABLE
    assert res.most_likely_estimate is None


@pytest.mark.asyncio
async def test_case_l_unsupported_assumptions_are_excluded():
    """CASE L: LLM adds contingency / tax lines with no evidence -> stripped."""
    interp = json.loads(json.dumps(GUARD_INTERP))
    interp["cost_components"].append({
        "component_id": "P9", "description": "10% contingency",
        "activity_ids": ["RA-001"], "cost_category": "contingency",
        "unit_cost": 5040, "unit_cost_basis": "ASSUMED", "currency": "INR",
        "amount_type": "COMPONENT", "source_reference_ids": [],
        "assumptions": ["assumed 10% contingency"],
    })
    res = await estimate_remediation_cost(
        finding_text=GUARD_FINDING,
        evidence_ledger=_guard_evidence(),
        client=FakeLLMClient(json.dumps(interp)),
        semantic_context=None,
    )
    # contingency carries no evidence -> not added to the total, no priced amount
    assert res.most_likely_estimate == 50400.0
    contingency = [c for c in res.cost_components if "contingency" in c.description.lower()]
    assert contingency and contingency[0].calculated_amount is None
