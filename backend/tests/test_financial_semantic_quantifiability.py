"""Proof that cost-factor identification and monetary quantification are
genuinely separate semantic dimensions -- a real architectural capability
added to fix a gap where a finding that clearly named its cost CATEGORY
(e.g. "18 additional rework hours") but supplied no rate/amount was
collapsed all the way down to NOT_ASSESSABLE, discarding the LLM's valid,
grounded cost-factor determination entirely.

Category B (FakeLLMClient contract tests, per this pass's Ollama Testing
Strategy): fast, deterministic, prove the validator/materialization/result-
construction plumbing is correct. Real-model proof lives in
tests/test_financial_semantic_real_ollama.py.
"""

from __future__ import annotations

import json

import pytest

from app.financial.models import FinancialEpistemicStatus
from app.financial.semantic_engine import analyze_financial_exposure_semantic
from app.models.agent import EvidenceItem, EvidenceStatus


class FakeLLMClient:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    async def chat_completion(self, messages, temperature=0.0, response_format_json=True, **kwargs):
        self.calls += 1
        return self.response


def _not_quantifiable_response(quantity: float, unit: str, factor: str, blocker: str, missing: list[str]) -> str:
    return json.dumps({
        "finding": {},
        "claims": [
            {"claim_id": "C0", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY",
             "value": quantity, "unit": unit, "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [],
        "calculation_proposals": [],
        "cost_factor": {"selected_factor": factor, "supporting_claim_ids": ["C0"],
                         "confidence": "HIGH", "rationale": f"the evidence describes {factor.lower()} activity"},
        "quantification": {"status": "NOT_QUANTIFIABLE", "blocker": blocker, "missing_inputs": missing},
    })


async def _run(finding_text: str, response: str):
    ledger = [EvidenceItem(claim=finding_text, status=EvidenceStatus.VERIFIED, source="S0")]
    return await analyze_financial_exposure_semantic(finding_text, ledger, client=FakeLLMClient(response))


class TestCostFactorWithoutQuantification:
    """Four different domains, same mechanism -- a factor is identified,
    grounded, and preserved, and NO monetary amount is ever fabricated."""

    async def test_rework_hours_without_rate(self):
        response = _not_quantifiable_response(
            24, "HOUR", "REWORK_COST",
            "No monetary labor rate or total repair cost is stated.",
            ["labor rate", "total repair cost"],
        )
        result = await _run("Additional repair labor was required for 24 hours.", response)
        assert result is not None
        fa, audit = result
        assert fa.status == FinancialEpistemicStatus.COST_FACTOR_IDENTIFIED_NOT_QUANTIFIABLE
        assert fa.confirmed_impact.financial_factor == "REWORK_COST"
        assert fa.confirmed_impact.quantification_status == "UNQUANTIFIED"
        assert fa.confirmed_impact.verified_gross_exposure is None
        # The semantic layer now stamps its own provenance: a
        # cost-factor-identified / not-quantifiable result is genuinely an
        # LLM semantic product, not a deterministic-regex one.
        assert fa.reasoning_source == "LLM_SEMANTIC"
        assert fa.financial_semantic_status == "OK"
        assert audit.outcome.validated_cost_factor == "REWORK_COST"

    async def test_downtime_hours_without_rate(self):
        response = _not_quantifiable_response(
            7, "HOUR", "DOWNTIME_COST",
            "No downtime cost rate or total disruption cost is stated.",
            ["downtime rate"],
        )
        result = await _run("The conveyor was unavailable for 7 hours.", response)
        assert result is not None
        fa, _ = result
        assert fa.confirmed_impact.financial_factor == "DOWNTIME_COST"
        assert fa.confirmed_impact.verified_gross_exposure is None
        assert fa.confirmed_impact.quantification_status == "UNQUANTIFIED"

    async def test_scrap_units_without_material_cost(self):
        response = _not_quantifiable_response(
            60, "UNIT", "SCRAP_COST",
            "No material cost per unit or total scrap value is stated.",
            ["material cost per unit"],
        )
        result = await _run("60 units were discarded as unusable scrap.", response)
        assert result is not None
        fa, _ = result
        assert fa.confirmed_impact.financial_factor == "SCRAP_COST"
        assert fa.confirmed_impact.verified_gross_exposure is None

    async def test_maintenance_visits_without_service_rate(self):
        response = _not_quantifiable_response(
            5, "VISIT", "DIRECT_LOSS",
            "No service rate or total maintenance cost is stated.",
            ["service rate per visit"],
        )
        result = await _run("The pump required 5 unscheduled maintenance visits.", response)
        assert result is not None
        fa, _ = result
        assert fa.confirmed_impact.financial_factor == "DIRECT_LOSS"
        assert fa.confirmed_impact.quantification_status == "UNQUANTIFIED"
        assert fa.confirmed_impact.verified_gross_exposure is None


class TestQuantifiabilityNeverOverridesRealCalculation:
    """The new NOT_QUANTIFIABLE path must never fire when a real
    calculation succeeds -- quantity x rate cases must be entirely
    unaffected by this change."""

    async def test_quantifiable_case_still_calculates_normally(self):
        response = json.dumps({
            "finding": {},
            "claims": [
                {"claim_id": "C0", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY",
                 "value": 320, "unit": "UNIT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
                {"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "RATE",
                 "value": 14, "unit": "UNIT", "currency": "EUR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            ],
            "relationships": [{"relationship_id": "R0", "type": "RATE_APPLIES_TO_QUANTITY",
                                "source_claim": "C1", "target_claim": "C0", "confidence": "HIGH", "evidence_basis": ["E0"]}],
            "calculation_proposals": [{"calculation_id": "CAL0", "operation": "MULTIPLY", "inputs": ["C0", "C1"],
                                        "relationship_ids": ["R0"], "reason": "quantity x rate"}],
            "cost_factor": {"selected_factor": "SCRAP_COST", "supporting_claim_ids": ["C0"],
                             "confidence": "HIGH", "rationale": "scrapped units"},
            "quantification": {"status": "QUANTIFIABLE", "blocker": "", "missing_inputs": []},
        })
        result = await _run("320 units were scrapped at a verified material cost of EUR 14 per unit.", response)
        assert result is not None
        fa, _ = result
        assert fa.confirmed_impact.verified_gross_exposure == 320 * 14
        assert fa.confirmed_impact.quantification_status != "UNQUANTIFIED"

    async def test_missing_quantification_field_still_surfaces_grounded_cost_factor(self):
        # Older-shaped LLM output with no "quantification" key at all --
        # must not raise. The system invariant (a grounded cost factor
        # must never be silently discarded merely because nothing else
        # could be quantified) means this still surfaces REWORK_COST with
        # no fabricated amount, rather than falling back to a bare None
        # that would erase the LLM's one genuinely established fact.
        response = json.dumps({
            "finding": {},
            "claims": [
                {"claim_id": "C0", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY",
                 "value": 24, "unit": "HOUR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            ],
            "relationships": [], "calculation_proposals": [],
            "cost_factor": {"selected_factor": "REWORK_COST", "supporting_claim_ids": ["C0"],
                             "confidence": "HIGH", "rationale": "rework activity"},
        })
        result = await _run("Additional repair labor was required for 24 hours.", response)
        assert result is not None
        fa, _ = result
        assert fa.confirmed_impact.financial_factor == "REWORK_COST"
        assert fa.confirmed_impact.verified_gross_exposure is None


class TestCostFactorSurvivesValidatorRejection:
    """System invariant: a grounded cost factor must surface even when
    the LLM believed a calculation was possible but the deterministic
    validator rejected it (e.g. incompatible units) -- not only when the
    LLM itself reported NOT_QUANTIFIABLE upfront. Previously this fell
    all the way back to a bare None, discarding the LLM's one genuinely
    established fact and producing no financial section at all."""

    async def test_rejected_calculation_still_surfaces_cost_factor_with_rejection_reason(self):
        response = json.dumps({
            "finding": {},
            "claims": [
                {"claim_id": "C0", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY",
                 "value": 5, "unit": "hour", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
                {"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "RATE",
                 "value": 100, "unit": "INR/kg", "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            ],
            "relationships": [{"relationship_id": "R0", "type": "RATE_APPLIES_TO_QUANTITY",
                                "source_claim": "C1", "target_claim": "C0", "confidence": "HIGH", "evidence_basis": ["E0"]}],
            "calculation_proposals": [{"calculation_id": "CAL0", "operation": "MULTIPLY", "inputs": ["C0", "C1"],
                                        "relationship_ids": ["R0"], "reason": "x"}],
            "cost_factor": {"selected_factor": "DOWNTIME_COST", "supporting_claim_ids": ["C0"],
                             "confidence": "HIGH", "rationale": "downtime activity"},
            "quantification": {"status": "QUANTIFIABLE", "blocker": "", "missing_inputs": []},
        })
        result = await _run("Equipment was unavailable for 5 hours.", response)
        assert result is not None
        fa, audit = result
        assert fa.status.value == "COST_FACTOR_IDENTIFIED_NOT_QUANTIFIABLE"
        assert fa.confirmed_impact.financial_factor == "DOWNTIME_COST"
        assert fa.confirmed_impact.verified_gross_exposure is None
        assert "Incompatible units" in fa.confirmed_impact.quantification_blocker
        assert audit.outcome.rejected
