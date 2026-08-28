"""Proof that the semantic layer's cost-factor CLASSIFICATION is LLM-driven
(selected from a fixed taxonomy, grounded in cited claims) rather than a
deterministic keyword lookup -- and that the validator never trusts an
ungrounded or low-confidence selection.

Before this pass, every MULTIPLY/ANNUALIZE materialization from the semantic
path was hardcoded to `FinancialAmountType.DIRECT_LOSS` regardless of what
the evidence actually described (idle equipment, scrapped material, rework,
a penalty, ...). That hardcoding is itself a "deterministic code discovers
semantics" violation of the architecture, independent of any individual
adversarial finding -- this file proves the fix generalizes across domains,
not that one finding now reports a nicer label.

Also includes the EQ-207 validation case named explicitly in this pass's
instructions, used only as one more generalization data point (10 HOURS x
INR 12,000/HOUR = INR 120,000, cost factor DOWNTIME_COST) -- nothing here
special-cases "EQ-207", "10 hours", "12000", or "per hour".
"""

from __future__ import annotations

import json

from app.financial.relationship_validator import validate_and_materialize
from app.financial.semantic_engine import analyze_financial_exposure_semantic
from app.financial.semantic_models import SemanticFindingInterpretation
from app.models.agent import EvidenceItem, EvidenceStatus


class FakeLLMClient:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    async def chat_completion(self, messages, temperature=0.0, response_format_json=True, **kwargs):
        self.calls += 1
        return self.response


def _quantity_rate_response(quantity: float, rate: float, currency: str, unit: str, cost_factor: dict) -> str:
    return json.dumps({
        "finding": {},
        "claims": [
            {"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY",
             "value": quantity, "unit": unit, "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "C2", "source_evidence_ids": ["E1"], "fact_type": "RATE",
             "value": rate, "unit": unit, "currency": currency, "population": "CURRENT_FINDING",
             "evidence_status": "VERIFIED"},
        ],
        "relationships": [
            {"relationship_id": "R1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "C2",
             "target_claim": "C1", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]},
        ],
        "calculation_proposals": [
            {"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["C1", "C2"],
             "relationship_ids": ["R1"], "reason": "quantity x unit rate"},
        ],
        "cost_factor": cost_factor,
    })


async def _run(finding_text: str, e1: str, e2: str, response: str):
    ledger = [
        EvidenceItem(claim=e1, status=EvidenceStatus.VERIFIED, source="S0"),
        EvidenceItem(claim=e2, status=EvidenceStatus.VERIFIED, source="S1"),
    ]
    client = FakeLLMClient(response)
    result, audit = await analyze_financial_exposure_semantic(finding_text, ledger, client=client)
    return result, audit


class TestLLMDrivenCostFactorAcrossDomains:
    """The same MULTIPLY mechanism, four unrelated domains, four different
    LLM-selected cost factors -- none hardcoded in the deterministic layer."""

    async def test_idle_equipment_is_downtime_cost(self):
        response = _quantity_rate_response(
            10, 12000, "INR", "HOUR",
            {"selected_factor": "DOWNTIME_COST", "supporting_claim_ids": ["C1"],
             "confidence": "HIGH", "rationale": "equipment sat idle for the stated duration"},
        )
        result, audit = await _run(
            "Equipment EQ-207 was unavailable for 10 hours.",
            "Equipment EQ-207 was unavailable for 10 hours.",
            "Production records verify an average production disruption cost of INR 12,000 per hour.",
            response,
        )
        assert result.confirmed_impact.verified_gross_exposure == 120000.0
        assert result.confirmed_impact.financial_factor == "DOWNTIME_COST"
        assert audit.outcome.validated_cost_factor == "DOWNTIME_COST"

    async def test_discarded_material_is_scrap_cost(self):
        response = _quantity_rate_response(
            2400, 18.50, "EUR", "COMPONENT",
            {"selected_factor": "SCRAP_COST", "supporting_claim_ids": ["C1"],
             "confidence": "HIGH", "rationale": "components were discarded, not reworked or returned"},
        )
        result, audit = await _run(
            "2,400 components were discarded as scrap.",
            "2,400 components were discarded as scrap.",
            "Manufacturing records indicate an average material value of EUR 18.50 per component.",
            response,
        )
        assert result.confirmed_impact.verified_gross_exposure == 2400 * 18.50
        assert result.confirmed_impact.financial_factor == "SCRAP_COST"
        assert audit.outcome.validated_cost_factor == "SCRAP_COST"

    async def test_defect_correction_is_rework_cost(self):
        response = _quantity_rate_response(
            300, 850, "INR", "BATCH",
            {"selected_factor": "REWORK_COST", "supporting_claim_ids": ["C1"],
             "confidence": "HIGH", "rationale": "batches were reworked to correct the defect"},
        )
        result, audit = await _run(
            "300 batches required rework.",
            "300 batches required rework.",
            "Finance confirmed a verified average rework cost of INR 850 per batch.",
            response,
        )
        assert result.confirmed_impact.verified_gross_exposure == 300 * 850
        assert result.confirmed_impact.financial_factor == "REWORK_COST"

    async def test_regulatory_charge_is_penalty(self):
        response = _quantity_rate_response(
            5, 20000, "USD", "VIOLATION",
            {"selected_factor": "PENALTY", "supporting_claim_ids": ["C1"],
             "confidence": "HIGH", "rationale": "regulator levied a fixed fine per violation"},
        )
        result, audit = await _run(
            "5 violations were cited by the regulator.",
            "5 violations were cited by the regulator.",
            "The regulator confirmed a fine of USD 20,000 per violation.",
            response,
        )
        assert result.confirmed_impact.verified_gross_exposure == 5 * 20000
        assert result.confirmed_impact.financial_factor == "PENALTY"


class TestCostFactorFailsClosed:
    """An LLM cost-factor selection is never trusted merely because it is
    present -- it must cite real claims and carry sufficient confidence, or
    the deterministic default (unchanged pre-existing behavior) applies."""

    def test_selection_citing_nonexistent_claim_is_ignored(self):
        interp = SemanticFindingInterpretation.model_validate({
            "claims": [{"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "AMOUNT",
                        "value": 5000, "currency": "USD", "population": "CURRENT_FINDING",
                        "evidence_status": "VERIFIED"}],
            "relationships": [],
            "calculation_proposals": [],
            "cost_factor": {"selected_factor": "PENALTY", "supporting_claim_ids": ["GHOST-CLAIM"],
                             "confidence": "HIGH", "rationale": "fabricated provenance"},
        })
        observations, outcome = validate_and_materialize(interp, evidence_count=1)
        assert outcome.validated_cost_factor is None
        assert observations[0].amount_type != "PENALTY"

    def test_low_confidence_selection_is_ignored(self):
        interp = SemanticFindingInterpretation.model_validate({
            "claims": [{"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "AMOUNT",
                        "value": 5000, "currency": "USD", "population": "CURRENT_FINDING",
                        "evidence_status": "VERIFIED"}],
            "relationships": [],
            "calculation_proposals": [],
            "cost_factor": {"selected_factor": "PENALTY", "supporting_claim_ids": ["C1"],
                             "confidence": "LOW", "rationale": "unsure"},
        })
        _, outcome = validate_and_materialize(interp, evidence_count=1)
        assert outcome.validated_cost_factor is None

    def test_remediation_cost_factor_never_becomes_a_financial_exposure_factor(self):
        # A finding whose only monetary content is an implementation/remediation
        # quotation must NOT produce a financial-exposure factor -- remediation
        # cost is analysed separately (spec sections 1, 2, 15). Financial
        # Factor: NOT ESTABLISHED here.
        for factor in ("REMEDIATION_COST", "PREVENTION_COST"):
            interp = SemanticFindingInterpretation.model_validate({
                "claims": [{"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "REMEDIATION_COST",
                            "value": 50000, "currency": "INR", "population": "REMEDIATION",
                            "evidence_status": "VERIFIED"}],
                "relationships": [],
                "calculation_proposals": [],
                "cost_factor": {"selected_factor": factor, "supporting_claim_ids": ["C1"],
                                 "confidence": "HIGH", "rationale": "supplier quotation"},
            })
            _, outcome = validate_and_materialize(interp, evidence_count=1)
            assert outcome.validated_cost_factor is None

    def test_not_established_selection_is_ignored(self):
        interp = SemanticFindingInterpretation.model_validate({
            "claims": [{"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "AMOUNT",
                        "value": 5000, "currency": "USD", "population": "CURRENT_FINDING",
                        "evidence_status": "VERIFIED"}],
            "relationships": [],
            "calculation_proposals": [],
            "cost_factor": {"selected_factor": "NOT_ESTABLISHED", "supporting_claim_ids": [],
                             "confidence": "LOW", "rationale": ""},
        })
        _, outcome = validate_and_materialize(interp, evidence_count=1)
        assert outcome.validated_cost_factor is None

    def test_missing_cost_factor_field_defaults_safely(self):
        # No cost_factor key at all in the LLM's JSON -- must not raise,
        # and the amount must still be reported (never dropped), but its
        # category must be the honest NOT_ESTABLISHED rather than a
        # fabricated DIRECT_LOSS guess -- fabricating a specific category
        # when none was ever confirmed is exactly the "deterministic code
        # decides financial semantics" failure mode this architecture
        # exists to prevent.
        interp = SemanticFindingInterpretation.model_validate({
            "claims": [{"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "AMOUNT",
                        "value": 5000, "currency": "USD", "population": "CURRENT_FINDING",
                        "evidence_status": "VERIFIED"}],
            "relationships": [],
            "calculation_proposals": [],
        })
        observations, outcome = validate_and_materialize(interp, evidence_count=1)
        assert outcome.validated_cost_factor is None
        assert observations[0].amount_type == "NOT_ESTABLISHED"
        assert observations[0].amount == 5000

    def test_recovery_claim_type_is_never_overridden_by_cost_factor(self):
        # RECOVERY already has unambiguous meaning from its own fact_type;
        # a general cost-factor selection about the finding overall must
        # never relabel a recovery claim as e.g. PENALTY.
        interp = SemanticFindingInterpretation.model_validate({
            "claims": [{"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "RECOVERY",
                        "value": 5000, "currency": "USD", "population": "RECOVERY",
                        "evidence_status": "VERIFIED"}],
            "relationships": [],
            "calculation_proposals": [],
            "cost_factor": {"selected_factor": "PENALTY", "supporting_claim_ids": ["C1"],
                             "confidence": "HIGH", "rationale": "irrelevant to a recovery claim"},
        })
        observations, _ = validate_and_materialize(interp, evidence_count=1)
        assert observations[0].amount_type == "RECOVERY"
