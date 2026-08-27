"""Component-level degradation of the LLM financial interpretation
(architecture spec sections 10, 11, 16, 33): one invalid / unresolvable
calculation proposal must NOT discard the valid financial components.

The reference fixture is the shape a real local model (qwen3:8b) actually
produced for the packaging-rework reproduction: a correct gross MULTIPLY,
plus two proposals that must be rejected --

  * a SUBTRACT whose `inputs` reference an earlier *calculation_id*
    ("CAL0") rather than a claim id, and
  * a MULTIPLY mixing a HISTORICAL quantity with a CURRENT_FINDING rate.

Expected: gross exposure, cost factor, recovery and remediation all
survive; only the two bad proposals are rejected, each with a reason.
"""

from __future__ import annotations

import json

import pytest

from app.financial.semantic_engine import analyze_financial_exposure_semantic
from app.models.agent import EvidenceItem, EvidenceStatus

_REAL_SHAPED = json.dumps({
    "finding": {"deviation": "packaging failures", "interpretation_confidence": "HIGH"},
    "claims": [
        {"claim_id": "C0", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 1000, "unit": "unit", "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        {"claim_id": "C1", "source_evidence_ids": ["E1"], "fact_type": "RATE", "value": 250, "unit": "INR/unit", "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        {"claim_id": "C2", "source_evidence_ids": ["E2"], "fact_type": "RECOVERY", "value": 40000, "unit": "INR", "currency": "INR", "population": "RECOVERY", "evidence_status": "REPORTED"},
        {"claim_id": "C3", "source_evidence_ids": ["E3"], "fact_type": "QUANTITY", "value": 10, "unit": "occurrence", "population": "HISTORICAL", "evidence_status": "VERIFIED"},
        {"claim_id": "C4", "source_evidence_ids": ["E4"], "fact_type": "REMEDIATION_COST", "value": 75000, "unit": "INR", "currency": "INR", "population": "REMEDIATION", "evidence_status": "VERIFIED"},
    ],
    "relationships": [
        {"relationship_id": "R0", "relationship_type": "rate_applies_to_quantity", "source_claim": "C1", "target_claim": "C0", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]},
        {"relationship_id": "R1", "relationship_type": "recovery_applies_to_gross", "source_claim": "C2", "target_claim": "C0", "confidence": "HIGH", "evidence_basis": ["E0", "E2"]},
        {"relationship_id": "R2", "relationship_type": "rate_applies_to_quantity", "source_claim": "C1", "target_claim": "C3", "confidence": "HIGH", "evidence_basis": ["E1", "E3"]},
    ],
    "calculation_proposals": [
        {"calculation_id": "CAL0", "operation": "MULTIPLY", "inputs": ["C0", "C1"], "relationship_ids": ["R0"], "proposed_result_value": 250000, "proposed_result_currency": "INR", "reason": "gross"},
        {"calculation_id": "CAL1", "operation": "SUBTRACT", "inputs": ["CAL0", "C2"], "relationship_ids": ["R1"], "proposed_result_value": 210000, "proposed_result_currency": "INR", "reason": "net"},
        {"calculation_id": "CAL2", "operation": "MULTIPLY", "inputs": ["C3", "C1"], "relationship_ids": ["R2"], "proposed_result_value": 25000, "proposed_result_currency": "INR", "reason": "historical"},
    ],
    "cost_factor": {"selected_factor": "REWORK_COST", "supporting_claim_ids": ["C0", "C1"], "confidence": "HIGH"},
    "financial_relevance": "CONFIRMED",
    "quantification": {"status": "QUANTIFIED"},
})

_LEDGER = [
    EvidenceItem(claim="Five confirmed packaging failures resulted in 1,000 units requiring rework.", status=EvidenceStatus.VERIFIED, source="C1"),
    EvidenceItem(claim="Production records verify an average rework cost of INR 250 per unit.", status=EvidenceStatus.VERIFIED, source="C2"),
    EvidenceItem(claim="Finance records confirm that INR 40,000 was recovered from the supplier.", status=EvidenceStatus.REPORTED, source="C3"),
    EvidenceItem(claim="Historical records show the same failure occurred 10 times during the previous 12 months.", status=EvidenceStatus.VERIFIED, source="C4"),
    EvidenceItem(claim="A proposed supplier-control improvement has a verified implementation cost of INR 75,000.", status=EvidenceStatus.VERIFIED, source="C5"),
]


class _FakeLLM:
    async def chat_completion(self, messages, **kwargs):
        return _REAL_SHAPED


@pytest.mark.asyncio
async def test_bad_proposals_rejected_valid_components_survive():
    result, audit = await analyze_financial_exposure_semantic(
        "Packaging failures caused rework.", evidence_ledger=_LEDGER, client=_FakeLLM()
    )
    ci = result.confirmed_impact

    assert result.financial_semantic_status == "OK"
    # never leak an internal state into the auditor-facing factor
    assert ci.financial_factor == "REWORK_COST"
    assert ci.verified_gross_exposure == 250000.0
    # recovery stays REPORTED and separate; no confirmed net loss from it
    assert ci.reported_recovery == 40000.0
    assert ci.verified_recovery is None
    assert ci.confirmed_net_loss is None
    # remediation stays separate, never folded into gross
    assert result.capa_economics.remediation_cost == 75000.0

    accepted = set(audit.outcome.accepted_calculation_ids)
    rejected = {r.calculation_id: r.reason_code for r in audit.outcome.rejected}
    assert accepted == {"CAL0"}
    assert rejected.get("CAL1") == "UNKNOWN_CLAIM"          # inputs referenced a calc id
    assert rejected.get("CAL2") == "POPULATION_MISMATCH"    # historical x current


@pytest.mark.asyncio
async def test_token_budget_is_large_enough_for_a_real_interpretation():
    """Guard the config value that caused the original failure: a complete
    5-claim interpretation is ~1300 output tokens against real qwen3:8b."""
    from app.config import get_settings

    assert get_settings().ollama_financial_semantic_max_tokens >= 1600
