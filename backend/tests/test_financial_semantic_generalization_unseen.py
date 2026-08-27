"""Unseen-wording generalization proof for the LLM semantic financial layer.

Every fixture in this file uses numbers, currencies, and a domain that do not
appear anywhere else in the test suite (no "1,000 units", no "INR 250", no
packaging/rework finding) -- the point is to prove the semantic path's
correctness comes from the deterministic validator + calculator being
wording-agnostic, not from any rule tuned to a particular finding's phrasing.

As with `test_semantic_financial_reasoning.py`, no real LLM/network call is
made: a `FakeLLMClient` simulates what a real provider's structured output
would look like for each unseen scenario, and the same downstream
validator/calculator that the regex-extraction path also uses is what
actually produces the authoritative number.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.financial.relationship_validator import validate_and_materialize
from app.financial.semantic_engine import analyze_financial_exposure_semantic
from app.financial.semantic_models import SemanticFindingInterpretation, SemanticRelationship
from app.models.agent import EvidenceItem, EvidenceStatus


class FakeLLMClient:
    def __init__(self, response: str | None = None, raise_exc: Exception | None = None):
        self.response = response
        self.raise_exc = raise_exc
        self.calls = 0

    async def chat_completion(self, messages, temperature=0.0, response_format_json=True, **kwargs):
        self.calls += 1
        if self.raise_exc:
            raise self.raise_exc
        return self.response


def _evidence(*pairs: tuple[str, EvidenceStatus]) -> list[EvidenceItem]:
    return [EvidenceItem(claim=text, status=status, source=f"S{i}") for i, (text, status) in enumerate(pairs)]


# ---------------------------------------------------------------------------
# 1. Novel domain, single sentence: "2,400 components ... EUR 18.50/component"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unseen_component_replacement_quantity_times_rate():
    """Task-spec example: 'Approximately 2,400 components required
    replacement. Manufacturing records indicate an average replacement
    charge of EUR 18.50 per component.' Nothing in the validator/calculator
    references 'components', 'replacement', or 18.50 -- the LLM's (wrong)
    proposed_result is also included to prove it is discarded."""
    response = json.dumps({
        "finding": {"deviation": "component replacement", "interpretation_confidence": "HIGH"},
        "claims": [
            {"claim_id": "Q1", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 2400, "unit": "UNIT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED", "explicit": True},
            {"claim_id": "R1", "source_evidence_ids": ["E0"], "fact_type": "RATE", "value": 18.50, "unit": "UNIT", "currency": "EUR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED", "explicit": True},
        ],
        "relationships": [{"relationship_id": "REL1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "R1", "target_claim": "Q1", "confidence": "HIGH", "evidence_basis": ["E0"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["Q1", "R1"], "relationship_ids": ["REL1"], "proposed_result_value": 43000, "proposed_result_currency": "EUR", "reason": "average replacement charge applies to the affected components"}],
    })
    client = FakeLLMClient(response=response)
    ledger = _evidence(("Approximately 2,400 components required replacement; manufacturing records indicate an average replacement charge of EUR 18.50 per component.", EvidenceStatus.VERIFIED))
    outcome = await analyze_financial_exposure_semantic("unseen finding", ledger, client=client)
    assert outcome is not None
    result, audit = outcome
    assert result.confirmed_impact.verified_gross_exposure == 44400.0
    assert result.currency == "EUR"
    # The LLM's wrong 43000 must never surface as the authoritative figure.
    assert any("43000" in d for d in audit.outcome.llm_disagreements)


# ---------------------------------------------------------------------------
# 2. Quantity and rate split across two separate evidence claims
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unseen_cross_evidence_hours_and_blended_rate():
    """Task-spec example: '850 support hours ... blended rate of $42/hour',
    with the quantity and the rate coming from two different evidence
    items -- proving cross-evidence relationship linking, not same-sentence
    pattern matching."""
    response = json.dumps({
        "finding": {"interpretation_confidence": "HIGH"},
        "claims": [
            {"claim_id": "HRS", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 850, "unit": "HOUR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED", "explicit": True},
            {"claim_id": "RATE", "source_evidence_ids": ["E1"], "fact_type": "RATE", "value": 42, "unit": "HOUR", "currency": "USD", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED", "explicit": True},
        ],
        "relationships": [{"relationship_id": "REL1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "RATE", "target_claim": "HRS", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["HRS", "RATE"], "relationship_ids": ["REL1"], "reason": "blended hourly rate applies to the support hours logged against the incident"}],
    })
    client = FakeLLMClient(response=response)
    ledger = _evidence(
        ("Approximately 850 support hours were spent addressing the incident.", EvidenceStatus.VERIFIED),
        ("Time-tracking records establish a blended rate of $42 per hour for this engagement.", EvidenceStatus.VERIFIED),
    )
    outcome = await analyze_financial_exposure_semantic("unseen finding", ledger, client=client)
    assert outcome is not None
    result, audit = outcome
    assert result.confirmed_impact.verified_gross_exposure == 35700.0
    assert result.currency == "USD"


# ---------------------------------------------------------------------------
# 3. Historical annualization, novel currency and period phrasing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unseen_historical_annualization_new_currency_and_period():
    """'Six similar occurrences were recorded over a six-month stretch, each
    averaging AED 8,000' -- proves annualization generalizes to a currency
    and a period phrasing absent from every other fixture."""
    response = json.dumps({
        "finding": {},
        "claims": [
            {"claim_id": "HQ", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 6, "unit": "OCCURRENCE", "population": "HISTORICAL", "evidence_status": "VERIFIED"},
            {"claim_id": "HR", "source_evidence_ids": ["E0"], "fact_type": "RATE", "value": 8000, "unit": "OCCURRENCE", "currency": "AED", "population": "HISTORICAL", "evidence_status": "VERIFIED"},
            {"claim_id": "HP", "source_evidence_ids": ["E0"], "fact_type": "OBSERVATION_PERIOD", "value": 6, "unit": "MONTH", "population": "HISTORICAL", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "R1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "HR", "target_claim": "HQ", "confidence": "HIGH", "evidence_basis": ["E0"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "ANNUALIZE", "inputs": ["HQ", "HR", "HP"], "relationship_ids": ["R1"], "reason": "annualize historical AED rate over the six-month observation window"}],
    })
    client = FakeLLMClient(response=response)
    ledger = _evidence(("Six similar occurrences were recorded over a six-month stretch, each averaging AED 8,000.", EvidenceStatus.VERIFIED))
    outcome = await analyze_financial_exposure_semantic("unseen finding", ledger, client=client)
    assert outcome is not None
    result, audit = outcome
    assert result.annualized_exposure.is_assessable is True
    assert result.annualized_exposure.annualized_amount == 96000.0
    assert result.annualized_exposure.currency == "AED"


# ---------------------------------------------------------------------------
# 4. Recovery REPORTED, gross VERIFIED -- net must not become VERIFIED
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unseen_recovery_reported_never_upgrades_net_to_verified():
    response = json.dumps({
        "finding": {},
        "claims": [
            {"claim_id": "GROSS", "source_evidence_ids": ["E0"], "fact_type": "AMOUNT", "value": 96500, "currency": "GBP", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED", "explicit": True},
            {"claim_id": "REC", "source_evidence_ids": ["E1"], "fact_type": "RECOVERY", "value": 15000, "currency": "GBP", "population": "RECOVERY", "evidence_status": "REPORTED", "explicit": True},
        ],
        "relationships": [{"relationship_id": "R1", "type": "RECOVERY_APPLIES_TO_GROSS", "source_claim": "REC", "target_claim": "GROSS", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "SUBTRACT", "inputs": ["GROSS", "REC"], "relationship_ids": ["R1"], "reason": "a supplier credit was reported against the verified exposure"}],
    })
    client = FakeLLMClient(response=response)
    ledger = _evidence(
        ("Verified invoices establish a GBP 96,500 exposure.", EvidenceStatus.VERIFIED),
        ("A supplier credit of GBP 15,000 was reportedly issued, pending finance confirmation.", EvidenceStatus.REPORTED),
    )
    outcome = await analyze_financial_exposure_semantic("unseen finding", ledger, client=client)
    assert outcome is not None
    result, audit = outcome
    assert result.confirmed_impact.verified_gross_exposure == 96500.0
    assert result.confirmed_impact.reported_recovery == 15000.0
    assert result.confirmed_impact.verified_recovery is None
    assert result.confirmed_impact.confirmed_net_loss is None


# ---------------------------------------------------------------------------
# 5. Cross-currency claims must never be silently combined
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unseen_cross_currency_exposure_and_remediation_not_combined():
    response = json.dumps({
        "finding": {},
        "claims": [
            {"claim_id": "EXP", "source_evidence_ids": ["E0"], "fact_type": "AMOUNT", "value": 12000, "currency": "USD", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED", "explicit": True},
            {"claim_id": "REMED", "source_evidence_ids": ["E1"], "fact_type": "REMEDIATION_COST", "value": 5000, "currency": "GBP", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED", "explicit": True},
        ],
        "relationships": [{"relationship_id": "R1", "type": "REMEDIATION_ENABLES_PAYBACK", "source_claim": "REMED", "target_claim": "EXP", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "DIVIDE", "inputs": ["REMED", "EXP"], "relationship_ids": ["R1"], "reason": "payback of remediation against exposure -- but currencies differ, no FX rate given"}],
    })
    interp = SemanticFindingInterpretation.model_validate(json.loads(response))
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    assert any(r.reason_code == "INCOMPATIBLE_CURRENCY" for r in outcome.rejected)
    # Each currency's own fact remains independently visible, never merged
    # into one figure and never silently converted.
    assert any(o.currency == "USD" and o.amount == 12000 for o in observations)
    assert any(o.currency == "GBP" and o.amount == 5000 for o in observations)
    assert not any(o.amount == 17000 for o in observations)


# ---------------------------------------------------------------------------
# 6. No currency stated anywhere -- refuse, never default to INR
# ---------------------------------------------------------------------------

def test_no_currency_stated_refuses_rather_than_defaulting_to_inr():
    interp = SemanticFindingInterpretation.model_validate({
        "claims": [
            {"claim_id": "Q1", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 500, "unit": "UNIT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "R1", "source_evidence_ids": ["E0"], "fact_type": "RATE", "value": 30, "unit": "UNIT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "REL1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "R1", "target_claim": "Q1", "confidence": "HIGH", "evidence_basis": ["E0"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["Q1", "R1"], "relationship_ids": ["REL1"], "reason": "no currency was ever stated for this rate"}],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=1)
    assert observations == []
    assert outcome.rejected[0].reason_code == "INCOMPATIBLE_CURRENCY"


def test_standalone_amount_with_no_currency_withheld_not_defaulted():
    interp = SemanticFindingInterpretation.model_validate({
        "claims": [
            {"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "REMEDIATION_COST", "value": 9000, "population": "REMEDIATION", "evidence_status": "VERIFIED"},
        ],
        "relationships": [],
        "calculation_proposals": [],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=1)
    assert observations == []
    assert any("C1" in d and "currency" in d for d in outcome.llm_disagreements)


# ---------------------------------------------------------------------------
# 7. Financial facts cannot become causal claims -- by schema, not by rule
# ---------------------------------------------------------------------------

def test_causal_relationship_type_label_is_inert_not_a_calculation_license():
    """`relationship_type` is free descriptive text (spec section 7), so a
    model may write "CAUSES" into it -- but that label carries zero
    operational meaning. It licenses no arithmetic on its own; a
    calculation still stands or falls purely on structural grounding
    (does the relationship connect the cited operands, are they numeric,
    compatible currency/dimensions). A stray causal label neither creates
    nor blocks a financial figure."""
    interp = SemanticFindingInterpretation.model_validate({
        "claims": [
            {"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 5, "population": "CURRENT_FINDING", "evidence_status": "REPORTED"},
            {"claim_id": "C2", "source_evidence_ids": ["E0"], "fact_type": "RATE", "value": 1200, "unit": "hour", "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "REPORTED"},
        ],
        "relationships": [{
            "relationship_id": "R1", "relationship_type": "CAUSES",
            "source_claim": "C1", "target_claim": "C2", "confidence": "HIGH", "evidence_basis": ["E0"],
        }],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["C1", "C2"], "relationship_ids": ["R1"], "reason": "hours x rate"}],
    })
    # The causal label neither blocked nor authorised anything: the
    # calculation is accepted because C1/C2 are numeric, currency- and
    # dimension-compatible, and R1 connects exactly them.
    observations, outcome = validate_and_materialize(interp, evidence_count=1)
    assert not outcome.rejected, outcome.rejected
    assert len(observations) == 1
    assert observations[0].event_count == 5
    assert observations[0].unit_amount == 1200
    assert interp.relationships[0].relationship_type == "CAUSES"
    assert interp.relationships[0].is_conflict is False
