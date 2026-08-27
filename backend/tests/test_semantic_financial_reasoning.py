"""Regression coverage for the LLM semantic-understanding financial layer
(app.financial.semantic_models / relationship_validator / semantic_engine /
app.services.semantic_evidence_interpreter).

No real LLM/network calls are made in this suite (network-gated, consistent
with every prior pass in this engagement) -- a `FakeLLMClient` stands in for
the provider, returning hand-authored JSON that simulates what a real LLM
would plausibly produce for each scenario. This is what makes the
GENERALIZATION claim testable at all: the same downstream validator +
deterministic calculator is exercised against structurally-different JSON
for each of several UNSEEN paraphrasings of the same underlying fact, and
all must produce the identical authoritative number -- proving the accuracy
comes from the validator/calculator being wording-agnostic, not from any
regex matching a specific phrase.

The single most important property under test throughout: the LLM's
`proposed_result_value` is NEVER what ends up in the rendered result --
only the independently-recomputed value from `_build_result_from_
observations` (the exact same function the regex-extraction engine calls)
is authoritative.
"""

from __future__ import annotations

import json

import pytest

from app.financial.relationship_validator import validate_and_materialize
from app.financial.semantic_engine import analyze_financial_exposure_semantic
from app.financial.semantic_models import SemanticFindingInterpretation
from app.models.agent import EvidenceItem, EvidenceStatus
from app.services.semantic_evidence_interpreter import interpret_evidence_semantically


class FakeLLMClient:
    """Stands in for a real LLM provider -- returns a fixed JSON string
    (or raises, for failure-mode tests) regardless of the prompt."""

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
# 1. Semantic generalization: five UNSEEN paraphrasings of the same fact
# ---------------------------------------------------------------------------

_QTY_RATE_PARAPHRASES = {
    "A_plant_reworked": json.dumps({
        "finding": {"deviation": "rework", "interpretation_confidence": "HIGH"},
        "claims": [
            {"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 1000, "unit": "UNIT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED", "explicit": True},
            {"claim_id": "C2", "source_evidence_ids": ["E0"], "fact_type": "RATE", "value": 250, "unit": "UNIT", "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED", "explicit": True},
        ],
        "relationships": [{"relationship_id": "R1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "C2", "target_claim": "C1", "confidence": "HIGH", "evidence_basis": ["E0"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["C1", "C2"], "relationship_ids": ["R1"], "proposed_result_value": 250000, "proposed_result_currency": "INR", "reason": "rate applies to quantity"}],
    }),
    "B_one_thousand_affected": json.dumps({
        "finding": {"interpretation_confidence": "HIGH"},
        "claims": [
            {"claim_id": "Q1", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 1000, "unit": "UNIT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "RATE1", "source_evidence_ids": ["E0"], "fact_type": "RATE", "value": 250, "unit": "UNIT", "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "REL_A", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "RATE1", "target_claim": "Q1", "confidence": "HIGH", "evidence_basis": ["E0"]}],
        "calculation_proposals": [{"calculation_id": "X1", "operation": "MULTIPLY", "inputs": ["Q1", "RATE1"], "relationship_ids": ["REL_A"], "reason": "average charge applies to the stated population"}],
    }),
    "C_each_of_the_reworked": json.dumps({
        "finding": {},
        "claims": [
            {"claim_id": "c_qty", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 1000, "unit": "UNIT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "c_rate", "source_evidence_ids": ["E0"], "fact_type": "RATE", "value": 250, "unit": "UNIT", "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "r1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "c_rate", "target_claim": "c_qty", "confidence": "HIGH", "evidence_basis": ["E0"]}],
        "calculation_proposals": [{"calculation_id": "calc_a", "operation": "MULTIPLY", "inputs": ["c_qty", "c_rate"], "relationship_ids": ["r1"], "reason": "per-unit amount incurred for each reworked unit"}],
    }),
    "D_rework_population": json.dumps({
        "finding": {},
        "claims": [
            {"claim_id": "pop", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 1000.0, "unit": "UNIT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "avg", "source_evidence_ids": ["E0"], "fact_type": "RATE", "value": 250.0, "unit": "UNIT", "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "link", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "avg", "target_claim": "pop", "confidence": "HIGH", "evidence_basis": ["E0"]}],
        "calculation_proposals": [{"calculation_id": "c1", "operation": "MULTIPLY", "inputs": ["pop", "avg"], "relationship_ids": ["link"], "reason": "average unit cost applied to rework population"}],
    }),
    "E_production_records_establish": json.dumps({
        "finding": {},
        "claims": [
            {"claim_id": "n_units", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 1000, "unit": "UNIT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "unit_rate", "source_evidence_ids": ["E0"], "fact_type": "RATE", "value": 250, "unit": "UNIT", "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "established_rate", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "unit_rate", "target_claim": "n_units", "confidence": "HIGH", "evidence_basis": ["E0"]}],
        "calculation_proposals": [{"calculation_id": "calc", "operation": "MULTIPLY", "inputs": ["n_units", "unit_rate"], "relationship_ids": ["established_rate"], "reason": "records establish a rework rate for the affected population"}],
    }),
}


@pytest.mark.parametrize("name,response", list(_QTY_RATE_PARAPHRASES.items()))
@pytest.mark.asyncio
async def test_unseen_paraphrase_produces_identical_authoritative_result(name, response):
    """Five structurally-different LLM outputs (simulating five different
    UNSEEN sentence phrasings of '1,000 units at INR 250/unit') must all
    produce the identical, independently-calculated INR 250,000 -- proving
    the result comes from the deterministic calculator, not from matching
    any particular wording."""
    client = FakeLLMClient(response=response)
    ledger = _evidence(("some paraphrase of the rework finding", EvidenceStatus.VERIFIED))
    outcome = await analyze_financial_exposure_semantic("finding text", ledger, client=client)
    assert outcome is not None, name
    result, audit = outcome
    assert result.confirmed_impact.verified_gross_exposure == 250000.0, name


# ---------------------------------------------------------------------------
# 2. LLM arithmetic is never trusted -- disagreement is recorded, not used
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_wrong_arithmetic_never_becomes_the_displayed_number():
    """The exact case from the task spec: LLM proposes 1000 x 250 = 240000
    (wrong). The deterministic calculator must independently compute and
    display 250000, and the disagreement must be recorded for audit."""
    response = json.dumps({
        "finding": {},
        "claims": [
            {"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 1000, "unit": "UNIT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "C2", "source_evidence_ids": ["E0"], "fact_type": "RATE", "value": 250, "unit": "UNIT", "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "R1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "C2", "target_claim": "C1", "confidence": "HIGH", "evidence_basis": ["E0"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["C1", "C2"], "relationship_ids": ["R1"], "proposed_result_value": 240000, "reason": "wrong LLM arithmetic"}],
    })
    client = FakeLLMClient(response=response)
    ledger = _evidence(("x", EvidenceStatus.VERIFIED))
    result, audit = await analyze_financial_exposure_semantic("x", ledger, client=client)
    assert result.confirmed_impact.verified_gross_exposure == 250000.0
    assert any("240000" in d for d in audit.outcome.llm_disagreements)


# ---------------------------------------------------------------------------
# 3. Relationship semantics: current/historical must never mix
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_historical_quantity_never_multiplied_by_current_rate():
    response = json.dumps({
        "finding": {},
        "claims": [
            {"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 1000, "population": "HISTORICAL", "evidence_status": "VERIFIED", "temporal_scope": "last year"},
            {"claim_id": "C2", "source_evidence_ids": ["E1"], "fact_type": "RATE", "value": 250, "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED", "temporal_scope": "current"},
        ],
        "relationships": [{"relationship_id": "R1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "C2", "target_claim": "C1", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["C1", "C2"], "relationship_ids": ["R1"], "reason": "incorrectly proposed cross-population link"}],
    })
    client = FakeLLMClient(response=response)
    ledger = _evidence(("historical quantity", EvidenceStatus.VERIFIED), ("current rate", EvidenceStatus.VERIFIED))
    outcome = await analyze_financial_exposure_semantic("x", ledger, client=client)
    # Nothing survives validation (POPULATION_MISMATCH) -> falls back (None)
    assert outcome is None


def test_relationship_validator_rejects_population_mismatch_directly():
    interp = SemanticFindingInterpretation.model_validate({
        "claims": [
            {"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 10, "population": "HISTORICAL", "evidence_status": "VERIFIED"},
            {"claim_id": "C2", "source_evidence_ids": ["E1"], "fact_type": "RATE", "value": 20000, "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "R1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "C2", "target_claim": "C1", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["C1", "C2"], "relationship_ids": ["R1"], "reason": "10 historical failures x current cost each"}],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    assert observations == []
    assert outcome.rejected[0].reason_code == "POPULATION_MISMATCH"


# ---------------------------------------------------------------------------
# 4. Evidence status is never upgraded by interpretation confidence
# ---------------------------------------------------------------------------

def test_high_interpretation_confidence_never_upgrades_evidence_status():
    interp = SemanticFindingInterpretation.model_validate({
        "finding": {"interpretation_confidence": "HIGH"},
        "claims": [
            {"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "AMOUNT", "value": 50000, "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "UNVERIFIED", "explicit": True},
        ],
        "relationships": [],
        "calculation_proposals": [],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=1)
    assert len(observations) == 1
    assert observations[0].verification_status == "UNVERIFIED"


# ---------------------------------------------------------------------------
# 5. Recovery / remediation as standalone facts (no relationship required)
# ---------------------------------------------------------------------------

def test_reported_recovery_materialized_without_a_relationship():
    interp = SemanticFindingInterpretation.model_validate({
        "claims": [
            {"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "RECOVERY", "value": 40000, "currency": "INR", "population": "RECOVERY", "evidence_status": "REPORTED"},
        ],
        "relationships": [],
        "calculation_proposals": [],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=1)
    assert len(observations) == 1
    assert observations[0].amount_type.value == "RECOVERY"
    assert observations[0].verification_status == "REPORTED"


# ---------------------------------------------------------------------------
# 6. Malformed / adversarial LLM output must fail safely
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_json_fails_honestly():
    client = FakeLLMClient(response="{not valid json")
    ledger = _evidence(("x", EvidenceStatus.VERIFIED))
    status, interpretation = await interpret_evidence_semantically("x", ledger, client=client)
    assert status == "LLM_INVALID"
    assert interpretation is None


@pytest.mark.asyncio
async def test_llm_connection_error_fails_honestly():
    client = FakeLLMClient(raise_exc=ConnectionError("provider unreachable"))
    ledger = _evidence(("x", EvidenceStatus.VERIFIED))
    status, interpretation = await interpret_evidence_semantically("x", ledger, client=client)
    assert status == "LLM_UNAVAILABLE"
    assert interpretation is None


def test_calculation_referencing_nonexistent_claim_rejected():
    interp = SemanticFindingInterpretation.model_validate({
        "claims": [{"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 10, "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"}],
        "relationships": [],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["C1", "GHOST"], "relationship_ids": [], "reason": "fabricated claim"}],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=1)
    assert observations == []
    assert outcome.rejected[0].reason_code == "UNKNOWN_CLAIM"


def test_calculation_citing_fabricated_evidence_id_rejected():
    interp = SemanticFindingInterpretation.model_validate({
        "claims": [
            {"claim_id": "C1", "source_evidence_ids": ["E99"], "fact_type": "QUANTITY", "value": 10, "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "C2", "source_evidence_ids": ["E0"], "fact_type": "RATE", "value": 100, "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "R1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "C2", "target_claim": "C1", "confidence": "HIGH", "evidence_basis": ["E0"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["C1", "C2"], "relationship_ids": ["R1"], "reason": "x"}],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=1)
    assert observations == []
    assert outcome.rejected[0].reason_code == "MISSING_PROVENANCE"


def test_incompatible_currency_rejected():
    interp = SemanticFindingInterpretation.model_validate({
        "claims": [
            {"claim_id": "GROSS", "source_evidence_ids": ["E0"], "fact_type": "AMOUNT", "value": 250000, "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "REC", "source_evidence_ids": ["E1"], "fact_type": "RECOVERY", "value": 40000, "currency": "USD", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "R1", "type": "RECOVERY_APPLIES_TO_GROSS", "source_claim": "REC", "target_claim": "GROSS", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "SUBTRACT", "inputs": ["GROSS", "REC"], "relationship_ids": ["R1"], "reason": "cross-currency subtraction"}],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    assert not any(o.amount_type.value == "DIRECT_LOSS" and o.amount == 210000 for o in observations)
    assert any(r.reason_code == "INCOMPATIBLE_CURRENCY" for r in outcome.rejected)
    # both currencies remain independently preserved as standalone facts
    # (each is self-sufficient on its own), never silently combined.
    assert any(o.currency == "INR" and o.amount == 250000 for o in observations)
    assert any(o.currency == "USD" and o.amount == 40000 for o in observations)


def test_low_confidence_relationship_treated_as_ambiguous_never_calculated():
    interp = SemanticFindingInterpretation.model_validate({
        "claims": [
            {"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 10, "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "C2", "source_evidence_ids": ["E1"], "fact_type": "RATE", "value": 100, "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "R1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "C2", "target_claim": "C1", "confidence": "LOW", "evidence_basis": ["E0", "E1"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["C1", "C2"], "relationship_ids": ["R1"], "reason": "uncertain link"}],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    assert observations == []
    assert outcome.rejected[0].reason_code == "AMBIGUOUS_RELATIONSHIP"


def test_contradicted_claim_never_participates():
    interp = SemanticFindingInterpretation.model_validate({
        "claims": [
            {"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 10, "population": "CURRENT_FINDING", "evidence_status": "CONTRADICTED"},
            {"claim_id": "C2", "source_evidence_ids": ["E1"], "fact_type": "RATE", "value": 100, "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "R1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "C2", "target_claim": "C1", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["C1", "C2"], "relationship_ids": ["R1"], "reason": "x"}],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    assert observations == []
    assert outcome.rejected[0].reason_code == "EVIDENCE_STATUS_INELIGIBLE"


def test_conflicting_claims_never_silently_combined():
    interp = SemanticFindingInterpretation.model_validate({
        "claims": [
            {"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "AMOUNT", "value": 20000, "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "C2", "source_evidence_ids": ["E1"], "fact_type": "AMOUNT", "value": 30000, "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [
            {"relationship_id": "RCONF", "type": "CONFLICTS_WITH", "source_claim": "C1", "target_claim": "C2", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]},
            {"relationship_id": "RSUM", "type": "CORROBORATES", "source_claim": "C1", "target_claim": "C2", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]},
        ],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "SUM", "inputs": ["C1", "C2"], "relationship_ids": ["RSUM"], "reason": "sum two conflicting amounts"}],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    assert not any(o.amount == 50000 for o in observations)
    assert any(r.reason_code == "CONFLICTING_CLAIMS" for r in outcome.rejected)


def test_unsupported_operation_without_materialization_rule_rejected():
    interp = SemanticFindingInterpretation.model_validate({
        "claims": [
            {"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "PERCENTAGE", "value": 40, "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "R1", "type": "CORROBORATES", "source_claim": "C1", "target_claim": "C1", "confidence": "HIGH", "evidence_basis": ["E0"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "SUM", "inputs": ["C1"], "relationship_ids": ["R1"], "reason": "a bare percentage has no direct materialization"}],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=1)
    assert observations == []
    assert outcome.rejected[0].reason_code == "UNSUPPORTED_OPERATION"


# ---------------------------------------------------------------------------
# 6b. ANNUALIZE and DIVIDE (payback) operations, not just MULTIPLY
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_annualize_operation_computes_historical_annual_rate():
    """Regression guard: an OBSERVATION_PERIOD claim's own unit (a time
    SPAN, e.g. 'MONTH') must never be compared for compatibility against
    the rate/quantity's occurrence unit -- they are different dimensions
    entirely. Caught by this test after the first implementation
    incorrectly rejected this exact shape as INCOMPATIBLE_UNITS."""
    response = json.dumps({
        "finding": {},
        "claims": [
            {"claim_id": "HQ", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 10, "unit": "OCCURRENCE", "population": "HISTORICAL", "evidence_status": "VERIFIED"},
            {"claim_id": "HR", "source_evidence_ids": ["E0"], "fact_type": "RATE", "value": 5000, "unit": "OCCURRENCE", "currency": "INR", "population": "HISTORICAL", "evidence_status": "VERIFIED"},
            {"claim_id": "HP", "source_evidence_ids": ["E0"], "fact_type": "OBSERVATION_PERIOD", "value": 12, "unit": "MONTH", "population": "HISTORICAL", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "R1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "HR", "target_claim": "HQ", "confidence": "HIGH", "evidence_basis": ["E0"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "ANNUALIZE", "inputs": ["HQ", "HR", "HP"], "relationship_ids": ["R1"], "reason": "annualize historical rate over 12 months"}],
    })
    client = FakeLLMClient(response=response)
    ledger = _evidence(("x", EvidenceStatus.VERIFIED))
    outcome = await analyze_financial_exposure_semantic("x", ledger, client=client)
    assert outcome is not None
    result, audit = outcome
    assert result.annualized_exposure.is_assessable is True
    assert result.annualized_exposure.annualized_amount == 50000.0


@pytest.mark.asyncio
async def test_remediation_payback_computed_against_verified_annualized_exposure():
    response = json.dumps({
        "finding": {},
        "claims": [
            {"claim_id": "HQ", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 10, "unit": "OCCURRENCE", "population": "HISTORICAL", "evidence_status": "VERIFIED"},
            {"claim_id": "HR", "source_evidence_ids": ["E0"], "fact_type": "RATE", "value": 5000, "unit": "OCCURRENCE", "currency": "INR", "population": "HISTORICAL", "evidence_status": "VERIFIED"},
            {"claim_id": "HP", "source_evidence_ids": ["E0"], "fact_type": "OBSERVATION_PERIOD", "value": 12, "unit": "MONTH", "population": "HISTORICAL", "evidence_status": "VERIFIED"},
            {"claim_id": "REMED", "source_evidence_ids": ["E1"], "fact_type": "REMEDIATION_COST", "value": 25000, "currency": "INR", "population": "REMEDIATION", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "R1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "HR", "target_claim": "HQ", "confidence": "HIGH", "evidence_basis": ["E0"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "ANNUALIZE", "inputs": ["HQ", "HR", "HP"], "relationship_ids": ["R1"], "reason": "annualize"}],
    })
    client = FakeLLMClient(response=response)
    ledger = _evidence(("historical", EvidenceStatus.VERIFIED), ("remediation", EvidenceStatus.VERIFIED))
    outcome = await analyze_financial_exposure_semantic("x", ledger, client=client)
    assert outcome is not None
    result, audit = outcome
    assert result.annualized_exposure.annualized_amount == 50000.0
    assert result.capa_economics.remediation_cost == 25000.0
    assert result.capa_economics.is_assessable is True
    assert result.capa_economics.indicative_payback_years == 0.5


# ---------------------------------------------------------------------------
# 7. The current adversarial finding (regression test, NOT the design target)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adversarial_finding_c1_to_c6_full_semantic_interpretation():
    """C1-C6 from the task spec. Verifies gross=250000 VERIFIED-eligible,
    recovery=40000 REPORTED (never upgraded, never dropped), historical
    population isolated (never mixed into current gross), remediation
    75000 VERIFIED, and no cross-population contamination."""
    response = json.dumps({
        "finding": {
            "deviation": "packaging failures resulting in rework",
            "affected_object": "packaging line",
            "interpretation_confidence": "HIGH",
        },
        "claims": [
            {"claim_id": "C1_QTY", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 1000, "unit": "UNIT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED", "explicit": True},
            {"claim_id": "C2_RATE", "source_evidence_ids": ["E1"], "fact_type": "RATE", "value": 250, "unit": "UNIT", "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED", "explicit": True},
            {"claim_id": "C3_REC", "source_evidence_ids": ["E2"], "fact_type": "RECOVERY", "value": 40000, "currency": "INR", "population": "RECOVERY", "evidence_status": "REPORTED", "explicit": True},
            {"claim_id": "C4_HIST_QTY", "source_evidence_ids": ["E3"], "fact_type": "QUANTITY", "value": 10, "unit": "OCCURRENCE", "population": "HISTORICAL", "temporal_scope": "previous 12 months", "evidence_status": "VERIFIED", "explicit": True},
            {"claim_id": "C4_HIST_PERIOD", "source_evidence_ids": ["E3"], "fact_type": "OBSERVATION_PERIOD", "value": 12, "unit": "MONTH", "population": "HISTORICAL", "evidence_status": "VERIFIED", "explicit": True},
            {"claim_id": "C6_REMED", "source_evidence_ids": ["E5"], "fact_type": "REMEDIATION_COST", "value": 75000, "currency": "INR", "population": "REMEDIATION", "evidence_status": "VERIFIED", "explicit": True},
        ],
        "relationships": [
            {"relationship_id": "R1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "C2_RATE", "target_claim": "C1_QTY", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]},
        ],
        "calculation_proposals": [
            {"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["C1_QTY", "C2_RATE"], "relationship_ids": ["R1"], "proposed_result_value": 250000, "proposed_result_currency": "INR", "reason": "verified per-unit rework rate applies to the 1,000 current-finding units"},
        ],
    })
    client = FakeLLMClient(response=response)
    ledger = _evidence(
        ("During January-June 2026, five confirmed packaging failures resulted in 1,000 units requiring rework. VERIFIED.", EvidenceStatus.VERIFIED),
        ("Production records verify an average rework cost of INR 250 per unit. VERIFIED.", EvidenceStatus.VERIFIED),
        ("Finance records confirm INR 40,000 was recovered from the supplier. REPORTED.", EvidenceStatus.REPORTED),
        ("Historical records show the same failure occurred 10 times during the previous 12 months. VERIFIED.", EvidenceStatus.VERIFIED),
        ("Current supplier contract remains active. VERIFIED.", EvidenceStatus.VERIFIED),
        ("Proposed supplier-control improvement costs INR 75,000. VERIFIED.", EvidenceStatus.VERIFIED),
    )
    result, audit = await analyze_financial_exposure_semantic("adversarial finding", ledger, client=client)

    # Gross exposure: 1000 x 250, current-finding only.
    assert result.confirmed_impact.verified_gross_exposure == 250000.0
    # Recovery preserved and correctly labeled REPORTED, never upgraded.
    assert result.confirmed_impact.reported_recovery == 40000.0
    assert result.confirmed_impact.verified_recovery is None
    # Net loss never fabricated from a non-verified recovery.
    assert result.confirmed_impact.confirmed_net_loss is None
    # Remediation preserved as its own population, VERIFIED, never
    # contaminating current gross exposure.
    assert result.capa_economics.remediation_cost == 75000.0
    assert result.capa_economics.remediation_cost_status == "VERIFIED"
    assert result.confirmed_impact.verified_gross_exposure == 250000.0  # unchanged by remediation


# ---------------------------------------------------------------------------
# 9. Provider-independent canonical conformance + honest-failure contract
# ---------------------------------------------------------------------------

def _duplicate_payment_payload_provider_a() -> str:
    """One plausible shape a provider returns for "Three duplicate payments
    of INR 50,000 each were reported." """
    return json.dumps({
        "finding": {"interpretation_confidence": "HIGH"},
        "financial_relevance": "MATERIAL",
        "claims": [
            {"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY",
             "value": 3, "unit": "payment", "population": "CURRENT_FINDING", "evidence_status": "REPORTED"},
            {"claim_id": "C2", "source_evidence_ids": ["E0"], "fact_type": "AMOUNT",
             "value": 50000, "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "REPORTED"},
        ],
        "relationships": [{"relationship_id": "R1", "relationship_type": "each payment carries the stated amount",
                            "source_claim": "C1", "target_claim": "C2", "confidence": "HIGH", "evidence_basis": ["E0"]}],
        "calculation_proposals": [{"calculation_id": "CAL1", "operation": "MULTIPLY", "inputs": ["C1", "C2"],
                                    "relationship_ids": ["R1"], "proposed_result_value": 150000, "reason": "count x per-payment amount"}],
        "cost_factor": {"selected_factor": "DUPLICATE_PAYMENT", "supporting_claim_ids": ["C1", "C2"],
                         "confidence": "HIGH", "rationale": "reported duplicate payments"},
    })


def _duplicate_payment_payload_provider_b() -> str:
    """A structurally different shape (different ids, key order, an
    operation-named relationship_type label, RATE fact_type instead of
    AMOUNT) another provider might return for the SAME fact."""
    return json.dumps({
        "cost_factor": {"rationale": "duplicate payments per finance", "confidence": "HIGH",
                         "supporting_claim_ids": ["q", "amt"], "selected_factor": "DUPLICATE_PAYMENT"},
        "calculation_proposals": [{"reason": "each", "operation": "MULTIPLY", "inputs": ["q", "amt"],
                                    "relationship_ids": ["rel0"], "calculation_id": "K0"}],
        "relationships": [{"evidence_basis": ["E0"], "confidence": "HIGH", "target_claim": "amt",
                            "source_claim": "q", "relationship_type": "MULTIPLY", "relationship_id": "rel0"}],
        "claims": [
            {"evidence_status": "REPORTED", "population": "CURRENT_FINDING", "value": 3, "fact_type": "QUANTITY",
             "unit": "payment", "source_evidence_ids": ["E0"], "claim_id": "q"},
            {"evidence_status": "REPORTED", "population": "CURRENT_FINDING", "value": 50000, "currency": "INR",
             "fact_type": "RATE", "unit": "payment", "source_evidence_ids": ["E0"], "claim_id": "amt"},
        ],
        "financial_relevance": "MATERIAL",
    })


@pytest.mark.parametrize("payload_fn", [
    _duplicate_payment_payload_provider_a,
    _duplicate_payment_payload_provider_b,
])
@pytest.mark.asyncio
async def test_materially_equivalent_providers_converge_to_same_canonical_result(payload_fn):
    client = FakeLLMClient(response=payload_fn())
    ledger = _evidence(("Finance records report three duplicate payments of INR 50,000 each.", EvidenceStatus.REPORTED))
    result, audit = await analyze_financial_exposure_semantic("finding", ledger, client=client)

    # Canonical semantic convergence -- independent of provider wording:
    assert result.confirmed_impact.financial_factor == "DUPLICATE_PAYMENT"
    assert result.confirmed_impact.reported_financial_exposure == 150000.0
    # Evidence status NEVER upgraded: REPORTED in -> REPORTED out.
    assert result.confirmed_impact.verified_gross_exposure is None
    assert result.reasoning_source == "LLM_SEMANTIC"
    assert result.financial_semantic_status == "OK"

    # Auditable calculation trace, with the LLM figure recorded but the
    # deterministic executor value authoritative and in agreement here.
    assert audit.outcome.traces
    tr = audit.outcome.traces[0]
    assert tr.operation == "MULTIPLY"
    assert tr.executor_result == 150000.0
    assert tr.evidence_status == "REPORTED"
    assert tr.disagreement is None


@pytest.mark.asyncio
async def test_llm_numeric_result_is_never_authoritative_trace_records_disagreement():
    payload = json.loads(_duplicate_payment_payload_provider_a())
    payload["calculation_proposals"][0]["proposed_result_value"] = 999999  # wrong on purpose
    client = FakeLLMClient(response=json.dumps(payload))
    ledger = _evidence(("Three duplicate payments of INR 50,000 each were reported.", EvidenceStatus.REPORTED))
    result, audit = await analyze_financial_exposure_semantic("finding", ledger, client=client)

    assert result.confirmed_impact.reported_financial_exposure == 150000.0  # executor, not 999999
    tr = audit.outcome.traces[0]
    assert tr.llm_proposed_result == 999999
    assert tr.executor_result == 150000.0
    assert tr.disagreement is not None


@pytest.mark.asyncio
async def test_honest_failure_when_provider_unavailable_no_fabricated_number():
    client = FakeLLMClient(raise_exc=ConnectionError("provider down"))
    ledger = _evidence(("Three duplicate payments of INR 50,000 each were reported.", EvidenceStatus.REPORTED))
    result, audit = await analyze_financial_exposure_semantic("finding", ledger, client=client)

    assert result is not None, "must fail honestly, not silently defer to a regex analyzer"
    assert result.financial_semantic_status == "LLM_UNAVAILABLE"
    assert result.confirmed_impact.reported_financial_exposure is None
    assert result.confirmed_impact.verified_gross_exposure is None
    assert result.confirmed_impact.financial_factor in (None, "NOT_ESTABLISHED")


@pytest.mark.asyncio
async def test_cost_factor_identified_without_amount_is_unquantified_not_fabricated():
    payload = json.dumps({
        "finding": {"interpretation_confidence": "HIGH"},
        "financial_relevance": "MATERIAL",
        "claims": [
            {"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY",
             "value": 18, "unit": "hour", "population": "CURRENT_FINDING", "evidence_status": "REPORTED"},
        ],
        "relationships": [],
        "calculation_proposals": [],
        "cost_factor": {"selected_factor": "REWORK_COST", "supporting_claim_ids": ["C1"],
                         "confidence": "HIGH", "rationale": "additional rework hours"},
        "quantification": {"status": "UNQUANTIFIED", "blocker": "No labor rate stated.", "missing_inputs": ["labor rate"]},
    })
    client = FakeLLMClient(response=payload)
    ledger = _evidence(("18 additional rework hours were required.", EvidenceStatus.REPORTED))
    result, audit = await analyze_financial_exposure_semantic("finding", ledger, client=client)

    assert result.confirmed_impact.financial_factor == "REWORK_COST"
    assert result.confirmed_impact.quantification_status == "UNQUANTIFIED"
    assert result.confirmed_impact.reported_financial_exposure is None
    assert result.confirmed_impact.verified_gross_exposure is None
    assert result.financial_semantic_status == "OK"
