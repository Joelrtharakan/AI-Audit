"""Provider-formatting robustness for the LLM financial semantic layer.

Architecture spec sections 15 & 22-25: a harmless provider-formatting
difference (enum casing, key spelling, a nested `{value, currency}` object
instead of the flat fields, a numeric string, a bare string where a list
is expected) MUST NOT be misread as a semantically invalid interpretation.
The canonical semantics that reach the validator/calculator must be
identical regardless of which surface form the provider used.

None of these fixtures introduce a NEW fact, relationship, cost factor, or
evidence status -- each is the SAME interpretation restated in a different
provider's house style.
"""

from __future__ import annotations

import json

import pytest

from app.financial.provider_normalization import normalize_to_canonical
from app.financial.semantic_models import SemanticFindingInterpretation
from app.models.agent import EvidenceItem, EvidenceStatus
from app.services.semantic_evidence_interpreter import interpret_evidence_semantically


class _FakeLLM:
    def __init__(self, response: str):
        self.response = response

    async def chat_completion(self, messages, **kwargs):
        return self.response


_LEDGER = [
    EvidenceItem(claim="1,000 units required rework at INR 250 per unit.", status=EvidenceStatus.VERIFIED, source="C1"),
]

# The "house style A" reference: exactly the canonical schema.
_CANONICAL = {
    "finding": {"interpretation_confidence": "HIGH"},
    "claims": [
        {"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 1000, "unit": "unit", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        {"claim_id": "C2", "source_evidence_ids": ["E0"], "fact_type": "RATE", "value": 250, "unit": "unit", "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
    ],
    "relationships": [
        {"relationship_id": "R1", "relationship_type": "per-unit rate", "source_claim": "C2", "target_claim": "C1", "confidence": "HIGH", "evidence_basis": ["E0"]},
    ],
    "calculation_proposals": [
        {"calculation_id": "K1", "operation": "MULTIPLY", "inputs": ["C1", "C2"], "relationship_ids": ["R1"], "proposed_result_value": 250000, "proposed_result_currency": "INR"},
    ],
    "cost_factor": {"selected_factor": "REWORK_COST", "supporting_claim_ids": ["C1", "C2"], "confidence": "HIGH"},
    "financial_relevance": "CONFIRMED",
    "quantification": {"status": "QUANTIFIED"},
}

# "house style B": lowercase enums, `from`/`to` endpoints, `type` alias,
# nested proposed_result, numeric string with grouping comma, bare-string
# evidence id, `factor`/`supporting_claims` aliases, `relevance` alias.
_HOUSE_B = {
    "finding": {"interpretation_confidence": "high"},
    "claims": [
        {"id": "C1", "source_evidence": "E0", "fact_type": "quantity", "value": "1,000", "unit": "unit", "population": "current_finding", "evidence_status": "verified"},
        {"id": "C2", "evidence_ids": ["E0"], "fact_type": "rate", "value": 250, "unit": "INR/unit", "currency": "INR", "population": "current_finding", "evidence_status": "verified"},
    ],
    "relationships": [
        {"relationship_id": "R1", "type": "per unit rate", "from": "C2", "to": "C1", "confidence": "high", "evidence_basis": ["E0"]},
    ],
    "calculation_proposals": [
        {"calc_id": "K1", "operation": "multiply", "operands": ["C1", "C2"], "relationship": ["R1"], "proposed_result": {"value": "250000", "currency": "INR"}},
    ],
    "cost_factor": {"factor": "rework_cost", "supporting_claims": ["C1", "C2"], "confidence": "high"},
    "relevance": "confirmed",
    "quantification": {"status": "quantifiable"},
}


def _canonical_semantics(interp: SemanticFindingInterpretation) -> dict:
    """The provider-independent facts that must match across house styles."""
    return {
        "claims": sorted(
            (c.claim_id, c.fact_type, c.value, c.currency, c.population, c.evidence_status)
            for c in interp.claims
        ),
        "relationships": sorted(
            (r.source_claim, r.target_claim, r.confidence, r.is_conflict) for r in interp.relationships
        ),
        "calculations": sorted(
            (c.operation, tuple(sorted(c.inputs)), tuple(sorted(c.relationship_ids))) for c in interp.calculation_proposals
        ),
        "cost_factor": interp.cost_factor.selected_factor,
        "cost_factor_support": sorted(interp.cost_factor.supporting_claim_ids),
        "financial_relevance": interp.financial_relevance,
        "quantification": interp.quantification.status,
    }


@pytest.mark.asyncio
async def test_house_style_b_parses_and_matches_canonical_semantics():
    status_a, interp_a = await interpret_evidence_semantically("x", _LEDGER, client=_FakeLLM(json.dumps(_CANONICAL)))
    status_b, interp_b = await interpret_evidence_semantically("x", _LEDGER, client=_FakeLLM(json.dumps(_HOUSE_B)))

    assert status_a == "OK" and interp_a is not None
    assert status_b == "OK", "a pure formatting variant must not be treated as LLM_INVALID"
    assert _canonical_semantics(interp_a) == _canonical_semantics(interp_b)


def test_nested_proposed_result_is_flattened_losslessly():
    out = normalize_to_canonical({
        "calculation_proposals": [
            {"calculation_id": "K1", "operation": "MULTIPLY", "inputs": ["C1", "C2"], "proposed_result": {"value": 250000, "currency": "INR"}},
        ],
    })
    calc = out["calculation_proposals"][0]
    assert calc["proposed_result_value"] == 250000
    assert calc["proposed_result_currency"] == "INR"
    assert "proposed_result" not in calc


def test_relationship_missing_endpoint_is_dropped_not_whole_interpretation():
    """One malformed relationship must not sink every valid claim/calc."""
    out = normalize_to_canonical({
        "claims": [{"claim_id": "C1", "fact_type": "AMOUNT", "value": 500}],
        "relationships": [
            {"relationship_id": "R1", "relationship_type": "x"},  # no endpoints
            {"relationship_id": "R2", "relationship_type": "y", "source_claim": "C1", "target_claim": "C1"},
        ],
    })
    assert [r["relationship_id"] for r in out["relationships"]] == ["R2"]
    SemanticFindingInterpretation.model_validate(out)  # must not raise


def test_normalizer_invents_nothing():
    """An empty provider payload stays empty -- no fabricated factor,
    relevance, claim, or calculation."""
    out = normalize_to_canonical({"claims": [], "relationships": []})
    interp = SemanticFindingInterpretation.model_validate(out)
    assert interp.claims == []
    assert interp.cost_factor.selected_factor == "NOT_ESTABLISHED"
    assert interp.financial_relevance is None
    assert interp.calculation_proposals == []


def test_generic_type_key_not_rewritten_outside_relationships():
    """The `type` -> `relationship_type` alias is relationship-scoped: a
    `type` key on a claim (some providers add one) is left untouched, not
    hoisted into a relationship field."""
    out = normalize_to_canonical({
        "claims": [{"claim_id": "C1", "fact_type": "AMOUNT", "value": 1, "type": "monetary"}],
        "relationships": [],
    })
    assert "relationship_type" not in out["claims"][0]
