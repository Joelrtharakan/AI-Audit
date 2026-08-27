"""Regression for a real defect surfaced by a live Ollama call: a quantity
claim naturally stated in the plural ("450 shipments") and a rate claim
naturally stated with a singular denominator ("USD 22 per shipment", i.e.
unit "USD per shipment") were rejected as INCOMPATIBLE_UNITS by a validator
that compared unit strings verbatim. Both spellings are correct English and
describe the same denominator -- the fix (app.financial.relationship_
validator._singularize, applied inside _rate_denominator) is a generic
regular -s/-es normalization, not a lookup table for "shipment" or any
other noun, so it must hold across arbitrary nouns/domains.
"""

from __future__ import annotations

from app.financial.relationship_validator import validate_and_materialize
from app.financial.semantic_models import SemanticFindingInterpretation


def _interp(qty_unit: str, rate_unit: str) -> SemanticFindingInterpretation:
    return SemanticFindingInterpretation.model_validate({
        "claims": [
            {"claim_id": "C0", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY",
             "value": 450, "unit": qty_unit, "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "C1", "source_evidence_ids": ["E1"], "fact_type": "RATE",
             "value": 22, "unit": rate_unit, "currency": "USD", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [
            {"relationship_id": "R0", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "C1",
             "target_claim": "C0", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]},
        ],
        "calculation_proposals": [
            {"calculation_id": "CAL0", "operation": "MULTIPLY", "inputs": ["C0", "C1"],
             "relationship_ids": ["R0"], "reason": "quantity x rate"},
        ],
    })


class TestPluralSingularUnitDenominatorCompatibility:
    def test_plural_quantity_unit_vs_singular_compound_rate_unit(self):
        observations, outcome = validate_and_materialize(_interp("shipments", "USD per shipment"), evidence_count=2)
        assert not outcome.rejected, outcome.rejected
        assert observations and observations[0].event_count == 450 and observations[0].unit_amount == 22

    def test_plural_quantity_unit_vs_plural_bare_rate_unit(self):
        observations, outcome = validate_and_materialize(_interp("transactions", "transactions"), evidence_count=2)
        assert not outcome.rejected, outcome.rejected
        assert observations

    def test_singular_quantity_unit_vs_plural_compound_rate_unit(self):
        observations, outcome = validate_and_materialize(_interp("component", "USD/components"), evidence_count=2)
        assert not outcome.rejected, outcome.rejected
        assert observations

    def test_genuinely_incompatible_units_still_rejected(self):
        # The fix must not become so lenient it stops catching real
        # mismatches -- an hourly rate must still never apply to a unit
        # count.
        observations, outcome = validate_and_materialize(_interp("units", "USD per hour"), evidence_count=2)
        assert not observations
        assert outcome.rejected and outcome.rejected[0].reason_code == "INCOMPATIBLE_UNITS"
