"""Generalization hardening for the LLM semantic financial layer's
QUANTITY x RATE relationship identification.

Root-cause context (see app/financial/relationship_validator.py): a
per-unit RATE that the interpreter mislabels as a flat AMOUNT claim (a
structurally self-contradictory shape -- a lump sum has no "per-X"
denominator) was previously either (a) silently materialized as a bare
flat amount via the standalone-fact fallback, discarding the quantity
entirely, or (b) invisible to materialization even when a correct
RATE_APPLIES_TO_QUANTITY relationship existed, because rate/quantity
role was derived from `fact_type` alone. Two GENERAL fixes (not tied to
any wording or specific numbers):

1. Rate/quantity role for MULTIPLY/ANNUALIZE is now derived from the
   explicit RATE_APPLIES_TO_QUANTITY relationship's source/target claims
   when the LLM cited one, falling back to fact_type lookup only when no
   such relationship role assignment exists.
2. A flat AMOUNT claim carrying a per-X `unit` is withheld (recorded for
   audit, never silently dropped) rather than materialized as a bare
   total, since it is unconsumed and rate-shaped.

Every fixture below uses a domain, currency, and numeric value that does
not appear in any other test file in this suite -- the point is to prove
the fix is general, not tuned to the one adversarial finding that
surfaced it.
"""

from __future__ import annotations

import json

import pytest

from app.financial.relationship_validator import validate_and_materialize
from app.financial.semantic_engine import analyze_financial_exposure_semantic
from app.financial.semantic_models import SemanticFindingInterpretation
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


def _qty_rate_interpretation(
    qty_value, qty_unit, rate_value, rate_unit, currency, *,
    qty_fact_type="QUANTITY", rate_fact_type="RATE",
    population="CURRENT_FINDING", qty_status="VERIFIED", rate_status="VERIFIED",
    proposed_result=None, cross_evidence=False,
):
    qty_ev = ["E0"]
    rate_ev = ["E1"] if cross_evidence else ["E0"]
    claims = [
        {"claim_id": "Q", "source_evidence_ids": qty_ev, "fact_type": qty_fact_type, "value": qty_value, "unit": qty_unit, "population": population, "evidence_status": qty_status},
        {"claim_id": "R", "source_evidence_ids": rate_ev, "fact_type": rate_fact_type, "value": rate_value, "unit": rate_unit, "currency": currency, "population": population, "evidence_status": rate_status},
    ]
    calc = {"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["Q", "R"], "relationship_ids": ["REL1"], "reason": "rate applies to quantity"}
    if proposed_result is not None:
        calc["proposed_result_value"] = proposed_result
    return json.dumps({
        "finding": {},
        "claims": claims,
        "relationships": [{"relationship_id": "REL1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "R", "target_claim": "Q", "confidence": "HIGH", "evidence_basis": sorted(set(qty_ev + rate_ev))}],
        "calculation_proposals": [calc],
    })


# ---------------------------------------------------------------------------
# 0. Regression reproduction: the exact defect, at all 3 pipeline levels
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_regression_rate_mislabeled_as_amount_still_multiplies_correctly():
    """Reproduces the reported defect: the LLM tags a per-unit rate as a
    flat AMOUNT claim (a very plausible interpretation slip) while still
    correctly identifying the RATE_APPLIES_TO_QUANTITY relationship.
    Validates all three pipeline levels independently, not just the final
    rendered number."""
    response = json.dumps({
        "finding": {"deviation": "scrap", "interpretation_confidence": "HIGH"},
        "claims": [
            {"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 800, "unit": "UNIT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "C2", "source_evidence_ids": ["E1"], "fact_type": "AMOUNT", "value": 450, "unit": "UNIT", "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "R1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "C2", "target_claim": "C1", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["C1", "C2"], "relationship_ids": ["R1"], "proposed_result_value": 450, "reason": "material cost applies to scrapped units"}],
    })

    # Level 1: semantic interpretation structure -- both claims present,
    # correct values/units/population, relationship correctly identified
    # regardless of the fact_type labeling slip.
    interp = SemanticFindingInterpretation.model_validate(json.loads(response))
    assert len(interp.claims) == 2
    assert {c.value for c in interp.claims} == {800.0, 450.0}
    assert interp.relationships[0].relationship_type == "RATE_APPLIES_TO_QUANTITY"
    assert interp.calculation_proposals[0].operation == "MULTIPLY"

    # Level 2: validated calculation proposal -- materializes as a single
    # rate x quantity observation, not a bare 450 amount.
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    assert len(observations) == 1
    assert observations[0].unit_amount == 450.0
    assert observations[0].event_count == 800
    assert observations[0].amount is None  # never a flat 450

    # Level 3: final deterministic result -- 800 x 450, LLM's wrong
    # proposed_result_value (450) never surfaces as authoritative.
    client = FakeLLMClient(response=response)
    ledger = _evidence(
        ("A production deviation resulted in 800 units being scrapped.", EvidenceStatus.VERIFIED),
        ("The verified material cost per unit was INR 450.", EvidenceStatus.VERIFIED),
    )
    outcome2 = await analyze_financial_exposure_semantic("adversarial finding", ledger, client=client)
    assert outcome2 is not None
    result, audit = outcome2
    assert result.confirmed_impact.verified_gross_exposure == 360_000.0
    assert result.confirmed_impact.verified_gross_exposure != 450.0
    assert any("450" in d for d in audit.outcome.llm_disagreements)


@pytest.mark.asyncio
async def test_correctly_typed_rate_also_still_works():
    """Same underlying finding, but the LLM correctly tags the rate as
    fact_type RATE (not AMOUNT) -- must produce the identical result,
    proving the fix didn't regress the already-correct path."""
    response = _qty_rate_interpretation(800, "UNIT", 450, "UNIT", "INR")
    client = FakeLLMClient(response=response)
    ledger = _evidence(("x", EvidenceStatus.VERIFIED), ("y", EvidenceStatus.VERIFIED))
    result, audit = await analyze_financial_exposure_semantic("x", ledger, client=client)
    assert result.confirmed_impact.verified_gross_exposure == 360_000.0


# ---------------------------------------------------------------------------
# 1-6. Generalization matrix: distinct domains, currencies, magnitudes
# ---------------------------------------------------------------------------

_DOMAIN_MATRIX = {
    "manufacturing_components": (2_400, "COMPONENT", 18.50, "COMPONENT", "EUR", 44_400.0),
    "services_support_hours": (850, "HOUR", 42, "HOUR", "USD", 35_700.0),
    "logistics_shipments": (1_200, "SHIPMENT", 35, "SHIPMENT", "AED", 42_000.0),
    "utilities_kwh": (75_000, "KWH", 8.20, "KWH", "INR", 615_000.0),
    "procurement_units": (430, "UNIT", 27, "UNIT", "GBP", 11_610.0),
    "quality_batches": (320, "BATCH", 115, "BATCH", "CHF", 36_800.0),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("name,params", list(_DOMAIN_MATRIX.items()))
async def test_domain_matrix_quantity_times_rate(name, params):
    qty, qty_unit, rate, rate_unit, currency, expected = params
    response = _qty_rate_interpretation(qty, qty_unit, rate, rate_unit, currency)
    client = FakeLLMClient(response=response)
    ledger = _evidence(("x", EvidenceStatus.VERIFIED), ("y", EvidenceStatus.VERIFIED))
    result, audit = await analyze_financial_exposure_semantic("x", ledger, client=client)
    assert result.confirmed_impact.verified_gross_exposure == pytest.approx(expected), name
    assert result.currency == currency, name


# ---------------------------------------------------------------------------
# 7-8. Cross-evidence and separate-sentence quantity/rate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_quantity_and_rate_in_separate_evidence_claims():
    response = _qty_rate_interpretation(560, "PALLET", 92.75, "PALLET", "USD", cross_evidence=True)
    client = FakeLLMClient(response=response)
    ledger = _evidence(
        ("Warehouse records confirm 560 pallets were affected.", EvidenceStatus.VERIFIED),
        ("Finance confirms a handling cost of $92.75 per pallet.", EvidenceStatus.VERIFIED),
    )
    result, audit = await analyze_financial_exposure_semantic("x", ledger, client=client)
    assert result.confirmed_impact.verified_gross_exposure == pytest.approx(51_940.0)


# ---------------------------------------------------------------------------
# 12-13. Decimal rate, large quantity
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_decimal_rate_and_large_quantity():
    response = _qty_rate_interpretation(184_600, "UNIT", 3.14159, "UNIT", "USD")
    client = FakeLLMClient(response=response)
    ledger = _evidence(("x", EvidenceStatus.VERIFIED), ("y", EvidenceStatus.VERIFIED))
    result, audit = await analyze_financial_exposure_semantic("x", ledger, client=client)
    assert result.confirmed_impact.verified_gross_exposure == pytest.approx(184_600 * 3.14159, rel=1e-6)


# ---------------------------------------------------------------------------
# 15-16. Historical population safety
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_historical_quantity_times_historical_rate_valid():
    response = _qty_rate_interpretation(14, "OCCURRENCE", 9_200, "OCCURRENCE", "ZAR", population="HISTORICAL")
    client = FakeLLMClient(response=response)
    ledger = _evidence(("x", EvidenceStatus.VERIFIED), ("y", EvidenceStatus.VERIFIED))
    outcome = await analyze_financial_exposure_semantic("x", ledger, client=client)
    assert outcome is not None
    result, audit = outcome
    # A HISTORICAL-population MULTIPLY does not populate CURRENT_FINDING
    # gross exposure -- it is a historical rate observation, not confused
    # with current exposure (existing, unchanged population semantics).
    assert result.confirmed_impact.verified_gross_exposure is None


def test_current_quantity_times_historical_rate_rejected():
    interp = SemanticFindingInterpretation.model_validate(json.loads(
        _qty_rate_interpretation(90, "UNIT", 1_250, "UNIT", "SGD")
    ))
    # Force a population mismatch: quantity CURRENT_FINDING, rate HISTORICAL.
    interp.claims[1].population = "HISTORICAL"
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    assert observations == []
    assert outcome.rejected[0].reason_code == "POPULATION_MISMATCH"


# ---------------------------------------------------------------------------
# 17. Current quantity + remediation cost -- must reject
# ---------------------------------------------------------------------------

def test_current_quantity_times_remediation_cost_rejected():
    interp = SemanticFindingInterpretation.model_validate({
        "claims": [
            {"claim_id": "Q", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 60, "unit": "UNIT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "R", "source_evidence_ids": ["E1"], "fact_type": "REMEDIATION_COST", "value": 4_400, "unit": "UNIT", "currency": "CAD", "population": "REMEDIATION", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "REL1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "R", "target_claim": "Q", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["Q", "R"], "relationship_ids": ["REL1"], "reason": "remediation cost incorrectly proposed as a per-unit rate"}],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    # The MULTIPLY is rejected -- population mismatch (CURRENT_FINDING x
    # REMEDIATION) -- but the remediation figure remains visible as its
    # own standalone fact (never silently dropped), just never multiplied
    # by the unrelated quantity.
    assert outcome.rejected[0].reason_code == "POPULATION_MISMATCH"
    assert not any(o.amount == 60 * 4_400 or o.unit_amount == 4_400 for o in observations)
    assert any(o.amount == 4_400 and o.amount_type.value == "REMEDIATION_COST" for o in observations)


# ---------------------------------------------------------------------------
# 18. Incompatible unit -- must reject
# ---------------------------------------------------------------------------

def test_incompatible_unit_rejected():
    interp = SemanticFindingInterpretation.model_validate(json.loads(
        _qty_rate_interpretation(300, "UNIT", 15, "HOUR", "NZD")
    ))
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    assert observations == []
    assert outcome.rejected[0].reason_code == "INCOMPATIBLE_UNITS"


# ---------------------------------------------------------------------------
# 19. Two unrelated amounts -- must not be arbitrarily multiplied
# ---------------------------------------------------------------------------

def test_unrelated_quantity_and_amount_never_auto_multiplied():
    """'340 units were inspected' + 'INR 12,000 remediation cost' -- no
    relationship links them (the LLM correctly did not propose one,
    since inspecting units and a flat remediation figure are unrelated
    facts). Both must remain standalone, never combined into 340 x 12000."""
    interp = SemanticFindingInterpretation.model_validate({
        "claims": [
            {"claim_id": "Q", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 340, "unit": "UNIT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "AMT", "source_evidence_ids": ["E1"], "fact_type": "REMEDIATION_COST", "value": 12_000, "currency": "INR", "population": "REMEDIATION", "evidence_status": "VERIFIED"},
        ],
        "relationships": [],
        "calculation_proposals": [],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    assert not any(o.amount == 340 * 12_000 for o in observations)
    # The remediation figure is still preserved as its own standalone fact.
    assert any(o.amount == 12_000 for o in observations)


# ---------------------------------------------------------------------------
# 20. One quantity, two competing rates -- must not arbitrarily choose
# ---------------------------------------------------------------------------

def test_competing_rates_for_one_quantity_not_arbitrarily_chosen():
    interp = SemanticFindingInterpretation.model_validate({
        "claims": [
            {"claim_id": "Q", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 500, "unit": "UNIT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "R1", "source_evidence_ids": ["E1"], "fact_type": "RATE", "value": 60, "unit": "UNIT", "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "R2", "source_evidence_ids": ["E2"], "fact_type": "RATE", "value": 75, "unit": "UNIT", "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [
            {"relationship_id": "REL1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "R1", "target_claim": "Q", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]},
            {"relationship_id": "REL2", "type": "CONFLICTS_WITH", "source_claim": "R1", "target_claim": "R2", "confidence": "HIGH", "evidence_basis": ["E1", "E2"]},
        ],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["Q", "R1"], "relationship_ids": ["REL1"], "reason": "one of two conflicting rates arbitrarily picked"}],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=3)
    assert not any(o.amount == 500 * 60 or o.unit_amount == 60 for o in observations)
    assert any(r.reason_code == "CONFLICTING_CLAIMS" for r in outcome.rejected)


# ---------------------------------------------------------------------------
# "Do not calculate from numbers alone" -- unrelated quantity/amount pairs
# must never combine merely because both are present.
# ---------------------------------------------------------------------------

def test_inspection_quantity_and_unrelated_remediation_cost_not_multiplied():
    """'X units were inspected' + 'Y was the remediation cost' -- these
    describe different things (an inspection count, a fix cost), not a
    per-unit charge. With no MULTIPLY proposal or RATE_APPLIES_TO_
    QUANTITY relationship at all, nothing should multiply them."""
    interp = SemanticFindingInterpretation.model_validate({
        "claims": [
            {"claim_id": "Q", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 210, "unit": "UNIT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "REM", "source_evidence_ids": ["E1"], "fact_type": "REMEDIATION_COST", "value": 8_500, "currency": "USD", "population": "REMEDIATION", "evidence_status": "VERIFIED"},
        ],
        "relationships": [],
        "calculation_proposals": [],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    assert not any(o.amount == 210 * 8_500 for o in observations)


def test_scrap_quantity_and_unrelated_recovery_not_multiplied():
    """'X units were scrapped' + 'Y was recovered from the supplier' --
    a recovery is not a per-unit rate for the scrapped quantity."""
    interp = SemanticFindingInterpretation.model_validate({
        "claims": [
            {"claim_id": "Q", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 640, "unit": "UNIT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "REC", "source_evidence_ids": ["E1"], "fact_type": "RECOVERY", "value": 1_100, "currency": "USD", "population": "RECOVERY", "evidence_status": "REPORTED"},
        ],
        "relationships": [],
        "calculation_proposals": [],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    assert not any(o.amount == 640 * 1_100 for o in observations)


# ---------------------------------------------------------------------------
# LLM adversarial tests A-G
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adversarial_a_correct_claims_wrong_proposed_result():
    response = _qty_rate_interpretation(275, "UNIT", 64, "UNIT", "USD", proposed_result=999_999)
    client = FakeLLMClient(response=response)
    ledger = _evidence(("x", EvidenceStatus.VERIFIED), ("y", EvidenceStatus.VERIFIED))
    result, audit = await analyze_financial_exposure_semantic("x", ledger, client=client)
    assert result.confirmed_impact.verified_gross_exposure == pytest.approx(275 * 64)
    assert any("999999" in d for d in audit.outcome.llm_disagreements)


def test_adversarial_b_differently_labeled_but_connecting_relationship_now_accepted():
    """Architectural relaxation: the validator no longer requires a cited
    relationship's TYPE to exactly match a fixed per-operation lookup
    table (e.g. MULTIPLY <- RATE_APPLIES_TO_QUANTITY only) -- the
    6-member RelationshipType taxonomy cannot cleanly label every real
    financial relationship (e.g. "3 repeated identical events, each
    valued at X" isn't cleanly a "rate"), and rejecting a semantically
    sound, well-grounded calculation merely because the LLM chose a
    different-but-still-cited, still-connecting relationship label was
    the deterministic layer second-guessing the LLM's semantic judgment.
    What IS still required (see the next test) is that the cited
    relationship actually connects these specific input claims."""
    interp = SemanticFindingInterpretation.model_validate({
        "claims": [
            {"claim_id": "Q", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 100, "unit": "UNIT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "R", "source_evidence_ids": ["E1"], "fact_type": "RATE", "value": 50, "unit": "UNIT", "currency": "USD", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "REL1", "type": "CORROBORATES", "source_claim": "R", "target_claim": "Q", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["Q", "R"], "relationship_ids": ["REL1"], "proposed_result_value": 5000, "reason": "quantity x rate, linked by a differently-labeled relationship"}],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    assert not outcome.rejected, outcome.rejected
    assert len(observations) == 1
    assert observations[0].event_count == 100 and observations[0].unit_amount == 50


def test_adversarial_b2_relationship_connecting_unrelated_claims_still_rejected():
    """The safety property that survives the relaxation above: a cited
    relationship that does not actually connect (any of) the
    calculation's own input claims -- i.e. it links some other,
    unrelated pair -- must still be rejected. This is what actually
    prevents arbitrary/coincidental number combination; the relationship
    TYPE label is not what does that work."""
    interp = SemanticFindingInterpretation.model_validate({
        "claims": [
            {"claim_id": "Q", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 100, "unit": "UNIT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "R", "source_evidence_ids": ["E1"], "fact_type": "RATE", "value": 50, "unit": "UNIT", "currency": "USD", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "X", "source_evidence_ids": ["E0"], "fact_type": "OTHER", "value": 9, "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "Y", "source_evidence_ids": ["E1"], "fact_type": "OTHER", "value": 3, "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "REL1", "type": "CORROBORATES", "source_claim": "X", "target_claim": "Y", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["Q", "R"], "relationship_ids": ["REL1"], "reason": "cites an unrelated relationship"}],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    assert observations == []
    assert outcome.rejected[0].reason_code == "UNSUPPORTED_RELATIONSHIP"


def test_adversarial_c_fabricated_claim_id_rejected():
    interp = SemanticFindingInterpretation.model_validate({
        "claims": [{"claim_id": "Q", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 100, "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"}],
        "relationships": [],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["Q", "GHOST_RATE"], "relationship_ids": [], "reason": "fabricated"}],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=1)
    assert observations == []
    assert outcome.rejected[0].reason_code == "UNKNOWN_CLAIM"


def test_adversarial_d_fabricated_evidence_id_rejected():
    interp = SemanticFindingInterpretation.model_validate({
        "claims": [
            {"claim_id": "Q", "source_evidence_ids": ["E77"], "fact_type": "QUANTITY", "value": 100, "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "R", "source_evidence_ids": ["E0"], "fact_type": "RATE", "value": 50, "currency": "USD", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "REL1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "R", "target_claim": "Q", "confidence": "HIGH", "evidence_basis": ["E0"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["Q", "R"], "relationship_ids": ["REL1"], "reason": "x"}],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=1)
    assert observations == []
    assert outcome.rejected[0].reason_code == "MISSING_PROVENANCE"


def test_adversarial_e_incompatible_rate_unit_rejected():
    interp = SemanticFindingInterpretation.model_validate(json.loads(
        _qty_rate_interpretation(400, "UNIT", 20, "DELIVERY", "USD")
    ))
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    assert observations == []
    assert outcome.rejected[0].reason_code == "INCOMPATIBLE_UNITS"


def test_adversarial_f_different_populations_rejected():
    interp = SemanticFindingInterpretation.model_validate(json.loads(
        _qty_rate_interpretation(150, "UNIT", 30, "UNIT", "USD")
    ))
    interp.claims[0].population = "CURRENT_FINDING"
    interp.claims[1].population = "HISTORICAL"
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    assert observations == []
    assert outcome.rejected[0].reason_code == "POPULATION_MISMATCH"


@pytest.mark.asyncio
async def test_adversarial_g_reported_rate_verified_quantity_status_preserved():
    """Quantity VERIFIED, rate REPORTED -- the calculation is materialized
    (evidence supports it) but its epistemic status must reflect the
    WEAKEST input status (REPORTED), never upgraded to VERIFIED merely
    because one side and the relationship's own confidence are HIGH."""
    response = _qty_rate_interpretation(320, "UNIT", 88, "UNIT", "USD", rate_status="REPORTED")
    client = FakeLLMClient(response=response)
    ledger = _evidence(("x", EvidenceStatus.VERIFIED), ("y", EvidenceStatus.REPORTED))
    result, audit = await analyze_financial_exposure_semantic("x", ledger, client=client)
    assert result.confirmed_impact.verified_gross_exposure is None
    assert result.confirmed_impact.reported_financial_exposure == pytest.approx(320 * 88)


def test_regression_relationship_source_target_reversed_still_correct():
    """Real defect found via a live Ollama call: the model correctly
    fact_type-tagged claims (QUANTITY=1000, RATE=250) but emitted the
    RATE_APPLIES_TO_QUANTITY relationship with source_claim=<quantity>,
    target_claim=<rate> -- the OPPOSITE of the "source=rate, target=
    quantity" convention relationship_validator.py used to trust
    positionally. That silently swapped which claim it materialized as
    unit_amount vs event_count. Invisible in MULTIPLY's total (1000*250 ==
    250*1000) but wrong in the per-unit breakdown and would be wrong for
    any non-commutative use. fact_type must be the primary signal;
    relationship position only a fallback when fact_type is ambiguous."""
    interp = SemanticFindingInterpretation.model_validate({
        "finding": {},
        "claims": [
            {"claim_id": "C0", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY",
             "value": 1000, "unit": "unit", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "C1", "source_evidence_ids": ["E1"], "fact_type": "RATE",
             "value": 250, "unit": "INR/unit", "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        # Reversed on purpose: source=quantity, target=rate.
        "relationships": [{"relationship_id": "R0", "type": "RATE_APPLIES_TO_QUANTITY",
                            "source_claim": "C0", "target_claim": "C1", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]}],
        "calculation_proposals": [{"calculation_id": "CAL0", "operation": "MULTIPLY", "inputs": ["C0", "C1"],
                                    "relationship_ids": ["R0"], "reason": "quantity x rate"}],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    assert not outcome.rejected, outcome.rejected
    assert len(observations) == 1
    obs = observations[0]
    assert obs.event_count == 1000, "event_count must be the QUANTITY claim's value, not the rate's"
    assert obs.unit_amount == 250, "unit_amount must be the RATE claim's value, not the quantity's"


def test_percentage_rate_converted_to_fraction_before_multiply():
    """Real defect found via a live Ollama call: a percentage-denominated
    rate (unit="%") was multiplied using its raw value (6.0) instead of
    the fraction it represents (0.06) -- 480,000 * 6 = 2,880,000 instead
    of 480,000 * 0.06 = 28,800. "6%" always means the fraction 0.06 in a
    multiplication, universally, regardless of domain -- this is a
    mechanical unit-conversion fact for the deterministic executor, not
    something the LLM's own arithmetic needs to get right."""
    interp = SemanticFindingInterpretation.model_validate({
        "finding": {},
        "claims": [
            {"claim_id": "C0", "source_evidence_ids": ["E0"], "fact_type": "AMOUNT",
             "value": 480000, "currency": "AED", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "C1", "source_evidence_ids": ["E1"], "fact_type": "RATE",
             "value": 6.0, "unit": "%", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "R0", "type": "RATE_APPLIES_TO_QUANTITY",
                            "source_claim": "C1", "target_claim": "C0", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]}],
        "calculation_proposals": [{"calculation_id": "CAL0", "operation": "MULTIPLY", "inputs": ["C0", "C1"],
                                    "relationship_ids": ["R0"], "reason": "percentage of contract value"}],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    assert not outcome.rejected, outcome.rejected
    assert len(observations) == 1
    assert observations[0].unit_amount == pytest.approx(0.06)
    assert observations[0].event_count == 480000


def test_percentage_fact_type_also_converted_regardless_of_unit_string():
    interp = SemanticFindingInterpretation.model_validate({
        "finding": {},
        "claims": [
            {"claim_id": "C0", "source_evidence_ids": ["E0"], "fact_type": "AMOUNT",
             "value": 200000, "currency": "USD", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "C1", "source_evidence_ids": ["E1"], "fact_type": "PERCENTAGE",
             "value": 3.5, "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "R0", "type": "RATE_APPLIES_TO_QUANTITY",
                            "source_claim": "C1", "target_claim": "C0", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]}],
        "calculation_proposals": [{"calculation_id": "CAL0", "operation": "MULTIPLY", "inputs": ["C0", "C1"],
                                    "relationship_ids": ["R0"], "reason": "percentage of amount"}],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    assert not outcome.rejected, outcome.rejected
    assert observations[0].unit_amount == pytest.approx(0.035)


def test_rate_unit_equal_to_currency_code_treated_as_unset():
    """Real defect found via a live Ollama call: a RATE claim's `unit`
    field sometimes just restates its own currency code ("INR") instead
    of a real per-X denominator ("failure"/"hour"/"unit") -- a reasonable
    way for the LLM to fill the field when the "per X" meaning is only
    implied grammatically ("Each resulted in a cost of INR 18,000", where
    "Each" carries the per-event meaning but no unit word follows the
    figure). That currency-code unit is NOT a real dimensional unit and
    must not be compared against a quantity's unit as a mismatch."""
    interp = SemanticFindingInterpretation.model_validate({
        "finding": {},
        "claims": [
            {"claim_id": "C0", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY",
             "value": 6, "unit": "failure", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "C1", "source_evidence_ids": ["E1"], "fact_type": "RATE",
             "value": 18000, "unit": "INR", "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "R0", "type": "RATE_APPLIES_TO_QUANTITY",
                            "source_claim": "C1", "target_claim": "C0", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]}],
        "calculation_proposals": [{"calculation_id": "CAL0", "operation": "MULTIPLY", "inputs": ["C0", "C1"],
                                    "relationship_ids": ["R0"], "reason": "count x average cost each"}],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    assert not outcome.rejected, outcome.rejected
    assert len(observations) == 1
    assert observations[0].event_count == 6
    assert observations[0].unit_amount == 18000


def test_genuinely_incompatible_units_still_rejected_even_with_currency_present():
    # The fix must not become so lenient it stops catching real
    # mismatches -- a currency-labeled hourly rate must still never apply
    # to an unrelated unit quantity.
    interp = SemanticFindingInterpretation.model_validate({
        "finding": {},
        "claims": [
            {"claim_id": "C0", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY",
             "value": 6, "unit": "unit", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "C1", "source_evidence_ids": ["E1"], "fact_type": "RATE",
             "value": 18000, "unit": "hour", "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "R0", "type": "RATE_APPLIES_TO_QUANTITY",
                            "source_claim": "C1", "target_claim": "C0", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]}],
        "calculation_proposals": [{"calculation_id": "CAL0", "operation": "MULTIPLY", "inputs": ["C0", "C1"],
                                    "relationship_ids": ["R0"], "reason": "x"}],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=2)
    assert not observations
    assert outcome.rejected and outcome.rejected[0].reason_code == "INCOMPATIBLE_UNITS"


def test_regression_amount_and_quantity_pair_reversed_relationship_still_correct():
    """Real defect found via a live Ollama call, same class as the RATE/
    QUANTITY swap above but for a QUANTITY+AMOUNT pair (no claim tagged
    RATE at all -- a legitimate LLM choice for "INR 50,000 each" with no
    explicit "per X" wording). event_count and unit_amount were swapped
    (3 payments x INR 50,000 became 50,000 "events" x INR 3) because,
    with no RATE-tagged claim to disambiguate, the code fell back to the
    LLM's (again reversed) relationship source/target ordering. Fixed by
    using the unambiguous QUANTITY tag to identify the other claim in a
    2-input MULTIPLY as the rate/multiplier by elimination, regardless of
    its own tag or relationship position."""
    interp = SemanticFindingInterpretation.model_validate({
        "finding": {},
        "claims": [
            {"claim_id": "C0", "source_evidence_ids": ["E0"], "fact_type": "AMOUNT",
             "value": 50000, "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY",
             "value": 3, "unit": "payment", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        # Reversed on purpose: source=quantity, target=amount.
        "relationships": [{"relationship_id": "R0", "type": "RATE_APPLIES_TO_QUANTITY",
                            "source_claim": "C1", "target_claim": "C0", "confidence": "HIGH", "evidence_basis": ["E0"]}],
        "calculation_proposals": [{"calculation_id": "CAL0", "operation": "MULTIPLY", "inputs": ["C1", "C0"],
                                    "relationship_ids": ["R0"], "reason": "quantity x amount each"}],
    })
    observations, outcome = validate_and_materialize(interp, evidence_count=1)
    assert not outcome.rejected, outcome.rejected
    assert len(observations) == 1
    obs = observations[0]
    assert obs.event_count == 3, "event_count must be the QUANTITY claim's value"
    assert obs.unit_amount == 50000, "unit_amount must be the AMOUNT claim's value, not the quantity's"

class TestRelationshipTypeIsDescriptiveMetadataOnly:
    """Architecture invariant (spec section 7): `relationship_type` is
    free-text descriptive metadata. It is NEVER an operation permission/
    license, and no fixed table maps an operation to a "required"
    relationship type. The validator's only structural reads off a
    relationship are (a) does it connect the claims the calculation cites
    and (b) `is_conflict`. Consequently an unusual, model-invented, or
    even operation-named string in `relationship_type` neither licenses
    nor blocks a calculation on its own -- structural grounding does."""

    async def test_unusual_relationship_type_string_does_not_block_calculation(self):
        response = json.dumps({
            "finding": {},
            "claims": [
                {"claim_id": "C0", "source_evidence_ids": ["E0"], "fact_type": "AMOUNT",
                 "value": 50000, "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
                {"claim_id": "C1", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY",
                 "value": 3, "unit": "payment", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            ],
            # An unusual / model-invented relationship_type string. It is
            # descriptive only -- the calculation stands on structural
            # grounding (the relationship connects C0 and C1), not on this
            # label matching any table.
            "relationships": [{"relationship_id": "R0", "relationship_type": "each-payment-carries-the-stated-amount",
                                "source_claim": "C1", "target_claim": "C0", "confidence": "HIGH", "evidence_basis": ["E0"]}],
            "calculation_proposals": [{"calculation_id": "CAL0", "operation": "MULTIPLY", "inputs": ["C1", "C0"],
                                        "relationship_ids": ["R0"], "reason": "quantity x amount each"}],
            "cost_factor": {"selected_factor": "DUPLICATE_PAYMENT", "supporting_claim_ids": ["C0", "C1"],
                             "confidence": "HIGH", "rationale": "duplicate payments"},
        })
        client = FakeLLMClient(response=response)
        ledger = _evidence(("Finance records confirm three duplicate payments.", EvidenceStatus.VERIFIED))
        result, audit = await analyze_financial_exposure_semantic("x", ledger, client=client)
        assert result is not None
        assert result.confirmed_impact.verified_gross_exposure == pytest.approx(3 * 50000)
        assert audit.interpretation.relationships[0].relationship_type == "each-payment-carries-the-stated-amount"
        assert audit.interpretation.relationships[0].is_conflict is False

    async def test_is_conflict_flag_not_the_type_label_blocks_silent_combination(self):
        # Two competing amounts for the same fact. What prevents them from
        # being silently summed is the structural `is_conflict` flag on the
        # relationship -- NOT any particular relationship_type spelling.
        response = json.dumps({
            "finding": {},
            "claims": [
                {"claim_id": "C0", "source_evidence_ids": ["E0"], "fact_type": "AMOUNT",
                 "value": 100000, "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "REPORTED"},
                {"claim_id": "C1", "source_evidence_ids": ["E1"], "fact_type": "AMOUNT",
                 "value": 80000, "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            ],
            "relationships": [{"relationship_id": "R0", "relationship_type": "two internal estimates disagree",
                                "is_conflict": True, "source_claim": "C0", "target_claim": "C1",
                                "confidence": "HIGH", "evidence_basis": ["E0", "E1"]}],
            "calculation_proposals": [{"calculation_id": "CAL0", "operation": "SUM", "inputs": ["C0", "C1"],
                                        "relationship_ids": ["R0"], "reason": "add the two amounts"}],
        })
        interp = SemanticFindingInterpretation.model_validate(json.loads(response))
        observations, outcome = validate_and_materialize(interp, evidence_count=2)
        assert not any(o.amount == 180000 for o in observations), "conflicting claims must never be silently summed"
        assert any(r.calculation_id == "CAL0" for r in outcome.rejected)
