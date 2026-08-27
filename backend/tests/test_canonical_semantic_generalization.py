"""Generalization tests for the canonical semantic architecture (promotion
pass follow-up): structurally different findings, never seen in any prior
pass's test fixtures, to prove the system generalizes via STRUCTURE
(fact_type/population/relationship/kind) rather than memorized wording.

None of these sentences (or anything resembling their specific nouns:
components, service incidents, delivery failures, vendor recovery, a
corrective program, "similar incidents", a missing-records control) appear
anywhere else in this codebase's source or prior test fixtures. Each test
constructs the SemanticClaim/CanonicalFindingContext a real LLM interpreter
would plausibly produce for that sentence, then verifies the SAME
deterministic validator/calculator/builder functions already proven correct
for the packaging-failure finding produce the semantically correct result
for this entirely different one -- proving the correctness lives in the
generic architecture, not in wording-specific rules.
"""

from __future__ import annotations

from app.agent.nodes.five_why_fallback import build_deterministic_five_why
from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
from app.financial.engine import _build_result_from_observations
from app.financial.relationship_validator import validate_and_materialize
from app.financial.semantic_models import SemanticFindingInterpretation
from app.models.agent import EvidenceItem, EvidenceStatus
from app.services.canonical_context_validator import get_affected_object_candidate, validate_canonical_context
from app.services.canonical_semantic_models import CanonicalFindingContext


def _evidence(*pairs: tuple[str, EvidenceStatus]) -> list[EvidenceItem]:
    return [EvidenceItem(claim=text, status=status, source=f"S{i}") for i, (text, status) in enumerate(pairs)]


def _financial_result(financial: dict, evidence_count: int):
    interp = SemanticFindingInterpretation.model_validate(financial)
    observations, _outcome = validate_and_materialize(interp, evidence_count=evidence_count)
    return _build_result_from_observations(observations)


# ---------------------------------------------------------------------------
# 1. "400 components required rework ... average cost of ₹80 per component."
# ---------------------------------------------------------------------------

def test_generalization_1_components_rework_quantity_times_rate():
    finding = "400 components required rework. Production records establish an average cost of INR 80 per component."
    ledger = _evidence((finding, EvidenceStatus.VERIFIED))
    financial = {
        "claims": [
            {"claim_id": "Q1", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 400, "unit": "COMPONENT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "R1", "source_evidence_ids": ["E0"], "fact_type": "RATE", "value": 80, "unit": "COMPONENT", "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "REL1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "R1", "target_claim": "Q1", "confidence": "HIGH", "evidence_basis": ["E0"]}],
        "calculation_proposals": [{"calculation_id": "C1", "operation": "MULTIPLY", "inputs": ["Q1", "R1"], "relationship_ids": ["REL1"], "reason": "average per-component cost applies to the 400 reworked components"}],
    }
    result = _financial_result(financial, len(ledger))
    assert result.confirmed_impact.verified_gross_exposure == 32000.0


# ---------------------------------------------------------------------------
# 2. "Twenty service incidents ... USD 500 in documented repair cost."
# ---------------------------------------------------------------------------

def test_generalization_2_service_incidents_usd_quantity_times_rate():
    finding = "Twenty service incidents were verified. Each incident generated USD 500 in documented repair cost."
    ledger = _evidence((finding, EvidenceStatus.VERIFIED))
    financial = {
        "claims": [
            {"claim_id": "Q1", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 20, "unit": "INCIDENT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "R1", "source_evidence_ids": ["E0"], "fact_type": "RATE", "value": 500, "unit": "INCIDENT", "currency": "USD", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "REL1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "R1", "target_claim": "Q1", "confidence": "HIGH", "evidence_basis": ["E0"]}],
        "calculation_proposals": [{"calculation_id": "C1", "operation": "MULTIPLY", "inputs": ["Q1", "R1"], "relationship_ids": ["REL1"], "reason": "documented repair cost per incident applies to the 20 verified incidents"}],
    }
    result = _financial_result(financial, len(ledger))
    assert result.confirmed_impact.verified_gross_exposure == 10000.0
    assert result.confirmed_impact.currency == "USD"


# ---------------------------------------------------------------------------
# 3. "The supplier agreement remains valid, but three delivery failures..."
# ---------------------------------------------------------------------------

def test_generalization_3_entity_state_separation_and_deviation():
    finding = "The supplier agreement remains valid, but three delivery failures occurred."
    context = CanonicalFindingContext.model_validate({
        "primary_deviation": "delivery failures",
        "primary_deviation_claim_id": "EV1",
        "primary_deviation_confidence": "HIGH",
        "entities": [
            {"entity_id": "ENT1", "name": "supplier agreement", "kind": "ENTITY", "state": "valid", "source_evidence_ids": ["E0"]},
            {"entity_id": "EV1", "name": "delivery failures", "kind": "EVENT", "source_evidence_ids": ["E0"]},
        ],
    })
    ledger = _evidence((finding, EvidenceStatus.VERIFIED))
    validated = validate_canonical_context(context, ledger, finding)
    assert validated.primary_deviation == "delivery failures"
    candidate = get_affected_object_candidate(validated)
    assert candidate == "supplier agreement"
    assert candidate != "valid"

    fw = build_deterministic_five_why(finding, ledger, semantic_context=validated)
    assert "delivery failures" in fw.steps[0].question.lower()


# ---------------------------------------------------------------------------
# 4. "EUR 12,000 was recovered from the vendor after the quality incident."
# ---------------------------------------------------------------------------

def test_generalization_4_recovery_never_becomes_primary_deviation_or_cause():
    finding = "EUR 12,000 was recovered from the vendor after the quality incident."
    context = CanonicalFindingContext.model_validate({
        "primary_deviation": None,
        "primary_deviation_confidence": "NOT_ESTABLISHED",
        "entities": [{"entity_id": "REC1", "name": "vendor recovery", "kind": "RECOVERY", "source_evidence_ids": ["E0"]}],
        "causal_claims": [{
            "claim_id": "CC1", "statement": "EUR 12,000 was recovered from the vendor",
            "is_causal": True, "cause_ref": "REC1", "effect_ref": "QUALITY_INCIDENT",
            "source_evidence_ids": ["E0"], "evidence_status": "VERIFIED",
        }],
    })
    ledger = _evidence((finding, EvidenceStatus.VERIFIED))
    validated = validate_canonical_context(context, ledger, finding)
    # A RECOVERY-kind entity can never be the cause side of a causal claim,
    # regardless of what the LLM's is_causal flag said.
    assert validated.causal_claims[0].is_causal is False
    assert validated.primary_deviation is None

    financial = {
        "claims": [{"claim_id": "REC1", "source_evidence_ids": ["E0"], "fact_type": "RECOVERY", "value": 12000, "currency": "EUR", "population": "RECOVERY", "evidence_status": "VERIFIED"}],
    }
    result = _financial_result(financial, len(ledger))
    assert result.confirmed_impact.verified_recovery == 12000.0
    assert result.confirmed_impact.currency == "EUR"


# ---------------------------------------------------------------------------
# 5. "A corrective program is estimated at CHF 20,000."
# ---------------------------------------------------------------------------

def test_generalization_5_remediation_never_loss_never_cause():
    finding = "A corrective program is estimated at CHF 20,000."
    ledger = _evidence((finding, EvidenceStatus.VERIFIED))
    financial = {
        "claims": [{"claim_id": "REM1", "source_evidence_ids": ["E0"], "fact_type": "REMEDIATION_COST", "value": 20000, "currency": "CHF", "population": "REMEDIATION", "evidence_status": "VERIFIED"}],
    }
    result = _financial_result(financial, len(ledger))
    assert result.confirmed_impact.verified_gross_exposure is None
    assert result.capa_economics.remediation_cost == 20000.0
    assert result.capa_economics.remediation_cost_status == "VERIFIED"
    assert result.capa_economics.is_assessable is False  # no verified annual avoided exposure


# ---------------------------------------------------------------------------
# 6. "Fifteen similar incidents occurred during the previous two years."
# ---------------------------------------------------------------------------

def test_generalization_6_historical_recurrence_never_implies_previous_capa():
    finding = "Fifteen similar incidents occurred during the previous two years."
    context = CanonicalFindingContext.model_validate({
        "explicit_previous_capa_reference": False,
        "entities": [{"entity_id": "H1", "name": "similar incidents", "kind": "HISTORICAL_CONTEXT", "source_evidence_ids": ["E0"]}],
    })
    ledger = _evidence((finding, EvidenceStatus.VERIFIED))
    validated = validate_canonical_context(context, ledger, finding)
    assert validated.explicit_previous_capa_reference is False

    _, plan = build_deterministic_investigation_plan(finding, ledger, semantic_context=validated)
    all_text = " ".join(plan.areas) + " " + " ".join(q.question for q in plan.questions)
    assert "capa" not in all_text.lower()


# ---------------------------------------------------------------------------
# 7. "The control was active during the audit, but required records were missing."
# ---------------------------------------------------------------------------

def test_generalization_7_active_state_never_leaks_when_real_deviation_exists():
    finding = "The control was active during the audit, but required records were missing."
    context = CanonicalFindingContext.model_validate({
        "primary_deviation": "missing records",
        "primary_deviation_claim_id": "EV1",
        "primary_deviation_confidence": "HIGH",
        "entities": [
            {"entity_id": "ENT1", "name": "control", "kind": "ENTITY", "state": "active", "source_evidence_ids": ["E0"]},
            {"entity_id": "EV1", "name": "missing records", "kind": "EVENT", "source_evidence_ids": ["E0"]},
        ],
    })
    ledger = _evidence((finding, EvidenceStatus.VERIFIED))
    validated = validate_canonical_context(context, ledger, finding)
    candidate = get_affected_object_candidate(validated)
    assert candidate != "active"
    assert candidate == "control"

    _, plan = build_deterministic_investigation_plan(finding, ledger, semantic_context=validated)
    all_text = " ".join(plan.areas)
    assert not any(a.lower().startswith("active") for a in plan.areas)

    fw = build_deterministic_five_why(finding, ledger, semantic_context=validated)
    assert "missing records" in fw.steps[0].question.lower()
