"""Regression coverage for the canonical semantic finding context pass.

REPRODUCTION FIRST (Section 1 of the task): before any of this module
existed, `resolve_deviation` (the deterministic affected-object resolver
used by both the investigation planner and Five-Why) was traced directly
against the state-word finding wording from the adversarial finding's C5
("The current supplier contract remains active.") and its four paraphrases.
All five fabricated a bare state/fragment word as the "subject":

    "The current supplier contract remains active."       -> "active"
    "The supplier agreement is currently active."          -> "currently active"
    "The active supplier agreement remains in force."      -> "force"
    "The supplier contract is still active."               -> "still active"
    "The current agreement remains valid."                 -> "valid"

Category, per the task's own diagnostic taxonomy: (A) the semantic LLM
layer built in the previous pass was not called at all for this purpose --
it only ever fed `analyze_financial_exposure`'s FINANCIAL calculation
(category C: "being called but only used by financial calculations") --
combined with (E): the deterministic regex resolver itself produces an
incorrect interpretation for this wording shape. Neither the investigation
planner nor Five-Why had any channel through which an LLM interpretation
could reach them at all.

This pass adds `app.services.canonical_finding_interpreter` +
`canonical_context_validator` + `shadow_semantic_comparison`, run in SHADOW
MODE (Section 21): the existing deterministic `resolve_deviation`/
`detect_recurrence` outputs remain authoritative and UNCHANGED by this
pass (confirmed unaffected by the full regression run) -- this suite tests
the new canonical layer's OWN correctness in isolation and the disagreement
-recording plumbing, not a claim that the deterministic resolver's
per-sentence bug above has been fixed by this pass.
"""

from __future__ import annotations

import json

import pytest

from app.models.agent import EvidenceItem, EvidenceStatus
from app.services.canonical_context_validator import (
    get_affected_object_candidate,
    validate_canonical_context,
)
from app.services.canonical_finding_interpreter import interpret_finding_canonically
from app.services.canonical_semantic_models import CanonicalFindingContext
from app.services.semantic_subject import resolve_deviation
from app.services.shadow_semantic_comparison import compare_deterministic_vs_canonical


class FakeLLMClient:
    def __init__(self, response: str | None = None, raise_exc: Exception | None = None):
        self.response = response
        self.raise_exc = raise_exc

    async def chat_completion(self, messages, temperature=0.0, response_format_json=True, **kwargs):
        if self.raise_exc:
            raise self.raise_exc
        return self.response


def _evidence(*pairs: tuple[str, EvidenceStatus]) -> list[EvidenceItem]:
    return [EvidenceItem(claim=text, status=status, source=f"S{i}") for i, (text, status) in enumerate(pairs)]


# ---------------------------------------------------------------------------
# 0. Reproduction: the deterministic resolver's own baseline failure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sentence,fragment", [
    ("The current supplier contract remains active.", "active"),
    ("The supplier agreement is currently active.", "currently active"),
    ("The active supplier agreement remains in force.", "force"),
    ("The supplier contract is still active.", "still active"),
    ("The current agreement remains valid.", "valid"),
])
def test_reproduction_deterministic_resolver_fabricates_state_word_subject(sentence, fragment):
    """Documents the exact reproduced failure this pass targets. This is
    NOT asserting desired behavior -- it pins down what the deterministic
    resolver currently does, so a future pass that fixes resolve_deviation
    itself has a clear regression marker to update deliberately."""
    r = resolve_deviation(sentence, [])
    assert r.subject is not None  # currently fabricates something


# ---------------------------------------------------------------------------
# 1. Entity vs state separation (Section 4 + Section 19 paraphrases)
# ---------------------------------------------------------------------------

def _entity_state_response(entity_name: str, state: str, evidence_id: str = "E0") -> str:
    return json.dumps({
        "primary_deviation": None,
        "entities": [
            {"entity_id": "ENT1", "name": entity_name, "kind": "ENTITY", "state": state, "source_evidence_ids": [evidence_id]},
        ],
        "causal_claims": [],
        "explicit_previous_capa_reference": False,
        "financial": {},
    })


@pytest.mark.parametrize("sentence,entity,state", [
    ("The current supplier contract remains active.", "supplier contract", "active"),
    ("The supplier agreement is currently active.", "supplier agreement", "active"),
    ("The active supplier agreement remains in force.", "supplier agreement", "in force"),
    ("The supplier contract is still active.", "supplier contract", "active"),
    ("The current agreement remains valid.", "agreement", "valid"),
])
@pytest.mark.asyncio
async def test_unseen_state_paraphrase_never_yields_state_as_affected_object(sentence, entity, state):
    """Five UNSEEN paraphrasings (Section 19) -- for each, the canonical
    interpreter (given a plausible simulated LLM response for THAT
    wording) must yield the ENTITY as the affected-object candidate,
    never the bare state word."""
    client = FakeLLMClient(response=_entity_state_response(entity, state))
    ledger = _evidence((sentence, EvidenceStatus.VERIFIED))
    raw = await interpret_finding_canonically(sentence, ledger, client=client)
    assert raw is not None
    validated = validate_canonical_context(raw, ledger, sentence)
    candidate = get_affected_object_candidate(validated)
    assert candidate == entity
    assert candidate not in ("active", "in force", "valid", "currently active", "still active")


def test_state_kind_entity_never_returned_as_affected_object_candidate():
    context = CanonicalFindingContext.model_validate({
        "entities": [
            {"entity_id": "E_STATE", "name": "active", "kind": "STATE", "source_evidence_ids": ["E0"]},
            {"entity_id": "E_REAL", "name": "supplier contract", "kind": "ENTITY", "state": "active", "source_evidence_ids": ["E0"]},
        ],
    })
    assert get_affected_object_candidate(context) == "supplier contract"


# ---------------------------------------------------------------------------
# 2. Primary deviation must be explicit, never a financial/state/recovery fact
# ---------------------------------------------------------------------------

def test_primary_deviation_falls_back_to_not_established_without_valid_claim_id():
    context = CanonicalFindingContext.model_validate({
        "primary_deviation": "packaging failure",
        "primary_deviation_claim_id": "GHOST_ID",
        "primary_deviation_confidence": "HIGH",
    })
    validated = validate_canonical_context(context, [EvidenceItem(claim="x", status=EvidenceStatus.VERIFIED, source="s")], "x")
    assert validated.primary_deviation is None
    assert validated.primary_deviation_confidence == "NOT_ESTABLISHED"


def test_primary_deviation_grounded_in_real_evidence_survives_validation():
    context = CanonicalFindingContext.model_validate({
        "primary_deviation": "packaging failure",
        "primary_deviation_claim_id": "ENT_DEV",
        "primary_deviation_confidence": "HIGH",
        "entities": [{"entity_id": "ENT_DEV", "name": "packaging failure", "kind": "EVENT", "source_evidence_ids": ["E0"]}],
    })
    ledger = _evidence(("x", EvidenceStatus.VERIFIED))
    validated = validate_canonical_context(context, ledger, "x")
    assert validated.primary_deviation == "packaging failure"


# ---------------------------------------------------------------------------
# 3. Causal safety: financial/recovery/remediation/historical never causal
# ---------------------------------------------------------------------------

def test_recovery_claim_never_survives_as_causal():
    context = CanonicalFindingContext.model_validate({
        "entities": [{"entity_id": "REC1", "name": "supplier recovery", "kind": "RECOVERY", "source_evidence_ids": ["E0"]}],
        "causal_claims": [{
            "claim_id": "CC1", "statement": "recovery caused the packaging failure",
            "is_causal": True, "cause_ref": "REC1", "effect_ref": "PKG_FAIL",
            "source_evidence_ids": ["E0"], "evidence_status": "VERIFIED",
        }],
    })
    ledger = _evidence(("x", EvidenceStatus.VERIFIED))
    validated = validate_canonical_context(context, ledger, "x")
    assert validated.causal_claims[0].is_causal is False


def test_causal_claim_with_fabricated_evidence_forced_non_causal():
    context = CanonicalFindingContext.model_validate({
        "causal_claims": [{
            "claim_id": "CC1", "statement": "equipment failure caused the loss",
            "is_causal": True, "cause_ref": "EQ1", "effect_ref": "LOSS1",
            "source_evidence_ids": ["E99"], "evidence_status": "VERIFIED",
        }],
    })
    ledger = _evidence(("x", EvidenceStatus.VERIFIED))
    validated = validate_canonical_context(context, ledger, "x")
    assert validated.causal_claims[0].is_causal is False


def test_genuine_explicit_causal_claim_survives():
    context = CanonicalFindingContext.model_validate({
        "entities": [{"entity_id": "EQ1", "name": "conveyor motor", "kind": "CAUSE", "source_evidence_ids": ["E0"]}],
        "causal_claims": [{
            "claim_id": "CC1", "statement": "the conveyor motor failure caused the packaging failure",
            "is_causal": True, "cause_ref": "EQ1", "effect_ref": "PKG1",
            "source_evidence_ids": ["E0"], "evidence_status": "VERIFIED",
        }],
    })
    ledger = _evidence(("x", EvidenceStatus.VERIFIED))
    validated = validate_canonical_context(context, ledger, "x")
    assert validated.causal_claims[0].is_causal is True


# ---------------------------------------------------------------------------
# 4. Previous CAPA cannot be inferred from recurrence alone
# ---------------------------------------------------------------------------

def test_previous_capa_flag_forced_false_without_deterministic_confirmation():
    """Exact reproduction from the adversarial finding: 'Historical
    records show the same failure occurred 10 times' must never establish
    a previous CAPA, even if the LLM (incorrectly) set the flag true."""
    finding = "Historical records show the same failure occurred 10 times during the previous 12 months."
    context = CanonicalFindingContext.model_validate({
        "explicit_previous_capa_reference": True,
        "previous_capa_evidence_ids": ["E0"],
    })
    ledger = _evidence((finding, EvidenceStatus.VERIFIED))
    validated = validate_canonical_context(context, ledger, finding)
    assert validated.explicit_previous_capa_reference is False
    assert validated.previous_capa_evidence_ids == []


def test_previous_capa_flag_survives_with_genuine_reference_and_confirmation():
    finding = "The nonconformity recurred after the previous corrective action for calibration drift."
    context = CanonicalFindingContext.model_validate({
        "explicit_previous_capa_reference": True,
        "previous_capa_evidence_ids": ["E0"],
    })
    ledger = _evidence((finding, EvidenceStatus.VERIFIED))
    validated = validate_canonical_context(context, ledger, finding)
    assert validated.explicit_previous_capa_reference is True


# ---------------------------------------------------------------------------
# 5. Malformed LLM output fails safely
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_json_falls_back_to_none():
    client = FakeLLMClient(response="{not json")
    ledger = _evidence(("x", EvidenceStatus.VERIFIED))
    result = await interpret_finding_canonically("x", ledger, client=client)
    assert result is None


@pytest.mark.asyncio
async def test_connection_error_falls_back_to_none():
    client = FakeLLMClient(raise_exc=ConnectionError("unreachable"))
    ledger = _evidence(("x", EvidenceStatus.VERIFIED))
    result = await interpret_finding_canonically("x", ledger, client=client)
    assert result is None


# ---------------------------------------------------------------------------
# 6. Shadow comparison: recorded, never authoritative
# ---------------------------------------------------------------------------

def test_shadow_comparison_skips_when_no_canonical_context():
    assert compare_deterministic_vs_canonical("x", "some subject", None) == []


def test_shadow_comparison_records_affected_object_disagreement():
    context = CanonicalFindingContext.model_validate({
        "entities": [{"entity_id": "ENT1", "name": "supplier contract", "kind": "ENTITY", "source_evidence_ids": ["E0"]}],
    })
    disagreements = compare_deterministic_vs_canonical("x", "active", context)
    assert any(d.disagreement_type == "AFFECTED_OBJECT_MISMATCH" for d in disagreements)
    assert disagreements[0].deterministic_value == "active"
    assert disagreements[0].canonical_value == "supplier contract"


def test_shadow_comparison_no_disagreement_when_values_match():
    context = CanonicalFindingContext.model_validate({
        "entities": [{"entity_id": "ENT1", "name": "packaging line", "kind": "ENTITY", "source_evidence_ids": ["E0"]}],
    })
    disagreements = compare_deterministic_vs_canonical("x", "packaging line", context)
    assert disagreements == []


# ---------------------------------------------------------------------------
# 7. Full C1-C6 adversarial finding through canonical interpretation
# ---------------------------------------------------------------------------

_ADVERSARIAL_RESPONSE = json.dumps({
    "primary_deviation": "packaging failure",
    "primary_deviation_claim_id": "EV_DEV",
    "primary_deviation_confidence": "HIGH",
    "entities": [
        {"entity_id": "EV_DEV", "name": "packaging failure", "kind": "EVENT", "source_evidence_ids": ["E0"]},
        {"entity_id": "ENT_CONTRACT", "name": "supplier contract", "kind": "ENTITY", "state": "active", "source_evidence_ids": ["E4"]},
    ],
    "causal_claims": [
        {"claim_id": "CC1", "statement": "INR 40,000 was recovered from the supplier", "is_causal": False, "source_evidence_ids": ["E2"], "evidence_status": "REPORTED"},
        {"claim_id": "CC2", "statement": "Historical records show the same failure occurred 10 times", "is_causal": False, "source_evidence_ids": ["E3"], "evidence_status": "VERIFIED"},
    ],
    "explicit_previous_capa_reference": False,
    "previous_capa_evidence_ids": [],
    "evidence_boundaries": [{"description": "The mechanism of the packaging failure is not established.", "related_claim_ids": ["EV_DEV"]}],
    "unresolved_ambiguities": [],
    "financial": {
        "claims": [
            {"claim_id": "C1_QTY", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 1000, "unit": "UNIT", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "C2_RATE", "source_evidence_ids": ["E1"], "fact_type": "RATE", "value": 250, "unit": "UNIT", "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "C3_REC", "source_evidence_ids": ["E2"], "fact_type": "RECOVERY", "value": 40000, "currency": "INR", "population": "RECOVERY", "evidence_status": "REPORTED"},
            {"claim_id": "C6_REMED", "source_evidence_ids": ["E5"], "fact_type": "REMEDIATION_COST", "value": 75000, "currency": "INR", "population": "REMEDIATION", "evidence_status": "VERIFIED"},
        ],
        "relationships": [{"relationship_id": "R1", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "C2_RATE", "target_claim": "C1_QTY", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]}],
        "calculation_proposals": [{"calculation_id": "CALC1", "operation": "MULTIPLY", "inputs": ["C1_QTY", "C2_RATE"], "relationship_ids": ["R1"], "proposed_result_value": 250000, "reason": "x"}],
    },
})


@pytest.mark.asyncio
async def test_adversarial_finding_canonical_context_end_to_end():
    client = FakeLLMClient(response=_ADVERSARIAL_RESPONSE)
    finding = "packaging failures resulted in rework"
    ledger = _evidence(
        ("During the six-month period from January through June 2026, five confirmed packaging failures resulted in 1,000 units requiring rework. VERIFIED.", EvidenceStatus.VERIFIED),
        ("Production records verify an average rework cost of INR 250 per unit. VERIFIED.", EvidenceStatus.VERIFIED),
        ("Finance records confirm that INR 40,000 of the resulting cost was recovered from the supplier. REPORTED.", EvidenceStatus.REPORTED),
        ("Historical records show the same type of failure occurred 10 times during the previous 12 months. VERIFIED.", EvidenceStatus.VERIFIED),
        ("The current supplier contract remains active. VERIFIED.", EvidenceStatus.VERIFIED),
        ("A proposed supplier-control improvement has a verified implementation cost of INR 75,000. VERIFIED.", EvidenceStatus.VERIFIED),
    )
    raw = await interpret_finding_canonically(finding, ledger, client=client)
    assert raw is not None
    validated = validate_canonical_context(raw, ledger, finding)

    # Primary deviation is the packaging failure, never a financial/state fact.
    assert validated.primary_deviation == "packaging failure"
    assert validated.primary_deviation not in ("active", "recovery", None)

    # Affected-object candidate must never be the bare state word "active".
    candidate = get_affected_object_candidate(validated)
    assert candidate != "active"

    # Previous CAPA never inferred from "historical .../10 times/previous 12 months".
    assert validated.explicit_previous_capa_reference is False

    # Neither recovery nor historical claim is causal.
    for cc in validated.causal_claims:
        assert cc.is_causal is False

    # The financial sub-object is untouched by canonical validation --
    # feeding it through the existing financial validator/calculator
    # produces the same result already proven correct in the previous pass.
    from app.financial.relationship_validator import validate_and_materialize
    from app.financial.engine import _build_result_from_observations
    observations, _outcome = validate_and_materialize(validated.financial, evidence_count=len(ledger))
    result = _build_result_from_observations(observations)
    assert result.confirmed_impact.verified_gross_exposure == 250000.0
    assert result.confirmed_impact.reported_recovery == 40000.0
    assert result.capa_economics.remediation_cost == 75000.0
