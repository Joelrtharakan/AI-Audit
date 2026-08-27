"""Regression coverage for promoting the canonical semantic finding context
from SHADOW-ONLY to an actual authoritative input for the deterministic
Five-Why and investigation-plan fallback builders.

Call path BEFORE this pass:
    plan_investigation_node -> build_deterministic_investigation_plan(...)
        (no channel for any LLM interpretation to reach it)
    core_synthesis_node / final_evidence_verification_node ->
        build_deterministic_five_why(...) (same)
    report_generator_node -> computed a canonical context SEPARATELY, late,
        purely for shadow comparison -- too late to influence the actual
        investigation_plan/five_why already stored in state.

Call path AFTER this pass:
    plan_investigation_node computes canonical_semantic_context ONCE
        (gated behind canonical_semantic_shadow_enabled, off by default)
        and stores it in state["canonical_semantic_context"]; passes it
        into its own build_deterministic_investigation_plan call.
    core_synthesis_node / final_evidence_verification_node reuse
        state["canonical_semantic_context"] (no second LLM call) for
        their own build_deterministic_five_why / build_deterministic_
        investigation_plan calls.
    report_generator_node reuses the same state key for shadow-diagnostic
        comparison, no longer recomputing it.

Both builder functions keep semantic_context defaulting to None, so every
existing call site not explicitly updated is provably unaffected -- this
is what the "backward-compatible, additive-only" requirement produces:
state C (semantic_context=None) is exactly the pre-existing behavior.
"""

from __future__ import annotations

from app.agent.nodes.five_why_fallback import build_deterministic_five_why
from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
from app.models.agent import EvidenceItem, EvidenceStatus
from app.services.canonical_semantic_models import CanonicalFindingContext


def _evidence(*pairs: tuple[str, EvidenceStatus]) -> list[EvidenceItem]:
    return [EvidenceItem(claim=text, status=status, source=f"S{i}") for i, (text, status) in enumerate(pairs)]


# ---------------------------------------------------------------------------
# State A: semantic_context present with a resolved entity -> used
# ---------------------------------------------------------------------------

def test_investigation_plan_uses_canonical_entity_over_raw_text_state_word():
    """The exact reproduction: 'active' must never appear as the affected
    object when a validated canonical context resolves the real entity."""
    finding = "The current supplier contract remains active."
    context = CanonicalFindingContext.model_validate({
        "entities": [{"entity_id": "ENT1", "name": "supplier contract", "kind": "ENTITY", "state": "active", "source_evidence_ids": ["E0"]}],
    })
    ledger = _evidence((finding, EvidenceStatus.VERIFIED))
    _, plan = build_deterministic_investigation_plan(finding, ledger, semantic_context=context)
    all_text = " ".join(plan.areas) + " " + " ".join(q.question for q in plan.questions)
    assert "supplier contract" in all_text.lower()
    assert not any(a.lower().startswith("active") for a in plan.areas)


def test_five_why_uses_canonical_primary_deviation_over_raw_text():
    finding = "packaging failures resulted in rework"
    context = CanonicalFindingContext.model_validate({
        "primary_deviation": "packaging failure",
        "primary_deviation_claim_id": "EV1",
        "primary_deviation_confidence": "HIGH",
        "entities": [{"entity_id": "EV1", "name": "packaging failure", "kind": "EVENT", "source_evidence_ids": ["E0"]}],
    })
    ledger = _evidence((finding, EvidenceStatus.VERIFIED))
    fw = build_deterministic_five_why(finding, ledger, semantic_context=context)
    assert fw.steps
    assert "packaging failure" in fw.steps[0].question.lower()


# ---------------------------------------------------------------------------
# State B: semantic_context present but explicitly unresolved -> generic
# placeholder, never a raw-text guess
# ---------------------------------------------------------------------------

def test_investigation_plan_state_b_never_falls_through_to_raw_text_guess():
    """Canonical context is present (state was attempted) but resolved no
    entity at all -- must use the generic placeholder, never re-derive a
    guess from resolve_deviation (which would produce 'active' here)."""
    finding = "The current supplier contract remains active."
    empty_context = CanonicalFindingContext.model_validate({})  # no entities at all
    ledger = _evidence((finding, EvidenceStatus.VERIFIED))
    _, plan = build_deterministic_investigation_plan(finding, ledger, semantic_context=empty_context)
    all_text = " ".join(plan.areas)
    assert "active" not in all_text.lower()
    assert "the affected process" in all_text.lower()


def test_five_why_state_b_never_falls_through_to_raw_text_guess():
    finding = "The current supplier contract remains active."
    empty_context = CanonicalFindingContext.model_validate({})
    ledger = _evidence((finding, EvidenceStatus.VERIFIED))
    fw = build_deterministic_five_why(finding, ledger, semantic_context=empty_context)
    combined = " ".join(s.question for s in fw.steps).lower()
    assert "active" not in combined


# ---------------------------------------------------------------------------
# State C: semantic_context=None -> fully unchanged legacy behavior
# ---------------------------------------------------------------------------

def test_state_c_none_context_reproduces_legacy_behavior_unchanged():
    """Regression pin: without a semantic_context (state C), the
    pre-existing (still-buggy-for-this-wording) deterministic behavior is
    completely unaffected by this pass's new parameter."""
    finding = "The current supplier contract remains active."
    ledger = _evidence((finding, EvidenceStatus.VERIFIED))
    _, plan_with_none = build_deterministic_investigation_plan(finding, ledger, semantic_context=None)
    _, plan_without_param = build_deterministic_investigation_plan(finding, ledger)
    assert plan_with_none.areas == plan_without_param.areas
    assert [q.question for q in plan_with_none.questions] == [q.question for q in plan_without_param.questions]


# ---------------------------------------------------------------------------
# Causal safety: financial/recovery/historical facts vetoed as mechanisms
# ---------------------------------------------------------------------------

def test_five_why_never_uses_recovery_as_mechanism_when_semantic_context_present():
    """The exact reproduction from the previous pass's live trace: without
    this fix, Why-2 was answered with the recovery evidence sentence
    verbatim."""
    finding = "packaging failures resulted in rework"
    context = CanonicalFindingContext.model_validate({
        "primary_deviation": "packaging failure",
        "primary_deviation_claim_id": "EV1",
        "primary_deviation_confidence": "HIGH",
        "entities": [{"entity_id": "EV1", "name": "packaging failure", "kind": "EVENT", "source_evidence_ids": ["E0"]}],
        "causal_claims": [{
            "claim_id": "CC1", "statement": "recovery", "is_causal": False,
            "source_evidence_ids": ["E2"], "evidence_status": "REPORTED",
        }],
    })
    ledger = _evidence(
        ("Five confirmed packaging failures resulted in 1,000 units requiring rework.", EvidenceStatus.VERIFIED),
        ("Production records verify an average rework cost of INR 250 per unit.", EvidenceStatus.VERIFIED),
        ("Finance records confirm that INR 40,000 of the resulting cost was recovered from the supplier.", EvidenceStatus.REPORTED),
    )
    fw = build_deterministic_five_why(finding, ledger, semantic_context=context)
    combined = " ".join((s.answer or "") for s in fw.steps).lower()
    assert "recovered" not in combined
    assert "40,000" not in combined


def test_five_why_vetoed_mechanism_falls_to_evidence_boundary_not_a_gap():
    """When the only candidate mechanism is vetoed as non-causal, the
    chain must stop at UNKNOWN, never silently produce an empty chain or
    crash."""
    finding = "packaging failures resulted in rework"
    context = CanonicalFindingContext.model_validate({
        "primary_deviation": "packaging failure",
        "primary_deviation_claim_id": "EV1",
        "entities": [{"entity_id": "EV1", "name": "packaging failure", "kind": "EVENT", "source_evidence_ids": ["E0"]}],
        "causal_claims": [{
            "claim_id": "CC1", "statement": "historical", "is_causal": False,
            "source_evidence_ids": ["E1"], "evidence_status": "VERIFIED",
        }],
    })
    ledger = _evidence(
        ("A packaging failure occurred.", EvidenceStatus.VERIFIED),
        ("Historical records show the same type of failure occurred 10 times during the previous 12 months.", EvidenceStatus.VERIFIED),
    )
    fw = build_deterministic_five_why(finding, ledger, semantic_context=context)
    assert fw.steps
    assert fw.steps[-1].status == "UNKNOWN"


# ---------------------------------------------------------------------------
# Previous CAPA gating uses semantic_context when present
# ---------------------------------------------------------------------------

def test_investigation_plan_previous_capa_gated_by_semantic_context():
    finding = "Historical records show the same type of failure occurred 10 times during the previous 12 months."
    context = CanonicalFindingContext.model_validate({
        "explicit_previous_capa_reference": False,
    })
    ledger = _evidence((finding, EvidenceStatus.VERIFIED))
    _, plan = build_deterministic_investigation_plan(finding, ledger, semantic_context=context)
    all_text = " ".join(plan.areas) + " " + " ".join(q.question for q in plan.questions)
    assert "capa" not in all_text.lower()


def test_five_why_previous_capa_gated_by_semantic_context():
    finding = "The nonconformity recurred."
    context = CanonicalFindingContext.model_validate({
        "explicit_previous_capa_reference": False,
    })
    ledger = _evidence((finding, EvidenceStatus.VERIFIED))
    fw = build_deterministic_five_why(finding, ledger, semantic_context=context)
    combined = " ".join(s.question for s in fw.steps).lower()
    assert "prior corrective action" not in combined


# ---------------------------------------------------------------------------
# Full C1-C6 adversarial finding, end-to-end through both builders
# ---------------------------------------------------------------------------

def _adversarial_context() -> CanonicalFindingContext:
    return CanonicalFindingContext.model_validate({
        "primary_deviation": "packaging failure",
        "primary_deviation_claim_id": "EV_DEV",
        "primary_deviation_confidence": "HIGH",
        "entities": [
            {"entity_id": "EV_DEV", "name": "packaging failure", "kind": "EVENT", "source_evidence_ids": ["E0"]},
            {"entity_id": "ENT_CONTRACT", "name": "supplier contract", "kind": "ENTITY", "state": "active", "source_evidence_ids": ["E4"]},
        ],
        "causal_claims": [
            {"claim_id": "CC1", "statement": "recovery", "is_causal": False, "source_evidence_ids": ["E2"], "evidence_status": "REPORTED"},
            {"claim_id": "CC2", "statement": "historical recurrence", "is_causal": False, "source_evidence_ids": ["E3"], "evidence_status": "VERIFIED"},
        ],
        "explicit_previous_capa_reference": False,
    })


def _adversarial_evidence() -> list[EvidenceItem]:
    return _evidence(
        ("During the six-month period from January through June 2026, five confirmed packaging failures resulted in 1,000 units requiring rework. VERIFIED.", EvidenceStatus.VERIFIED),
        ("Production records verify an average rework cost of INR 250 per unit. VERIFIED.", EvidenceStatus.VERIFIED),
        ("Finance records confirm that INR 40,000 of the resulting cost was recovered from the supplier. REPORTED.", EvidenceStatus.REPORTED),
        ("Historical records show the same type of failure occurred 10 times during the previous 12 months. VERIFIED.", EvidenceStatus.VERIFIED),
        ("The current supplier contract remains active. VERIFIED.", EvidenceStatus.VERIFIED),
        ("A proposed supplier-control improvement has a verified implementation cost of INR 75,000. VERIFIED.", EvidenceStatus.VERIFIED),
    )


def test_adversarial_finding_investigation_plan_end_to_end():
    context = _adversarial_context()
    ledger = _adversarial_evidence()
    finding = "packaging failures resulted in rework"
    _, plan = build_deterministic_investigation_plan(finding, ledger, semantic_context=context)
    all_text = " ".join(plan.areas) + " " + " ".join(q.question for q in plan.questions)
    assert not any(a.lower().startswith("active") for a in plan.areas)
    assert "capa" not in all_text.lower()


def test_adversarial_finding_five_why_end_to_end():
    context = _adversarial_context()
    ledger = _adversarial_evidence()
    finding = "packaging failures resulted in rework"
    fw = build_deterministic_five_why(finding, ledger, semantic_context=context)
    assert fw.steps
    assert "packaging failure" in fw.steps[0].question.lower()
    combined_answers = " ".join((s.answer or "") for s in fw.steps).lower()
    assert "recovered" not in combined_answers
    assert "40,000" not in combined_answers
    assert fw.steps[-1].status == "UNKNOWN"
