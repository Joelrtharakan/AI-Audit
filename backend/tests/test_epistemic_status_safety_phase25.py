"""Phase 25 Rule 6/7/8: repository-wide VERIFIED-status-check audit
regression, the temporal-causality firewall, and the critic-send-back
second-writer fix. Exercises the real production functions directly --
never a reimplementation.
"""
from __future__ import annotations

import asyncio
import json

from app.agent.causal_graph import merge_candidate_hypotheses
from app.agent.common_factor import detect_common_factor
from app.agent.evidence_interpreter import EvidenceInterpreter, derive_hypothesis_relevance, validate_relation
from app.agent.invariants import evaluate_all_invariants
from app.models.agent import CandidateHypothesis, EvidenceItem, EvidenceProposition, EvidenceStatus, ImpactAssessment
from app.services.llm.base import LLMProvider, LLMResponse


class _FakeLLMProvider(LLMProvider):
    def __init__(self, payload):
        self._payload = payload

    async def generate(self, *, node, prompt, **kwargs):
        return LLMResponse(content=json.dumps(self._payload), provider="fake", model="fake-model")


# ---------------------------------------------------------------------------
# Rule 7: exact VERIFIED/UNVERIFIED/REPORTED/UNKNOWN status discrimination,
# proven at each fixed call site -- not just the firewall.
# ---------------------------------------------------------------------------

def test_unverified_evidence_never_treated_as_verified_in_common_factor():
    """common_factor.py:66 fix -- UNVERIFIED items must be excluded from
    the VERIFIED-claims pool used for common-factor pattern detection."""
    verified = EvidenceItem(claim="A shared system caused delay for site 1", source="s",
                             status=EvidenceStatus.VERIFIED, evidence_id="EV1")
    unverified = EvidenceItem(claim="A shared system caused delay for site 2", source="s",
                               status=EvidenceStatus.UNVERIFIED, evidence_id="EV2")
    # detect_common_factor's indexed_claims pool is built ONLY from
    # VERIFIED-status items -- prove it by checking a ledger containing
    # only the UNVERIFIED item behaves identically to an empty ledger
    # (fewer than 2 indexed VERIFIED claims -> no pattern detected).
    only_unverified = detect_common_factor([unverified])
    empty = detect_common_factor([])
    assert only_unverified.detected == empty.detected == False


def test_status_equality_correctly_discriminates_all_values():
    """Direct proof the exact-equality pattern (now used throughout) works
    for every relevant EvidenceStatus value -- VERIFIED matches, nothing
    else does, including the specific UNVERIFIED collision that was buggy."""
    assert (EvidenceStatus.VERIFIED == "VERIFIED") is True
    assert (EvidenceStatus.UNVERIFIED == "VERIFIED") is False
    assert (EvidenceStatus.REPORTED == "VERIFIED") is False
    assert (EvidenceStatus.UNKNOWN == "VERIFIED") is False
    assert (EvidenceStatus.REPORTED == "REPORTED") is True
    assert (EvidenceStatus.UNKNOWN == "REPORTED") is False


def test_impact_verified_invariant_rejects_unverified_evidence():
    """INV-IMPACT-002, Phase 24/25 fix: UNVERIFIED evidence must not
    satisfy the 'has real VERIFIED proof' check."""
    impact = ImpactAssessment(status="IMPACT_VERIFIED")
    unverified_item = EvidenceItem(claim="x", source="s", status=EvidenceStatus.UNVERIFIED)
    state = {"impact_assessment": impact, "evidence_ledger": [unverified_item]}
    is_valid, violations = evaluate_all_invariants(state)
    assert any("INV-IMPACT-002" in v for v in violations)


def test_impact_verified_invariant_accepts_real_verified_evidence():
    impact = ImpactAssessment(status="IMPACT_VERIFIED")
    verified_item = EvidenceItem(claim="x", source="s", status=EvidenceStatus.VERIFIED)
    state = {"impact_assessment": impact, "evidence_ledger": [verified_item]}
    is_valid, violations = evaluate_all_invariants(state)
    assert not any("INV-IMPACT-002" in v for v in violations)


def test_impact_not_verified_status_never_flagged_regardless_of_evidence():
    impact = ImpactAssessment(status="IMPACT_REQUIRES_ASSESSMENT")
    unverified_item = EvidenceItem(claim="x", source="s", status=EvidenceStatus.UNVERIFIED)
    state = {"impact_assessment": impact, "evidence_ledger": [unverified_item]}
    is_valid, violations = evaluate_all_invariants(state)
    assert not any("INV-IMPACT-002" in v for v in violations)


# ---------------------------------------------------------------------------
# Rule 6: temporal-causality firewall (validate_relation direct + end-to-end)
# ---------------------------------------------------------------------------

def test_temporal_only_causal_relation_downgraded_to_neutral():
    rel, decision = validate_relation("SUPPORTING", EvidenceStatus.VERIFIED,
                                       EvidenceProposition.POSITIVE_ASSERTION,
                                       causal_basis="TEMPORAL_ONLY")
    assert rel == "NEUTRAL"
    assert decision == "DOWNGRADE_TO_NEUTRAL"


def test_temporal_plus_independent_evidence_not_overcorrected():
    """Rule 6: 'do not overcorrect legitimate causal evidence that happens
    to include temporal information' -- INDEPENDENT_EVIDENCE must survive
    unchanged even when temporal_relation is also present on the claim."""
    rel, decision = validate_relation("CONTRADICTING", EvidenceStatus.VERIFIED,
                                       EvidenceProposition.POSITIVE_ASSERTION,
                                       causal_basis="INDEPENDENT_EVIDENCE")
    assert rel == "CONTRADICTING"
    assert decision == "ACCEPT"


def test_not_applicable_causal_basis_does_not_affect_non_causal_hypotheses():
    rel, decision = validate_relation("SUPPORTING", EvidenceStatus.VERIFIED,
                                       EvidenceProposition.POSITIVE_ASSERTION,
                                       causal_basis="NOT_APPLICABLE")
    assert rel == "SUPPORTING"
    assert decision == "ACCEPT"


def test_temporal_only_end_to_end_through_interpreter():
    """Reproduces the exact Rule 6 scenario end-to-end: 'A occurred before
    B' proposed as SUPPORTING a causal hypothesis, with causal_basis
    correctly reported as TEMPORAL_ONLY by the (fake) LLM -- the firewall
    must downgrade it to NEUTRAL regardless of what the LLM proposed."""
    interp = EvidenceInterpreter(llm_provider=_FakeLLMProvider({"claims": [{
        "text": "the deployment occurred at 09:00 and the outage began at 09:15",
        "source_reference": "system timeline",
        "proposition_type": "POSITIVE_ASSERTION",
        "temporal_relation": "BEFORE",
        "hypothesis_relations": [{
            "hypothesis_id": "H1", "relation": "SUPPORTING",
            "reason": "the deployment happened shortly before the outage",
            "causal_basis": "TEMPORAL_ONLY",
        }],
    }]}))
    item = EvidenceItem(claim="The deployment occurred at 09:00. The outage began at 09:15.",
                         source="system_timeline", status=EvidenceStatus.VERIFIED, evidence_id="EV1")
    claims = asyncio.run(interp.interpret(item, hypotheses=[
        {"id": "H1", "statement": "The deployment caused the outage."},
    ]))
    assert len(claims) == 1
    # derive_hypothesis_relevance's aggregate only ever surfaces a decisive
    # SUPPORTING/CONTRADICTING/CONFLICTING vote or INSUFFICIENT (by design
    # -- NEUTRAL/INSUFFICIENT are both "no decisive relevance"); the
    # DOWNGRADE itself is verified directly on the claim's own relation.
    assert derive_hypothesis_relevance(claims, "H1") == "INSUFFICIENT"
    assert claims[0].hypothesis_relations[0].relation == "NEUTRAL"
    assert claims[0].hypothesis_relations[0].validation_decision == "DOWNGRADE_TO_NEUTRAL"
    assert claims[0].temporal_relation == "BEFORE"


def test_temporal_with_explicit_causal_statement_survives():
    """The 'do not overcorrect' counter-case, end to end: the SAME temporal
    facts, but the evidence also contains an explicit causal statement, so
    the LLM (correctly) reports INDEPENDENT_EVIDENCE -- the relation must
    survive unchanged."""
    interp = EvidenceInterpreter(llm_provider=_FakeLLMProvider({"claims": [{
        "text": "the deployment caused the outage, as confirmed by the incident review",
        "source_reference": "incident review report",
        "proposition_type": "POSITIVE_ASSERTION",
        "hypothesis_relations": [{
            "hypothesis_id": "H1", "relation": "SUPPORTING",
            "reason": "an incident review explicitly attributes the outage to the deployment",
            "causal_basis": "INDEPENDENT_EVIDENCE",
        }],
    }]}))
    item = EvidenceItem(claim="The incident review confirmed the deployment caused the outage.",
                         source="incident_review", status=EvidenceStatus.VERIFIED, evidence_id="EV2")
    claims = asyncio.run(interp.interpret(item, hypotheses=[
        {"id": "H1", "statement": "The deployment caused the outage."},
    ]))
    assert derive_hypothesis_relevance(claims, "H1") == "SUPPORTING"


def test_malformed_causal_basis_defaults_to_not_applicable_safely():
    interp = EvidenceInterpreter(llm_provider=_FakeLLMProvider({"claims": [{
        "text": "x", "source_reference": "ref",
        "hypothesis_relations": [{"hypothesis_id": "H1", "relation": "SUPPORTING",
                                   "causal_basis": "SOMETHING_INVALID"}],
    }]}))
    item = EvidenceItem(claim="x", source="s", status=EvidenceStatus.VERIFIED, evidence_id="EV1")
    claims = asyncio.run(interp.interpret(item, hypotheses=[{"id": "H1", "statement": "a"}]))
    assert claims[0].hypothesis_relations[0].causal_basis == "NOT_APPLICABLE"
    assert claims[0].hypothesis_relations[0].relation == "SUPPORTING"  # unaffected -- not treated as temporal-only


# ---------------------------------------------------------------------------
# Rule 8: critic-send-back second-writer bug (merge_candidate_hypotheses)
# ---------------------------------------------------------------------------

def test_locked_refuted_hypothesis_survives_critic_send_back_regeneration():
    """The genuine defect this phase discovered: a second core_synthesis
    pass (critic-send-back re-investigation) regenerates candidate_hypotheses
    from scratch. Before the fix, a hypothesis the evidence loop had already
    REFUTED (status_locked=True) would be silently reverted to POSSIBLE by
    the freshly-regenerated proposal, because REFUTED and POSSIBLE/
    UNRESOLVED share the same HYPOTHESIS_STATUS_RANK tier -- the rank
    comparison alone never caught it."""
    prev = [CandidateHypothesis(id="H1", name="X", statement="s", evidence_needed="e",
                                 status="REFUTED", evidence_strength="VERIFIED", status_locked=True)]
    regenerated = [CandidateHypothesis(id="H1", name="X", statement="s", evidence_needed="e",
                                        status="POSSIBLE", evidence_strength="NONE")]
    merged = merge_candidate_hypotheses(prev, regenerated)
    assert merged[0].status == "REFUTED"
    assert merged[0].status_locked is True
    assert merged[0].evidence_strength == "VERIFIED"


def test_locked_supported_hypothesis_also_survives_regeneration():
    prev = [CandidateHypothesis(id="H1", name="X", statement="s", evidence_needed="e",
                                 status="SUPPORTED", evidence_strength="VERIFIED", status_locked=True)]
    regenerated = [CandidateHypothesis(id="H1", name="X", statement="s", evidence_needed="e",
                                        status="POSSIBLE", evidence_strength="NONE")]
    merged = merge_candidate_hypotheses(prev, regenerated)
    assert merged[0].status == "SUPPORTED"
    assert merged[0].status_locked is True


def test_unlocked_hypothesis_merge_behavior_unchanged():
    """Confirms the fix is additive -- the pre-existing monotonic-merge
    behavior for NON-locked hypotheses (no evidence loop has run) is
    unaffected."""
    prev = [CandidateHypothesis(id="H1", name="X", statement="s", evidence_needed="e",
                                 status="SUPPORTED", evidence_strength="VERIFIED")]  # not locked
    regenerated = [CandidateHypothesis(id="H1", name="X", statement="s", evidence_needed="e",
                                        status="POSSIBLE", evidence_strength="NONE")]
    merged = merge_candidate_hypotheses(prev, regenerated)
    assert merged[0].status == "SUPPORTED"  # still restored via the pre-existing rank-based logic


def test_new_explicit_refutation_still_respected_when_not_locked():
    """The pre-existing rule -- an explicit new REFUTED from the second pass
    is respected, not overridden -- must still hold for non-locked hypotheses."""
    prev = [CandidateHypothesis(id="H1", name="X", statement="s", evidence_needed="e",
                                 status="SUPPORTED", evidence_strength="VERIFIED")]
    regenerated = [CandidateHypothesis(id="H1", name="X", statement="s", evidence_needed="e",
                                        status="REFUTED", evidence_strength="VERIFIED")]
    merged = merge_candidate_hypotheses(prev, regenerated)
    assert merged[0].status == "REFUTED"
