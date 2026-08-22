"""Phase 24: deterministic relation-validation firewall, absence-of-evidence
vs evidence-of-absence, quantitative arithmetic safety, temporal structure,
and downstream CAPA/Impact/status-authority regression. Exercises the real
production functions (validate_relation, verify_quantitative, EvidenceInterpreter,
reconcile_hypothesis_from_evidence) -- never a reimplementation.
"""
from __future__ import annotations

import asyncio
import json

from app.agent.evidence_interpreter import (
    EvidenceInterpreter,
    derive_hypothesis_relevance,
    validate_relation,
    verify_quantitative,
)
from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.evidence_acquisition import reconcile_hypothesis_from_evidence
from app.agent.nodes.plan_investigation_fallback import build_conditional_capa_actions
from app.models.agent import (
    CandidateHypothesis,
    EvidenceProposition,
    EvidenceItem,
    EvidenceStatus,
    ImpactAssessment,
    QuantitativeAssertion,
    RootCauseAnalysis,
    RootCauseStatus,
)
from app.services.llm.base import LLMProvider, LLMResponse


class _FakeLLMProvider(LLMProvider):
    def __init__(self, response_json=None):
        self._response_json = response_json

    async def generate(self, *, node, prompt, **kwargs):
        return LLMResponse(content=json.dumps(self._response_json), provider="fake", model="fake-model")


def _item(evidence_id="EV1", claim="the record shows the step was completed", status=EvidenceStatus.VERIFIED):
    return EvidenceItem(claim=claim, source="s", status=status, evidence_id=evidence_id)


def _interp(payload):
    return EvidenceInterpreter(llm_provider=_FakeLLMProvider(payload))


# ---------------------------------------------------------------------------
# Part D: validate_relation firewall -- pure function, direct tests
# ---------------------------------------------------------------------------

def test_firewall_accepts_supporting_on_positive_assertion():
    rel, decision = validate_relation("SUPPORTING", EvidenceStatus.VERIFIED, EvidenceProposition.POSITIVE_ASSERTION)
    assert (rel, decision) == ("SUPPORTING", "ACCEPT")


def test_firewall_downgrades_absence_of_evidence_supporting():
    rel, decision = validate_relation("SUPPORTING", EvidenceStatus.VERIFIED, EvidenceProposition.ABSENCE_OF_EVIDENCE)
    assert rel == "INSUFFICIENT"
    assert decision == "DOWNGRADE_TO_INSUFFICIENT"


def test_firewall_downgrades_absence_of_evidence_contradicting():
    rel, decision = validate_relation("CONTRADICTING", EvidenceStatus.REPORTED, EvidenceProposition.ABSENCE_OF_EVIDENCE)
    assert rel == "INSUFFICIENT"
    assert decision == "DOWNGRADE_TO_INSUFFICIENT"


def test_firewall_allows_evidence_of_absence_to_stand():
    """The Part B/E exception: an evidence text that itself establishes the
    source is authoritative/exhaustive MAY license CONTRADICTING."""
    rel, decision = validate_relation("CONTRADICTING", EvidenceStatus.VERIFIED, EvidenceProposition.EVIDENCE_OF_ABSENCE)
    assert rel == "CONTRADICTING"
    assert decision == "ACCEPT"


def test_firewall_rejects_unknown_relation_value():
    rel, decision = validate_relation("DEFINITELY_TRUE", EvidenceStatus.VERIFIED, EvidenceProposition.POSITIVE_ASSERTION)
    assert rel == "INSUFFICIENT"
    assert decision == "REJECT"


def test_firewall_downgrades_unknown_status_relational_claim():
    rel, decision = validate_relation("SUPPORTING", EvidenceStatus.UNKNOWN, EvidenceProposition.POSITIVE_ASSERTION)
    assert rel == "INSUFFICIENT"
    assert decision == "DOWNGRADE_TO_INSUFFICIENT"


def test_firewall_downgrades_on_bad_arithmetic():
    rel, decision = validate_relation("SUPPORTING", EvidenceStatus.VERIFIED, EvidenceProposition.POSITIVE_ASSERTION,
                                       quantitative_valid=False)
    assert rel == "INSUFFICIENT"
    assert decision == "DOWNGRADE_TO_INSUFFICIENT"


def test_firewall_never_touches_neutral_or_insufficient():
    """NEUTRAL/INSUFFICIENT are already safe -- the firewall must not
    "upgrade" them or otherwise interfere."""
    rel, decision = validate_relation("NEUTRAL", EvidenceStatus.UNKNOWN, EvidenceProposition.ABSENCE_OF_EVIDENCE)
    assert (rel, decision) == ("NEUTRAL", "ACCEPT")
    rel, decision = validate_relation("INSUFFICIENT", EvidenceStatus.VERIFIED, EvidenceProposition.ABSENCE_OF_EVIDENCE)
    assert (rel, decision) == ("INSUFFICIENT", "ACCEPT")


# ---------------------------------------------------------------------------
# Part B/E: the exact Phase 23 defect, fixed structurally
# ---------------------------------------------------------------------------

def test_absence_of_evidence_end_to_end_no_longer_supports():
    """Reproduces the Phase 23 'missing_evidence -> SUPPORTING' defect
    structurally: an LLM that (incorrectly) proposes SUPPORTING for an
    ABSENCE_OF_EVIDENCE claim is now overridden by the deterministic
    firewall, end to end through EvidenceInterpreter.interpret()."""
    interp = _interp({"claims": [{
        "text": "no record of the required review could be located after a full search",
        "source_reference": "document management system search",
        "proposition_type": "ABSENCE_OF_EVIDENCE",
        "hypothesis_relations": [{"hypothesis_id": "H1", "relation": "SUPPORTING",
                                   "reason": "the model incorrectly treats absence as proof"}],
    }]})
    claims = asyncio.run(interp.interpret(
        _item(claim="A full search of the document management system found no record of the required review."),
        hypotheses=[{"id": "H1", "statement": "The required review was never conducted."}],
    ))
    assert len(claims) == 1
    assert derive_hypothesis_relevance(claims, "H1") == "INSUFFICIENT"
    assert claims[0].hypothesis_relations[0].validation_decision == "DOWNGRADE_TO_INSUFFICIENT"


def test_evidence_of_absence_permitted_to_contradict():
    """The other half of Part E ('do not overcorrect'): an authoritative,
    exhaustive record explicitly establishing non-occurrence IS allowed to
    contradict a hypothesis."""
    interp = _interp({"claims": [{
        "text": "the complete, gapless access log for the server room shows no entry for the "
                "suspect during the incident window",
        "source_reference": "complete access control log",
        "proposition_type": "EVIDENCE_OF_ABSENCE",
        "hypothesis_relations": [{"hypothesis_id": "H1", "relation": "CONTRADICTING",
                                   "reason": "an exhaustive log with no entry affirmatively rules this out"}],
    }]})
    claims = asyncio.run(interp.interpret(
        _item(claim="The complete, gapless access control log shows no entry for the suspect "
                    "during the incident window."),
        hypotheses=[{"id": "H1", "statement": "The suspect accessed the server room during the incident window."}],
    ))
    assert derive_hypothesis_relevance(claims, "H1") == "CONTRADICTING"


# ---------------------------------------------------------------------------
# Part F: quantitative arithmetic safety
# ---------------------------------------------------------------------------

def test_verify_quantitative_true_when_arithmetic_holds():
    q = QuantitativeAssertion(left=4850, operator="GT", right=3520, unit="particles/m3")
    assert verify_quantitative(q) is True


def test_verify_quantitative_false_when_arithmetic_does_not_hold():
    q = QuantitativeAssertion(left=100, operator="GT", right=3520)  # 100 is NOT > 3520
    assert verify_quantitative(q) is False


def test_verify_quantitative_none_is_trivially_valid():
    assert verify_quantitative(None) is True


def test_inconsistent_quantitative_claim_downgrades_relation():
    interp = _interp({"claims": [{
        "text": "the measured value of 100 exceeded the limit of 3520", "source_reference": "sensor log",
        "quantitative": {"left": 100, "operator": "GT", "right": 3520},  # arithmetically false
        "hypothesis_relations": [{"hypothesis_id": "H1", "relation": "SUPPORTING"}],
    }]})
    claims = asyncio.run(interp.interpret(_item(claim="the measured value of 100 exceeded the limit of 3520"),
                                           hypotheses=[{"id": "H1", "statement": "the limit was exceeded"}]))
    assert len(claims) == 1
    assert claims[0].quantitative is None  # inconsistent assertion dropped
    assert derive_hypothesis_relevance(claims, "H1") == "INSUFFICIENT"


def test_consistent_quantitative_claim_is_preserved_and_relation_stands():
    interp = _interp({"claims": [{
        "text": "the measured value of 4850 exceeded the limit of 3520", "source_reference": "sensor log",
        "quantitative": {"left": 4850, "operator": "GT", "right": 3520, "unit": "particles/m3"},
        "hypothesis_relations": [{"hypothesis_id": "H1", "relation": "SUPPORTING"}],
    }]})
    claims = asyncio.run(interp.interpret(_item(claim="the measured value of 4850 exceeded the limit of 3520"),
                                           hypotheses=[{"id": "H1", "statement": "the limit was exceeded"}]))
    assert claims[0].quantitative.left == 4850
    assert derive_hypothesis_relevance(claims, "H1") == "SUPPORTING"


# ---------------------------------------------------------------------------
# Part G: temporal structure -- never inferred as causal proof
# ---------------------------------------------------------------------------

def test_temporal_relation_recorded_structurally():
    interp = _interp({"claims": [{
        "text": "the door was accessed after the authorized window closed", "source_reference": "badge log",
        "temporal_relation": "AFTER",
        "hypothesis_relations": [{"hypothesis_id": "H1", "relation": "SUPPORTING"}],
    }]})
    claims = asyncio.run(interp.interpret(_item(claim="the door was accessed after the authorized window closed"),
                                           hypotheses=[{"id": "H1", "statement": "access occurred outside the window"}]))
    assert claims[0].temporal_relation == "AFTER"


# ---------------------------------------------------------------------------
# Part I/K: multi-hypothesis matrix, unchanged and reconfirmed
# ---------------------------------------------------------------------------

def test_one_evidence_three_hypotheses_independent_relations_all_preserved():
    interp = _interp({"claims": [{
        "text": "the record shows the step was completed and signed off", "source_reference": "ref",
        "proposition_type": "POSITIVE_ASSERTION",
        "hypothesis_relations": [
            {"hypothesis_id": "H1", "relation": "SUPPORTING"},
            {"hypothesis_id": "H2", "relation": "CONTRADICTING"},
            {"hypothesis_id": "H3", "relation": "INSUFFICIENT"},
        ],
    }]})
    claims = asyncio.run(interp.interpret(_item(), hypotheses=[
        {"id": "H1", "statement": "a"}, {"id": "H2", "statement": "b"}, {"id": "H3", "statement": "c"},
    ]))
    assert derive_hypothesis_relevance(claims, "H1") == "SUPPORTING"
    assert derive_hypothesis_relevance(claims, "H2") == "CONTRADICTING"
    assert derive_hypothesis_relevance(claims, "H3") == "INSUFFICIENT"


# ---------------------------------------------------------------------------
# Part L/M/N: status-transition policy, root-cause safety, CAPA/Impact
# ---------------------------------------------------------------------------

def test_single_supporting_reported_claim_does_not_promote_to_supported():
    """Part L: supporting evidence != sufficient evidence for confirmation.
    A single REPORTED-status supporting claim must not promote to SUPPORTED."""
    hyp = CandidateHypothesis(id="H1", name="H1", statement="a", evidence_needed="e")
    item = _item(status=EvidenceStatus.REPORTED)
    item.hypothesis_relevance = None
    interp = _interp({"claims": [{
        "text": "a witness said the step was completed", "source_reference": "interview",
        "hypothesis_relations": [{"hypothesis_id": "H1", "relation": "SUPPORTING"}],
    }]})
    claims = asyncio.run(interp.interpret(item, hypotheses=[{"id": "H1", "statement": "a"}]))
    status, strength, _ = reconcile_hypothesis_from_evidence(hyp, [item], claims)
    assert status == "POSSIBLE"  # not SUPPORTED
    assert strength == "REPORTED"


def test_refuted_hypothesis_never_produces_capa():
    hyp = CandidateHypothesis(id="H1", name="X", statement="s", evidence_needed="e", status="REFUTED")
    actions = build_conditional_capa_actions([hyp], "record", "record")
    assert actions == []


def test_possible_hypothesis_capa_remains_conditional():
    hyp = CandidateHypothesis(id="H1", name="X", statement="s", evidence_needed="e", status="POSSIBLE")
    actions = build_conditional_capa_actions([hyp], "record", "record")
    assert actions and all(a.if_cause_confirmed.startswith("IF ") for a in actions)


def test_impact_verified_requires_verified_evidence_invariant_still_enforced():
    impact = ImpactAssessment(status="IMPACT_VERIFIED")
    unknown_item = _item(status=EvidenceStatus.UNVERIFIED)
    state = {"impact_assessment": impact, "evidence_ledger": [unknown_item]}
    is_valid, violations = evaluate_all_invariants(state)
    assert any("INV-IMPACT-002" in v for v in violations)


# ---------------------------------------------------------------------------
# Part K/status-authority: unaffected by Phase 24 changes
# ---------------------------------------------------------------------------

def test_single_epistemic_authority_invariant_still_holds():
    hyp = CandidateHypothesis(id="H1", name="H1", statement="a", evidence_needed="e",
                               status="SUPPORTED", status_locked=True)
    from app.models.agent import HypothesisStatusChange
    history = [HypothesisStatusChange(hypothesis_id="H1", previous_status="POSSIBLE", new_status="SUPPORTED")]
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[hyp])
    state = {"root_cause": rc, "hypothesis_history": history}
    is_valid, violations = evaluate_all_invariants(state)
    assert not any("INV-INVEST-028" in v for v in violations)
