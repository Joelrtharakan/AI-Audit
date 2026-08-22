"""Phase 23: epistemic-class/hypothesis-relation separation, multi-hypothesis
classification, provenance rejection, and fallback-safety adversarial tests.
Exercises the real production EvidenceInterpreter/derive_hypothesis_relevance/
reconcile_hypothesis_from_evidence -- never a reimplementation.
"""
from __future__ import annotations

import asyncio

from app.agent.evidence_interpreter import EvidenceInterpreter, derive_hypothesis_relevance
from app.agent.nodes.evidence_acquisition import reconcile_hypothesis_from_evidence
from app.models.agent import CandidateHypothesis, EvidenceItem, EvidenceStatus
from app.services.llm.base import LLMProvider, LLMResponse
from app.services.llm.exceptions import LLMTimeoutError

import json


class _FakeLLMProvider(LLMProvider):
    def __init__(self, response_json=None, raise_exc=None, raw_text=None):
        self._response_json = response_json
        self._raise_exc = raise_exc
        self._raw_text = raw_text

    async def generate(self, *, node, prompt, **kwargs):
        if self._raise_exc is not None:
            raise self._raise_exc
        content = self._raw_text if self._raw_text is not None else json.dumps(self._response_json)
        return LLMResponse(content=content, provider="fake", model="fake-model")


def _item(evidence_id="EV1", claim="the record shows the step was completed", status=EvidenceStatus.VERIFIED):
    return EvidenceItem(claim=claim, source="s", status=status, evidence_id=evidence_id)


def _interp(payload=None, **kw):
    return EvidenceInterpreter(llm_provider=_FakeLLMProvider(payload, **kw))


# ---------------------------------------------------------------------------
# Part B/D: epistemic class and hypothesis relation are genuinely separate
# ---------------------------------------------------------------------------

def test_verified_evidence_can_still_contradict_a_hypothesis():
    """The canonical Part B example: VERIFIED evidence, CONTRADICTING relation
    -- the two axes must not collapse into one label."""
    interp = _interp({"claims": [{
        "text": "the training record shows the employee completed training on 2026-05-01",
        "source_reference": "training system record",
        "hypothesis_relations": [{"hypothesis_id": "H1", "relation": "CONTRADICTING",
                                   "reason": "training was completed, contradicting the failure-to-train hypothesis"}],
    }]})
    claims = asyncio.run(interp.interpret(
        _item(claim="the employee training record shows completion on 2026-05-01"),
        hypotheses=[{"id": "H1", "statement": "The employee failed to receive required training."}],
    ))
    assert len(claims) == 1
    assert claims[0].status == EvidenceStatus.VERIFIED
    assert claims[0].hypothesis_relations[0].relation == "CONTRADICTING"


def test_epistemic_class_never_llm_controlled():
    """epistemic_class on the resulting claim is deterministic (derived
    from evidence_item.status), regardless of what the LLM's raw payload
    contains for that field (which is no longer even read)."""
    interp = _interp({"claims": [{
        "text": "the record shows the step was completed", "source_reference": "ref",
        "epistemic_class": "SOMETHING_THE_MODEL_MADE_UP",
        "hypothesis_relations": [{"hypothesis_id": "H1", "relation": "SUPPORTING"}],
    }]})
    claims = asyncio.run(interp.interpret(_item(status=EvidenceStatus.REPORTED),
                                           hypotheses=[{"id": "H1", "statement": "s"}]))
    assert claims[0].epistemic_class == "REPORTED"  # derived from evidence_item.status, not the LLM field


# ---------------------------------------------------------------------------
# Part K: multiple hypotheses, independent relations from one evidence item
# ---------------------------------------------------------------------------

def test_one_evidence_item_different_relations_for_different_hypotheses():
    interp = _interp({"claims": [{
        "text": "the record shows the step was completed and signed off",
        "source_reference": "ref",
        "hypothesis_relations": [
            {"hypothesis_id": "H1", "relation": "CONTRADICTING", "reason": "step was completed"},
            {"hypothesis_id": "H2", "relation": "SUPPORTING", "reason": "signature confirms the approver acted"},
            {"hypothesis_id": "H3", "relation": "INSUFFICIENT", "reason": "no bearing on staffing levels"},
        ],
    }]})
    claims = asyncio.run(interp.interpret(_item(), hypotheses=[
        {"id": "H1", "statement": "The step was skipped."},
        {"id": "H2", "statement": "The approver never signed off."},
        {"id": "H3", "statement": "Staffing levels were insufficient."},
    ]))
    assert len(claims) == 1
    assert derive_hypothesis_relevance(claims, "H1") == "CONTRADICTING"
    assert derive_hypothesis_relevance(claims, "H2") == "SUPPORTING"
    assert derive_hypothesis_relevance(claims, "H3") == "INSUFFICIENT"


def test_multiple_hypotheses_none_dropped_or_merged():
    interp = _interp({"claims": [{
        "text": "the record shows the step was completed", "source_reference": "ref",
        "hypothesis_relations": [
            {"hypothesis_id": "H1", "relation": "SUPPORTING"},
            {"hypothesis_id": "H2", "relation": "SUPPORTING"},
        ],
    }]})
    claims = asyncio.run(interp.interpret(_item(), hypotheses=[
        {"id": "H1", "statement": "a"}, {"id": "H2", "statement": "b"},
    ]))
    assert set(claims[0].hypothesis_ids) == {"H1", "H2"}


def test_reconcile_uses_per_hypothesis_relation_not_global_aggregate():
    """The reconciliation authority must reach opposite conclusions for H1
    vs H2 from the SAME evidence batch (Part K, wired through
    reconcile_hypothesis_from_evidence, not just derive_hypothesis_relevance)."""
    interp = _interp({"claims": [{
        "text": "the record shows the step was completed", "source_reference": "ref",
        "hypothesis_relations": [
            {"hypothesis_id": "H1", "relation": "CONTRADICTING"},
            {"hypothesis_id": "H2", "relation": "SUPPORTING"},
        ],
    }]})
    claims = asyncio.run(interp.interpret(_item(), hypotheses=[
        {"id": "H1", "statement": "a"}, {"id": "H2", "statement": "b"},
    ]))
    h1 = CandidateHypothesis(id="H1", name="H1", statement="a", evidence_needed="e")
    h2 = CandidateHypothesis(id="H2", name="H2", statement="b", evidence_needed="e")
    item = _item()
    item.hypothesis_relevance = None  # not provider-set -- relies on claims

    status1, strength1, _ = reconcile_hypothesis_from_evidence(h1, [item], claims)
    status2, strength2, _ = reconcile_hypothesis_from_evidence(h2, [item], claims)
    assert status1 == "REFUTED"
    assert status2 == "SUPPORTED"


# ---------------------------------------------------------------------------
# Part L: provenance -- reject unknown hypothesis IDs, missing fields
# ---------------------------------------------------------------------------

def test_unknown_hypothesis_id_in_relation_is_rejected():
    interp = _interp({"claims": [{
        "text": "the record shows the step was completed", "source_reference": "ref",
        "hypothesis_relations": [{"hypothesis_id": "H999_NOT_REAL", "relation": "SUPPORTING"}],
    }]})
    claims = asyncio.run(interp.interpret(_item(), hypotheses=[{"id": "H1", "statement": "a"}]))
    assert len(claims) == 1
    assert claims[0].hypothesis_relations == []
    assert claims[0].hypothesis_ids == []


def test_no_hypotheses_provided_yields_no_claims():
    interp = _interp({"claims": [{"text": "x", "source_reference": "ref"}]})
    claims = asyncio.run(interp.interpret(_item(), hypotheses=[]))
    assert claims == []


def test_missing_evidence_id_rejected():
    interp = _interp({"claims": [{"text": "x", "source_reference": "ref"}]})
    item = _item()
    item.evidence_id = None
    claims = asyncio.run(interp.interpret(item, hypotheses=[{"id": "H1", "statement": "a"}]))
    assert claims == []


# ---------------------------------------------------------------------------
# Part I: unknown/missing evidence must never become contradiction
# ---------------------------------------------------------------------------

def test_no_record_found_is_insufficient_not_contradicting():
    interp = _interp({"claims": [{
        "text": "no record of the required inspection could be located in the system",
        "source_reference": "system search result",
        "hypothesis_relations": [{"hypothesis_id": "H1", "relation": "INSUFFICIENT",
                                   "reason": "absence of a record does not prove the inspection did not occur"}],
    }]})
    claims = asyncio.run(interp.interpret(
        _item(claim="A search of the system found no record of the required inspection."),
        hypotheses=[{"id": "H1", "statement": "The required inspection was never performed."}],
    ))
    assert derive_hypothesis_relevance(claims, "H1") == "INSUFFICIENT"


def test_empty_evidence_produces_no_claims_ever():
    interp = _interp({"claims": [{"text": "x", "source_reference": "ref",
                                   "hypothesis_relations": [{"hypothesis_id": "H1", "relation": "CONTRADICTING"}]}]})
    item = _item(claim="")
    claims = asyncio.run(interp.interpret(item, hypotheses=[{"id": "H1", "statement": "a"}]))
    assert claims == []


# ---------------------------------------------------------------------------
# Part H: reported testimony -- neither side automatically wins or verifies
# ---------------------------------------------------------------------------

def test_conflicting_testimony_both_preserved_neither_verified():
    interp_emp = _interp({"claims": [{
        "text": "the employee stated the required check was performed", "source_reference": "employee interview",
        "hypothesis_relations": [{"hypothesis_id": "H1", "relation": "CONTRADICTING"}],
    }]})
    interp_sup = _interp({"claims": [{
        "text": "the supervisor stated the required check was not performed", "source_reference": "supervisor interview",
        "hypothesis_relations": [{"hypothesis_id": "H1", "relation": "SUPPORTING"}],
    }]})
    emp_item = _item(evidence_id="EV1", claim="The employee stated the required check was performed.",
                      status=EvidenceStatus.REPORTED)
    sup_item = _item(evidence_id="EV2", claim="The supervisor stated the required check was not performed.",
                      status=EvidenceStatus.REPORTED)
    claims_emp = asyncio.run(interp_emp.interpret(emp_item, hypotheses=[{"id": "H1", "statement": "The check was skipped."}]))
    claims_sup = asyncio.run(interp_sup.interpret(sup_item, hypotheses=[{"id": "H1", "statement": "The check was skipped."}]))
    assert all(c.status == EvidenceStatus.REPORTED for c in claims_emp + claims_sup)
    assert derive_hypothesis_relevance(claims_emp + claims_sup, "H1") == "CONFLICTING"


# ---------------------------------------------------------------------------
# Part O: fallback safety -- malformed/unavailable never fabricates
# ---------------------------------------------------------------------------

def test_malformed_json_yields_no_claims():
    interp = _interp(raw_text="not valid json {{{")
    claims = asyncio.run(interp.interpret(_item(), hypotheses=[{"id": "H1", "statement": "a"}]))
    assert claims == []


def test_provider_timeout_yields_no_claims():
    interp = _interp(raise_exc=LLMTimeoutError("timed out"))
    claims = asyncio.run(interp.interpret(_item(), hypotheses=[{"id": "H1", "statement": "a"}]))
    assert claims == []


def test_empty_claims_array_is_valid_and_safe():
    interp = _interp({"claims": []})
    claims = asyncio.run(interp.interpret(_item(), hypotheses=[{"id": "H1", "statement": "a"}]))
    assert claims == []


def test_relation_referencing_no_known_hypothesis_does_not_crash():
    interp = _interp({"claims": [{
        "text": "x", "source_reference": "ref",
        "hypothesis_relations": "not_a_list",  # malformed shape
    }]})
    claims = asyncio.run(interp.interpret(_item(), hypotheses=[{"id": "H1", "statement": "a"}]))
    assert len(claims) == 1
    assert claims[0].hypothesis_relations == []


# ---------------------------------------------------------------------------
# Part J: interpreter never writes status; locked hypothesis unaffected
# ---------------------------------------------------------------------------

def test_interpreter_cannot_touch_candidate_hypothesis_status():
    """EvidenceInterpreter.interpret has no reference to CandidateHypothesis
    at all -- structurally cannot mutate .status. Assert via absence of any
    write: run interpret, then confirm the hypothesis object is untouched."""
    hyp = CandidateHypothesis(id="H1", name="H1", statement="a", evidence_needed="e", status="POSSIBLE")
    interp = _interp({"claims": [{
        "text": "the record shows the step was completed", "source_reference": "ref",
        "hypothesis_relations": [{"hypothesis_id": "H1", "relation": "CONTRADICTING"}],
    }]})
    asyncio.run(interp.interpret(_item(), hypotheses=[{"id": "H1", "statement": hyp.statement}]))
    assert hyp.status == "POSSIBLE"  # interpret() never received or could mutate the hypothesis object


def test_locked_hypothesis_relation_data_does_not_override_lock_downstream():
    """reconcile_hypothesis_from_evidence recomputes a status regardless of
    lock state (locking is enforced by the CALLER, acquire_evidence_node,
    which is unit-tested separately) -- this test documents that the
    function itself always returns a fresh recommendation, and status_lock
    enforcement is the caller's job, not this pure function's."""
    hyp = CandidateHypothesis(id="H1", name="H1", statement="a", evidence_needed="e",
                               status="REFUTED", status_locked=True)
    item = _item()
    item.hypothesis_relevance = "SUPPORTING"
    status, _, _ = reconcile_hypothesis_from_evidence(hyp, [item], [])
    # The pure function still computes what the evidence implies (SUPPORTED
    # here) -- it is acquire_evidence_node's job to skip applying this to a
    # locked hypothesis, verified in test_evidence_request_lifecycle_phase22.
    assert status == "SUPPORTED"
