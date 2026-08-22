"""Phase 21: EvidenceClaim / EvidenceInterpreter / claim-reconciliation
adversarial and integration coverage.

Exercises the real production components (app.agent.evidence_interpreter,
app.agent.claim_extractor.detect_evidence_conflicts, app.agent.nodes.
evidence_acquisition.reconcile_hypothesis_from_evidence) -- never
reimplements the pipeline inside the test. A FakeLLMProvider stands in for
a concrete LLM backend only where a deterministic canned response is
needed (unit-level adversarial cases); test_full_compiled_graph_evidence_claim_authority
drives the REAL compiled LangGraph via graph.ainvoke with a test-only
EvidenceProvider (external data access boundary) and a real
EvidenceInterpreter wired to the fake provider. test_live_ollama_* uses a
real local Ollama server and is skipped (not faked) when unavailable.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.agent.claim_extractor import detect_evidence_conflicts
from app.agent.evidence_interpreter import EvidenceInterpreter, derive_hypothesis_relevance
from app.agent.graph import build_agent_graph
from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.evidence_acquisition import reconcile_hypothesis_from_evidence
from app.models.agent import (
    AgentTraceStep,
    CandidateHypothesis,
    EvidenceItem,
    EvidenceRequest,
    EvidenceStatus,
    InvestigateRequest,
)
from app.services.evidence_provider import EvidenceProvider
from app.services.llm.base import LLMProvider, LLMResponse


class _FakeLLMProvider(LLMProvider):
    """Test-only canned-response provider implementing the real
    LLMProvider ABC -- EvidenceInterpreter never knows this isn't Ollama."""

    def __init__(self, response_json: dict | None = None, raise_exc: Exception | None = None,
                 raw_text: str | None = None):
        self._response_json = response_json
        self._raise_exc = raise_exc
        self._raw_text = raw_text
        self.calls: list[str] = []

    async def generate(self, *, node: str, prompt: str, **kwargs) -> LLMResponse:
        self.calls.append(prompt)
        if self._raise_exc is not None:
            raise self._raise_exc
        content = self._raw_text if self._raw_text is not None else json.dumps(self._response_json)
        return LLMResponse(content=content, provider="fake", model="fake-model")


def _evidence_item(evidence_id="EV1", claim="The training log shows the operator completed training on 2026-01-05.",
                    status=EvidenceStatus.VERIFIED, source="training_log"):
    return EvidenceItem(claim=claim, source=source, status=status, evidence_id=evidence_id)


def _hypothesis(hid="H1", statement="The operator was not trained on the revised procedure."):
    return CandidateHypothesis(id=hid, name="TRAINING_GAP", statement=statement, evidence_needed="training records")


# ---------------------------------------------------------------------------
# Adversarial: malformed / unsafe LLM output must never become a claim
# ---------------------------------------------------------------------------

def test_claim_without_evidence_id_is_rejected():
    """An evidence item with no evidence_id cannot license any claim."""
    interp = EvidenceInterpreter(llm_provider=_FakeLLMProvider({"claims": [{
        "text": "x", "epistemic_class": "OBSERVED", "source_reference": "ref",
    }]}))
    item = EvidenceItem(claim="something happened", source="s", status=EvidenceStatus.VERIFIED, evidence_id=None)
    claims = asyncio.run(interp.interpret(item, "hyp", "H1"))
    assert claims == []


def test_evidence_without_claim_text_produces_no_claims():
    interp = EvidenceInterpreter(llm_provider=_FakeLLMProvider({"claims": []}))
    item = EvidenceItem(claim="", source="s", status=EvidenceStatus.VERIFIED, evidence_id="EV1")
    claims = asyncio.run(interp.interpret(item, "hyp", "H1"))
    assert claims == []


def test_malformed_llm_output_returns_empty_not_invented():
    interp = EvidenceInterpreter(llm_provider=_FakeLLMProvider(raw_text="not json at all {{{"))
    claims = asyncio.run(interp.interpret(_evidence_item(), "hyp", "H1"))
    assert claims == []


def test_unknown_hypothesis_relation_is_rejected():
    """Phase 23/24: an unrecognized relation value is never trusted as-is.
    Phase 24 Part D folded this into the deterministic relation-validation
    firewall (validate_relation): rather than silently dropping the entry
    (Phase 23 behavior), it is REJECTed and replaced with the safe
    INSUFFICIENT value, with the rejection recorded in
    validation_decision -- auditable, not silent (Part Q: "record ...
    validation decision")."""
    interp = EvidenceInterpreter(llm_provider=_FakeLLMProvider({"claims": [{
        "text": "the operator completed training",
        "source_reference": "log line 1",
        "hypothesis_relations": [{"hypothesis_id": "H1", "relation": "DEFINITELY_TRUE"}],
    }]}))
    claims = asyncio.run(interp.interpret(_evidence_item(), "hyp", "H1"))
    assert len(claims) == 1
    assert len(claims[0].hypothesis_relations) == 1
    assert claims[0].hypothesis_relations[0].relation == "INSUFFICIENT"
    assert claims[0].hypothesis_relations[0].validation_decision == "REJECT"


def test_missing_source_reference_is_rejected():
    interp = EvidenceInterpreter(llm_provider=_FakeLLMProvider({"claims": [{
        "text": "the operator completed training",
        "epistemic_class": "OBSERVED",
        "source_reference": "",
    }]}))
    claims = asyncio.run(interp.interpret(_evidence_item(), "hyp", "H1"))
    assert claims == []


def test_hallucinated_claim_with_no_grounding_is_rejected():
    """A claim whose text shares no vocabulary at all with the evidence
    text is rejected even though the schema is otherwise valid -- the
    cheap grounding defense against hallucination."""
    interp = EvidenceInterpreter(llm_provider=_FakeLLMProvider({"claims": [{
        "text": "the warehouse forklift was overdue for inspection",
        "epistemic_class": "OBSERVED",
        "source_reference": "log line 1",
    }]}))
    claims = asyncio.run(interp.interpret(_evidence_item(), "hyp", "H1"))
    assert claims == []


def test_provider_exception_fails_safe_not_crash():
    interp = EvidenceInterpreter(llm_provider=_FakeLLMProvider(raise_exc=httpx.ConnectError("no route")))
    claims = asyncio.run(interp.interpret(_evidence_item(), "hyp", "H1"))
    assert claims == []


def test_provider_timeout_fails_safe():
    from app.services.llm.exceptions import LLMTimeoutError
    interp = EvidenceInterpreter(llm_provider=_FakeLLMProvider(raise_exc=LLMTimeoutError("timed out")))
    claims = asyncio.run(interp.interpret(_evidence_item(), "hyp", "H1"))
    assert claims == []


# ---------------------------------------------------------------------------
# Well-formed claims: provenance, status capping, epistemic classes
# ---------------------------------------------------------------------------

def test_valid_supporting_claim_is_produced_with_provenance():
    interp = EvidenceInterpreter(llm_provider=_FakeLLMProvider({"claims": [{
        "text": "the training log shows the operator completed training",
        "subject": "operator", "predicate": "completed", "object": "training",
        "source_reference": "training log entry",
        "temporal_context": "2026-01-05",
        "hypothesis_relations": [{"hypothesis_id": "H1", "relation": "CONTRADICTING",
                                   "reason": "contradicts the not-trained hypothesis"}],
    }]}))
    claims = asyncio.run(interp.interpret(_evidence_item(), "operator was not trained", "H1"))
    assert len(claims) == 1
    c = claims[0]
    assert c.evidence_ids == ["EV1"]
    assert c.hypothesis_ids == ["H1"]
    assert c.hypothesis_relations[0].relation == "CONTRADICTING"
    assert c.epistemic_class == "OBSERVED"  # deterministic, derived from evidence item's own VERIFIED status
    assert c.status == EvidenceStatus.VERIFIED  # evidence item itself was VERIFIED
    assert c.extraction_status == "EXTRACTED"


def test_reported_evidence_never_upgrades_claim_to_verified():
    """Part D: REPORTED evidence must never produce a VERIFIED claim, even
    if the LLM's epistemic_class implies directness."""
    item = _evidence_item(claim="A supervisor said the operator completed training.",
                           status=EvidenceStatus.REPORTED)
    interp = EvidenceInterpreter(llm_provider=_FakeLLMProvider({"claims": [{
        "text": "the supervisor said the operator completed training",
        "epistemic_class": "OBSERVED",  # LLM incorrectly claims directness
        "source_reference": "supervisor statement",
    }]}))
    claims = asyncio.run(interp.interpret(item, "hyp", "H1"))
    assert len(claims) == 1
    assert claims[0].status == EvidenceStatus.REPORTED  # capped, never VERIFIED


def test_missing_evidence_never_becomes_negative_claim():
    """UNKNOWN epistemic_class must never be interpreted as CONTRADICTING."""
    interp = EvidenceInterpreter(llm_provider=_FakeLLMProvider({"claims": [{
        "text": "the training log shows the operator completed training",
        "epistemic_class": "UNKNOWN",
        "source_reference": "training log entry",
    }]}))
    claims = asyncio.run(interp.interpret(_evidence_item(), "hyp", "H1"))
    assert len(claims) == 1
    relevance = derive_hypothesis_relevance(claims)
    assert relevance == "INSUFFICIENT"  # not CONTRADICTING, not SUPPORTING


# ---------------------------------------------------------------------------
# Claim reconciliation: contradiction is preserved, never auto-resolved
# ---------------------------------------------------------------------------

def test_contradictory_claims_are_both_preserved_with_conflict_record():
    ev1 = _evidence_item(evidence_id="EV1", source="supervisor_statement", status=EvidenceStatus.REPORTED)
    ev2 = _evidence_item(evidence_id="EV2", source="operator_statement", status=EvidenceStatus.REPORTED)
    interp1 = EvidenceInterpreter(llm_provider=_FakeLLMProvider({"claims": [{
        "text": "the operator completed the required training",
        "epistemic_class": "REPORTED", "source_reference": "supervisor statement",
    }]}))
    interp2 = EvidenceInterpreter(llm_provider=_FakeLLMProvider({"claims": [{
        "text": "the operator did not complete the required training",
        "epistemic_class": "REPORTED", "source_reference": "operator statement",
    }]}))
    claims1 = asyncio.run(interp1.interpret(ev1, "hyp", "H1"))
    claims2 = asyncio.run(interp2.interpret(ev2, "hyp", "H1"))
    assert len(claims1) == 1 and len(claims2) == 1

    conflicts = detect_evidence_conflicts(claims1 + claims2)
    assert len(conflicts) >= 1, "contradictory REPORTED claims about the same proposition must produce a conflict record"
    conflict = conflicts[0]
    assert conflict.status == "UNRESOLVED"
    assert set(conflict.claims) == {claims1[0].claim_id, claims2[0].claim_id}
    # Neither claim was deleted or silently overwritten -- both preserved.
    assert claims1[0].text != claims2[0].text


def test_duplicate_equivalent_claims_do_not_manufacture_a_conflict():
    ev1 = _evidence_item(evidence_id="EV1")
    ev2 = _evidence_item(evidence_id="EV2")
    interp = EvidenceInterpreter(llm_provider=_FakeLLMProvider({"claims": [{
        "text": "the operator completed training", "epistemic_class": "REPORTED",
        "source_reference": "ref",
    }]}))
    c1 = asyncio.run(interp.interpret(ev1, "hyp", "H1"))
    c2 = asyncio.run(interp.interpret(ev2, "hyp", "H1"))
    conflicts = detect_evidence_conflicts(c1 + c2)
    assert conflicts == []


# ---------------------------------------------------------------------------
# derive_hypothesis_relevance: deterministic aggregation, no LLM authority
# ---------------------------------------------------------------------------

def test_derive_hypothesis_relevance_supporting_only():
    interp = EvidenceInterpreter(llm_provider=_FakeLLMProvider({"claims": [{
        "text": "the operator completed training", "source_reference": "ref",
        "hypothesis_relations": [{"hypothesis_id": "H1", "relation": "SUPPORTING"}],
    }]}))
    claims = asyncio.run(interp.interpret(_evidence_item(claim="the operator completed training"), "hyp", "H1"))
    assert derive_hypothesis_relevance(claims, "H1") == "SUPPORTING"


def test_derive_hypothesis_relevance_conflicting_when_both_present():
    a = EvidenceInterpreter(llm_provider=_FakeLLMProvider({"claims": [{
        "text": "the operator completed training", "source_reference": "r1",
        "hypothesis_relations": [{"hypothesis_id": "H1", "relation": "SUPPORTING"}],
    }]}))
    b = EvidenceInterpreter(llm_provider=_FakeLLMProvider({"claims": [{
        "text": "the operator completed training late", "source_reference": "r2",
        "hypothesis_relations": [{"hypothesis_id": "H1", "relation": "CONTRADICTING"}],
    }]}))
    claims_a = asyncio.run(a.interpret(_evidence_item(evidence_id="EV1", claim="the operator completed training"), "hyp", "H1"))
    claims_b = asyncio.run(b.interpret(_evidence_item(evidence_id="EV2", claim="the operator completed training late"), "hyp", "H1"))
    assert derive_hypothesis_relevance(claims_a + claims_b, "H1") == "CONFLICTING"


def test_derive_hypothesis_relevance_none_when_no_claims():
    assert derive_hypothesis_relevance([]) is None


# ---------------------------------------------------------------------------
# reconcile_hypothesis_from_evidence: claims inform but do not bypass the
# single authoritative evaluator (Phase 20 contract unchanged)
# ---------------------------------------------------------------------------

def test_reconcile_with_claims_still_deterministic_and_capped():
    hyp = _hypothesis()
    ev = _evidence_item(status=EvidenceStatus.REPORTED)
    ev.hypothesis_relevance = "SUPPORTING"
    status, strength, reason = reconcile_hypothesis_from_evidence(hyp, [ev], claims=[])
    assert status == "POSSIBLE"  # unchanged -- REPORTED can't promote to SUPPORTED
    assert strength == "REPORTED"


def test_reconcile_status_escalation_requires_verified_evidence_not_just_claim_count():
    hyp = _hypothesis()
    ev = _evidence_item(status=EvidenceStatus.REPORTED)
    ev.hypothesis_relevance = "SUPPORTING"
    fake_claims = [object()] * 5  # many "claims" -- must not matter, only evidence status does
    status, strength, reason = reconcile_hypothesis_from_evidence(hyp, [ev], claims=fake_claims)
    assert status != "SUPPORTED"


# ---------------------------------------------------------------------------
# INV-INVEST-029: dangling / provenance-less claims must never validate
# ---------------------------------------------------------------------------

def test_invariant_rejects_claim_with_no_evidence_ids():
    from app.models.agent import EvidenceClaim
    bad_claim = EvidenceClaim(claim_id="C1", text="x", source="s", status=EvidenceStatus.VERIFIED, evidence_ids=[])
    state = {"evidence_claims": [bad_claim], "evidence_ledger": []}
    is_valid, violations = evaluate_all_invariants(state)
    assert any("INV-INVEST-029" in v for v in violations)


def test_invariant_rejects_dangling_evidence_id():
    from app.models.agent import EvidenceClaim
    bad_claim = EvidenceClaim(claim_id="C1", text="x", source="s", status=EvidenceStatus.VERIFIED,
                               evidence_ids=["EV_DOES_NOT_EXIST"])
    state = {"evidence_claims": [bad_claim], "evidence_ledger": [_evidence_item(evidence_id="EV1")]}
    is_valid, violations = evaluate_all_invariants(state)
    assert any("INV-INVEST-029" in v for v in violations)


def test_invariant_passes_for_claim_with_real_provenance():
    from app.models.agent import EvidenceClaim
    good_claim = EvidenceClaim(claim_id="C1", text="x", source="s", status=EvidenceStatus.VERIFIED,
                                evidence_ids=["EV1"])
    state = {"evidence_claims": [good_claim], "evidence_ledger": [_evidence_item(evidence_id="EV1")]}
    is_valid, violations = evaluate_all_invariants(state)
    assert not any("INV-INVEST-029" in v for v in violations)


# ---------------------------------------------------------------------------
# Compiled LangGraph end-to-end: EvidenceItem -> EvidenceInterpreter ->
# EvidenceClaim -> reconciliation -> status transition, through graph.ainvoke
# ---------------------------------------------------------------------------

class _ClaimBackedEvidenceProvider(EvidenceProvider):
    """Test-only (external data access boundary): returns real evidence
    TEXT but leaves hypothesis_relevance unset, so the production
    EvidenceInterpreter -> reconcile pipeline (not the provider) decides
    relevance -- proving the LLM interpretation path is what's exercised,
    not a provider shortcut."""

    def __init__(self, target_hid: str):
        self.target_hid = target_hid

    async def acquire(self, request: EvidenceRequest) -> EvidenceItem:
        if self.target_hid in request.hypothesis_ids:
            return EvidenceItem(
                claim="The training log shows the operator completed the revised inspection checklist training.",
                source="training_log", status=EvidenceStatus.VERIFIED,
            )
        return EvidenceItem(claim="", source="none", status=EvidenceStatus.UNVERIFIED,
                             hypothesis_relevance="UNAVAILABLE")


def test_full_compiled_graph_evidence_claim_authority():
    """Section Y: real graph.ainvoke() exercising EvidenceRequest ->
    EvidenceProvider -> EvidenceItem -> EvidenceInterpreter -> EvidenceClaim
    -> reconcile_hypothesis_from_evidence -> HypothesisStatusChange, with no
    manual mutation of CandidateHypothesis.status anywhere in this test."""
    graph = build_agent_graph()
    fake_llm = _FakeLLMProvider({"claims": [{
        "text": "the operator completed the revised inspection checklist training",
        "subject": "operator", "predicate": "completed", "object": "training",
        "source_reference": "training log",
        "hypothesis_relations": [{"hypothesis_id": "H1", "relation": "CONTRADICTING",
                                   "reason": "contradicts a not-trained hypothesis"}],
    }]})
    interpreter = EvidenceInterpreter(llm_provider=fake_llm)
    provider = _ClaimBackedEvidenceProvider(target_hid="H1")

    state = {
        "request": InvestigateRequest(finding_text=(
            "Four employees failed to complete the revised inspection checklist. "
            "One employee reported insufficient training. "
            "Another employee reported workload pressure. "
            "The supervisor reported poor discipline."
        )),
        "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0,
        "evidence_ledger": [], "errors": [], "trace": [AgentTraceStep.ok("start")],
        "evidence_provider": provider,
        "evidence_interpreter": interpreter,
    }

    from unittest.mock import patch
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        final_state = asyncio.run(graph.ainvoke(state))

    claims = final_state.get("evidence_claims") or []
    assert any(
        r.relation == "CONTRADICTING" for c in claims for r in c.hypothesis_relations
    ), "the real EvidenceInterpreter must have produced a runtime EvidenceClaim, not a test double"
    assert all(c.evidence_ids for c in claims), "every claim must carry evidence provenance"

    history = final_state.get("hypothesis_history") or []
    assert history, "the real evidence acquisition -> claim -> reconciliation loop must have executed"

    is_valid, violations = evaluate_all_invariants(final_state)
    assert not any("INV-INVEST-028" in v for v in violations)
    assert not any("INV-INVEST-029" in v for v in violations)


# ---------------------------------------------------------------------------
# Live Ollama validation (Part AO) -- real network call, real model, no
# fabrication. Skipped (never faked) when Ollama is unreachable.
# ---------------------------------------------------------------------------

def _ollama_reachable() -> bool:
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.mark.live_ollama
@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama is not reachable at localhost:11434")
def test_live_ollama_evidence_interpretation():
    """Real end-to-end call through app.services.llm.factory.get_llm_provider()
    -> OllamaProvider -> qwen3:8b. Not mocked. If the live model's output
    fails strict schema/grounding validation, EvidenceInterpreter must fail
    safe (empty list), not fabricate -- this test asserts that safety
    property holds against a REAL model, whatever it actually returns."""
    from app.services.llm.factory import get_llm_provider

    provider = get_llm_provider("ollama")
    interp = EvidenceInterpreter(llm_provider=provider)
    item = _evidence_item(
        claim="Training system record: operator J. Smith completed the revised inspection "
              "checklist training module on 2026-01-05, confirmed by digital signature.",
        status=EvidenceStatus.VERIFIED,
    )
    claims = asyncio.run(interp.interpret(
        item, hypothesis_statement="The operator was not trained on the revised procedure.",
        hypothesis_id="H1", question="Was the operator trained on the revised checklist?",
    ))
    # Whatever the live model produced, every surviving claim must satisfy
    # the strict provenance/vocabulary contract -- proving validation, not
    # the model's cooperation, is what guarantees safety.
    for c in claims:
        assert c.evidence_ids == ["EV1"]
        assert c.epistemic_class in {"SUPPORTING", "CONTRADICTING", "OBSERVED", "REPORTED", "UNKNOWN"}
        assert c.extraction_status == "EXTRACTED"
    print(f"\nLIVE OLLAMA RESULT: {len(claims)} claim(s) survived validation: "
          f"{[(c.epistemic_class, c.text) for c in claims]}")
