"""Phase 22: EvidenceRequest lifecycle (identity/dedup/parent-child/
idempotency), CAPA/Impact epistemic safety, and provider-failure adversarial
coverage. Exercises the real production functions -- never reimplements
their logic inline.
"""
from __future__ import annotations

import asyncio

from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.evidence_acquisition import (
    acquire_evidence_node,
    build_evidence_requests,
    reconcile_evidence_request,
    request_identity,
)
from app.agent.nodes.plan_investigation_fallback import build_conditional_capa_actions
from app.models.agent import (
    AgentTraceStep,
    CandidateHypothesis,
    CapaAnalysis,
    CapaStatus,
    ConditionalCapaAction,
    EvidenceItem,
    EvidenceRequest,
    EvidenceStatus,
    ImpactAssessment,
    InvestigationQuestion,
    RootCauseAnalysis,
    RootCauseStatus,
)
from app.services.evidence_provider import EvidenceProvider


def _req(target_node_id="N1", hypothesis_ids=None, required_artifacts=None, objective="", request_id=None):
    import uuid
    return EvidenceRequest(
        request_id=request_id or f"REQ_{uuid.uuid4().hex[:8]}",
        target_node_id=target_node_id,
        hypothesis_ids=hypothesis_ids or ["H1"],
        required_artifacts=required_artifacts or ["calibration_certificate"],
        objective=objective,
    )


# ---------------------------------------------------------------------------
# Part B: structured identity (wording-independent)
# ---------------------------------------------------------------------------

def test_identity_ignores_wording_and_question_id():
    a = _req(objective="Was the instrument calibrated before use?", request_id="R1")
    b = _req(objective="Confirm calibration status prior to the affected batch.", request_id="R2")
    assert request_identity(a) == request_identity(b)


def test_identity_differs_for_different_required_artifacts():
    a = _req(required_artifacts=["calibration_certificate"])
    b = _req(required_artifacts=["maintenance_log"])
    assert request_identity(a) != request_identity(b)


def test_identity_differs_for_different_hypothesis():
    a = _req(hypothesis_ids=["H1"])
    b = _req(hypothesis_ids=["H2"])
    assert request_identity(a) != request_identity(b)


# ---------------------------------------------------------------------------
# Part C: deduplication -- exact structural duplicate reused
# ---------------------------------------------------------------------------

def test_duplicate_structured_request_is_reused_not_recreated():
    existing = _req(request_id="R1")
    existing.status = "REQUESTED"
    dup = _req(request_id="R2", objective="differently worded but same target")
    result, action = reconcile_evidence_request(dup, [existing])
    assert action == "REUSED"
    assert result.request_id == "R1"


def test_fulfilled_duplicate_is_reused_not_recreated():
    existing = _req(request_id="R1")
    existing.status = "FULFILLED"
    dup = _req(request_id="R2")
    result, action = reconcile_evidence_request(dup, [existing])
    assert action == "REUSED"
    assert result.status == "FULFILLED"


def test_genuinely_different_request_is_created():
    existing = _req(request_id="R1", hypothesis_ids=["H1"])
    existing.status = "FULFILLED"
    different = _req(request_id="R2", hypothesis_ids=["H2"])
    result, action = reconcile_evidence_request(different, [existing])
    assert action == "CREATED"
    assert result.request_id == "R2"


# ---------------------------------------------------------------------------
# Part D: legitimate refinement -- parent preserved, never erased
# ---------------------------------------------------------------------------

def test_refinement_of_unresolved_request_sets_parent_and_supersedes():
    parent = _req(request_id="R1", hypothesis_ids=["H1"], required_artifacts=["general_execution_record"])
    parent.status = "UNRESOLVED"
    child = _req(request_id="R2", hypothesis_ids=["H1"], required_artifacts=["authenticated_execution_log"])
    result, action = reconcile_evidence_request(child, [parent])
    assert action == "REFINED"
    assert result.parent_request_id == "R1"
    assert parent.status == "SUPERSEDED"  # never deleted -- still in the ledger, just marked


def test_refinement_not_triggered_by_unrelated_unresolved_request():
    unrelated = _req(request_id="R1", hypothesis_ids=["H9"], target_node_id="N9")
    unrelated.status = "UNRESOLVED"
    fresh = _req(request_id="R2", hypothesis_ids=["H1"], target_node_id="N1")
    result, action = reconcile_evidence_request(fresh, [unrelated])
    assert action == "CREATED"
    assert result.parent_request_id is None


# ---------------------------------------------------------------------------
# Part F: idempotency through the real node -- repeated acquisition
# ---------------------------------------------------------------------------

class _CountingProvider(EvidenceProvider):
    def __init__(self):
        self.call_count = 0

    async def acquire(self, request):
        self.call_count += 1
        return EvidenceItem(
            claim="Objective record confirms the hypothesis.", source="test",
            status=EvidenceStatus.VERIFIED, hypothesis_relevance="SUPPORTING",
        )


def _plan_and_state(provider):
    plan = type("Plan", (), {"questions": [
        InvestigationQuestion(question="q", target_node_id="N1", target_hypothesis_ids=["H1"], evidence_required="record"),
    ]})()
    hyp = CandidateHypothesis(id="H1", name="H", statement="s", evidence_needed="e")
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[hyp])
    return {
        "causal_investigation_plan": plan, "root_cause": rc, "evidence_provider": provider,
        "evidence_ledger": [], "hypothesis_history": [], "evidence_claims": [], "evidence_conflicts": [],
        "evidence_requests": [], "causal_graph_version": 0, "investigation_iteration": 0,
        "trace": [AgentTraceStep.ok("start")],
    }


def test_repeated_acquisition_is_idempotent_for_the_same_target():
    provider = _CountingProvider()
    state = _plan_and_state(provider)

    state1 = asyncio.run(acquire_evidence_node(state))
    assert provider.call_count == 1
    assert len(state1["evidence_ledger"]) == 1
    assert len(state1["hypothesis_history"]) == 1

    # Same plan/target run again (as the real loop does on every Stage-B
    # pass) -- the request is now FULFILLED, so it must be REUSED, not
    # re-acquired, and no new EvidenceItem/HypothesisStatusChange created.
    state2 = asyncio.run(acquire_evidence_node(state1))
    assert provider.call_count == 1, "provider must not be called again for an already-fulfilled request"
    assert len(state2["evidence_ledger"]) == 1
    assert len(state2["hypothesis_history"]) == 1
    assert len(state2["evidence_requests"]) == 1


# ---------------------------------------------------------------------------
# Part M: provider failure -- fail safe, request marked UNRESOLVED
# ---------------------------------------------------------------------------

class _FailingProvider(EvidenceProvider):
    async def acquire(self, request):
        raise ConnectionError("backend unreachable")


def test_provider_failure_leaves_request_unresolved_and_no_status_change():
    provider = _FailingProvider()
    state = _plan_and_state(provider)
    result = asyncio.run(acquire_evidence_node(state))
    assert result["hypothesis_history"] == []
    reqs = result["evidence_requests"]
    assert len(reqs) == 1
    assert reqs[0].status == "UNRESOLVED"


# ---------------------------------------------------------------------------
# Part G/H: CAPA epistemic safety (domain: lab calibration, not training)
# ---------------------------------------------------------------------------

def test_capa_from_refuted_hypothesis_produces_no_action():
    hyps = [CandidateHypothesis(id="H1", name="CALIBRATION_LAPSED", statement="s", evidence_needed="e", status="REFUTED")]
    actions = build_conditional_capa_actions(hyps, "calibration record", "calibration")
    assert actions == []


def test_capa_from_possible_hypothesis_stays_conditional():
    hyps = [CandidateHypothesis(id="H1", name="CALIBRATION_LAPSED", statement="the gauge was used past its calibration due date", evidence_needed="e", status="POSSIBLE")]
    actions = build_conditional_capa_actions(hyps, "calibration record", "calibration")
    assert actions
    for a in actions:
        assert a.if_cause_confirmed.startswith("IF ")
        assert a.root_cause_hypothesis_id == "H1"


def test_capa_from_supported_hypothesis_still_traceable():
    hyps = [CandidateHypothesis(
        id="H1", name="VENDOR_INVOICE_UNAPPROVED", statement="the invoice was paid without required approval",
        evidence_needed="e", status="SUPPORTED", supporting_claim_ids=["C7"],
    )]
    actions = build_conditional_capa_actions(hyps, "approval record", "approval")
    assert actions
    assert all(a.root_cause_hypothesis_id == "H1" for a in actions)
    assert all("C7" in a.supporting_claim_ids for a in actions)


def test_invariant_rejects_capa_conditioned_on_refuted_hypothesis():
    hyp = CandidateHypothesis(id="H1", name="X", statement="s", evidence_needed="e", status="REFUTED")
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[hyp])
    bad_action = ConditionalCapaAction(if_cause_confirmed="IF H1 is confirmed", recommended_action="x",
                                        root_cause_hypothesis_id="H1")
    capa = CapaAnalysis(status=CapaStatus.INVESTIGATION_REQUIRED, conditional_actions=[bad_action])
    state = {"root_cause": rc, "capa_analysis": capa}
    is_valid, violations = evaluate_all_invariants(state)
    assert any("INV-CAPA-001" in v for v in violations)


# ---------------------------------------------------------------------------
# Part I/J: Impact epistemic safety
# ---------------------------------------------------------------------------

def test_invariant_rejects_impact_verified_without_verified_evidence():
    impact = ImpactAssessment(status="IMPACT_VERIFIED")
    reported_item = EvidenceItem(claim="a witness said X", source="s", status=EvidenceStatus.REPORTED)
    state = {"impact_assessment": impact, "evidence_ledger": [reported_item]}
    is_valid, violations = evaluate_all_invariants(state)
    assert any("INV-IMPACT-002" in v for v in violations)


def test_invariant_allows_impact_verified_with_verified_evidence():
    impact = ImpactAssessment(status="IMPACT_VERIFIED")
    verified_item = EvidenceItem(claim="objective record", source="s", status=EvidenceStatus.VERIFIED)
    state = {"impact_assessment": impact, "evidence_ledger": [verified_item]}
    is_valid, violations = evaluate_all_invariants(state)
    assert not any("INV-IMPACT-002" in v for v in violations)


def test_impact_requires_assessment_never_flagged():
    impact = ImpactAssessment(status="IMPACT_REQUIRES_ASSESSMENT")
    state = {"impact_assessment": impact, "evidence_ledger": []}
    is_valid, violations = evaluate_all_invariants(state)
    assert not any("INV-IMPACT-002" in v for v in violations)
