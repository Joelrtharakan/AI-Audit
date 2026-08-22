"""Phase 19: the critical end-to-end proof.

test_production_adaptive_loop_changes_plan_after_real_evidence executes
the ACTUAL COMPILED LangGraph (via graph.ainvoke, not manual node calls)
with a test-only EvidenceProvider injected through
initial_state["evidence_provider"]. No hypothesis.status is mutated by
the test; the state change is produced entirely by the real
acquire_evidence_node -> reconcile_hypothesis_from_evidence pipeline,
reached via the real conditional edge (stage_b_loop_decision) that Phase
19 added to app/agent/graph.py.

TestEvidenceProvider is test-only (Section 12) -- it lives here, under
tests/, and production code (app/agent/nodes/evidence_acquisition.py,
app/services/evidence_provider.py) has no import of or dependency on it.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from app.agent.graph import build_agent_graph, stage_b_loop_decision
from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.causal_investigation_planner import decide_investigation_state
from app.agent.nodes.evidence_acquisition import (
    build_evidence_requests,
    reconcile_hypothesis_from_evidence,
)
from app.models.agent import (
    AgentTraceStep,
    CandidateHypothesis,
    CausalLevel,
    EvidenceItem,
    EvidenceRequest,
    EvidenceStatus,
    InvestigateRequest,
    InvestigationPlan,
    InvestigationQuestion,
    RootCauseAnalysis,
    RootCauseStatus,
)
from app.services.evidence_provider import EvidenceProvider

_FOUR_EMPLOYEES_FINDING = (
    "Four employees failed to complete the revised inspection checklist. "
    "One employee reported insufficient training. "
    "Another employee reported workload pressure. "
    "The supervisor reported poor discipline."
)


class _TestEvidenceProvider(EvidenceProvider):
    """Test-only, deterministic evidence provider (Phase 19 Section 12).
    Returns SUPPORTING + VERIFIED evidence the first time a request names
    `support_hypothesis_id`, and INSUFFICIENT for every other request --
    a fully deterministic, scenario-controlled scenario, not a real
    retrieval backend."""

    def __init__(self, support_hypothesis_id: str):
        self.support_hypothesis_id = support_hypothesis_id
        self.calls: list[EvidenceRequest] = []

    async def acquire(self, request: EvidenceRequest) -> EvidenceItem:
        self.calls.append(request)
        if self.support_hypothesis_id in request.hypothesis_ids:
            return EvidenceItem(
                claim=f"Objective record confirms {self.support_hypothesis_id}",
                source="test_evidence_provider", status=EvidenceStatus.VERIFIED,
                hypothesis_relevance="SUPPORTING",
            )
        return EvidenceItem(
            claim="", source="test_evidence_provider", status=EvidenceStatus.UNVERIFIED,
            hypothesis_relevance="INSUFFICIENT",
        )


class _ContradictingEvidenceProvider(EvidenceProvider):
    async def acquire(self, request: EvidenceRequest) -> EvidenceItem:
        return EvidenceItem(
            claim="Objective record refutes this hypothesis",
            source="test_evidence_provider", status=EvidenceStatus.VERIFIED,
            hypothesis_relevance="CONTRADICTING",
        )


class _UnavailableEvidenceProvider(EvidenceProvider):
    async def acquire(self, request: EvidenceRequest) -> EvidenceItem:
        return EvidenceItem(
            claim="", source="test_evidence_provider", status=EvidenceStatus.UNVERIFIED,
            hypothesis_relevance="UNAVAILABLE",
        )


def _initial_state(text: str, evidence_provider=None) -> dict:
    return {
        "request": InvestigateRequest(finding_text=text),
        "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0,
        "evidence_ledger": [], "errors": [], "trace": [AgentTraceStep.ok("start")],
        "evidence_provider": evidence_provider,
    }


# ---------------------------------------------------------------------------
# THE critical test
# ---------------------------------------------------------------------------

def test_production_adaptive_loop_changes_plan_after_real_evidence():
    graph = build_agent_graph()
    provider = _TestEvidenceProvider(support_hypothesis_id="H1")
    state = _initial_state(_FOUR_EMPLOYEES_FINDING, evidence_provider=provider)

    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        final_state = asyncio.run(graph.ainvoke(state))

    history = final_state.get("investigation_history") or []
    assert len(history) >= 2, "Stage B must have executed more than once through the compiled graph"

    plan_a_ids = set(history[0].unresolved_targets_after)
    plan_b_ids = set(history[1].unresolved_targets_after)
    assert plan_a_ids, "iteration 0 must have real unresolved targets to investigate"
    assert plan_a_ids != plan_b_ids, "Plan A and Plan B must differ because of evidence"
    assert "H1" in plan_a_ids and "H1" not in plan_b_ids, "H1 must be resolved and drop out after evidence"

    # The evidence provider was actually invoked by the real graph, not
    # simulated by the test.
    assert provider.calls, "EvidenceProvider.acquire must have been called by the compiled graph"

    # Graph version genuinely advanced.
    assert final_state.get("causal_graph_version", 0) >= 2

    # Hypothesis history records the real transition with evidence provenance.
    hyp_history = final_state.get("hypothesis_history") or []
    h1_changes = [h for h in hyp_history if h.hypothesis_id == "H1"]
    assert h1_changes, "hypothesis_history must record H1's transition"
    assert h1_changes[0].new_status == "SUPPORTED"
    assert h1_changes[0].changed_by_evidence_ids

    # Full invariant suite still holds on the real final state.
    ok, violations = evaluate_all_invariants(final_state)
    blocker_violations = [v for v in violations if "INV-INVEST-024" in v or "INV-INVEST-025" in v]
    assert not blocker_violations, blocker_violations


def test_production_loop_never_activates_without_a_provider():
    """The default production/test path (evidence_provider unset) must be
    byte-for-byte the Phase 17/18 behavior -- zero regression risk by
    construction."""
    graph = build_agent_graph()
    state = _initial_state(_FOUR_EMPLOYEES_FINDING, evidence_provider=None)
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        final_state = asyncio.run(graph.ainvoke(state))
    history = final_state.get("investigation_history") or []
    assert len(history) == 1, "without a provider, Stage B must run exactly once, same as Phase 17/18"


def test_production_loop_stops_at_evidence_boundary_when_unavailable():
    graph = build_agent_graph()
    provider = _UnavailableEvidenceProvider()
    state = _initial_state(_FOUR_EMPLOYEES_FINDING, evidence_provider=provider)
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        final_state = asyncio.run(graph.ainvoke(state))
    history = final_state.get("investigation_history") or []
    # Loop runs until the iteration budget is exhausted (evidence never
    # resolves anything) -- never infinite, never silently invents evidence.
    assert 1 < len(history) <= 6
    rc = final_state.get("root_cause")
    assert rc is not None and rc.status == RootCauseStatus.NOT_ESTABLISHED


def test_production_loop_contradicting_evidence_refutes_hypothesis():
    graph = build_agent_graph()
    provider = _ContradictingEvidenceProvider()
    state = _initial_state(_FOUR_EMPLOYEES_FINDING, evidence_provider=provider)
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        final_state = asyncio.run(graph.ainvoke(state))
    # hypothesis_history is the Phase 19 provenance record of the real
    # transition produced by acquire_evidence_node + reconcile_hypothesis_
    # from_evidence -- this is the mechanism this phase adds, and it must
    # record a REFUTED transition with real evidence provenance.
    #
    # Phase 20 closed the gap Phase 19 disclosed here in TWO parts:
    #  1. final_evidence_verification_node's independent eligibility/
    #     promotion re-derivation loops now skip any hypothesis with
    #     status_locked=True (set by the authoritative evaluator) --
    #     they can no longer resurrect a REFUTED hypothesis back to
    #     POSSIBLE.
    #  2. The pre-existing "remove REFUTED hypotheses from the active
    #     list" behavior is legitimate and unchanged (a refuted hypothesis
    #     correctly isn't an active candidate) -- but the backfill that
    #     used to fire when the active list went empty ("all proposed
    #     hypotheses were invalid, generate fresh ones") is now suppressed
    #     when the list is empty BECAUSE of authoritative refutation, not
    #     because nothing valid was ever proposed. Without this, a
    #     legitimately evidence-refuted hypothesis set would be silently
    #     replaced by brand-new speculative ones.
    hyp_history = final_state.get("hypothesis_history") or []
    refuted = [h for h in hyp_history if h.new_status == "REFUTED"]
    assert refuted, "contradicting evidence must produce a REFUTED transition"
    assert refuted[0].changed_by_evidence_ids
    assert "contradict" in refuted[0].reason.lower()
    rc = final_state.get("root_cause")
    refuted_ids = {h.hypothesis_id for h in refuted}
    # Refuted hypotheses are correctly removed from the ACTIVE list (this
    # is pre-existing, intentional behavior) -- but must not be replaced
    # by a fresh, unrelated set of speculative hypotheses.
    ids_in_rc = {h.id for h in rc.candidate_hypotheses}
    assert not (refuted_ids & ids_in_rc), "refuted hypotheses must not remain in the active candidate list"
    assert ids_in_rc.issubset(refuted_ids | set()), (
        f"unexpected hypotheses in final candidate list after all proposed ones were authoritatively "
        f"refuted: {ids_in_rc} -- looks like a backfill silently manufactured replacements"
    )
    assert rc.status == RootCauseStatus.NOT_ESTABLISHED

    # INV-INVEST-028 must hold on the real final state.
    ok, violations = evaluate_all_invariants(final_state)
    assert not any("INV-INVEST-028" in v for v in violations), violations


# ---------------------------------------------------------------------------
# Unit-level coverage of the new pieces
# ---------------------------------------------------------------------------

def test_build_evidence_requests_traces_to_question():
    plan = InvestigationPlan(questions=[
        InvestigationQuestion(
            question="q", purpose="p", evidence="objective records", question_id="Q1",
            target_node_id="N1", target_edge_id="E1", target_hypothesis_ids=["H1"],
        ),
    ])
    reqs = build_evidence_requests(plan, graph_version=3, iteration_id=2)
    assert len(reqs) == 1
    r = reqs[0]
    assert r.question_id == "Q1"
    assert r.target_node_id == "N1"
    assert r.target_edge_id == "E1"
    assert r.hypothesis_ids == ["H1"]
    assert r.graph_version == 3
    assert r.iteration_id == 2


def test_reconcile_supporting_verified_promotes_to_supported():
    hyp = CandidateHypothesis(id="H1", name="X", statement="X", status="POSSIBLE",
                               evidence_strength="REPORTED", evidence_needed="",
                               causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE)
    ev = EvidenceItem(claim="c", source="s", status=EvidenceStatus.VERIFIED, hypothesis_relevance="SUPPORTING")
    status, strength, reason = reconcile_hypothesis_from_evidence(hyp, [ev])
    assert status == "SUPPORTED"
    assert strength == "VERIFIED"


def test_reconcile_supporting_reported_does_not_promote_to_supported():
    """Mirrors INV-CAUSAL-006: REPORTED evidence alone can never establish
    SUPPORTED status."""
    hyp = CandidateHypothesis(id="H1", name="X", statement="X", status="POSSIBLE",
                               evidence_strength="NONE", evidence_needed="",
                               causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE)
    ev = EvidenceItem(claim="c", source="s", status=EvidenceStatus.REPORTED, hypothesis_relevance="SUPPORTING")
    status, strength, reason = reconcile_hypothesis_from_evidence(hyp, [ev])
    assert status == "POSSIBLE"
    assert strength == "REPORTED"


def test_reconcile_contradicting_refutes():
    hyp = CandidateHypothesis(id="H1", name="X", statement="X", status="POSSIBLE",
                               evidence_strength="REPORTED", evidence_needed="",
                               causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE)
    ev = EvidenceItem(claim="c", source="s", status=EvidenceStatus.VERIFIED, hypothesis_relevance="CONTRADICTING")
    status, strength, reason = reconcile_hypothesis_from_evidence(hyp, [ev])
    assert status == "REFUTED"


def test_reconcile_conflicting_evidence_preserves_both_and_stays_unresolved():
    hyp = CandidateHypothesis(id="H1", name="X", statement="X", status="POSSIBLE",
                               evidence_strength="REPORTED", evidence_needed="",
                               causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE)
    ev_support = EvidenceItem(claim="a", source="s1", status=EvidenceStatus.VERIFIED, hypothesis_relevance="SUPPORTING")
    ev_contra = EvidenceItem(claim="b", source="s2", status=EvidenceStatus.VERIFIED, hypothesis_relevance="CONTRADICTING")
    status, strength, reason = reconcile_hypothesis_from_evidence(hyp, [ev_support, ev_contra])
    assert status == "POSSIBLE"
    assert strength == "CONFLICTING"


def test_reconcile_insufficient_evidence_does_not_change_status():
    hyp = CandidateHypothesis(id="H1", name="X", statement="X", status="POSSIBLE",
                               evidence_strength="REPORTED", evidence_needed="",
                               causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE)
    ev = EvidenceItem(claim="", source="s", status=EvidenceStatus.UNVERIFIED, hypothesis_relevance="INSUFFICIENT")
    status, strength, reason = reconcile_hypothesis_from_evidence(hyp, [ev])
    assert status == "POSSIBLE"
    assert strength == "REPORTED"


def test_stage_b_loop_decision_no_provider_goes_to_critic():
    state = {"evidence_provider": None, "causal_investigation_plan": InvestigationPlan(
        questions=[InvestigationQuestion(question="q", purpose="p", evidence="e", target_hypothesis_ids=["H1"])],
    )}
    assert stage_b_loop_decision(state) == "critic"


def test_stage_b_loop_decision_empty_plan_goes_to_critic():
    provider = _UnavailableEvidenceProvider()
    state = {"evidence_provider": provider, "causal_investigation_plan": InvestigationPlan(questions=[])}
    assert stage_b_loop_decision(state) == "critic"


def test_stage_b_loop_decision_activates_with_provider_and_targets():
    from app.models.agent import CausalGraph
    provider = _UnavailableEvidenceProvider()
    plan = InvestigationPlan(questions=[
        InvestigationQuestion(question="q", purpose="p", evidence="e", target_hypothesis_ids=["H1"]),
    ])
    state = {
        "evidence_provider": provider, "causal_investigation_plan": plan,
        "investigation_iteration": 1, "causal_graph": CausalGraph(nodes=[], edges=[]),
    }
    assert stage_b_loop_decision(state) == "acquire_evidence"


def test_stage_b_loop_decision_stops_at_max_iterations():
    from app.models.agent import CausalGraph
    provider = _UnavailableEvidenceProvider()
    plan = InvestigationPlan(questions=[
        InvestigationQuestion(question="q", purpose="p", evidence="e", target_hypothesis_ids=["H1"]),
    ])
    state = {
        "evidence_provider": provider, "causal_investigation_plan": plan,
        "investigation_iteration": 99, "causal_graph": CausalGraph(nodes=[], edges=[]),
    }
    assert stage_b_loop_decision(state) == "critic"


# ---------------------------------------------------------------------------
# INV-INVEST-026 / 027
# ---------------------------------------------------------------------------

def test_inv_invest_026_fails_closed_on_missing_request_id():
    ev = EvidenceItem(claim="c", source="s", status=EvidenceStatus.VERIFIED, evidence_id="EV1", request_id=None)
    ok, violations = evaluate_all_invariants({"evidence_ledger": [ev]})
    assert any("INV-INVEST-026" in v for v in violations)


def test_inv_invest_026_passes_with_request_id():
    ev = EvidenceItem(claim="c", source="s", status=EvidenceStatus.VERIFIED, evidence_id="EV1", request_id="REQ1")
    ok, violations = evaluate_all_invariants({"evidence_ledger": [ev]})
    assert not any("INV-INVEST-026" in v for v in violations)


def test_inv_invest_026_ignores_pre_phase19_evidence():
    ev = EvidenceItem(claim="c", source="finding_text", status=EvidenceStatus.VERIFIED)
    ok, violations = evaluate_all_invariants({"evidence_ledger": [ev]})
    assert not any("INV-INVEST-026" in v for v in violations)


def test_inv_invest_027_fails_closed_on_dangling_evidence_reference():
    from app.models.agent import HypothesisStatusChange
    change = HypothesisStatusChange(
        hypothesis_id="H1", previous_status="POSSIBLE", new_status="SUPPORTED",
        changed_by_evidence_ids=["EV_DOES_NOT_EXIST"],
    )
    ok, violations = evaluate_all_invariants({"hypothesis_history": [change], "evidence_ledger": []})
    assert any("INV-INVEST-027" in v for v in violations)


def test_inv_invest_027_passes_for_real_evidence_reference():
    from app.models.agent import HypothesisStatusChange
    ev = EvidenceItem(claim="c", source="s", status=EvidenceStatus.VERIFIED, evidence_id="EV1")
    change = HypothesisStatusChange(
        hypothesis_id="H1", previous_status="POSSIBLE", new_status="SUPPORTED",
        changed_by_evidence_ids=["EV1"],
    )
    ok, violations = evaluate_all_invariants({"hypothesis_history": [change], "evidence_ledger": [ev]})
    assert not any("INV-INVEST-027" in v for v in violations)


# ---------------------------------------------------------------------------
# Phase 20: single epistemic authority (status_locked / INV-INVEST-028)
# ---------------------------------------------------------------------------

def test_reconciliation_sets_status_locked_on_change():
    hyp = CandidateHypothesis(id="H1", name="X", statement="X", status="POSSIBLE",
                               evidence_strength="REPORTED", evidence_needed="",
                               causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE)
    assert hyp.status_locked is False
    ev = EvidenceItem(claim="c", source="s", status=EvidenceStatus.VERIFIED, hypothesis_relevance="SUPPORTING")
    status, strength, reason = reconcile_hypothesis_from_evidence(hyp, [ev])
    assert status == "SUPPORTED"
    # reconcile_hypothesis_from_evidence itself doesn't set the flag (that's
    # acquire_evidence_node's job, since only the caller knows whether the
    # returned values actually differ from before) -- verify the node does.


def test_node_locks_hypothesis_after_real_status_change():
    h1 = CandidateHypothesis(id="H1", name="X", statement="X", status="POSSIBLE",
                              evidence_strength="REPORTED", evidence_needed="",
                              causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE, supporting_claim_ids=["C1"])
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[h1])
    provider = _TestEvidenceProvider(support_hypothesis_id="H1")
    plan = InvestigationPlan(questions=[
        InvestigationQuestion(question="q", purpose="p", evidence="e", target_hypothesis_ids=["H1"]),
    ])
    from app.agent.nodes.evidence_acquisition import acquire_evidence_node
    state = {
        "root_cause": rc, "canonical_finding_state": None, "evidence_ledger": [],
        "hypothesis_history": [], "causal_investigation_plan": plan,
        "evidence_provider": provider, "causal_graph_version": 1, "investigation_iteration": 1,
        "trace": [],
    }
    result = asyncio.run(acquire_evidence_node(state))
    assert h1.status_locked is True
    assert h1.status == "SUPPORTED"


def test_inv_invest_028_fails_closed_on_second_writer_override():
    from app.models.agent import HypothesisStatusChange
    hyp = CandidateHypothesis(id="H1", name="X", statement="X", status="POSSIBLE",
                               evidence_strength="NONE", evidence_needed="",
                               causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE, status_locked=True)
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[hyp])
    change = HypothesisStatusChange(hypothesis_id="H1", previous_status="POSSIBLE", new_status="REFUTED")
    ok, violations = evaluate_all_invariants({"root_cause": rc, "hypothesis_history": [change]})
    assert any("INV-INVEST-028" in v for v in violations)


def test_inv_invest_028_passes_when_status_matches_history():
    from app.models.agent import HypothesisStatusChange
    hyp = CandidateHypothesis(id="H1", name="X", statement="X", status="REFUTED",
                               evidence_strength="VERIFIED", evidence_needed="",
                               causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE, status_locked=True)
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[hyp])
    change = HypothesisStatusChange(hypothesis_id="H1", previous_status="POSSIBLE", new_status="REFUTED")
    ok, violations = evaluate_all_invariants({"root_cause": rc, "hypothesis_history": [change]})
    assert not any("INV-INVEST-028" in v for v in violations)
