"""Phase 16: graph-first investigation planning.

Audited constraint (see app/agent/nodes/graph_investigation_planner.py's
module docstring): `plan_investigation_node` runs before `core_synthesis`
in the LangGraph ordering (app/agent/graph.py), so CandidateHypothesis
objects and the licensed CausalGraph do not exist yet when investigation
planning runs. This phase's graph-first authority is therefore real but
scoped to what genuinely exists at that point: SemanticGraph-derived
unresolved structure (via the existing build_causal_uncertainty_graph).

These tests exercise the REAL runtime entry point
(plan_investigation_graph_first / plan_investigation_node), not just
helper functions in isolation, per Phase 16 Section 29/30's explicit
requirement.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.graph_investigation_planner import (
    build_discriminating_questions_for_competing_hypotheses,
    plan_investigation_graph_first,
)
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.agent.nodes.understanding import understand_finding_node
from app.models.agent import (
    CandidateHypothesis,
    CanonicalFindingState,
    CausalLevel,
    InvestigateRequest,
    SemanticEdge,
    SemanticGraph,
    SemanticNode,
    SemanticNodeType,
    SemanticRelationType,
)


def _canonical_with_unresolved(dev_label: str, req_label: str, claim_ids: list[str] | None = None) -> CanonicalFindingState:
    dev = SemanticNode(id="N1", node_type=SemanticNodeType.EVENT, label=dev_label)
    req = SemanticNode(id="N2", node_type=SemanticNodeType.REQUIREMENT, label=req_label)
    edge = SemanticEdge(id="E1", source_id="N1", target_id="N2", relation_type=SemanticRelationType.VIOLATES,
                         source_claim_ids=claim_ids or [])
    graph = SemanticGraph(nodes=[dev, req], edges=[edge])
    return CanonicalFindingState(
        raw_finding=dev_label, finding_subject=req_label, affected_object=req_label,
        affected_process="UNKNOWN", affected_activity="UNKNOWN", deviation=dev_label,
        observed_deviation=dev_label, facts=[dev_label], semantic_graph=graph,
    )


# ---------------------------------------------------------------------------
# 1. TARGET_UNRESOLVED — genuinely new state, insufficient structure
# ---------------------------------------------------------------------------

def test_no_canonical_state_is_target_unresolved():
    result = plan_investigation_graph_first(None)
    assert result.planner_mode == "TARGET_UNRESOLVED"
    assert result.plan is None


def test_missing_semantic_graph_is_target_unresolved():
    canonical = CanonicalFindingState(
        raw_finding="x", finding_subject="x", affected_object="x", affected_process="UNKNOWN",
        affected_activity="UNKNOWN", deviation="x", observed_deviation="x", facts=["x"],
    )
    result = plan_investigation_graph_first(canonical)
    assert result.planner_mode == "TARGET_UNRESOLVED"
    assert result.plan is None


# ---------------------------------------------------------------------------
# 2. NO_ACTIONABLE_UNCERTAINTY — genuinely new state
# ---------------------------------------------------------------------------

def test_non_actionable_finding_is_no_actionable_uncertainty():
    canonical = _canonical_with_unresolved("x deviated", "x requirement")
    canonical.is_actionable = False
    result = plan_investigation_graph_first(canonical)
    assert result.planner_mode == "NO_ACTIONABLE_UNCERTAINTY"
    assert result.plan.questions == []
    assert result.plan.status == "NO_ACTIONABLE_UNCERTAINTY"


def test_semantic_graph_with_no_recognized_trigger_relation_falls_back_rather_than_claiming_no_action():
    """Phase 17 correction: a semantic_graph with zero unresolved nodes of a
    recognized trigger-relation type is a coverage gap in the trigger-
    relation set, not a verified absence of uncertainty (found via real
    Ollama + full-suite regression testing — see graph_investigation_planner.py).
    This must route to TARGET_UNRESOLVED (legacy planner runs), never
    silently claim NO_ACTIONABLE_UNCERTAINTY."""
    dev = SemanticNode(id="N1", node_type=SemanticNodeType.EVENT, label="An event occurred")
    graph = SemanticGraph(nodes=[dev], edges=[])  # no VIOLATES-style edge at all
    canonical = CanonicalFindingState(
        raw_finding="x", finding_subject="x", affected_object="x", affected_process="UNKNOWN",
        affected_activity="UNKNOWN", deviation="x", observed_deviation="x", facts=["x"],
        semantic_graph=graph,
    )
    result = plan_investigation_graph_first(canonical)
    assert result.planner_mode == "TARGET_UNRESOLVED"
    assert result.plan is None


# ---------------------------------------------------------------------------
# 3. GRAPH_FIRST content when real unresolved structure exists
# ---------------------------------------------------------------------------

def test_unresolved_relation_produces_graph_first_targets():
    canonical = _canonical_with_unresolved(
        "The Blimtor cycle exceeded the qualifying window.", "Kestrion process directive", claim_ids=["C1", "C2"],
    )
    result = plan_investigation_graph_first(canonical)
    assert result.planner_mode == "GRAPH_FIRST"
    assert result.graph_targets_considered == 1
    assert result.graph_targets_selected == 1
    q = result.plan.questions[0]
    assert q.target_node_id and q.target_edge_id and q.source_node_id
    assert q.information_gain_band == "HIGH"


# ---------------------------------------------------------------------------
# 4. CRITICAL ACCEPTANCE TEST — plan_investigation_node uses the graph-first
#    result to OWN the plan for the two new authoritative states, and to
#    honestly label the plan otherwise (Section 30, scoped per the disclosed
#    dependency gap — see module docstring).
# ---------------------------------------------------------------------------

def test_plan_investigation_node_owns_no_actionable_uncertainty():
    class _FakeCanonical:
        is_actionable = False

    state = {
        "request": InvestigateRequest(finding_text="irrelevant"),
        "canonical_finding_state": _FakeCanonical(),
        "trace": [], "errors": [],
    }
    result = asyncio.run(plan_investigation_node(state))
    plan = result["investigation_plan"]
    assert plan.status == "NO_ACTIONABLE_UNCERTAINTY"
    assert plan.questions == []


def test_plan_investigation_node_labels_fallback_honestly():
    """When the graph-first planner identifies real unresolved structure,
    plan_investigation_node must label the resulting plan's content source
    honestly (LEGACY_FALLBACK with a disclosed reason) rather than claim
    GRAPH_FIRST authority it does not actually have for question content."""
    state = {
        "request": InvestigateRequest(finding_text="The Blimtor cycle exceeded the qualifying window."),
        "trace": [], "errors": [], "evidence_ledger": [],
    }
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None):
        state = asyncio.run(understand_finding_node(state))
    result = asyncio.run(plan_investigation_node(state))
    plan = result["investigation_plan"]
    # The real semantic-graph builder may or may not find a normative
    # violation in this synthetic sentence -- either real, honest outcome
    # is acceptable here; what must NEVER happen is an unlabeled plan
    # silently claiming GRAPH_FIRST content authority it doesn't have.
    assert plan.planner_mode in ("LEGACY_FALLBACK", "TARGET_UNRESOLVED", "NO_ACTIONABLE_UNCERTAINTY")
    if plan.planner_mode == "LEGACY_FALLBACK":
        assert plan.fallback_reason
        assert len(plan.questions) >= 1
    elif plan.planner_mode == "NO_ACTIONABLE_UNCERTAINTY":
        assert plan.questions == []


def test_causal_uncertainty_graph_is_populated_in_state():
    state = {
        "request": InvestigateRequest(finding_text="The Blimtor cycle exceeded the qualifying window."),
        "trace": [], "errors": [], "evidence_ledger": [],
    }
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None):
        state = asyncio.run(understand_finding_node(state))
    result = asyncio.run(plan_investigation_node(state))
    assert result.get("causal_uncertainty_graph") is not None


# ---------------------------------------------------------------------------
# 5. Competing-hypothesis discrimination (real, tested, NOT yet wired into
#    the live single-pass planner -- see module docstring for why).
# ---------------------------------------------------------------------------

def _hyp(**kwargs) -> CandidateHypothesis:
    defaults = dict(evidence_needed="", causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE)
    defaults.update(kwargs)
    return CandidateHypothesis(**defaults)


def test_competing_hypotheses_produce_one_discriminating_question():
    hyps = [
        _hyp(id="H1", name="MECHANISM_A", statement="Mechanism A caused it", status="POSSIBLE",
             evidence_strength="REPORTED", supporting_claim_ids=["C1"]),
        _hyp(id="H2", name="MECHANISM_B", statement="Mechanism B caused it", status="POSSIBLE",
             evidence_strength="REPORTED", supporting_claim_ids=["C2"]),
    ]
    qs = build_discriminating_questions_for_competing_hypotheses(hyps)
    assert len(qs) == 1
    assert set(qs[0].target_hypothesis_ids) == {"H1", "H2"}
    assert qs[0].information_gain_band == "HIGH"


def test_chained_hypotheses_are_not_treated_as_competing():
    hyps = [
        _hyp(id="H1", name="IMMEDIATE", statement="Immediate mechanism", status="SUPPORTED",
             evidence_strength="VERIFIED", supporting_claim_ids=["C1"]),
        _hyp(id="H2", name="SYSTEMIC", statement="Systemic cause", status="SUPPORTED",
             evidence_strength="VERIFIED", supporting_claim_ids=["C1"], deepens_hypothesis_id="H1"),
    ]
    qs = build_discriminating_questions_for_competing_hypotheses(hyps)
    assert qs == []


def test_three_independent_hypotheses_produce_bounded_pairwise_questions():
    hyps = [
        _hyp(id="H1", name="A", statement="A", status="POSSIBLE", evidence_strength="REPORTED", supporting_claim_ids=["C1"]),
        _hyp(id="H2", name="B", statement="B", status="POSSIBLE", evidence_strength="REPORTED", supporting_claim_ids=["C2"]),
        _hyp(id="H3", name="C", statement="C", status="POSSIBLE", evidence_strength="REPORTED", supporting_claim_ids=["C3"]),
    ]
    qs = build_discriminating_questions_for_competing_hypotheses(hyps)
    # No hypothesis is silently dropped -- every pair is represented.
    covered_ids = set()
    for q in qs:
        covered_ids.update(q.target_hypothesis_ids)
    assert covered_ids == {"H1", "H2", "H3"}
    assert len(qs) == 3  # C(3,2) pairs, each exactly once


def test_single_hypothesis_produces_no_discriminating_question():
    hyps = [_hyp(id="H1", name="A", statement="A", status="POSSIBLE", evidence_strength="REPORTED")]
    assert build_discriminating_questions_for_competing_hypotheses(hyps) == []


# ---------------------------------------------------------------------------
# 6. INV-INVEST-019
# ---------------------------------------------------------------------------

def test_inv_invest_019_passes_when_graph_has_no_unresolved_nodes():
    from app.models.agent import CausalGraph, CausalGraphNode, CausalGraphNodeType, EpistemicSource, EvidenceStatus, InvestigationPlan
    dev = CausalGraphNode(node_id="D", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="dev",
                           causal_level=CausalLevel.L0_OBSERVATION, epistemic_status=EvidenceStatus.VERIFIED,
                           provenance=EpistemicSource.AUDIT_OBSERVATION)
    graph = CausalGraph(nodes=[dev], edges=[])
    plan = InvestigationPlan(status="NO_ACTIONABLE_UNCERTAINTY", questions=[])
    ok, violations = evaluate_all_invariants({"investigation_plan": plan, "causal_uncertainty_graph": graph})
    assert not any("INV-INVEST-019" in v for v in violations)


def test_inv_invest_019_fails_closed_on_inconsistent_claim():
    from app.models.agent import (
        CausalGraph, CausalGraphEdge, CausalGraphEdgeStatus, CausalGraphNode, CausalGraphNodeType,
        EpistemicSource, EvidenceStatus, InvestigationPlan,
    )
    dev = CausalGraphNode(node_id="D", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="dev",
                           causal_level=CausalLevel.L0_OBSERVATION, epistemic_status=EvidenceStatus.VERIFIED,
                           provenance=EpistemicSource.AUDIT_OBSERVATION)
    unresolved = CausalGraphNode(node_id="U1", node_type=CausalGraphNodeType.UNRESOLVED, label="unresolved thing",
                                  causal_level=CausalLevel.EVIDENCE_STATE, epistemic_status=EvidenceStatus.UNKNOWN,
                                  provenance=EpistemicSource.AUDIT_OBSERVATION)
    edge = CausalGraphEdge(edge_id="E1", source_node_id="D", target_node_id="U1",
                            status=CausalGraphEdgeStatus.UNKNOWN, causal_level_transition="D->U")
    graph = CausalGraph(nodes=[dev, unresolved], edges=[edge])
    plan = InvestigationPlan(status="NO_ACTIONABLE_UNCERTAINTY", questions=[])
    ok, violations = evaluate_all_invariants({"investigation_plan": plan, "causal_uncertainty_graph": graph})
    assert any("INV-INVEST-019" in v for v in violations)
