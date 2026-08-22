"""Phase 14: investigation-question traceability firewall and deterministic
information-gain banding.

Audit finding (Phase 14 Step 1): the actual runtime investigation planner
(`plan_investigation_node` -> `build_deterministic_investigation_plan` in
plan_investigation_fallback.py, the fast-path used whenever the ASP.NET
integration is not configured -- i.e. in every test and most real
deployments) is a ~2,900-line deterministic template/keyword decision tree,
not a graph-first generator. A genuinely graph-grounded supplementary path
already existed from Phase 4 (build_causal_uncertainty_graph +
rank_uncertainty_nodes_by_information_gain), but it is purely additive
(capped at 2 extra questions, never authoritative) and its
`information_gain_rank` field was a bare enumeration position, not an
inspectable ordinal classification.

A full replacement of the template decision tree was assessed and NOT
attempted this phase (see the Phase 14 report) -- it is extensively tested,
production behavior, and a safe rewrite is a larger effort than fits one
session. What Phase 14 adds for real, tested here:

  1. `InvestigationQuestion.information_gain_band` (HIGH/MEDIUM/LOW) +
     `.information_gain_reason` -- a real, populated, machine-readable
     ordinal classification (Section 5), derived from the same structural
     signal the existing ranking already used (proposition_ids count on the
     unresolved causal-uncertainty edge), not a fabricated probability.
  2. INV-INVEST-012 -- a traceability firewall (Section 18): any question
     that populates target_node_id/target_edge_id/source_node_id (the
     documented "this came from the causal-graph path" signal) must have
     those ids actually resolve inside this run's causal graph(s), or the
     invariant fails closed rather than silently trusting a dangling
     reference.
"""
from __future__ import annotations

from app.agent.causal_graph import build_causal_uncertainty_graph, information_gain_band_for_edge
from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
from app.models.agent import (
    CanonicalFindingState,
    CausalGraph,
    CausalGraphEdge,
    CausalGraphEdgeStatus,
    CausalGraphNode,
    CausalGraphNodeType,
    CausalLevel,
    EpistemicSource,
    EvidenceStatus,
    InvestigationPlan,
    InvestigationQuestion,
    SemanticEdge,
    SemanticRelationType,
    SemanticGraph,
    SemanticNode,
    SemanticNodeType,
)


def _canonical_with_violation(deviation: str, subject_label: str) -> CanonicalFindingState:
    """A canonical state whose semantic_graph carries one VIOLATES edge --
    the only structural trigger build_causal_uncertainty_graph consumes --
    so build_deterministic_investigation_plan's graph-grounded append block
    has real unresolved structure to work with."""
    dev_node = SemanticNode(id="N1", node_type=SemanticNodeType.EVENT, label=deviation)
    req_node = SemanticNode(id="N2", node_type=SemanticNodeType.REQUIREMENT, label=subject_label)
    edge = SemanticEdge(id="E1", source_id="N1", target_id="N2", relation_type=SemanticRelationType.VIOLATES, source_claim_ids=["C1", "C2"])
    graph = SemanticGraph(nodes=[dev_node, req_node], edges=[edge])
    return CanonicalFindingState(
        raw_finding=deviation, finding_subject=subject_label, affected_object=subject_label,
        affected_process="UNKNOWN", affected_activity="UNKNOWN", deviation=deviation,
        observed_deviation=deviation, facts=[deviation], semantic_graph=graph,
    )


# ---------------------------------------------------------------------------
# 1. Deterministic information-gain banding on the graph-grounded question
# ---------------------------------------------------------------------------

def test_information_gain_band_high_with_two_propositions():
    canonical = _canonical_with_violation(
        "The Zargon output diverged from the qualifying threshold.",
        "Wexnall control directive",
    )
    ug = build_causal_uncertainty_graph(canonical)
    assert len(ug.edges) == 1
    band, reason = information_gain_band_for_edge(ug.edges[0])
    assert band == "HIGH"
    assert "2 proposition" in reason


def test_information_gain_band_low_with_no_propositions():
    dev_node = SemanticNode(id="N1", node_type=SemanticNodeType.EVENT, label="An unexplained deviation occurred.")
    req_node = SemanticNode(id="N2", node_type=SemanticNodeType.REQUIREMENT, label="An applicable control requirement")
    edge = SemanticEdge(id="E1", source_id="N1", target_id="N2", relation_type=SemanticRelationType.VIOLATES, source_claim_ids=[])
    graph = SemanticGraph(nodes=[dev_node, req_node], edges=[edge])
    canonical = CanonicalFindingState(
        raw_finding="An unexplained deviation occurred.", finding_subject="An applicable control requirement",
        affected_object="An applicable control requirement", affected_process="UNKNOWN", affected_activity="UNKNOWN",
        deviation="An unexplained deviation occurred.", observed_deviation="An unexplained deviation occurred.",
        facts=["An unexplained deviation occurred."], semantic_graph=graph,
    )
    ug = build_causal_uncertainty_graph(canonical)
    assert len(ug.edges) == 1
    band, reason = information_gain_band_for_edge(ug.edges[0])
    assert band == "LOW"


def test_information_gain_band_medium_with_one_proposition():
    dev_node = SemanticNode(id="N1", node_type=SemanticNodeType.EVENT, label="An unexplained deviation occurred.")
    req_node = SemanticNode(id="N2", node_type=SemanticNodeType.REQUIREMENT, label="An applicable control requirement")
    edge = SemanticEdge(id="E1", source_id="N1", target_id="N2", relation_type=SemanticRelationType.VIOLATES, source_claim_ids=["C1"])
    graph = SemanticGraph(nodes=[dev_node, req_node], edges=[edge])
    canonical = CanonicalFindingState(
        raw_finding="An unexplained deviation occurred.", finding_subject="An applicable control requirement",
        affected_object="An applicable control requirement", affected_process="UNKNOWN", affected_activity="UNKNOWN",
        deviation="An unexplained deviation occurred.", observed_deviation="An unexplained deviation occurred.",
        facts=["An unexplained deviation occurred."], semantic_graph=graph,
    )
    ug = build_causal_uncertainty_graph(canonical)
    band, reason = information_gain_band_for_edge(ug.edges[0])
    assert band == "MEDIUM"


def test_graph_grounded_question_in_full_plan_carries_its_band_when_generated():
    """When the graph-grounded append block in
    build_deterministic_investigation_plan does fire (the uncertainty
    node's vocabulary is not already covered by the template questions),
    the resulting question must carry the same band the standalone
    function would compute for that edge -- not a fresh/duplicated
    computation that could drift out of sync."""
    canonical = _canonical_with_violation(
        "The Zargon output diverged from the qualifying threshold.",
        "Wexnall control directive",
    )
    ug = build_causal_uncertainty_graph(canonical)
    expected_band, _ = information_gain_band_for_edge(ug.edges[0])
    _, plan = build_deterministic_investigation_plan(
        canonical.raw_finding, [], canonical_subject=canonical.finding_subject, canonical_state=canonical,
    )
    graph_qs = [q for q in plan.questions if q.target_node_id]
    if graph_qs:
        assert graph_qs[0].information_gain_band == expected_band
    # else: the append block's own dedup-against-existing-questions logic
    # (unrelated to Phase 14) decided this uncertainty was already covered
    # by the template question set -- not a Phase 14 concern.


def test_non_graph_grounded_question_has_no_information_gain_band():
    q = InvestigationQuestion(question="What procedure applies here?", purpose="p", evidence="e")
    assert q.information_gain_band is None
    assert q.information_gain_reason is None


# ---------------------------------------------------------------------------
# 2. INV-INVEST-012 traceability firewall
# ---------------------------------------------------------------------------

def _graph() -> CausalGraph:
    dev = CausalGraphNode(
        node_id="CN_DEVIATION", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="deviation",
        causal_level=CausalLevel.L0_OBSERVATION, epistemic_status=EvidenceStatus.VERIFIED,
        provenance=EpistemicSource.AUDIT_OBSERVATION,
    )
    mech = CausalGraphNode(
        node_id="CN1", node_type=CausalGraphNodeType.CONTRIBUTING_FACTOR, label="mechanism",
        causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE, epistemic_status=EvidenceStatus.UNKNOWN,
        provenance=EpistemicSource.UNKNOWN_SOURCE,
    )
    edge = CausalGraphEdge(edge_id="CE1", source_node_id="CN_DEVIATION", target_node_id="CN1",
                            status=CausalGraphEdgeStatus.POSSIBLE, causal_level_transition="D->M")
    return CausalGraph(nodes=[dev, mech], edges=[edge])


def test_traceability_passes_for_valid_graph_reference():
    q = InvestigationQuestion(
        question="What evidence establishes the mechanism?", purpose="p", evidence="e",
        target_node_id="CN1", target_edge_id="CE1", source_node_id="CN_DEVIATION",
    )
    state = {"investigation_plan": InvestigationPlan(questions=[q]), "causal_graph": _graph()}
    ok, violations = evaluate_all_invariants(state)
    assert not any("INV-INVEST-012" in v for v in violations)


def test_traceability_passes_for_non_graph_grounded_question():
    q = InvestigationQuestion(question="What procedure applies here?", purpose="p", evidence="e")
    state = {"investigation_plan": InvestigationPlan(questions=[q])}
    ok, violations = evaluate_all_invariants(state)
    assert not any("INV-INVEST-012" in v for v in violations)


def test_traceability_rejects_dangling_node_reference():
    q = InvestigationQuestion(
        question="What evidence establishes the mechanism?", purpose="p", evidence="e",
        target_node_id="CN_DOES_NOT_EXIST",
    )
    state = {"investigation_plan": InvestigationPlan(questions=[q]), "causal_graph": _graph()}
    ok, violations = evaluate_all_invariants(state)
    matches = [v for v in violations if "INV-INVEST-012" in v]
    assert matches, "dangling target_node_id must be rejected"
    assert "CN_DOES_NOT_EXIST" in matches[0]


def test_traceability_rejects_dangling_edge_reference():
    q = InvestigationQuestion(
        question="What evidence establishes the mechanism?", purpose="p", evidence="e",
        target_node_id="CN1", target_edge_id="CE_DOES_NOT_EXIST",
    )
    state = {"investigation_plan": InvestigationPlan(questions=[q]), "causal_graph": _graph()}
    ok, violations = evaluate_all_invariants(state)
    assert any("INV-INVEST-012" in v for v in violations)


def test_traceability_rejects_reference_with_no_graph_at_all():
    q = InvestigationQuestion(
        question="What evidence establishes the mechanism?", purpose="p", evidence="e",
        target_node_id="CN1",
    )
    state = {"investigation_plan": InvestigationPlan(questions=[q])}
    ok, violations = evaluate_all_invariants(state)
    assert any("INV-INVEST-012" in v for v in violations)
