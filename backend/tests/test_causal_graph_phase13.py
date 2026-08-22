"""Phase 13: single-hop causal edge selection, boundary semantics, and the
deterministic 5-Why authority gate.

Phase 12 Step 10 widened the graph-authoritative 5-Why swap to also fire on
plain DIRECT (single-hop) edges, not just multi-hop EXPLICIT/
EVIDENCE_CORRELATED chains, and measured 60 regressions across the full
suite. This phase root-caused all 60 to three structural gaps (see the long
comment above `is_graph_authoritative_for_five_why` in
app.agent.causal_graph) and fixed each at its source:

  1. `build_graph_grounded_five_why` unconditionally appended a second,
     duplicate EVIDENCE_BOUNDARY marker step even when the single emitted
     transition step was itself already the boundary (a POSSIBLE-status
     edge renders as status="UNKNOWN") -- INV-WHY-007's "padded to 2 UNKNOWN
     steps instead of stopping", the dominant failure (56 of 60).
  2. The graph walk silently picked one hypothesis among several
     independent siblings of the observed deviation and discarded the
     others, while the prose generator represents each sibling as its own
     step -- a manufactured winner among competing, unresolved hypotheses.
  3. A VERIFIED, directly-stated `immediate_mechanism` (or a semantic_type
     requiring structured phrasing, e.g. EVENT_SEQUENCE_CONTROL) that has
     no corresponding causal-graph node would silently vanish from the
     rendered chain if the graph became authoritative.

These tests exercise the resulting `is_graph_authoritative_for_five_why`
predicate and the corrected boundary-marker behavior directly, independent
of any full-pipeline LLM call.
"""
from __future__ import annotations

from app.agent.causal_graph import build_causal_graph, is_graph_authoritative_for_five_why
from app.agent.causal_graph_traversal import build_graph_grounded_five_why
from app.models.agent import (
    CandidateHypothesis,
    CanonicalFindingState,
    CausalGraph,
    CausalGraphEdge,
    CausalGraphEdgeStatus,
    CausalGraphNode,
    CausalGraphNodeType,
    CausalLevel,
    EpistemicSource,
    EvidenceItem,
    EvidenceStatus,
    RootCauseAnalysis,
    RootCauseStatus,
)


def _canonical(deviation: str = "An anomaly was recorded in the system.", **kwargs) -> CanonicalFindingState:
    return CanonicalFindingState(
        raw_finding=deviation, finding_subject="subject", affected_object="subject",
        affected_process="UNKNOWN", affected_activity="UNKNOWN", deviation=deviation,
        observed_deviation=deviation, facts=[deviation], **kwargs,
    )


def _hyp(**kwargs) -> CandidateHypothesis:
    defaults = dict(evidence_needed="", supporting_claim_ids=["c1"], supporting_evidence=["e1"])
    defaults.update(kwargs)
    return CandidateHypothesis(**defaults)


# ---------------------------------------------------------------------------
# 1. Single-edge authority cases
# ---------------------------------------------------------------------------

def test_single_verified_edge_is_authoritative():
    rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
        _hyp(id="H1", name="X", statement="A verified mechanism explains the deviation",
             status="SUPPORTED", evidence_strength="VERIFIED", causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM),
    ])
    g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
    ok, reason = is_graph_authoritative_for_five_why(g, _canonical())
    assert ok, reason


def test_single_possible_edge_is_authoritative_and_produces_no_padding():
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
        _hyp(id="H1", name="X", statement="A reported factor possibly contributed",
             status="POSSIBLE", evidence_strength="INDICATIVE", causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE),
    ])
    g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
    ok, reason = is_graph_authoritative_for_five_why(g, _canonical())
    assert ok, reason
    fw = build_graph_grounded_five_why(g)
    assert fw is not None
    # Exactly one step: the transition step already IS the boundary, no
    # second duplicate marker (INV-WHY-007).
    assert len(fw.steps) == 1
    assert fw.steps[0].boundary_status == "EVIDENCE_BOUNDARY"
    assert fw.steps[0].status == "UNKNOWN"


def test_single_reported_edge_is_authoritative():
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
        _hyp(id="H1", name="X", statement="A reported factor",
             status="POSSIBLE", evidence_strength="REPORTED", causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE),
    ])
    g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
    ok, reason = is_graph_authoritative_for_five_why(g, _canonical())
    assert ok, reason


def test_no_causal_edge_is_not_authoritative():
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[])
    g = build_causal_graph(_canonical(), rc, [])
    ok, reason = is_graph_authoritative_for_five_why(g, _canonical())
    assert not ok
    assert "no licensed causal edge" in reason


def test_empty_graph_is_not_authoritative():
    ok, reason = is_graph_authoritative_for_five_why(CausalGraph(nodes=[], edges=[]), None)
    assert not ok


# ---------------------------------------------------------------------------
# 2. Competing / independent hypotheses must not manufacture a winner
# ---------------------------------------------------------------------------

def test_competing_possible_hypotheses_block_authority():
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
        _hyp(id="H1", name="TRAINING", statement="Employee lacked training",
             status="POSSIBLE", evidence_strength="REPORTED", causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE),
        _hyp(id="H2", name="WORKLOAD", statement="Employee experienced workload pressure",
             status="POSSIBLE", evidence_strength="REPORTED", causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE),
    ])
    g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
    ok, reason = is_graph_authoritative_for_five_why(g, _canonical())
    assert not ok
    assert "independent hypotheses remain siblings" in reason


def test_independent_supported_and_possible_hypotheses_block_authority():
    """A VERIFIED/SUPPORTED hypothesis alongside an independent POSSIBLE one
    (e.g. 'the rule was disabled' + 'operators reported fatigue') must not
    let the graph silently pick the supported one and drop the other as a
    restated-and-marked-UNKNOWN step (INV-CAUSAL-001/003)."""
    rc = RootCauseAnalysis(status=RootCauseStatus.ESTABLISHED, candidate_hypotheses=[
        _hyp(id="H1", name="RULE_DISABLED", statement="The validation rule was disabled",
             status="SUPPORTED", evidence_strength="VERIFIED", causal_level=CausalLevel.L5_SYSTEMIC_CAUSE,
             supporting_claim_ids=["c1"]),
        _hyp(id="H2", name="FATIGUE", statement="Operators reported high shift fatigue",
             status="POSSIBLE", evidence_strength="REPORTED", causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE,
             supporting_claim_ids=["c2"]),
    ])
    g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
    ok, reason = is_graph_authoritative_for_five_why(g, _canonical())
    assert not ok


# ---------------------------------------------------------------------------
# 3. Explicit multi-hop chains still activate (regression guard on the old
#    `_has_multihop_chain` behavior, now subsumed by the same predicate)
# ---------------------------------------------------------------------------

def test_explicit_chain_still_authoritative_after_reparenting():
    rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
        _hyp(id="H1", name="IMMEDIATE", statement="Calibration verification was bypassed",
             status="SUPPORTED", evidence_strength="VERIFIED", causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM),
        _hyp(id="H2", name="SYSTEMIC", statement="Bypass authority was never revoked",
             status="SUPPORTED", evidence_strength="VERIFIED", causal_level=CausalLevel.L5_SYSTEMIC_CAUSE,
             deepens_hypothesis_id="H1"),
    ])
    g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
    # Exactly one edge should remain sourced at the deviation after re-parenting.
    root_edges = [e for e in g.edges if e.source_node_id == "CN_DEVIATION"]
    assert len(root_edges) == 1
    ok, reason = is_graph_authoritative_for_five_why(g, _canonical())
    assert ok, reason


# ---------------------------------------------------------------------------
# 4. Structural information-loss guards
# ---------------------------------------------------------------------------

def test_unrepresented_verified_immediate_mechanism_blocks_authority():
    canonical = _canonical(
        deviation="Inspection execution — missed",
        immediate_mechanism="A completely different verified mechanism sentence about calibration drift",
        immediate_mechanism_status="VERIFIED",
    )
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
        _hyp(id="H1", name="TRAINING", statement="Insufficient training contributed",
             status="POSSIBLE", evidence_strength="NONE", causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE),
    ])
    g = build_causal_graph(canonical, rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
    ok, reason = is_graph_authoritative_for_five_why(g, canonical)
    assert not ok
    assert "not represented as a VERIFIED node" in reason


def test_immediate_mechanism_restating_the_deviation_does_not_block_authority():
    """A different phrasing of the SAME observed fact (not a genuinely
    deeper mechanism) must not spuriously block authority."""
    canonical = _canonical(
        deviation="Inspection execution — missed",
        immediate_mechanism="Technician missed the inspection.",
        immediate_mechanism_status="VERIFIED",
    )
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
        _hyp(id="H1", name="TRAINING", statement="Insufficient training contributed",
             status="POSSIBLE", evidence_strength="NONE", causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE),
    ])
    g = build_causal_graph(canonical, rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
    ok, reason = is_graph_authoritative_for_five_why(g, canonical)
    assert ok, reason


def test_event_sequence_control_semantic_type_blocks_authority():
    canonical = _canonical(semantic_type="EVENT_SEQUENCE_CONTROL")
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
        _hyp(id="H1", name="OVERRIDE", statement="A transition occurred without justification",
             status="POSSIBLE", evidence_strength="REPORTED", causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE),
    ])
    g = build_causal_graph(canonical, rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
    ok, reason = is_graph_authoritative_for_five_why(g, canonical)
    assert not ok
    assert "EVENT_SEQUENCE_CONTROL" in reason


# ---------------------------------------------------------------------------
# 5. Cycle safety in the traversal itself (defense in depth on top of
#    construction-time cycle rejection)
# ---------------------------------------------------------------------------

def test_traversal_does_not_loop_on_a_malformed_cyclic_graph():
    """build_graph_grounded_five_why must terminate even if fed a
    hand-built graph containing a cycle (never producible by
    build_causal_graph itself, but the traversal must fail safe regardless
    of how the graph was constructed)."""
    nodes = [
        CausalGraphNode(node_id="D", node_type=CausalGraphNodeType.OBSERVED_DEVIATION,
                         label="deviation", causal_level=CausalLevel.L0_OBSERVATION,
                         epistemic_status=EvidenceStatus.VERIFIED, provenance=EpistemicSource.AUDIT_OBSERVATION),
        CausalGraphNode(node_id="A", node_type=CausalGraphNodeType.IMMEDIATE_MECHANISM,
                         label="mechanism A", causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM,
                         epistemic_status=EvidenceStatus.VERIFIED, provenance=EpistemicSource.OBJECTIVE_RECORD),
        CausalGraphNode(node_id="B", node_type=CausalGraphNodeType.UNDERLYING_CAUSE,
                         label="mechanism B", causal_level=CausalLevel.L4_ROOT_CAUSE,
                         epistemic_status=EvidenceStatus.VERIFIED, provenance=EpistemicSource.OBJECTIVE_RECORD),
    ]
    edges = [
        CausalGraphEdge(edge_id="E1", source_node_id="D", target_node_id="A",
                         status=CausalGraphEdgeStatus.VERIFIED, causal_level_transition="D->A"),
        CausalGraphEdge(edge_id="E2", source_node_id="A", target_node_id="B",
                         status=CausalGraphEdgeStatus.VERIFIED, causal_level_transition="A->B"),
        # Malformed cycle: B -> A, which would loop forever without the
        # visited-set guard in build_graph_grounded_five_why.
        CausalGraphEdge(edge_id="E3", source_node_id="B", target_node_id="A",
                         status=CausalGraphEdgeStatus.VERIFIED, causal_level_transition="B->A"),
    ]
    g = CausalGraph(nodes=nodes, edges=edges)
    fw = build_graph_grounded_five_why(g)
    assert fw is not None
    assert len(fw.steps) <= 5
    # Every node visited at most once across the emitted steps.
    visited_targets = [s.target_node_id for s in fw.steps if s.target_node_id]
    assert len(visited_targets) == len(set(visited_targets))
