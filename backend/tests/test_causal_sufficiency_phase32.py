"""Final output quality pass, Section 2/3: mechanism vs. root cause vs.
systemic cause depth distinction. Populates the previously-dead
RootCauseAnalysis.causal_sufficiency field (CausalSufficiencyAssessment
already existed on the model, declared but never assigned anywhere in
production code) via a new deterministic derivation function reusing the
SAME graph-depth/status vocabulary INV-CAUSAL-005 already licenses. Uses
abstract synthetic node labels only.
"""
from __future__ import annotations

from app.agent.causal_graph import derive_causal_sufficiency
from app.agent.invariants import evaluate_all_invariants
from app.models.agent import (
    CausalGraph,
    CausalGraphEdge,
    CausalGraphNode,
    CausalGraphNodeType,
    EvidenceStatus,
    RootCauseAnalysis,
    RootCauseStatus,
)


def _node(node_id, node_type):
    return CausalGraphNode(node_id=node_id, node_type=node_type, label="x", epistemic_status=EvidenceStatus.VERIFIED)


def _edge(edge_id, source, target, status="VERIFIED"):
    return CausalGraphEdge(edge_id=edge_id, source_node_id=source, target_node_id=target, status=status)


# ---------------------------------------------------------------------------
# derive_causal_sufficiency: direct unit tests, Section 3's 4 worked cases
# ---------------------------------------------------------------------------

def test_case_a_no_mechanism_edge_at_all():
    """Observation VERIFIED, mechanism UNKNOWN, root cause NOT_ESTABLISHED."""
    graph = CausalGraph(nodes=[_node("N1", CausalGraphNodeType.OBSERVED_DEVIATION)], edges=[])
    suff = derive_causal_sufficiency(graph)
    assert suff.mechanism_sufficiency == "UNKNOWN"
    assert suff.root_cause_sufficiency == "NOT_ESTABLISHED"
    assert suff.systemic_sufficiency == "UNKNOWN"


def test_case_b_mechanism_supported_root_cause_not_established():
    """Mechanism SUPPORTED (REPORTED-strength edge, not VERIFIED), root
    cause and systemic cause not reached."""
    graph = CausalGraph(
        nodes=[_node("N1", CausalGraphNodeType.OBSERVED_DEVIATION), _node("N2", CausalGraphNodeType.IMMEDIATE_MECHANISM)],
        edges=[_edge("E1", "N1", "N2", status="REPORTED")],
    )
    suff = derive_causal_sufficiency(graph)
    assert suff.mechanism_sufficiency == "SUPPORTED"
    assert suff.root_cause_sufficiency == "NOT_ESTABLISHED"
    assert suff.systemic_sufficiency == "UNKNOWN"


def test_case_c_mechanism_and_root_cause_established_systemic_unknown():
    graph = CausalGraph(
        nodes=[
            _node("N1", CausalGraphNodeType.OBSERVED_DEVIATION),
            _node("N2", CausalGraphNodeType.IMMEDIATE_MECHANISM),
            _node("N3", CausalGraphNodeType.UNDERLYING_CAUSE),
        ],
        edges=[_edge("E1", "N1", "N2"), _edge("E2", "N2", "N3")],
    )
    suff = derive_causal_sufficiency(graph)
    assert suff.mechanism_sufficiency == "ESTABLISHED"
    assert suff.root_cause_sufficiency == "ESTABLISHED"
    assert suff.systemic_sufficiency == "UNKNOWN"


def test_case_d_all_three_levels_established():
    graph = CausalGraph(
        nodes=[
            _node("N1", CausalGraphNodeType.OBSERVED_DEVIATION),
            _node("N2", CausalGraphNodeType.IMMEDIATE_MECHANISM),
            _node("N3", CausalGraphNodeType.UNDERLYING_CAUSE),
            _node("N4", CausalGraphNodeType.SYSTEMIC_ROOT_CAUSE),
        ],
        edges=[_edge("E1", "N1", "N2"), _edge("E2", "N2", "N3"), _edge("E3", "N3", "N4")],
    )
    suff = derive_causal_sufficiency(graph)
    assert suff.mechanism_sufficiency == "ESTABLISHED"
    assert suff.root_cause_sufficiency == "ESTABLISHED"
    assert suff.systemic_sufficiency == "ESTABLISHED"


def test_empty_graph_returns_safe_defaults():
    suff = derive_causal_sufficiency(None)
    assert suff.mechanism_sufficiency == "UNKNOWN"
    assert suff.root_cause_sufficiency == "NOT_ESTABLISHED"
    assert suff.systemic_sufficiency == "UNKNOWN"


def test_systemic_reached_directly_also_satisfies_root_cause_sufficiency():
    """The exact regression this fix corrects: a hypothesis licensed
    straight from the observation to SYSTEMIC_ROOT_CAUSE (skipping an
    intermediate UNDERLYING_CAUSE node -- a real production shape) must
    ALSO report root_cause_sufficiency=ESTABLISHED, mirroring
    INV-CAUSAL-005's own OR condition over both node types."""
    graph = CausalGraph(
        nodes=[_node("N1", CausalGraphNodeType.OBSERVED_DEVIATION), _node("N2", CausalGraphNodeType.SYSTEMIC_ROOT_CAUSE)],
        edges=[_edge("E1", "N1", "N2", status="VERIFIED")],
    )
    suff = derive_causal_sufficiency(graph)
    assert suff.systemic_sufficiency == "ESTABLISHED"
    assert suff.root_cause_sufficiency == "ESTABLISHED"
    assert suff.mechanism_sufficiency == "UNKNOWN"


def test_unreached_tier_does_not_inherit_a_shallower_tiers_status():
    """A VERIFIED edge to the mechanism node must not cause the root-cause
    or systemic tiers (never reached by any edge) to also read ESTABLISHED
    -- each tier's status comes only from edges that actually target it."""
    graph = CausalGraph(
        nodes=[_node("N1", CausalGraphNodeType.OBSERVED_DEVIATION), _node("N2", CausalGraphNodeType.IMMEDIATE_MECHANISM)],
        edges=[_edge("E1", "N1", "N2", status="VERIFIED")],
    )
    suff = derive_causal_sufficiency(graph)
    assert suff.mechanism_sufficiency == "ESTABLISHED"
    assert suff.root_cause_sufficiency == "NOT_ESTABLISHED"
    assert suff.systemic_sufficiency == "UNKNOWN"


# ---------------------------------------------------------------------------
# INV-REPORT-002: mechanism-only ESTABLISHED must never render as root cause
# ---------------------------------------------------------------------------

def test_invariant_rejects_established_root_cause_backed_only_by_mechanism():
    graph = CausalGraph(
        nodes=[_node("N1", CausalGraphNodeType.OBSERVED_DEVIATION), _node("N2", CausalGraphNodeType.IMMEDIATE_MECHANISM)],
        edges=[_edge("E1", "N1", "N2")],
    )
    rc = RootCauseAnalysis(status=RootCauseStatus.ESTABLISHED, causal_sufficiency=derive_causal_sufficiency(graph))
    state = {"root_cause": rc}
    is_valid, violations = evaluate_all_invariants(state)
    assert any("INV-REPORT-002" in v for v in violations)


def test_invariant_accepts_established_root_cause_backed_by_underlying_cause():
    graph = CausalGraph(
        nodes=[
            _node("N1", CausalGraphNodeType.OBSERVED_DEVIATION),
            _node("N2", CausalGraphNodeType.IMMEDIATE_MECHANISM),
            _node("N3", CausalGraphNodeType.UNDERLYING_CAUSE),
        ],
        edges=[_edge("E1", "N1", "N2"), _edge("E2", "N2", "N3")],
    )
    rc = RootCauseAnalysis(status=RootCauseStatus.ESTABLISHED, causal_sufficiency=derive_causal_sufficiency(graph))
    state = {"root_cause": rc}
    is_valid, violations = evaluate_all_invariants(state)
    assert not any("INV-REPORT-002" in v for v in violations)


def test_invariant_never_flagged_when_causal_sufficiency_unpopulated():
    """Backward compatibility: a state that never populated
    causal_sufficiency (e.g. an older code path, or a test predating this
    field) must not be penalized -- INV-CAUSAL-005 remains the authority
    in that case."""
    rc = RootCauseAnalysis(status=RootCauseStatus.ESTABLISHED)
    state = {"root_cause": rc}
    is_valid, violations = evaluate_all_invariants(state)
    assert not any("INV-REPORT-002" in v for v in violations)


def test_invariant_never_flagged_for_not_established_root_cause():
    graph = CausalGraph(nodes=[_node("N1", CausalGraphNodeType.OBSERVED_DEVIATION)], edges=[])
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, causal_sufficiency=derive_causal_sufficiency(graph))
    state = {"root_cause": rc}
    is_valid, violations = evaluate_all_invariants(state)
    assert not any("INV-REPORT-002" in v for v in violations)
