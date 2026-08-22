"""Phase 26: 5-Why <-> causal graph semantic alignment and downstream
epistemic-status provenance. Uses abstract, domain-neutral node/hypothesis
labels (N1, N2, EV1, ...) -- no finding-specific vocabulary, per this
phase's explicit instruction. Exercises the real production invariants
(INV-CGRAPH-004, INV-CAUSAL-008, INV-TRACE-001) directly.
"""
from __future__ import annotations

from app.agent.invariants import evaluate_all_invariants
from app.models.agent import (
    CausalGraph,
    CausalGraphEdge,
    CausalGraphNode,
    EvidenceItem,
    EvidenceStatus,
    FiveWhyAnalysis,
    FiveWhyStep,
)


def _edge(edge_id="E1", source="N1", target="N2"):
    return CausalGraphEdge(edge_id=edge_id, source_node_id=source, target_node_id=target)


def _graph(edges=None, nodes=None):
    return CausalGraph(nodes=nodes or [], edges=edges or [])


def _step(question="why", answer="because", status="SUPPORTED", source=None, target=None,
          edge_id=None, evidence_ids=None):
    return FiveWhyStep(
        question=question, answer=answer, status=status,
        source_node_id=source, target_node_id=target, causal_edge_id=edge_id,
        evidence_ids=evidence_ids or [],
    )


def _violations(state):
    is_valid, violations = evaluate_all_invariants(state)
    return violations


# ---------------------------------------------------------------------------
# INV-CAUSAL-008: step must describe the SAME transition its edge licenses
# ---------------------------------------------------------------------------

def test_step_matching_its_edge_is_valid():
    graph = _graph(edges=[_edge("E1", "N1", "N2")])
    fw = FiveWhyAnalysis(steps=[_step(source="N1", target="N2", edge_id="E1")])
    state = {"five_why": fw, "causal_graph": graph}
    assert not any("INV-CAUSAL-008" in v for v in _violations(state))


def test_step_source_disconnected_from_its_own_edge_is_rejected():
    """The exact defect this phase targets: a structurally-valid edge
    reference (INV-CGRAPH-004 would pass) whose step claims a DIFFERENT
    source than the edge actually has -- semantically disconnected."""
    graph = _graph(edges=[_edge("E1", "N1", "N2")])
    fw = FiveWhyAnalysis(steps=[_step(source="N9_UNRELATED", target="N2", edge_id="E1")])
    state = {"five_why": fw, "causal_graph": graph}
    assert any("INV-CAUSAL-008" in v for v in _violations(state))


def test_step_target_disconnected_from_its_own_edge_is_rejected():
    graph = _graph(edges=[_edge("E1", "N1", "N2")])
    fw = FiveWhyAnalysis(steps=[_step(source="N1", target="N9_UNRELATED", edge_id="E1")])
    state = {"five_why": fw, "causal_graph": graph}
    assert any("INV-CAUSAL-008" in v for v in _violations(state))


def test_step_with_no_edge_id_is_not_flagged():
    """A boundary step (no causal_edge_id) is not this invariant's concern
    -- it never claims a transition in the first place."""
    fw = FiveWhyAnalysis(steps=[_step(source=None, target=None, edge_id=None)])
    state = {"five_why": fw, "causal_graph": _graph()}
    assert not any("INV-CAUSAL-008" in v for v in _violations(state))


def test_multiple_independent_hops_each_validated_independently():
    """Independent facts represented as separate, correctly-grounded steps
    (not silently chained) -- each hop must resolve to its OWN edge."""
    graph = _graph(edges=[_edge("E1", "N1", "N2"), _edge("E2", "N2", "N3")])
    fw = FiveWhyAnalysis(steps=[
        _step(source="N1", target="N2", edge_id="E1"),
        _step(source="N2", target="N3", edge_id="E2"),
    ])
    state = {"five_why": fw, "causal_graph": graph}
    assert not any("INV-CAUSAL-008" in v for v in _violations(state))


def test_fabricated_hop_between_two_real_but_unconnected_edges_is_rejected():
    """Two real, valid edges exist (N1->N2, N4->N5) but a step falsely
    claims a N2->N4 transition citing one of them -- must be rejected."""
    graph = _graph(edges=[_edge("E1", "N1", "N2"), _edge("E2", "N4", "N5")])
    fw = FiveWhyAnalysis(steps=[_step(source="N2", target="N4", edge_id="E1")])  # E1 is actually N1->N2
    state = {"five_why": fw, "causal_graph": graph}
    assert any("INV-CAUSAL-008" in v for v in _violations(state))


# ---------------------------------------------------------------------------
# INV-TRACE-001: FiveWhyStep.evidence_ids provenance + no downstream
# epistemic-status upgrade
# ---------------------------------------------------------------------------

def test_evidence_ids_resolving_to_real_ledger_entries_is_valid():
    ledger = [EvidenceItem(claim="x", source="s", status=EvidenceStatus.VERIFIED, evidence_id="EV1")]
    fw = FiveWhyAnalysis(steps=[_step(status="VERIFIED", evidence_ids=["EV1"])])
    state = {"five_why": fw, "evidence_ledger": ledger}
    assert not any("INV-TRACE-001" in v for v in _violations(state))


def test_dangling_evidence_id_is_rejected():
    ledger = [EvidenceItem(claim="x", source="s", status=EvidenceStatus.VERIFIED, evidence_id="EV1")]
    fw = FiveWhyAnalysis(steps=[_step(status="SUPPORTED", evidence_ids=["EV_DOES_NOT_EXIST"])])
    state = {"five_why": fw, "evidence_ledger": ledger}
    assert any("INV-TRACE-001" in v for v in _violations(state))


def test_step_claims_verified_but_cited_evidence_is_reported_is_rejected():
    """The exact Section 3 downstream-mismatch defect: authoritative ledger
    status is REPORTED, but the 5-Why step independently relabels it
    VERIFIED."""
    ledger = [EvidenceItem(claim="x", source="s", status=EvidenceStatus.REPORTED, evidence_id="EV1")]
    fw = FiveWhyAnalysis(steps=[_step(status="VERIFIED", evidence_ids=["EV1"])])
    state = {"five_why": fw, "evidence_ledger": ledger}
    assert any("INV-TRACE-001" in v for v in _violations(state))


def test_step_claims_verified_with_mixed_evidence_one_verified_is_accepted():
    """If AT LEAST ONE cited item is genuinely VERIFIED, the step's VERIFIED
    label is grounded -- not required that ALL cited evidence be VERIFIED."""
    ledger = [
        EvidenceItem(claim="a", source="s", status=EvidenceStatus.REPORTED, evidence_id="EV1"),
        EvidenceItem(claim="b", source="s", status=EvidenceStatus.VERIFIED, evidence_id="EV2"),
    ]
    fw = FiveWhyAnalysis(steps=[_step(status="VERIFIED", evidence_ids=["EV1", "EV2"])])
    state = {"five_why": fw, "evidence_ledger": ledger}
    assert not any("INV-TRACE-001" in v for v in _violations(state))


def test_non_verified_step_status_never_flagged_regardless_of_evidence():
    ledger = [EvidenceItem(claim="x", source="s", status=EvidenceStatus.UNKNOWN, evidence_id="EV1")]
    fw = FiveWhyAnalysis(steps=[_step(status="REPORTED", evidence_ids=["EV1"])])
    state = {"five_why": fw, "evidence_ledger": ledger}
    assert not any("INV-TRACE-001" in v for v in _violations(state))


def test_step_with_no_evidence_ids_not_flagged():
    fw = FiveWhyAnalysis(steps=[_step(status="VERIFIED", evidence_ids=[])])
    state = {"five_why": fw, "evidence_ledger": []}
    assert not any("INV-TRACE-001" in v for v in _violations(state))


def test_empty_ledger_with_no_evidence_id_addressable_items_is_not_this_invariants_concern():
    """No evidence_id-addressable ledger this run (e.g. legacy finding-text
    claims) -- nothing to cross-check, must not false-positive."""
    ledger = [EvidenceItem(claim="x", source="s", status=EvidenceStatus.VERIFIED)]  # no evidence_id
    fw = FiveWhyAnalysis(steps=[_step(status="VERIFIED", evidence_ids=["EV1"])])
    state = {"five_why": fw, "evidence_ledger": ledger}
    assert not any("INV-TRACE-001" in v for v in _violations(state))
