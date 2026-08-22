"""Final output intelligence pass, Section 15: report consistency firewall.
Closes a genuine, confirmed gap: root_cause.status=ESTABLISHED/SUPPORTED
could coexist with a 5-Why chain whose every step reports no causal
mechanism (all UNKNOWN/NOT_ESTABLISHED/REQUIRES_EVIDENCE) -- two sections
of the same report disagreeing about whether a mechanism was found, which
none of the pre-existing invariants (including INV-CAUSAL-005, which only
checks the causal GRAPH, not the rendered 5-Why narrative) caught. Uses
abstract synthetic labels only.
"""
from __future__ import annotations

from app.agent.invariants import evaluate_all_invariants
from app.models.agent import (
    CandidateHypothesis,
    CausalGraph,
    CausalGraphEdge,
    CausalGraphNode,
    CausalGraphNodeType,
    EpistemicSource,
    EvidenceItem,
    EvidenceStatus,
    FiveWhyAnalysis,
    FiveWhyStep,
    RootCauseAnalysis,
    RootCauseStatus,
)


def _grounded_established_state(five_why_steps):
    hyp = CandidateHypothesis(
        id="H1", name="X", statement="s", evidence_needed="e", status="SUPPORTED",
        evidence_strength="VERIFIED", status_locked=True, causal_node_id="N2",
        supporting_evidence=["x"], supporting_claim_ids=["EV1"],
    )
    rc = RootCauseAnalysis(status=RootCauseStatus.ESTABLISHED, candidate_hypotheses=[hyp],
                            leading_hypothesis="H1", leading_hypothesis_status="SELECTED")
    graph = CausalGraph(
        nodes=[
            CausalGraphNode(node_id="N1", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="dev",
                             epistemic_status=EvidenceStatus.VERIFIED),
            CausalGraphNode(node_id="N2", node_type=CausalGraphNodeType.SYSTEMIC_ROOT_CAUSE, label="cause",
                             epistemic_status=EvidenceStatus.VERIFIED),
        ],
        edges=[CausalGraphEdge(edge_id="E1", source_node_id="N1", target_node_id="N2", status="VERIFIED",
                                evidence_ids=["EV1"], provenance=EpistemicSource.OBJECTIVE_RECORD)],
    )
    ledger = [EvidenceItem(claim="x", source="s", status=EvidenceStatus.VERIFIED, evidence_id="EV1", request_id="R1")]
    fw = FiveWhyAnalysis(steps=five_why_steps)
    return {"root_cause": rc, "causal_graph": graph, "five_why": fw, "evidence_ledger": ledger}


def _has(violations, inv_id):
    return any(inv_id in v for v in violations)


def test_established_root_cause_with_entirely_unknown_five_why_is_rejected():
    steps = [FiveWhyStep(question="why?", answer=None, status="UNKNOWN")]
    is_valid, violations = evaluate_all_invariants(_grounded_established_state(steps))
    assert not is_valid
    assert _has(violations, "INV-REPORT-001")


def test_established_root_cause_with_all_not_established_five_why_is_rejected():
    steps = [
        FiveWhyStep(question="why did the condition occur?", answer=None, status="NOT_ESTABLISHED"),
        FiveWhyStep(question="why did that condition exist?", answer=None, status="REQUIRES_EVIDENCE"),
    ]
    is_valid, violations = evaluate_all_invariants(_grounded_established_state(steps))
    assert not is_valid
    assert _has(violations, "INV-REPORT-001")


def test_established_root_cause_with_at_least_one_grounded_step_is_accepted():
    steps = [FiveWhyStep(
        question="why did the condition occur?", answer="a confirmed mechanism explains the transition",
        status="VERIFIED", source_node_id="N1", target_node_id="N2", causal_edge_id="E1",
    )]
    is_valid, violations = evaluate_all_invariants(_grounded_established_state(steps))
    assert not _has(violations, "INV-REPORT-001")


def test_supported_root_cause_with_supported_five_why_step_is_accepted():
    steps = [FiveWhyStep(question="why?", answer="mechanism is supported by available evidence", status="SUPPORTED")]
    state = _grounded_established_state(steps)
    state["root_cause"].status = RootCauseStatus.SUPPORTED
    is_valid, violations = evaluate_all_invariants(state)
    assert not _has(violations, "INV-REPORT-001")


def test_not_established_root_cause_with_unknown_five_why_is_never_flagged():
    """The invariant only applies to ESTABLISHED/SUPPORTED -- a genuinely
    unresolved investigation with an UNKNOWN 5-Why chain is fully
    consistent and must never be flagged."""
    hyp = CandidateHypothesis(id="H1", name="X", statement="s", evidence_needed="e", status="POSSIBLE")
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[hyp])
    fw = FiveWhyAnalysis(steps=[FiveWhyStep(question="why?", answer=None, status="UNKNOWN")])
    state = {"root_cause": rc, "five_why": fw}
    is_valid, violations = evaluate_all_invariants(state)
    assert not _has(violations, "INV-REPORT-001")


def test_established_root_cause_with_no_five_why_steps_at_all_not_flagged_by_this_invariant():
    """An empty 5-Why (no steps rendered) is a different, separate concern
    -- not this invariant's job to flag (it only compares CONTRADICTING
    content, not absence)."""
    state = _grounded_established_state([])
    is_valid, violations = evaluate_all_invariants(state)
    assert not _has(violations, "INV-REPORT-001")
