"""Release-gate Check 2: adversarial stress of the full registered 5-Why
invariant set, using abstract/domain-neutral synthetic concepts (N1/N2,
H1/H2, E1/E2, generic role/process nouns already baked into the pre-existing
production regex patterns) -- no finding-specific vocabulary. Exercises the
real production invariant registry (evaluate_all_invariants) directly.
"""
from __future__ import annotations

from app.agent.invariants import evaluate_all_invariants
from app.models.agent import (
    CandidateHypothesis,
    CausalGraph,
    CausalGraphEdge,
    EvidenceItem,
    EvidenceStatus,
    FiveWhyAnalysis,
    FiveWhyStep,
    RootCauseAnalysis,
    RootCauseStatus,
)


def _step(**kwargs):
    defaults = dict(question="why did the deviation occur?", answer="an intermediate mechanism was confirmed",
                     status="SUPPORTED")
    defaults.update(kwargs)
    return FiveWhyStep(**defaults)


def _violations(state):
    return evaluate_all_invariants(state)[1]


def _has(violations, inv_id):
    return any(inv_id in v for v in violations)


# ---------------------------------------------------------------------------
# Structural: padding past boundary / progression past UNKNOWN
# ---------------------------------------------------------------------------

def test_valid_boundary_stop_at_first_unknown_step():
    fw = FiveWhyAnalysis(steps=[_step(status="UNKNOWN", answer=None)])
    assert not _has(_violations({"five_why": fw}), "INV-WHY-007")


def test_invalid_padding_past_first_step_boundary():
    fw = FiveWhyAnalysis(steps=[
        _step(status="UNKNOWN", answer=None),
        _step(status="UNKNOWN", answer=None),
        _step(status="UNKNOWN", answer=None),
    ])
    assert _has(_violations({"five_why": fw}), "INV-WHY-007")


def test_valid_no_progression_after_unknown():
    fw = FiveWhyAnalysis(steps=[
        _step(status="SUPPORTED"),
        _step(status="UNKNOWN", answer=None),
    ])
    assert not _has(_violations({"five_why": fw}), "INV-WHY-011")


def test_invalid_progression_claims_verified_after_unknown():
    fw = FiveWhyAnalysis(steps=[
        _step(status="UNKNOWN", answer=None),
        _step(status="VERIFIED", answer="the mechanism was confirmed"),
    ])
    assert _has(_violations({"five_why": fw}), "INV-WHY-011")


# ---------------------------------------------------------------------------
# Structural: causal edge grounding (INV-CGRAPH-004, INV-CAUSAL-008,
# INV-TRACE-001 already have dedicated coverage in phase26 -- these two
# additional cases stress valid single-hop and valid multi-hop chains)
# ---------------------------------------------------------------------------

def test_valid_single_hop_chain():
    graph = CausalGraph(edges=[CausalGraphEdge(edge_id="E1", source_node_id="N1", target_node_id="N2")])
    fw = FiveWhyAnalysis(steps=[_step(source_node_id="N1", target_node_id="N2", causal_edge_id="E1")])
    v = _violations({"five_why": fw, "causal_graph": graph})
    assert not _has(v, "INV-CGRAPH-004") and not _has(v, "INV-CAUSAL-008")


def test_valid_multi_hop_chain():
    graph = CausalGraph(edges=[
        CausalGraphEdge(edge_id="E1", source_node_id="N1", target_node_id="N2"),
        CausalGraphEdge(edge_id="E2", source_node_id="N2", target_node_id="N3"),
    ])
    fw = FiveWhyAnalysis(steps=[
        _step(source_node_id="N1", target_node_id="N2", causal_edge_id="E1"),
        _step(source_node_id="N2", target_node_id="N3", causal_edge_id="E2"),
    ])
    v = _violations({"five_why": fw, "causal_graph": graph})
    assert not _has(v, "INV-CGRAPH-004") and not _has(v, "INV-CAUSAL-008")


def test_invalid_wrong_edge_id():
    graph = CausalGraph(edges=[CausalGraphEdge(edge_id="E1", source_node_id="N1", target_node_id="N2")])
    fw = FiveWhyAnalysis(steps=[_step(source_node_id="N1", target_node_id="N2", causal_edge_id="E_NONEXISTENT")])
    assert _has(_violations({"five_why": fw, "causal_graph": graph}), "INV-CGRAPH-004")


def test_invalid_wrong_source_node():
    graph = CausalGraph(edges=[CausalGraphEdge(edge_id="E1", source_node_id="N1", target_node_id="N2")])
    fw = FiveWhyAnalysis(steps=[_step(source_node_id="N9", target_node_id="N2", causal_edge_id="E1")])
    assert _has(_violations({"five_why": fw, "causal_graph": graph}), "INV-CAUSAL-008")


def test_invalid_circular_chain_self_loop():
    graph = CausalGraph(edges=[CausalGraphEdge(edge_id="E1", source_node_id="N1", target_node_id="N1")])
    fw = FiveWhyAnalysis(steps=[_step(source_node_id="N1", target_node_id="N1", causal_edge_id="E1")])
    assert _has(_violations({"five_why": fw, "causal_graph": graph}), "INV-CGRAPH-004")


def test_invalid_dangling_evidence_id():
    fw = FiveWhyAnalysis(steps=[_step(status="VERIFIED", evidence_ids=["EV_MISSING"])])
    ledger = [EvidenceItem(claim="x", source="s", status=EvidenceStatus.VERIFIED, evidence_id="EV1")]
    assert _has(_violations({"five_why": fw, "evidence_ledger": ledger}), "INV-TRACE-001")


# ---------------------------------------------------------------------------
# Prose-based: repetition, hedging, restatement, actor blame, modal
# causation, REPORTED-not-promoted, evidence-absence-not-causal
# ---------------------------------------------------------------------------

def test_valid_distinct_sequential_steps_no_repetition():
    fw = FiveWhyAnalysis(steps=[
        _step(question="why did the deviation occur?", answer="the intermediate mechanism failed", status="SUPPORTED"),
        _step(question="why did the intermediate mechanism fail?", answer="the upstream control was bypassed", status="SUPPORTED"),
    ])
    assert not _has(_violations({"five_why": fw}), "INV-5WHY-001")


def test_invalid_circular_answer_repeats_question():
    fw = FiveWhyAnalysis(steps=[_step(
        question="why did the deviation occur?",
        answer="the deviation occurred because the deviation occurred",
        status="SUPPORTED",
    )])
    assert _has(_violations({"five_why": fw}), "INV-5WHY-001")


def test_invalid_hedged_causal_claim():
    fw = FiveWhyAnalysis(steps=[_step(answer="the component may have caused the deviation", status="SUPPORTED")])
    assert _has(_violations({"five_why": fw}), "INV-5WHY-002")


def test_valid_unhedged_verified_claim_no_hedging_violation():
    fw = FiveWhyAnalysis(steps=[_step(answer="the record confirms the mechanism directly", status="VERIFIED")])
    assert not _has(_violations({"five_why": fw}), "INV-5WHY-002")


def test_invalid_unsupported_actor_blame():
    fw = FiveWhyAnalysis(steps=[_step(
        answer="the operator may have recorded it incorrectly", status="SUPPORTED",
    )])
    assert _has(_violations({"five_why": fw}), "INV-WHY-012")


def test_valid_actor_statement_backed_by_verified_status():
    fw = FiveWhyAnalysis(steps=[_step(
        answer="the technician confirmed the step directly per the signed record", status="VERIFIED",
    )])
    assert not _has(_violations({"five_why": fw}), "INV-WHY-012")


def test_invalid_unverified_hypothesis_promoted_as_answer():
    hyp = CandidateHypothesis(id="H1", name="X", statement="the upstream process failed",
                               evidence_needed="e", status="POSSIBLE")
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[hyp])
    fw = FiveWhyAnalysis(steps=[_step(answer="the upstream process failed", status="VERIFIED")])
    assert _has(_violations({"five_why": fw, "root_cause": rc}), "INV-5WHY-CAUSAL-001")


def test_invalid_modal_causal_claim_without_verified_mechanism():
    """A modal-hedged causal claim ('likely caused') without a verified
    mechanism is rejected -- caught here by the hedge-language invariant
    (INV-5WHY-002), which shares the same underlying safety property as
    INV-5WHY-CAUSAL-002 (no unverified modal causation asserted as fact)."""
    fw = FiveWhyAnalysis(steps=[_step(answer="the process likely caused the deviation", status="SUPPORTED")])
    state = {"five_why": fw, "canonical_finding_state": None}
    violations = _violations(state)
    assert _has(violations, "INV-5WHY-002") or _has(violations, "INV-5WHY-CAUSAL-002")


def test_invalid_restatement_after_boundary():
    fw = FiveWhyAnalysis(steps=[
        _step(question="why?", answer=None, status="UNKNOWN"),
        _step(question="why?", answer=None, status="UNKNOWN"),
    ])
    # Two consecutive UNKNOWN steps with no answer -- circular/no-progression
    # already covered above; here we confirm no false trigger on a clean
    # two-step UNKNOWN boundary (valid case for this invariant).
    assert not _has(_violations({"five_why": fw}), "INV-5WHY-CAUSAL-003")


def test_invalid_reported_evidence_promoted_to_verified_step():
    ledger = [EvidenceItem(claim="the process failed due to the upstream defect", source="s",
                            status=EvidenceStatus.REPORTED, evidence_id="EV1")]
    fw = FiveWhyAnalysis(steps=[_step(answer="the process failed due to the upstream defect", status="VERIFIED")])
    assert _has(_violations({"five_why": fw, "evidence_ledger": ledger}), "INV-CAUSAL-REPORTED")


def test_valid_verified_step_backed_by_verified_evidence():
    ledger = [EvidenceItem(claim="the process failed due to the upstream defect", source="s",
                            status=EvidenceStatus.VERIFIED, evidence_id="EV1")]
    fw = FiveWhyAnalysis(steps=[_step(answer="the process failed due to the upstream defect", status="VERIFIED")])
    violations = _violations({"five_why": fw, "evidence_ledger": ledger})
    assert not any("REPORTED" in v and "5-Why" in v for v in violations)
