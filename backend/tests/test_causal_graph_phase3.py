"""Phase 3: real runtime CausalGraph structural test suite.

All fixtures are domain-neutral (constructed directly from typed model
objects — CandidateHypothesis/RootCauseAnalysis/EvidenceItem — never from
finding text in a specific industry vocabulary), so these tests exercise
the graph builder's STRUCTURE, not any particular domain's wording.
"""
from __future__ import annotations

from app.agent.causal_graph import build_causal_graph
from app.agent.causal_graph_traversal import build_graph_grounded_five_why, ground_five_why_steps
from app.agent.invariants import evaluate_all_invariants
from app.agent.output_quality_scorer import compute_output_quality_score
from app.models.agent import (
    CandidateHypothesis,
    CanonicalFindingState,
    CausalGraph,
    CausalGraphEdge,
    CausalGraphEdgeStatus,
    CausalGraphNode,
    CausalGraphNodeType,
    CausalLevel,
    EvidenceItem,
    EvidenceStatus,
    FiveWhyAnalysis,
    FiveWhyStep,
    RootCauseAnalysis,
    RootCauseStatus,
)


def _canonical(deviation: str = "Deviation X was observed") -> CanonicalFindingState:
    return CanonicalFindingState(
        raw_finding=deviation, finding_subject="subject", affected_object="subject",
        affected_process="UNKNOWN", affected_activity="UNKNOWN", deviation=deviation,
        observed_deviation=deviation, facts=[deviation],
    )


def _hyp(**kwargs) -> CandidateHypothesis:
    defaults = dict(id="H1", name="N", statement="A mechanism statement", status="POSSIBLE",
                     evidence_needed="", evidence_strength="NONE",
                     causal_level=CausalLevel.L3_IMMEDIATE_MECHANISM)
    defaults.update(kwargs)
    return CandidateHypothesis(**defaults)


def _ledger(*statuses: EvidenceStatus) -> list[EvidenceItem]:
    return [EvidenceItem(claim=f"claim {i}", source="finding", status=s) for i, s in enumerate(statuses)]


class TestCausalGraphConstruction:
    def test_1_verified_observation_no_mechanism(self):
        """A finding with a verified observation and zero candidate hypotheses
        produces a graph with only the OBSERVED_DEVIATION node."""
        g = build_causal_graph(_canonical(), RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED), _ledger(EvidenceStatus.VERIFIED))
        assert len(g.nodes) == 1
        assert g.nodes[0].node_type == CausalGraphNodeType.OBSERVED_DEVIATION
        assert not g.edges

    def test_2_verified_mechanism_unknown_deeper_cause(self):
        """A VERIFIED immediate mechanism with no deeper hypothesis: mechanism
        edge is VERIFIED, and there is no fabricated deeper node/edge."""
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            _hyp(id="H1", statement="Immediate mechanism", status="SUPPORTED",
                 evidence_strength="VERIFIED", causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM,
                 supporting_evidence=["e1"]),
        ])
        g = build_causal_graph(_canonical(), rc, _ledger(EvidenceStatus.VERIFIED))
        assert len(g.edges) == 1
        assert g.edges[0].status == CausalGraphEdgeStatus.VERIFIED
        underlying = [n for n in g.nodes if n.node_type == CausalGraphNodeType.UNDERLYING_CAUSE]
        assert not underlying, "No deeper cause was licensed — none must be fabricated"

    def test_3_reported_mechanism(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
            _hyp(id="H1", statement="Reported mechanism", status="POSSIBLE", evidence_strength="REPORTED"),
        ])
        g = build_causal_graph(_canonical(), rc, _ledger(EvidenceStatus.VERIFIED))
        assert g.edges[0].status == CausalGraphEdgeStatus.REPORTED

    def test_4_possible_mechanism(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
            _hyp(id="H1", statement="Possible mechanism", status="POSSIBLE", evidence_strength="INDICATIVE"),
        ])
        g = build_causal_graph(_canonical(), rc, _ledger(EvidenceStatus.VERIFIED))
        assert g.edges[0].status == CausalGraphEdgeStatus.POSSIBLE

    def test_5_competing_causal_hypotheses(self):
        """Two independently-licensed hypotheses both get nodes and edges —
        neither suppresses the other."""
        rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
            _hyp(id="H1", statement="Hypothesis A", status="POSSIBLE", evidence_strength="INDICATIVE"),
            _hyp(id="H2", statement="Hypothesis B", status="POSSIBLE", evidence_strength="INDICATIVE"),
        ])
        g = build_causal_graph(_canonical(), rc, _ledger(EvidenceStatus.VERIFIED))
        assert len(g.edges) == 2

    def test_6_conflicting_causal_evidence(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
            _hyp(id="H1", statement="Disputed mechanism", status="POSSIBLE", evidence_strength="CONFLICTING"),
        ])
        g = build_causal_graph(_canonical(), rc, _ledger(EvidenceStatus.VERIFIED))
        assert g.edges[0].status == CausalGraphEdgeStatus.DISPUTED

    def test_7_multiple_independent_causal_chains(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
            _hyp(id="H1", statement="Chain A mechanism", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"]),
            _hyp(id="H2", statement="Chain B mechanism", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e2"]),
        ])
        g = build_causal_graph(_canonical(), rc, _ledger(EvidenceStatus.VERIFIED))
        assert len({e.target_node_id for e in g.edges}) == 2

    def test_8_unrelated_evidence_does_not_create_edge(self):
        """A REFUTED, zero-evidence hypothesis produces no node/edge at all."""
        rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
            _hyp(id="H1", statement="Refuted speculation", status="REFUTED", evidence_strength="NONE"),
        ])
        g = build_causal_graph(_canonical(), rc, _ledger(EvidenceStatus.VERIFIED))
        assert len(g.nodes) == 1  # only OBSERVED_DEVIATION
        assert not g.edges

    def test_9_evidence_absence_does_not_create_causal_edge(self):
        """An UNVERIFIED hypothesis with NONE evidence strength (i.e. purely
        an absence-of-evidence situation) gets no licensed edge — evidence
        absence must never become a causal claim (Section 7)."""
        rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
            _hyp(id="H1", statement="No record was available", status="UNVERIFIED", evidence_strength="NONE"),
        ])
        g = build_causal_graph(_canonical(), rc, [])
        assert not g.edges

    def test_10_fully_established_root_cause(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.ESTABLISHED, candidate_hypotheses=[
            _hyp(id="H1", statement="Systemic root cause", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L5_SYSTEMIC_CAUSE, supporting_evidence=["e1"]),
        ])
        g = build_causal_graph(_canonical(), rc, _ledger(EvidenceStatus.VERIFIED))
        node = next(n for n in g.nodes if n.node_type == CausalGraphNodeType.SYSTEMIC_ROOT_CAUSE)
        edge = next(e for e in g.edges if e.target_node_id == node.node_id)
        assert edge.status == CausalGraphEdgeStatus.VERIFIED

    def test_11_skipped_causal_level_allowed(self):
        """A hypothesis jumping straight to L5_SYSTEMIC_CAUSE with no L2/L3/L4
        node present is structurally valid — levels may be skipped."""
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            _hyp(id="H1", statement="Direct systemic finding", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L5_SYSTEMIC_CAUSE, supporting_evidence=["e1"]),
        ])
        g = build_causal_graph(_canonical(), rc, _ledger(EvidenceStatus.VERIFIED))
        levels_present = {n.node_type for n in g.nodes}
        assert CausalGraphNodeType.IMMEDIATE_MECHANISM not in levels_present
        assert CausalGraphNodeType.SYSTEMIC_ROOT_CAUSE in levels_present

    def test_18_missing_traceability_downgrades_verified_edge(self):
        """A SUPPORTED+VERIFIED-labeled hypothesis backed by NO verified item
        in the actual evidence ledger must not retain VERIFIED edge status —
        the label alone is not evidence (evidence_ledger is load-bearing)."""
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            _hyp(id="H1", statement="Claims verification", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"]),
        ])
        g = build_causal_graph(_canonical(), rc, evidence_ledger=[])  # no VERIFIED item in ledger
        assert g.edges[0].status != CausalGraphEdgeStatus.VERIFIED

    def test_19_orphan_causal_edge_rejected_by_invariant(self):
        """A hand-crafted edge referencing a non-existent node must be caught
        by INV-CGRAPH-002-adjacent structural checks via the quality scorer."""
        cg = CausalGraph(
            nodes=[CausalGraphNode(node_id="CN_DEVIATION", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="dev")],
            edges=[CausalGraphEdge(edge_id="CE1", source_node_id="CN_DEVIATION", target_node_id="CN_GHOST",
                                    status=CausalGraphEdgeStatus.VERIFIED, evidence_ids=["e1"])],
        )
        score = compute_output_quality_score({"causal_graph": cg})
        causal_dim = next(d for d in score.dimensions if d.name == "Causal graph integrity")
        assert causal_dim.score == 0

    def test_20_invalid_causal_edge_self_loop_rejected(self):
        cg = CausalGraph(
            nodes=[CausalGraphNode(node_id="CN1", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="dev")],
            edges=[CausalGraphEdge(edge_id="CE1", source_node_id="CN1", target_node_id="CN1")],
        )
        state = {"causal_graph": cg}
        is_valid, violations = evaluate_all_invariants(state)
        assert not is_valid
        assert any("INV-CGRAPH-002" in v for v in violations)


class TestFiveWhyGraphGrounding:
    def test_12_max_depth_traversal_capped_at_five(self):
        nodes = [CausalGraphNode(node_id="CN_DEVIATION", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="dev")]
        edges = []
        for i in range(8):
            nid = f"CN{i}"
            nodes.append(CausalGraphNode(node_id=nid, node_type=CausalGraphNodeType.CONTRIBUTING_FACTOR,
                                          label=f"factor {i}", causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE))
            edges.append(CausalGraphEdge(edge_id=f"CE{i}", source_node_id="CN_DEVIATION", target_node_id=nid,
                                          status=CausalGraphEdgeStatus.POSSIBLE))
        g = CausalGraph(nodes=nodes, edges=edges)
        fw = build_graph_grounded_five_why(g)
        assert len(fw.steps) <= 5, "Traversal must cap at 5 steps even when more licensed edges exist"

    def test_13_restatement_rejected_by_source_ne_target_invariant(self):
        fw = FiveWhyAnalysis(steps=[
            FiveWhyStep(question="Why did X happen?", answer="X happened", status="UNKNOWN",
                        source_node_id="CN1", target_node_id="CN1", causal_edge_id="CE1"),
        ])
        cg = CausalGraph(
            nodes=[CausalGraphNode(node_id="CN1", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="X")],
            edges=[],
        )
        is_valid, violations = evaluate_all_invariants({"five_why": fw, "causal_graph": cg})
        assert not is_valid
        assert any("INV-CGRAPH-004" in v for v in violations)

    def test_22_stale_edge_id_rejected(self):
        fw = FiveWhyAnalysis(steps=[
            FiveWhyStep(question="Why?", answer="Because", status="VERIFIED", causal_edge_id="CE_FABRICATED"),
        ])
        cg = CausalGraph(nodes=[], edges=[])
        is_valid, violations = evaluate_all_invariants({"five_why": fw, "causal_graph": cg})
        assert not is_valid
        assert any("INV-CGRAPH-004" in v for v in violations)

    def test_14_evidence_boundary_no_edges_returns_none(self):
        """Section 13: when no licensed edge exists at all, the traversal must
        return None rather than fabricate a step to reach a target count."""
        g = CausalGraph(
            nodes=[CausalGraphNode(node_id="CN_DEVIATION", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="dev")],
            edges=[],
        )
        assert build_graph_grounded_five_why(g) is None

    def test_grounding_never_mutates_question_answer_status(self):
        """ground_five_why_steps only sets traceability fields; it must never
        rewrite question/answer/status text (Section 26 firewall)."""
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            _hyp(id="H1", statement="Root mechanism identified", status="SUPPORTED",
                 evidence_strength="VERIFIED", causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM,
                 supporting_evidence=["e1"]),
        ])
        g = build_causal_graph(_canonical(), rc, _ledger(EvidenceStatus.VERIFIED))
        original_question = "Why did the deviation occur?"
        original_answer = "Root mechanism identified"
        original_status = "VERIFIED"
        fw = FiveWhyAnalysis(steps=[FiveWhyStep(question=original_question, answer=original_answer, status=original_status)])
        ground_five_why_steps(fw, g)
        assert fw.steps[0].question == original_question
        assert fw.steps[0].answer == original_answer
        assert fw.steps[0].status == original_status
        assert fw.steps[0].causal_edge_id is not None, "A real match should have been grounded"


class TestDomainIndependence:
    """Section 21/22: the same construction logic must behave identically
    across completely unrelated domain vocabularies — proving node/edge
    typing is driven by structured fields (status/evidence_strength/
    causal_level), never by the hypothesis statement's wording."""

    def test_15_paraphrase_and_unseen_vocabulary_identical_structure(self):
        for statement in [
            "Welding joint integrity was compromised due to unqualified technician assignment",
            "Financial reconciliation discrepancy arose from an unvalidated exchange rate feed",
            "Irrigation valve failure occurred due to unmaintained solenoid actuator",
        ]:
            rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
                _hyp(id="H1", statement=statement, status="SUPPORTED", evidence_strength="VERIFIED",
                     causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"]),
            ])
            g = build_causal_graph(_canonical(), rc, _ledger(EvidenceStatus.VERIFIED))
            assert len(g.nodes) == 2
            assert len(g.edges) == 1
            assert g.edges[0].status == CausalGraphEdgeStatus.VERIFIED


class TestAdversarialMutation:
    """Section 22: mutating one causal dimension must not silently mutate
    unrelated canonical dimensions."""

    def test_16_mutate_evidence_strength_only_status_changes(self):
        base_kwargs = dict(id="H1", statement="A mechanism", status="POSSIBLE",
                            causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE)
        g_reported = build_causal_graph(_canonical(), RootCauseAnalysis(
            status=RootCauseStatus.NOT_ESTABLISHED,
            candidate_hypotheses=[_hyp(evidence_strength="REPORTED", **base_kwargs)],
        ), _ledger(EvidenceStatus.VERIFIED))
        g_indicative = build_causal_graph(_canonical(), RootCauseAnalysis(
            status=RootCauseStatus.NOT_ESTABLISHED,
            candidate_hypotheses=[_hyp(evidence_strength="INDICATIVE", **base_kwargs)],
        ), _ledger(EvidenceStatus.VERIFIED))
        # Only edge status differs; node label/causal_level are unaffected by the mutation.
        assert g_reported.nodes[1].label == g_indicative.nodes[1].label
        assert g_reported.nodes[1].causal_level == g_indicative.nodes[1].causal_level
        assert g_reported.edges[0].status != g_indicative.edges[0].status

    def test_17_mutate_causal_level_only_node_type_changes(self):
        base_kwargs = dict(id="H1", statement="A mechanism", status="SUPPORTED",
                            evidence_strength="VERIFIED", supporting_evidence=["e1"])
        g_l2 = build_causal_graph(_canonical(), RootCauseAnalysis(
            status=RootCauseStatus.SUPPORTED,
            candidate_hypotheses=[_hyp(causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, **base_kwargs)],
        ), _ledger(EvidenceStatus.VERIFIED))
        g_l5 = build_causal_graph(_canonical(), RootCauseAnalysis(
            status=RootCauseStatus.SUPPORTED,
            candidate_hypotheses=[_hyp(causal_level=CausalLevel.L5_SYSTEMIC_CAUSE, **base_kwargs)],
        ), _ledger(EvidenceStatus.VERIFIED))
        assert g_l2.nodes[1].node_type != g_l5.nodes[1].node_type
        # Edge status (VERIFIED) is unaffected by the causal-level mutation.
        assert g_l2.edges[0].status == g_l5.edges[0].status == CausalGraphEdgeStatus.VERIFIED
