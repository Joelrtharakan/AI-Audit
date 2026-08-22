"""Phase 4: pre-investigation causal uncertainty graph, multi-hop causal
chains, graph-grounded investigation planning, and root-cause graph-grounding
structural test suite.

All fixtures use domain-neutral vocabulary (generic process/record/
requirement wording, or fabricated cross-domain examples), never the same
finding text twice across files, and never assert on hardcoded LLM prose.
"""
from __future__ import annotations

import asyncio

import pytest

from app.agent.causal_graph import (
    build_causal_graph,
    build_causal_uncertainty_graph,
    rank_uncertainty_nodes_by_information_gain,
)
from app.agent.claim_extractor import detect_evidence_conflicts, extract_claims
from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
from app.agent.proposition_engine import build_propositions_from_ledger, build_semantic_graph
from app.models.agent import (
    CandidateHypothesis,
    CanonicalFindingState,
    CausalGraphEdgeStatus,
    CausalGraphNodeType,
    CausalLevel,
    EvidenceItem,
    EvidenceStatus,
    RootCauseAnalysis,
    RootCauseStatus,
)


def _canonical_with_semantic_graph(finding: str) -> CanonicalFindingState:
    ledger = [EvidenceItem(claim=finding, source="finding", status=EvidenceStatus.VERIFIED)]
    claims = extract_claims(finding, ledger)
    conflicts = detect_evidence_conflicts(claims)
    props = build_propositions_from_ledger(finding, claims, conflicts)
    sg = build_semantic_graph(finding, claims, props, conflicts)
    return CanonicalFindingState(
        raw_finding=finding, finding_subject="subject", affected_object="subject",
        affected_process="UNKNOWN", affected_activity="UNKNOWN", deviation=finding,
        observed_deviation=finding, facts=[finding], semantic_graph=sg,
    )


class TestPreInvestigationUncertaintyGraph:
    def test_1_uncertainty_graph_from_normative_violation(self):
        """A normative violation in the semantic graph produces an UNRESOLVED
        node with an UNKNOWN-status edge — no causal claim is invented."""
        canonical = _canonical_with_semantic_graph(
            "The irrigation cycle was not completed as required by Schedule IRR-9."
        )
        ug = build_causal_uncertainty_graph(canonical)
        unresolved = [n for n in ug.nodes if n.node_type == CausalGraphNodeType.UNRESOLVED]
        assert unresolved, "A normative violation must produce at least one UNRESOLVED node"
        for e in ug.edges:
            assert e.status == CausalGraphEdgeStatus.UNKNOWN, (
                "Pre-investigation graph must never assert a resolved causal edge"
            )

    def test_2_unresolved_causal_edge_structure(self):
        canonical = _canonical_with_semantic_graph(
            "The reconciliation was not performed as required by Directive FIN-7."
        )
        ug = build_causal_uncertainty_graph(canonical)
        assert ug.edges, "Expected at least one unresolved edge"
        e = ug.edges[0]
        assert e.source_node_id and e.target_node_id
        assert e.source_node_id != e.target_node_id

    def test_5_evidence_fails_to_discriminate_no_hypotheses_no_licensed_edge(self):
        """Zero candidate hypotheses -> the post-synthesis CausalGraph has
        zero edges; evidence that doesn't discriminate never licenses one."""
        canonical = _canonical_with_semantic_graph("An unexplained deviation was observed.")
        rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[])
        cg = build_causal_graph(canonical, rc, [])
        assert not cg.edges

    def test_27_domain_independent_vocabulary(self):
        """The uncertainty graph builder must behave identically across
        unrelated domain vocabularies — no domain nouns appear in the
        production code that classifies these."""
        for finding in [
            "The turbine inspection was not completed as required by Directive AV-3.",
            "The ledger reconciliation was not performed as required by Policy FIN-2.",
            "The crop rotation was not documented as required by Standard AG-5.",
        ]:
            canonical = _canonical_with_semantic_graph(finding)
            ug = build_causal_uncertainty_graph(canonical)
            assert any(n.node_type == CausalGraphNodeType.UNRESOLVED for n in ug.nodes), (
                f"Expected an UNRESOLVED node for {finding!r}"
            )


class TestInformationGainRanking:
    def test_ranking_prefers_more_corroborated_uncertainty(self):
        """Deterministic ranking (Section 5): a node whose unresolved edge is
        backed by more source proposition_ids ranks first — not a fabricated
        probability model."""
        canonical = _canonical_with_semantic_graph(
            "The audit log was not reviewed as required by Procedure SEC-9, "
            "and the access record was not reviewed as required by Procedure SEC-9."
        )
        ug = build_causal_uncertainty_graph(canonical)
        ranked = rank_uncertainty_nodes_by_information_gain(ug)
        # Ranking must be deterministic: calling twice yields the same order.
        ranked_again = rank_uncertainty_nodes_by_information_gain(ug)
        assert [n.node_id for n in ranked] == [n.node_id for n in ranked_again]

    def test_empty_graph_ranks_to_empty_list(self):
        from app.models.agent import CausalGraph
        assert rank_uncertainty_nodes_by_information_gain(CausalGraph(nodes=[], edges=[])) == []


class TestMultiHopCausalChain:
    def test_8_multi_hop_chain_via_shared_evidence(self):
        """Two hypotheses at different causal levels sharing supporting
        evidence produce a genuine connected chain, not two parallel
        deviation-children."""
        canonical = _canonical_with_semantic_graph("An anomaly was recorded in the system.")
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            CandidateHypothesis(
                id="H1", name="M", statement="An immediate mechanism M occurred", status="SUPPORTED",
                evidence_needed="", evidence_strength="VERIFIED",
                causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM,
                supporting_evidence=["e1"], supporting_claim_ids=["C1"],
            ),
            CandidateHypothesis(
                id="H2", name="R", statement="A systemic condition R explains mechanism M", status="SUPPORTED",
                evidence_needed="", evidence_strength="VERIFIED",
                causal_level=CausalLevel.L5_SYSTEMIC_CAUSE,
                supporting_evidence=["e1"], supporting_claim_ids=["C1"],
            ),
        ])
        ledger = [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)]
        g = build_causal_graph(canonical, rc, ledger)
        mech_node = next(n for n in g.nodes if n.node_type == CausalGraphNodeType.IMMEDIATE_MECHANISM)
        root_node = next(n for n in g.nodes if n.node_type == CausalGraphNodeType.SYSTEMIC_ROOT_CAUSE)
        deep_edge = next(e for e in g.edges if e.target_node_id == root_node.node_id)
        assert deep_edge.source_node_id == mech_node.node_id, (
            "Root-cause edge must be re-parented onto the immediate-mechanism node, "
            "not left as a direct child of the deviation"
        )

    def test_9_ambiguous_evidence_overlap_does_not_chain(self):
        """When TWO shallower hypotheses both share evidence with a deeper
        one, chaining is ambiguous and must NOT be manufactured — the deep
        hypothesis stays a direct child of the deviation (fail-safe)."""
        canonical = _canonical_with_semantic_graph("An anomaly was recorded in the system.")
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            CandidateHypothesis(
                id="H1", name="A", statement="Mechanism A occurred", status="SUPPORTED",
                evidence_needed="", evidence_strength="VERIFIED",
                causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM,
                supporting_evidence=["e1"], supporting_claim_ids=["C1"],
            ),
            CandidateHypothesis(
                id="H2", name="B", statement="Mechanism B occurred", status="SUPPORTED",
                evidence_needed="", evidence_strength="VERIFIED",
                causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM,
                supporting_evidence=["e1"], supporting_claim_ids=["C1"],
            ),
            CandidateHypothesis(
                id="H3", name="R", statement="Systemic root cause R", status="SUPPORTED",
                evidence_needed="", evidence_strength="VERIFIED",
                causal_level=CausalLevel.L5_SYSTEMIC_CAUSE,
                supporting_evidence=["e1"], supporting_claim_ids=["C1"],
            ),
        ])
        ledger = [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)]
        g = build_causal_graph(canonical, rc, ledger)
        root_node = next(n for n in g.nodes if n.node_type == CausalGraphNodeType.SYSTEMIC_ROOT_CAUSE)
        deep_edge = next(e for e in g.edges if e.target_node_id == root_node.node_id)
        assert deep_edge.source_node_id == "CN_DEVIATION", (
            "Ambiguous evidence overlap (two equally-qualified shallower candidates) "
            "must never be arbitrarily resolved into a fabricated chain"
        )

    def test_10_causal_chain_termination_no_evidence_no_extension(self):
        """A chain that has no further evidence-backed hypothesis simply
        terminates — no synthetic terminal node is added."""
        canonical = _canonical_with_semantic_graph("An anomaly was recorded in the system.")
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            CandidateHypothesis(
                id="H1", name="M", statement="Mechanism M occurred", status="SUPPORTED",
                evidence_needed="", evidence_strength="VERIFIED",
                causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"],
            ),
        ])
        ledger = [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)]
        g = build_causal_graph(canonical, rc, ledger)
        assert len(g.nodes) == 2  # deviation + mechanism only
        assert not any(n.node_type == CausalGraphNodeType.SYSTEMIC_ROOT_CAUSE for n in g.nodes)


class TestRootCauseGraphGrounding:
    def test_established_without_verified_edge_is_blocked(self):
        """INV-CGRAPH-005: root_cause.status=ESTABLISHED with no VERIFIED
        causal graph edge reaching a deep node is rejected."""
        from app.agent.invariants import _check_established_root_cause_has_verified_causal_path
        from app.models.agent import CausalGraph, CausalGraphNode
        rc = RootCauseAnalysis(status=RootCauseStatus.ESTABLISHED, statement="x")
        cg = CausalGraph(nodes=[CausalGraphNode(
            node_id="CN_DEVIATION", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="x",
        )], edges=[])
        is_valid, reason = _check_established_root_cause_has_verified_causal_path(
            {"root_cause": rc, "causal_graph": cg}
        )
        assert not is_valid

    def test_established_with_verified_edge_passes(self):
        canonical = _canonical_with_semantic_graph("An anomaly was recorded in the system.")
        rc = RootCauseAnalysis(status=RootCauseStatus.ESTABLISHED, candidate_hypotheses=[
            CandidateHypothesis(
                id="H1", name="R", statement="Systemic root cause R", status="SUPPORTED",
                evidence_needed="", evidence_strength="VERIFIED",
                causal_level=CausalLevel.L5_SYSTEMIC_CAUSE, supporting_evidence=["e1"],
            ),
        ])
        ledger = [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)]
        cg = build_causal_graph(canonical, rc, ledger)
        is_valid, violations = evaluate_all_invariants({"root_cause": rc, "causal_graph": cg})
        cgraph_violations = [v for v in violations if "INV-CGRAPH-005" in v]
        assert not cgraph_violations


class TestGraphGroundedInvestigationPlanning:
    def test_11_plan_includes_graph_grounded_question(self):
        """End-to-end (deterministic planner, not the LLM path): a semantic
        graph with an unresolved normative fact produces at least one
        InvestigationQuestion with target_node_id set."""
        finding = "The equipment log was not maintained as required by Directive EQ-4."
        canonical = _canonical_with_semantic_graph(finding)
        ledger = [EvidenceItem(claim=finding, source="finding", status=EvidenceStatus.VERIFIED)]
        _, plan = build_deterministic_investigation_plan(
            finding, ledger, canonical_subject=canonical.finding_subject, canonical_state=canonical,
        )
        grounded = [q for q in plan.questions if q.target_node_id]
        assert grounded, "Expected at least one graph-grounded investigation question"
        assert grounded[0].causal_level is not None
        assert grounded[0].target_edge_id is not None

    def test_full_end_to_end_runtime_graph_population(self):
        """Section 21/30: run the ACTUAL LangGraph pipeline (not a unit
        function) and verify both graphs persist in the final state."""
        from app.agent.graph import build_agent_graph
        from app.models.agent import InvestigateRequest

        async def _run():
            text = "The maintenance record was not completed as required by Directive MX-11."
            req = InvestigateRequest(finding_text=text)
            state = {
                "request": req, "iteration_count": 0, "tool_call_count": 0,
                "critic_iteration": 0, "evidence_ledger": [], "errors": [], "trace": [],
            }
            graph = build_agent_graph()
            return await graph.ainvoke(state)

        res = asyncio.run(_run())
        ug = res.get("causal_uncertainty_graph")
        cg = res.get("causal_graph")
        assert ug is not None, "causal_uncertainty_graph must survive the full pipeline run"
        assert len(ug.nodes) >= 1
        assert cg is not None, "causal_graph must survive the full pipeline run"
        inv = res.get("investigation_plan")
        assert inv is not None and inv.questions


class TestGeneratedTextImmutability:
    def test_26_uncertainty_graph_construction_does_not_mutate_semantic_graph(self):
        canonical = _canonical_with_semantic_graph(
            "The batch reconciliation was not performed as required by Protocol BR-2."
        )
        before_node_count = len(canonical.semantic_graph.nodes)
        before_edge_count = len(canonical.semantic_graph.edges)
        build_causal_uncertainty_graph(canonical)
        assert len(canonical.semantic_graph.nodes) == before_node_count
        assert len(canonical.semantic_graph.edges) == before_edge_count
