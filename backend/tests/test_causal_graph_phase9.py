"""Phase 9: real CausalPath objects (Step 6) and competing-hypothesis
independence (Step 5) — a pure derivation over already-licensed CausalGraph
edges, never constructed from prose or fabricated to increase density.

Domain-neutral fixtures throughout.
"""
from __future__ import annotations

from app.agent.causal_graph import build_causal_graph, build_causal_paths
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
    RootCauseAnalysis,
    RootCauseStatus,
)


def _canonical(deviation: str = "An anomaly was recorded in the system.") -> CanonicalFindingState:
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


class TestCausalPathConstruction:
    def test_1_single_hop_path(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            _hyp(id="H1", statement="Mechanism M", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"]),
        ])
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        paths = build_causal_paths(g)
        assert len(paths) == 1
        assert paths[0].ordered_node_ids == ["CN_DEVIATION", "CN1"]
        assert paths[0].hypothesis_id == "H1"
        assert paths[0].epistemic_status == "VERIFIED"

    def test_4_explicit_chain_produces_multi_hop_path(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            _hyp(id="H1", statement="Mechanism M", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"]),
            _hyp(id="H2", statement="Root cause R", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L5_SYSTEMIC_CAUSE, supporting_evidence=["e1"],
                 deepens_hypothesis_id="H1"),
        ])
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        paths = build_causal_paths(g)
        h2_path = next(p for p in paths if p.hypothesis_id == "H2")
        assert h2_path.ordered_node_ids == ["CN_DEVIATION", "CN1", "CN2"]
        assert len(h2_path.ordered_edge_ids) == 2
        assert h2_path.starting_level == "L0_OBSERVATION"
        assert h2_path.terminal_level == "L5_SYSTEMIC_CAUSE"

    def test_7_independent_hypotheses_get_independent_paths(self):
        """H1 and H2 are unrelated — must never share a path or merge."""
        rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
            _hyp(id="H1", statement="Independent mechanism A", status="POSSIBLE", evidence_strength="INDICATIVE"),
            _hyp(id="H2", statement="Independent mechanism B", status="POSSIBLE", evidence_strength="INDICATIVE"),
        ])
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        paths = build_causal_paths(g)
        assert len(paths) == 2
        assert paths[0].path_id != paths[1].path_id
        assert set(paths[0].ordered_node_ids) & set(paths[1].ordered_node_ids) == {"CN_DEVIATION"}

    def test_unresolved_candidate_gets_no_path(self):
        """A dangling candidate (no licensed edge) correctly gets zero
        paths, not a fabricated one."""
        rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
            _hyp(id="H1", statement="Unresolved candidate", status="UNRESOLVED", evidence_strength="NONE"),
        ])
        g = build_causal_graph(_canonical(), rc, [])
        assert any(n.node_id == "CN1" for n in g.nodes)  # node exists
        assert not g.edges  # but no edge is licensed
        paths = build_causal_paths(g)
        assert paths == []

    def test_9_five_level_path(self):
        levels = [
            CausalLevel.L1_EVENT, CausalLevel.L2_IMMEDIATE_MECHANISM,
            CausalLevel.L3_CONTRIBUTING_CAUSE, CausalLevel.L4_ROOT_CAUSE,
            CausalLevel.L5_SYSTEMIC_CAUSE,
        ]
        hyps = []
        prev_id = None
        for i, level in enumerate(levels, start=1):
            hyps.append(_hyp(
                id=f"H{i}", statement=f"Cause at level {i}", status="SUPPORTED",
                evidence_strength="VERIFIED", causal_level=level, supporting_evidence=[f"e{i}"],
                deepens_hypothesis_id=prev_id,
            ))
            prev_id = f"H{i}"
        rc = RootCauseAnalysis(status=RootCauseStatus.ESTABLISHED, candidate_hypotheses=hyps)
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        paths = build_causal_paths(g)
        deepest_path = next(p for p in paths if p.hypothesis_id == "H5")
        assert len(deepest_path.ordered_node_ids) == 6  # deviation + 5 levels
        assert len(deepest_path.ordered_edge_ids) == 5

    def test_12_cycle_never_produces_a_path(self):
        """A hand-constructed cyclic graph (bypassing build_causal_graph's
        own cycle prevention) must never produce a path via build_causal_paths
        — the walk detects the revisit and aborts that path rather than
        looping or fabricating a truncated one."""
        nodes = [
            CausalGraphNode(node_id="CN_DEVIATION", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="dev"),
            CausalGraphNode(node_id="CN1", node_type=CausalGraphNodeType.IMMEDIATE_MECHANISM, label="A", causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM),
            CausalGraphNode(node_id="CN2", node_type=CausalGraphNodeType.CONTRIBUTING_FACTOR, label="B", causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE),
        ]
        edges = [
            CausalGraphEdge(edge_id="CE1", source_node_id="CN1", target_node_id="CN2", status=CausalGraphEdgeStatus.VERIFIED),
            CausalGraphEdge(edge_id="CE2", source_node_id="CN2", target_node_id="CN1", status=CausalGraphEdgeStatus.VERIFIED),
        ]
        g = CausalGraph(nodes=nodes, edges=edges)
        paths = build_causal_paths(g)
        assert paths == [], "A cyclic subgraph disconnected from the deviation must never yield a fabricated path"


class TestCausalPathStamping:
    def test_hypothesis_stamped_with_own_path_id(self):
        from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
        import asyncio
        from unittest.mock import patch
        from app.agent.nodes.understanding import understand_finding_node
        from app.models.agent import FiveWhyAnalysis, FiveWhyStep, InvestigateRequest

        async def _run():
            text = "The batch reconciliation was not performed as required by Protocol BR-9."
            state = {
                "request": InvestigateRequest(finding_text=text),
                "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0,
                "evidence_ledger": [], "errors": [], "trace": [],
            }
            with patch("app.agent.nodes.understanding.get_llm_client", return_value=None):
                state = await understand_finding_node(state)
            state["root_cause"] = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
                _hyp(id="H1", statement="Ledger totals were not cross-checked before month close",
                     status="SUPPORTED", evidence_strength="VERIFIED", causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM,
                     supporting_evidence=["e1"]),
            ])
            state["five_why"] = FiveWhyAnalysis(steps=[FiveWhyStep(question="Why?", answer="x", status="MIXED")])
            state["evidence_ledger"] = [
                EvidenceItem(claim=text, source="finding", status=EvidenceStatus.VERIFIED),
                EvidenceItem(claim="Ledger totals were not cross-checked before month close, per audit trail.",
                             source="record", status=EvidenceStatus.VERIFIED),
            ]
            state["capa_analysis"] = None
            state["impact_assessment"] = None
            state["investigation_plan"] = None
            state["ca_draft"] = None
            return await final_evidence_verification_node(state)

        result = asyncio.run(_run())
        rc = result["root_cause"]
        assert rc.candidate_hypotheses[0].causal_path_id is not None
        paths = result["causal_paths"]
        assert any(p.path_id == rc.candidate_hypotheses[0].causal_path_id for p in paths)
