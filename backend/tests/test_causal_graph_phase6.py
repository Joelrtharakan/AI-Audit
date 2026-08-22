"""Phase 6: hypothesis graph-nativeness, explicit edge derivation, and pure
structural RCA/Impact projections from CausalGraph.

Domain-neutral fixtures; production logic keys only on CausalGraphNodeType /
CausalGraphEdgeStatus / EvidenceStatus / graph topology.
"""
from __future__ import annotations

import asyncio

import pytest

from app.agent.causal_graph import (
    build_causal_graph,
    build_impact_from_graph,
    build_rca_from_causal_graph,
)
from app.models.agent import (
    CandidateHypothesis,
    CanonicalFindingState,
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


class TestHypothesisGraphNativeness:
    def test_a_licensed_hypothesis_stamped_with_graph_ids(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            _hyp(id="H1", statement="Mechanism M", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"]),
        ])
        build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        h = rc.candidate_hypotheses[0]
        assert h.causal_node_id is not None
        assert h.causal_edge_id is not None
        assert h.causal_edge_source_node_id == "CN_DEVIATION"

    def test_refuted_hypothesis_never_stamped(self):
        """A REFUTED, zero-evidence hypothesis is not licensed to exist as a
        graph node at all — its graph-native fields must stay unset, not be
        fabricated to satisfy the schema."""
        rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
            _hyp(id="H1", statement="Refuted speculation", status="REFUTED", evidence_strength="NONE"),
        ])
        build_causal_graph(_canonical(), rc, [])
        h = rc.candidate_hypotheses[0]
        assert h.causal_node_id is None
        assert h.causal_edge_id is None

    def test_d_competing_hypotheses_both_stamped_independently(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
            _hyp(id="H1", statement="Hypothesis A", status="POSSIBLE", evidence_strength="INDICATIVE"),
            _hyp(id="H2", statement="Hypothesis B", status="POSSIBLE", evidence_strength="INDICATIVE"),
        ])
        build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        h1, h2 = rc.candidate_hypotheses
        assert h1.causal_node_id != h2.causal_node_id
        assert h1.causal_edge_id != h2.causal_edge_id


class TestExplicitEdgeDerivation:
    def test_direct_edge_marked_direct(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            _hyp(id="H1", statement="Mechanism M", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"]),
        ])
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        assert g.edges[0].derivation == "DIRECT"

    def test_i_multihop_edge_marked_evidence_correlated(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            _hyp(id="H1", statement="Mechanism M", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"], supporting_claim_ids=["C1"]),
            _hyp(id="H2", statement="Root cause R", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L5_SYSTEMIC_CAUSE, supporting_evidence=["e1"], supporting_claim_ids=["C1"]),
        ])
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        chained_edge = next(e for e in g.edges if e.source_node_id != "CN_DEVIATION")
        assert chained_edge.derivation == "EVIDENCE_CORRELATED"


class TestRCAProjection:
    def test_c_verified_mechanism_and_verified_root(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.ESTABLISHED, candidate_hypotheses=[
            _hyp(id="H1", statement="Mechanism M", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"], supporting_claim_ids=["C1"]),
            _hyp(id="H2", statement="Root cause R", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L5_SYSTEMIC_CAUSE, supporting_evidence=["e1"], supporting_claim_ids=["C1"]),
        ])
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        rca = build_rca_from_causal_graph(g)
        assert rca.root_cause_status == "ESTABLISHED"
        assert rca.systemic_root_causes and rca.systemic_root_causes[0].label == "Root cause R"
        assert rca.immediate_mechanisms and rca.immediate_mechanisms[0].label == "Mechanism M"

    def test_b_verified_mechanism_unknown_root(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            _hyp(id="H1", statement="Mechanism M", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"]),
        ])
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        rca = build_rca_from_causal_graph(g)
        assert rca.root_cause_status == "NOT_ESTABLISHED"
        assert not rca.systemic_root_causes
        assert not rca.underlying_causes

    def test_a_unknown_root_cause_no_hypotheses(self):
        """No candidate hypotheses at all -> RCA has only the deviation, no
        fabricated cause of any kind."""
        rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[])
        g = build_causal_graph(_canonical(), rc, [])
        rca = build_rca_from_causal_graph(g)
        assert rca.observed_deviation is not None
        assert not rca.immediate_mechanisms
        assert not rca.contributing_factors
        assert not rca.underlying_causes
        assert not rca.systemic_root_causes
        assert rca.root_cause_status == "NOT_ESTABLISHED"

    def test_d_competing_hypotheses_surfaced_not_collapsed(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
            _hyp(id="H1", statement="Hypothesis A", status="POSSIBLE", evidence_strength="INDICATIVE",
                 causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE),
            _hyp(id="H2", statement="Hypothesis B", status="POSSIBLE", evidence_strength="INDICATIVE",
                 causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE),
        ])
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        rca = build_rca_from_causal_graph(g)
        assert len(rca.competing_hypotheses) == 2, "Both tied candidates must be surfaced, never collapsed to one"

    def test_ah_incomplete_causal_chain_reports_evidence_boundary(self):
        """A hypothesis with status="UNRESOLVED" (neither SUPPORTED nor
        POSSIBLE, so no edge is licensed per `_edge_status_for`, but also
        neither REFUTED nor zero-evidence-UNVERIFIED, so the node itself
        still exists) produces a dangling candidate node — a genuine
        evidence boundary, not an omission."""
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            _hyp(id="H1", statement="Mechanism M", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"]),
            _hyp(id="H2", statement="Unresolved deeper candidate", status="UNRESOLVED", evidence_strength="NONE"),
        ])
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        assert any(n.node_id == "CN2" for n in g.nodes), "Sanity: the dangling node must exist"
        assert not any(e.target_node_id == "CN2" for e in g.edges), "Sanity: no edge must be licensed for it"
        rca = build_rca_from_causal_graph(g)
        assert rca.evidence_boundary_reached is True


class TestImpactProjection:
    def test_v_observed_impact_is_the_verified_deviation(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[])
        g = build_causal_graph(_canonical(), rc, [])
        impact = build_impact_from_graph(_canonical(), g)
        assert len(impact.observed) == 1
        assert impact.observed[0].basis == "OBSERVED"

    def test_w_potential_impact_from_unconfirmed_mechanism(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
            _hyp(id="H1", statement="An unconfirmed downstream effect", status="POSSIBLE", evidence_strength="REPORTED"),
        ])
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        impact = build_impact_from_graph(_canonical(), g)
        assert impact.potential, "A REPORTED-strength node must surface as potential impact"
        assert impact.potential[0].basis == "POTENTIAL"

    def test_impact_never_fabricated_from_severity_wording(self):
        """A finding whose text sounds severe but has zero hypotheses/edges
        must not produce any potential/unknown impact entries — only the
        observed deviation itself."""
        canonical = _canonical("A catastrophic total system failure occurred.")
        rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[])
        g = build_causal_graph(canonical, rc, [])
        impact = build_impact_from_graph(canonical, g)
        assert len(impact.observed) == 1
        assert not impact.potential
        assert not impact.unknown_investigation_required


class TestExplicitSequentialChainParsing:
    """Phase 7 Section 5: exercises the ACTUAL core_synthesis_node parsing
    path (not just build_causal_graph in isolation) with a mocked LLM
    response, proving deepens_hypothesis_id survives real JSON parsing and
    validation, and that a dangling reference is dropped."""

    @pytest.mark.asyncio
    async def test_valid_deepens_reference_survives_parsing(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from app.agent.nodes.core_synthesis import core_synthesis_node
        from app.agent.nodes.understanding import understand_finding_node
        from app.models.agent import InvestigateRequest

        text = "The calibration record was not signed as required by Directive CAL-3."
        state = {
            "request": InvestigateRequest(finding_text=text),
            "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0,
            "evidence_ledger": [], "errors": [], "trace": [],
        }
        with patch("app.agent.nodes.understanding.get_llm_client", return_value=None):
            state = await understand_finding_node(state)

        raw_json = (
            '{"root_cause": {"status": "NOT_ESTABLISHED", "category": "TO_BE_CONFIRMED", '
            '"statement": null, "root_cause_basis": "pending", "evidence_required": [], '
            '"candidate_hypotheses": ['
            '{"id": "H1", "name": "MECH", "statement": "The signature step was skipped", '
            '"supporting_claim_ids": ["C1"], "contradicting_claim_ids": [], "status": "POSSIBLE", '
            '"evidence_needed": "record", "confirms_if": "x", "refutes_if": "y"}, '
            '{"id": "H2", "name": "DEEPER", "statement": "The sign-off step was never assigned to a role", '
            '"supporting_claim_ids": ["C1"], "contradicting_claim_ids": [], "status": "POSSIBLE", '
            '"evidence_needed": "record", "confirms_if": "x", "refutes_if": "y", '
            '"deepens_hypothesis_id": "H1"}, '
            '{"id": "H3", "name": "DANGLING", "statement": "An unrelated speculative cause", '
            '"supporting_claim_ids": ["C1"], "contradicting_claim_ids": [], "status": "POSSIBLE", '
            '"evidence_needed": "record", "confirms_if": "x", "refutes_if": "y", '
            '"deepens_hypothesis_id": "H99"}'
            '], "narrative": "pending"}, '
            '"five_why": {"steps": [], "is_complete": false, "status_note": "pending"}, '
            '"contributing_factors": []}'
        )
        mock_client = MagicMock()
        mock_client.chat_completion = AsyncMock(return_value=raw_json)
        with patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=mock_client):
            result = await core_synthesis_node(state)

        rc = result["root_cause"]
        by_id = {h.id: h for h in rc.candidate_hypotheses}
        assert "H2" in by_id, f"Expected H2 among parsed hypotheses: {list(by_id)}"
        assert by_id["H2"].deepens_hypothesis_id == "H1", (
            "A valid deepens_hypothesis_id referencing a real id in the same response must survive parsing"
        )
        if "H3" in by_id:
            assert by_id["H3"].deepens_hypothesis_id is None, (
                "A dangling deepens_hypothesis_id (H99 does not exist) must be dropped, not trusted"
            )


class TestFullPipelineProjections:
    def test_full_end_to_end_rca_and_impact_projections_populate(self):
        from app.agent.graph import build_agent_graph
        from app.models.agent import InvestigateRequest

        async def _run():
            text = "The calibration certificate was not renewed as required by Directive CAL-8."
            req = InvestigateRequest(finding_text=text)
            state = {
                "request": req, "iteration_count": 0, "tool_call_count": 0,
                "critic_iteration": 0, "evidence_ledger": [], "errors": [], "trace": [],
            }
            graph = build_agent_graph()
            return await graph.ainvoke(state)

        res = asyncio.run(_run())
        report = res.get("report")
        assert report is not None
        assert report.rca_projection is not None
        assert report.rca_projection.observed_deviation is not None
        assert report.impact_graph_projection is not None
        assert report.impact_graph_projection.observed
