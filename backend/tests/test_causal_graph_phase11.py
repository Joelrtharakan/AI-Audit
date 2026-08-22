"""Phase 11: epistemically correct rendering of graph-derived causal
transitions — the fix that closed the Phase 10 regression (44 test
failures against INV-5WHY-CAUSAL-001/002).

Domain-neutral fixtures; `render_causal_transition_answer` and
`_humanize_concept_name` are purely mechanical (status-keyed templates,
case/underscore transforms) — no domain vocabulary anywhere.
"""
from __future__ import annotations

from app.agent.causal_graph import build_causal_graph, _humanize_concept_name
from app.agent.causal_graph_traversal import (
    build_graph_grounded_five_why,
    render_causal_transition_answer,
)
from app.agent.causal_guard import (
    answer_selects_unverified_hypothesis,
    five_why_answer_contains_unverified_modal_causation,
)
from app.models.agent import (
    CandidateHypothesis,
    CanonicalFindingState,
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


class TestConceptRefHumanization:
    def test_underscored_name_humanized(self):
        assert _humanize_concept_name("TRAINING_NOT_COMPLETED") == "Training Not Completed"

    def test_none_name_returns_none(self):
        assert _humanize_concept_name(None) is None

    def test_hypothesis_node_carries_concept_ref(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            CandidateHypothesis(id="H1", name="MECHANISM_A_FAILURE", statement="Mechanism A may have failed",
                                 status="SUPPORTED", evidence_needed="", evidence_strength="VERIFIED",
                                 causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"]),
        ])
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        node = next(n for n in g.nodes if n.node_id == "CN1")
        assert node.concept_ref == "Mechanism A Failure"


class TestEpistemicallyCorrectRendering:
    def _nodes(self, target_label: str, concept_ref: str | None = "Neutral Concept"):
        src = CausalGraphNode(node_id="A", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="the deviation")
        tgt = CausalGraphNode(node_id="B", node_type=CausalGraphNodeType.CONTRIBUTING_FACTOR,
                               label=target_label, concept_ref=concept_ref)
        return src, tgt

    def test_2_verified_renders_target_label_directly(self):
        src, tgt = self._nodes("A verified mechanism explains the deviation")
        ans = render_causal_transition_answer(src, tgt, CausalGraphEdgeStatus.VERIFIED)
        assert ans == tgt.label

    def test_4_reported_uses_concept_ref_not_raw_statement(self):
        src, tgt = self._nodes("Someone reported that X may have caused Y", concept_ref="Reported Factor")
        ans = render_causal_transition_answer(src, tgt, CausalGraphEdgeStatus.REPORTED)
        assert "Reported Factor" in ans
        assert "may have caused" not in ans

    def test_3_requires_evidence_uses_neutral_boundary_phrasing(self):
        src, tgt = self._nodes("A candidate mechanism possibly contributed, may have occurred", concept_ref="Candidate Mechanism")
        ans = render_causal_transition_answer(src, tgt, CausalGraphEdgeStatus.POSSIBLE)
        assert "does not establish whether" in ans
        assert "possibly contributed" not in ans
        assert "may have occurred" not in ans

    def test_5_unknown_uses_neutral_boundary_phrasing(self):
        src, tgt = self._nodes("An unverified mechanism", concept_ref="Unverified Mechanism")
        ans = render_causal_transition_answer(src, tgt, CausalGraphEdgeStatus.UNKNOWN)
        assert "does not establish whether" in ans

    def test_7_possible_never_renders_as_established_fact(self):
        """The core Phase 10->11 regression check, run directly against
        the invariant function rather than the full pipeline."""
        hyp = CandidateHypothesis(id="H1", name="WEAK_HYPOTHESIS", statement="This may possibly have happened",
                                   status="POSSIBLE", evidence_needed="", evidence_strength="INDICATIVE")
        src, tgt = self._nodes(hyp.statement, concept_ref=_humanize_concept_name(hyp.name))
        ans = render_causal_transition_answer(src, tgt, CausalGraphEdgeStatus.POSSIBLE)
        assert answer_selects_unverified_hypothesis(ans, [hyp], status="REQUIRES_EVIDENCE") is None
        assert five_why_answer_contains_unverified_modal_causation(ans, "REQUIRES_EVIDENCE", has_verified_mechanism=False) is False

    def test_8_requires_evidence_never_renders_as_established_fact(self):
        hyp = CandidateHypothesis(id="H1", name="UNCERTAIN_CAUSE", statement="The unconfirmed cause likely explains this",
                                   status="UNRESOLVED", evidence_needed="", evidence_strength="NONE")
        src, tgt = self._nodes(hyp.statement, concept_ref=_humanize_concept_name(hyp.name))
        ans = render_causal_transition_answer(src, tgt, CausalGraphEdgeStatus.UNKNOWN)
        assert answer_selects_unverified_hypothesis(ans, [hyp], status="UNKNOWN") is None

    def test_13_does_not_copy_unsupported_hypothesis_wording(self):
        hyp_statement = "The operator possibly mishandled the equipment, which may have caused the deviation"
        src, tgt = self._nodes(hyp_statement, concept_ref="Equipment Mishandling")
        ans = render_causal_transition_answer(src, tgt, CausalGraphEdgeStatus.POSSIBLE)
        assert hyp_statement not in ans


class TestBoundaryStepEmission:
    def test_9_verified_mechanism_remains_verified_with_unknown_deeper_cause(self):
        """A single VERIFIED mechanism with no deeper chain must keep its
        VERIFIED transition step and get an explicit EVIDENCE_BOUNDARY
        marker after it — never silently drop the verified fact, never
        fabricate a deeper cause."""
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            CandidateHypothesis(id="H1", name="MECH", statement="Mechanism M", status="SUPPORTED",
                                 evidence_needed="", evidence_strength="VERIFIED",
                                 causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"]),
        ])
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        fw = build_graph_grounded_five_why(g)
        transition = [s for s in fw.steps if s.boundary_status == "TRANSITION"]
        boundary = [s for s in fw.steps if s.boundary_status == "EVIDENCE_BOUNDARY"]
        assert transition[0].status == "VERIFIED"
        assert transition[0].answer == "Mechanism M"
        assert len(boundary) == 1
        assert boundary[0].status == "UNKNOWN"
        assert boundary[0].causal_edge_id is None
        assert boundary[0].target_node_id is None

    def test_11_graph_derived_step_has_complete_ids(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            CandidateHypothesis(id="H1", name="MECH", statement="Mechanism M", status="SUPPORTED",
                                 evidence_needed="", evidence_strength="VERIFIED",
                                 causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"]),
        ])
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        fw = build_graph_grounded_five_why(g)
        step = next(s for s in fw.steps if s.boundary_status == "TRANSITION")
        assert step.source_node_id and step.target_node_id and step.causal_edge_id
        assert step.causal_level is not None
