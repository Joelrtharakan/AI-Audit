"""Phase 12: canonical measurement/comparison projection onto the causal
graph, so graph-derived 5-Why rendering no longer loses quantitative
precision for comparison-type findings.

All fixtures are synthetic and domain-neutral (generic labels like
"quantity A" / "reference value", generic comparison subtypes) — no
finding/industry-specific vocabulary anywhere in this file. Property/
invariant tests use randomized synthetic inputs, not fixed fixtures.
"""
from __future__ import annotations

import random

import pytest

from app.agent.causal_graph import build_causal_graph
from app.agent.causal_graph_traversal import (
    build_graph_grounded_five_why,
    render_causal_transition_answer,
    _graph_node_why_question,
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
    SemanticMeasurement,
)


def _canonical(**kwargs) -> CanonicalFindingState:
    base = dict(
        raw_finding="A discrepancy was recorded.", finding_subject="subject",
        affected_object="subject", affected_process="UNKNOWN",
        affected_activity="UNKNOWN", deviation="A discrepancy was recorded.",
        observed_deviation="A discrepancy was recorded.",
        facts=["A discrepancy was recorded."],
    )
    base.update(kwargs)
    return CanonicalFindingState(**base)


def _hyp(status: str = "POSSIBLE", evidence_strength: str = "NONE") -> CandidateHypothesis:
    return CandidateHypothesis(
        id="H1", name="CANDIDATE_MECHANISM_ONE", statement="A candidate mechanism possibly explains this",
        status=status, evidence_needed="", evidence_strength=evidence_strength,
        causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM,
    )


def _rc(hyp: CandidateHypothesis | None = None) -> RootCauseAnalysis:
    return RootCauseAnalysis(
        status=RootCauseStatus.NOT_ESTABLISHED,
        candidate_hypotheses=[hyp] if hyp else [],
    )


def _ledger() -> list[EvidenceItem]:
    return [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)]


# ---------------------------------------------------------------------------
# Group 1: canonical -> graph node projection (does the data actually flow?)
# ---------------------------------------------------------------------------

class TestMeasurementProjectionOntoGraph:
    def test_comparison_fields_flow_onto_deviation_node(self):
        canonical = _canonical(
            comparison_type="MISMATCH", comparison_left="quantity A",
            comparison_left_qualifier="recorded", comparison_right="reference value",
            comparison_subtype="GENERIC_SUBTYPE",
        )
        g = build_causal_graph(canonical, _rc(_hyp()), _ledger())
        deviation = next(n for n in g.nodes if n.node_type == CausalGraphNodeType.OBSERVED_DEVIATION)
        assert deviation.comparison_type == "MISMATCH"
        assert deviation.comparison_left == "quantity A"
        assert deviation.comparison_left_qualifier == "recorded"
        assert deviation.comparison_right == "reference value"

    def test_measurement_value_flows_onto_deviation_node(self):
        canonical = _canonical(
            comparison_type="MISMATCH", comparison_left="quantity A", comparison_right="reference value",
            measurement=SemanticMeasurement(value=7.5, unit="units", qualifier="approximately", evidence_status="VERIFIED"),
        )
        g = build_causal_graph(canonical, _rc(_hyp()), _ledger())
        deviation = next(n for n in g.nodes if n.node_type == CausalGraphNodeType.OBSERVED_DEVIATION)
        assert deviation.measurement_value == 7.5
        assert deviation.measurement_unit == "units"
        assert deviation.measurement_qualifier == "approximately"
        assert deviation.measurement_evidence_status == "VERIFIED"

    def test_no_comparison_data_leaves_fields_none(self):
        canonical = _canonical()
        g = build_causal_graph(canonical, _rc(_hyp()), _ledger())
        deviation = next(n for n in g.nodes if n.node_type == CausalGraphNodeType.OBSERVED_DEVIATION)
        assert deviation.comparison_type is None
        assert deviation.measurement_value is None

    def test_measurement_without_comparison_type_does_not_fabricate_comparison(self):
        canonical = _canonical(measurement=SemanticMeasurement(value=3.0, unit=None, evidence_status="VERIFIED"))
        g = build_causal_graph(canonical, _rc(_hyp()), _ledger())
        deviation = next(n for n in g.nodes if n.node_type == CausalGraphNodeType.OBSERVED_DEVIATION)
        assert deviation.comparison_type is None
        assert deviation.measurement_value == 3.0

    def test_hypothesis_nodes_do_not_carry_deviation_measurement(self):
        """Only the OBSERVED_DEVIATION node is genuinely supported by the
        finding's own comparison data — a hypothesis node must not silently
        inherit it."""
        canonical = _canonical(
            comparison_type="MISMATCH", comparison_left="quantity A", comparison_right="reference value",
            measurement=SemanticMeasurement(value=7.5, unit="units", evidence_status="VERIFIED"),
        )
        g = build_causal_graph(canonical, _rc(_hyp()), _ledger())
        hyp_node = next((n for n in g.nodes if n.node_type != CausalGraphNodeType.OBSERVED_DEVIATION), None)
        if hyp_node is not None:
            assert hyp_node.measurement_value is None
            assert hyp_node.comparison_type is None


# ---------------------------------------------------------------------------
# Group 2: rendering — precision preserved, domain-neutral, never upgrades
# epistemic certainty
# ---------------------------------------------------------------------------

class TestComparisonAwareRendering:
    def _nodes_with_comparison(self, edge_status_source_label="A discrepancy was recorded."):
        src = CausalGraphNode(
            node_id="A", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label=edge_status_source_label,
            comparison_type="MISMATCH", comparison_left="quantity A", comparison_left_qualifier="recorded",
            comparison_right="reference value", comparison_subtype="GENERIC_SUBTYPE",
            measurement_value=9.9, measurement_unit="units", measurement_qualifier="approximately",
        )
        tgt = CausalGraphNode(node_id="B", node_type=CausalGraphNodeType.CONTRIBUTING_FACTOR,
                               label="A candidate mechanism possibly explains this", concept_ref="Candidate Mechanism One")
        return src, tgt

    def test_measurement_value_appears_in_rendered_answer(self):
        src, tgt = self._nodes_with_comparison()
        ans = render_causal_transition_answer(src, tgt, CausalGraphEdgeStatus.POSSIBLE)
        assert "9.9" in ans
        assert "quantity A" in ans
        assert "reference value" in ans

    def test_measurement_precision_does_not_upgrade_possible_to_established(self):
        src, tgt = self._nodes_with_comparison()
        ans = render_causal_transition_answer(src, tgt, CausalGraphEdgeStatus.POSSIBLE)
        assert "does not establish whether" in ans

    def test_measurement_precision_does_not_leak_hypothesis_modal_language(self):
        src, tgt = self._nodes_with_comparison()
        ans = render_causal_transition_answer(src, tgt, CausalGraphEdgeStatus.POSSIBLE)
        assert "possibly explains" not in ans

    def test_verified_edge_with_comparison_context_still_renders_target_label_directly(self):
        """VERIFIED rendering must not be altered by the presence of
        comparison data on the source node — the epistemic gate is
        unaffected by whether a measurement exists."""
        src, tgt = self._nodes_with_comparison()
        ans = render_causal_transition_answer(src, tgt, CausalGraphEdgeStatus.VERIFIED)
        assert ans == tgt.label

    def test_no_domain_specific_template_branch_for_percentage_or_calibration(self):
        """Renderer output for two different generic subtypes must come
        from the same shared code path (no special-cased vocabulary
        injected per subtype beyond the pre-existing mechanism-category
        table)."""
        src1, tgt = self._nodes_with_comparison()
        src1.comparison_subtype = "SUBTYPE_ONE"
        ans1 = render_causal_transition_answer(src1, tgt, CausalGraphEdgeStatus.POSSIBLE)
        src2, _ = self._nodes_with_comparison()
        src2.comparison_subtype = "SUBTYPE_TWO_NEVER_SEEN"
        ans2 = render_causal_transition_answer(src2, tgt, CausalGraphEdgeStatus.POSSIBLE)
        # Both must still produce the safe boundary phrasing; an unknown
        # subtype falls back to the shared default vocabulary rather than
        # erroring or inventing subtype-specific text.
        assert "does not establish whether" in ans1
        assert "does not establish whether" in ans2

    def test_answer_never_reveals_raw_hypothesis_statement(self):
        src, tgt = self._nodes_with_comparison()
        tgt.label = "The operator possibly mishandled something, which may have caused this"
        ans = render_causal_transition_answer(src, tgt, CausalGraphEdgeStatus.POSSIBLE)
        assert tgt.label not in ans


# ---------------------------------------------------------------------------
# Group 3: question construction — domain-general, comparison-type-keyed
# ---------------------------------------------------------------------------

class TestComparisonAwareQuestion:
    def test_comparison_question_uses_structured_fields(self):
        node = CausalGraphNode(
            node_id="A", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="A discrepancy was recorded.",
            comparison_type="MISMATCH", comparison_left="quantity A", comparison_left_qualifier="recorded",
            comparison_right="reference value",
        )
        q = _graph_node_why_question(node)
        assert q == "Why did the recorded quantity A differ from the reference value?"

    def test_no_comparison_data_falls_back_to_generic_question(self):
        node = CausalGraphNode(node_id="A", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="X occurred")
        q = _graph_node_why_question(node)
        assert q == "Why did the following occur: X occurred?"

    def test_unrecognized_comparison_type_falls_back_to_generic_question(self):
        node = CausalGraphNode(
            node_id="A", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="X occurred",
            comparison_type="NEVER_SEEN_TYPE", comparison_left="quantity A", comparison_right="reference value",
        )
        q = _graph_node_why_question(node)
        assert q == "Why did the following occur: X occurred?"

    @pytest.mark.parametrize("comparison_type", [
        "MISMATCH", "EXCEEDED", "BELOW", "INCONSISTENT", "RECONCILIATION_FAILURE", "MISSING", "DUPLICATE",
    ])
    def test_every_known_comparison_type_produces_a_question(self, comparison_type):
        node = CausalGraphNode(
            node_id="A", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="X occurred",
            comparison_type=comparison_type, comparison_left="quantity A", comparison_right="reference value",
        )
        q = _graph_node_why_question(node)
        assert q.startswith("Why ")
        assert q.endswith("?")
        assert "quantity A" in q
        assert "reference value" in q


# ---------------------------------------------------------------------------
# Group 4: end-to-end graph-grounded 5-Why with comparison context
# ---------------------------------------------------------------------------

class TestGraphGroundedFiveWhyWithComparison:
    def _build(self, status="POSSIBLE", evidence_strength="NONE"):
        canonical = _canonical(
            comparison_type="MISMATCH", comparison_left="quantity A", comparison_left_qualifier="recorded",
            comparison_right="reference value", comparison_subtype="GENERIC_SUBTYPE",
            measurement=SemanticMeasurement(value=4.2, unit="%", qualifier="approximately", evidence_status="VERIFIED"),
        )
        rc = _rc(_hyp(status=status, evidence_strength=evidence_strength))
        g = build_causal_graph(canonical, rc, _ledger())
        return build_graph_grounded_five_why(g)

    def test_possible_edge_status_is_unknown_not_established(self):
        fw = self._build()
        step = fw.steps[0]
        assert step.status == "UNKNOWN"

    def test_possible_edge_never_satisfies_unverified_hypothesis_guard(self):
        fw = self._build()
        step = fw.steps[0]
        hyp = _hyp()
        assert answer_selects_unverified_hypothesis(step.answer, [hyp], status=step.status) is None
        assert five_why_answer_contains_unverified_modal_causation(step.answer, step.status, has_verified_mechanism=False) is False

    def test_measurement_content_present_in_first_step(self):
        fw = self._build()
        step = fw.steps[0]
        assert "4.2" in step.answer
        assert "quantity A" in step.answer
        assert "reference value" in step.answer

    def test_question_uses_comparison_phrasing(self):
        fw = self._build()
        step = fw.steps[0]
        assert step.question == "Why did the recorded quantity A differ from the reference value?"

    def test_zero_hypotheses_still_returns_none_no_fabricated_steps(self):
        """A finding with comparison data but literally no hypotheses must
        not fabricate a causal transition just because measurement data
        exists — measurement is descriptive, not a substitute for a causal
        edge."""
        canonical = _canonical(
            comparison_type="MISMATCH", comparison_left="quantity A", comparison_right="reference value",
        )
        g = build_causal_graph(canonical, _rc(None), _ledger())
        fw = build_graph_grounded_five_why(g)
        assert fw is None


# ---------------------------------------------------------------------------
# Group 5: property / invariant tests over randomized synthetic inputs
# ---------------------------------------------------------------------------

_SYNTHETIC_LEFT_LABELS = ["quantity A", "value X", "recorded amount", "measured total", "logged figure"]
_SYNTHETIC_RIGHT_LABELS = ["reference value", "expected total", "baseline figure", "target amount"]
_SYNTHETIC_SUBTYPES = ["SUBTYPE_ONE", "SUBTYPE_TWO", "SUBTYPE_UNKNOWN", None]
_ALL_STATUSES = [CausalGraphEdgeStatus.VERIFIED, CausalGraphEdgeStatus.REPORTED,
                  CausalGraphEdgeStatus.POSSIBLE, CausalGraphEdgeStatus.UNKNOWN]


class TestRandomizedInvariants:
    @pytest.mark.parametrize("seed", range(20))
    def test_non_verified_rendering_never_leaks_target_modal_language(self, seed):
        rnd = random.Random(seed)
        left = rnd.choice(_SYNTHETIC_LEFT_LABELS)
        right = rnd.choice(_SYNTHETIC_RIGHT_LABELS)
        value = rnd.uniform(0.1, 99.9)
        modal_statement = f"Mechanism {seed} possibly caused this, which may have resulted in the deviation"
        src = CausalGraphNode(
            node_id="A", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="A discrepancy was recorded.",
            comparison_type="MISMATCH", comparison_left=left, comparison_right=right,
            comparison_subtype=rnd.choice(_SYNTHETIC_SUBTYPES), measurement_value=round(value, 2), measurement_unit="units",
        )
        tgt = CausalGraphNode(node_id="B", node_type=CausalGraphNodeType.CONTRIBUTING_FACTOR,
                               label=modal_statement, concept_ref=f"Mechanism {seed}")
        status = rnd.choice([CausalGraphEdgeStatus.POSSIBLE, CausalGraphEdgeStatus.UNKNOWN, CausalGraphEdgeStatus.REPORTED])
        ans = render_causal_transition_answer(src, tgt, status)
        if status != CausalGraphEdgeStatus.REPORTED:
            assert "possibly caused" not in ans
            assert "may have resulted" not in ans
        assert modal_statement not in ans

    @pytest.mark.parametrize("seed", range(10))
    def test_measurement_value_always_present_when_supplied(self, seed):
        rnd = random.Random(seed)
        value = round(rnd.uniform(0.1, 999.9), 2)
        src = CausalGraphNode(
            node_id="A", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="A discrepancy was recorded.",
            comparison_type="MISMATCH", comparison_left="quantity A", comparison_right="reference value",
            measurement_value=value, measurement_unit="units",
        )
        tgt = CausalGraphNode(node_id="B", node_type=CausalGraphNodeType.CONTRIBUTING_FACTOR, label="candidate", concept_ref="Candidate")
        ans = render_causal_transition_answer(src, tgt, CausalGraphEdgeStatus.POSSIBLE)
        assert str(value) in ans

    def test_verified_status_always_bypasses_comparison_branch_unchanged(self):
        """Regardless of what comparison data is attached, a VERIFIED edge
        renders the target label directly — comparison context must never
        change VERIFIED-branch behavior."""
        for _ in range(10):
            src = CausalGraphNode(
                node_id="A", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="dev",
                comparison_type="MISMATCH", comparison_left="quantity A", comparison_right="reference value",
                measurement_value=1.0,
            )
            tgt = CausalGraphNode(node_id="B", node_type=CausalGraphNodeType.CONTRIBUTING_FACTOR, label="A verified fact")
            ans = render_causal_transition_answer(src, tgt, CausalGraphEdgeStatus.VERIFIED)
            assert ans == tgt.label


# ---------------------------------------------------------------------------
# Group 6: no accidental fabrication of a causal edge from measurement alone
# ---------------------------------------------------------------------------

class TestMeasurementDoesNotSubstituteForCausation:
    def test_measurement_alone_never_produces_verified_step_status(self):
        canonical = _canonical(
            comparison_type="MISMATCH", comparison_left="quantity A", comparison_right="reference value",
            measurement=SemanticMeasurement(value=50.0, unit="%", evidence_status="VERIFIED"),
        )
        g = build_causal_graph(canonical, _rc(_hyp(status="POSSIBLE", evidence_strength="NONE")), _ledger())
        fw = build_graph_grounded_five_why(g)
        assert fw is not None
        assert all(s.status != "VERIFIED" for s in fw.steps if s.step_id == "GW1")

    def test_high_evidence_status_measurement_does_not_override_possible_hypothesis_status(self):
        canonical = _canonical(
            comparison_type="MISMATCH", comparison_left="quantity A", comparison_right="reference value",
            measurement=SemanticMeasurement(value=0.01, unit="%", evidence_status="VERIFIED"),
        )
        g = build_causal_graph(canonical, _rc(_hyp(status="POSSIBLE", evidence_strength="NONE")), _ledger())
        fw = build_graph_grounded_five_why(g)
        assert fw.steps[0].status == "UNKNOWN"
