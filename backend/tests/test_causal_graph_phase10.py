"""Phase 10: guard-rail test for a real regression discovered and reverted
this phase, plus verification of the Step 1 runtime audit finding that
capa.py/impact.py are dead legacy code — the live CAPA/Impact path
(core_synthesis.py's `_derive_deterministic_impact` /
`build_conditional_capa_actions`) is already structurally-derived, not
LLM-narrative-parsed.
"""
from __future__ import annotations

from app.agent.causal_graph import build_causal_graph
from app.agent.causal_guard import answer_selects_unverified_hypothesis
from app.agent.causal_graph_traversal import build_graph_grounded_five_why
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


class TestGraphFiveWhyPossibleStatusGuard:
    """Phase 10 attempted to widen the 5-Why graph-authority swap to fire on
    ANY licensed edge (not just multi-hop chains), which caused 44 real
    regression failures — a POSSIBLE-status graph edge's rendered answer
    (the raw hypothesis statement text, unhedged, embedding the
    hypothesis's own modal language) tripped INV-5WHY-CAUSAL-001/002.

    Phase 11 fixed the actual root cause (not the invariant): added
    `CausalGraphNode.concept_ref` — a short, controlled-vocabulary
    identifier mechanically derived from the hypothesis's `name`, never
    from its prose `statement` — and rewrote `render_causal_transition_
    answer()` to render non-VERIFIED edges from `concept_ref` inside a
    domain-neutral evidence-boundary template, instead of copying the raw
    statement. This test now asserts the CORRECT (fixed) behavior; the
    assertion direction changed from Phase 10 deliberately, and is
    explained here rather than silently flipped."""

    def test_possible_status_step_answer_no_longer_trips_the_unverified_check(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
            CandidateHypothesis(
                id="H1", name="POSSIBLE_FACTOR", statement="A reported factor possibly contributed to the deviation",
                status="POSSIBLE", evidence_needed="", evidence_strength="INDICATIVE",
                causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE,
            ),
        ])
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        fw = build_graph_grounded_five_why(g)
        # Phase 13: a single POSSIBLE-status edge's transition step already
        # IS the evidence boundary (there is nothing beyond it to walk to),
        # so it is relabeled EVIDENCE_BOUNDARY in place rather than followed
        # by a second, duplicate marker step (INV-WHY-007) -- exactly one
        # step is produced here, not two.
        assert fw is not None
        assert len(fw.steps) == 1
        step = fw.steps[0]
        assert step.boundary_status == "EVIDENCE_BOUNDARY"
        # The rendered answer must use the neutral concept_ref, never the
        # raw hedged hypothesis statement, and must not trip the invariant.
        assert "possibly contributed" not in (step.answer or ""), (
            "The raw hypothesis statement's own modal language must never leak into the "
            "rendered answer"
        )
        matched = answer_selects_unverified_hypothesis(step.answer, rc.candidate_hypotheses, status=step.status)
        assert matched is None, (
            f"Fixed this phase via concept_ref-based rendering — a regression here means the "
            f"hedging fix broke: {step.answer!r}"
        )

    def test_verified_status_step_answer_does_not_trip_the_check(self):
        """Sanity: the existing swap condition (VERIFIED/REPORTED-backed
        multi-hop chains) does NOT have this problem — confirms the revert
        target is genuinely safe, not just untested."""
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            CandidateHypothesis(
                id="H1", name="VERIFIED_MECH", statement="A verified mechanism explains the deviation",
                status="SUPPORTED", evidence_needed="", evidence_strength="VERIFIED",
                causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"],
            ),
        ])
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        fw = build_graph_grounded_five_why(g)
        step = fw.steps[0]
        matched = answer_selects_unverified_hypothesis(step.answer, rc.candidate_hypotheses, status=step.status)
        assert matched is None


class TestLiveCapaImpactPathIsAlreadyStructural:
    """Step 1 runtime audit finding: app/agent/nodes/capa.py and impact.py
    carry an explicit "LEGACY NODE — NOT PART OF LIVE GRAPH" docstring —
    verified here that the ACTUAL live path (core_synthesis.py) does not
    import or call either legacy module, and that its real deterministic
    functions take structured canonical/hypothesis state as input, never
    root_cause.narrative text."""

    def test_legacy_capa_module_not_imported_by_core_synthesis(self):
        import inspect

        import app.agent.nodes.core_synthesis as core_synthesis_mod
        src = inspect.getsource(core_synthesis_mod)
        assert "from app.agent.nodes.capa import" not in src
        assert "from app.agent.nodes.impact import" not in src

    def test_legacy_modules_carry_explicit_disclaimer(self):
        import app.agent.nodes.capa as capa_mod
        import app.agent.nodes.impact as impact_mod
        assert "LEGACY NODE" in (capa_mod.__doc__ or "")
        assert "LEGACY NODE" in (impact_mod.__doc__ or "")

    def test_live_impact_derivation_does_not_take_narrative_text_as_input(self):
        import inspect

        from app.agent.nodes.core_synthesis import _derive_deterministic_impact
        sig = inspect.signature(_derive_deterministic_impact)
        assert "narrative" not in sig.parameters
        assert "root_cause" not in sig.parameters
