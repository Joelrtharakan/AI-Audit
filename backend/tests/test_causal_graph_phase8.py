"""Phase 8: architecture must not depend on the LLM voluntarily emitting
`deepens_hypothesis_id` (Phase C). 5-Why graph authority must activate on
ANY genuine multi-hop causal chain — whether asserted explicitly or
independently resolved from shared evidence — not only the explicit path.

Domain-neutral fixtures; production logic keys only on typed graph fields.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.agent.causal_graph import build_causal_graph
from app.agent.causal_graph_traversal import build_graph_grounded_five_why
from app.models.agent import (
    CandidateHypothesis,
    CanonicalFindingState,
    CausalLevel,
    EvidenceItem,
    EvidenceStatus,
    FiveWhyAnalysis,
    FiveWhyStep,
    InvestigateRequest,
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


class TestGraphAuthorityWithoutOptionalField:
    """Item 27/35: the LLM never populating deepens_hypothesis_id must not
    disable graph-authoritative 5-Why — connectivity is independently
    resolvable from shared supporting_claim_ids (Phase 4's heuristic,
    which predates and does not depend on Phase 7's optional field)."""

    def test_evidence_correlated_chain_alone_yields_graph_traversal(self):
        """No deepens_hypothesis_id anywhere in this fixture."""
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            _hyp(id="H1", statement="Mechanism M", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"],
                 supporting_claim_ids=["C1"]),
            _hyp(id="H2", statement="Root cause R", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L5_SYSTEMIC_CAUSE, supporting_evidence=["e1"],
                 supporting_claim_ids=["C1"]),
        ])
        assert all(h.deepens_hypothesis_id is None for h in rc.candidate_hypotheses)
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        assert any(e.derivation == "EVIDENCE_CORRELATED" for e in g.edges)
        assert not any(e.derivation == "EXPLICIT" for e in g.edges)
        fw = build_graph_grounded_five_why(g)
        # Phase 11 Step 8: a terminal EVIDENCE_BOUNDARY marker step is now
        # appended after the last real transition — re-asserted via the
        # transition-only count rather than raw len(fw.steps).
        transition_steps = [s for s in fw.steps if s.boundary_status == "TRANSITION"]
        assert fw is not None and len(transition_steps) == 2
        assert transition_steps[1].source_node_id == transition_steps[0].target_node_id
        assert fw.steps[-1].boundary_status == "EVIDENCE_BOUNDARY"

    @pytest.mark.asyncio
    async def test_final_verification_swaps_on_evidence_correlated_chain_alone(self):
        """Exercises the REAL final_evidence_verification_node: proves the
        5-Why authority swap fires from EVIDENCE_CORRELATED chaining with
        zero use of deepens_hypothesis_id anywhere in the fixture."""
        from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
        from app.agent.nodes.understanding import understand_finding_node

        text = "The reagent expiry check was not completed as required by Directive RX-4."
        state = {
            "request": InvestigateRequest(finding_text=text),
            "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0,
            "evidence_ledger": [], "errors": [], "trace": [],
        }
        with patch("app.agent.nodes.understanding.get_llm_client", return_value=None):
            state = await understand_finding_node(state)

        state["root_cause"] = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            _hyp(id="H1", statement="Workstation gauge alignment drifted outside tolerance during the shift",
                 status="SUPPORTED", evidence_strength="VERIFIED", causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM,
                 supporting_evidence=["e1"], supporting_claim_ids=["C1"]),
            _hyp(id="H2", statement="Gauge alignment drift monitoring frequency was reduced after the vendor changeover",
                 status="SUPPORTED", evidence_strength="VERIFIED", causal_level=CausalLevel.L5_SYSTEMIC_CAUSE,
                 supporting_evidence=["e1"], supporting_claim_ids=["C1"]),
        ])
        assert all(h.deepens_hypothesis_id is None for h in state["root_cause"].candidate_hypotheses), (
            "Sanity: this fixture must not use the optional field at all"
        )
        state["five_why"] = FiveWhyAnalysis(steps=[
            FiveWhyStep(question="Why was the expiry check not completed?",
                        answer="PROSE-ONLY-ANSWER-MUST-BE-SUPERSEDED", status="MIXED"),
        ])
        # Distinctive per-hypothesis vocabulary avoids both (a) overlap with
        # the finding subject, which the pre-existing causal_model.py
        # eligibility gate excludes from its overlap check, and (b) the
        # pre-existing evaluate_root_cause_eligibility() regex proof-pattern
        # matcher (keywords like "bypass"/"disabled"/"override"), which
        # would otherwise promote BOTH hypotheses to the same causal_level
        # regardless of what this test assigns them — discovered by tracing
        # actual runtime causal_level values through the real pipeline.
        state["evidence_ledger"] = [
            EvidenceItem(claim=text, source="finding", status=EvidenceStatus.VERIFIED),
            EvidenceItem(claim="Workstation gauge alignment drifted outside tolerance during the shift, per calibration log.",
                         source="record", status=EvidenceStatus.VERIFIED),
            EvidenceItem(claim="Gauge alignment drift monitoring frequency was reduced after the vendor changeover.",
                         source="record", status=EvidenceStatus.VERIFIED),
        ]
        state["capa_analysis"] = None
        state["impact_assessment"] = None
        state["investigation_plan"] = None
        state["ca_draft"] = None

        result = await final_evidence_verification_node(state)
        fw = result.get("five_why")
        assert fw is not None
        assert not any("PROSE-ONLY-ANSWER-MUST-BE-SUPERSEDED" in (s.answer or "") for s in fw.steps), (
            "Graph traversal must supersede independently-generated prose even without an "
            "explicit deepens_hypothesis_id assertion anywhere in the fixture"
        )
        assert any(s.causal_edge_id is not None for s in fw.steps)
        cg = result.get("causal_graph")
        assert any(e.derivation == "EVIDENCE_CORRELATED" for e in cg.edges)
        assert not any(e.derivation == "EXPLICIT" for e in cg.edges)

    def test_direct_only_graph_does_not_trigger_authority_swap(self):
        """Two independent (parallel, non-chained) hypotheses must NOT be
        misrepresented as a sequential 5-Why chain — DIRECT-only edges are
        breadth, not depth, and must not trigger the swap."""
        rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
            _hyp(id="H1", statement="Independent hypothesis A", status="POSSIBLE", evidence_strength="INDICATIVE"),
            _hyp(id="H2", statement="Independent hypothesis B", status="POSSIBLE", evidence_strength="INDICATIVE"),
        ])
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        assert all(e.derivation == "DIRECT" for e in g.edges)
        _has_multihop = any(e.derivation in ("EXPLICIT", "EVIDENCE_CORRELATED") for e in g.edges)
        assert not _has_multihop
