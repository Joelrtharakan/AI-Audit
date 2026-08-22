"""Phase 7: explicit sequential causal-chain generation, graph-authoritative
5-Why (when a real chain exists), and RCA/Impact projection depth.

Domain-neutral throughout; no fixture text is reused verbatim across phases.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.causal_graph import build_causal_graph, build_rca_from_causal_graph
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


def _hyp(**kwargs) -> CandidateHypothesis:
    defaults = dict(id="H1", name="N", statement="A mechanism statement", status="POSSIBLE",
                     evidence_needed="", evidence_strength="NONE",
                     causal_level=CausalLevel.L3_IMMEDIATE_MECHANISM)
    defaults.update(kwargs)
    return CandidateHypothesis(**defaults)


class TestExplicitChainDerivation:
    def test_2_two_hop_explicit_chain(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            _hyp(id="H1", statement="Mechanism M", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"]),
            _hyp(id="H2", statement="Root cause R", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L5_SYSTEMIC_CAUSE, supporting_evidence=["e1"],
                 deepens_hypothesis_id="H1"),
        ])
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        chained = [e for e in g.edges if e.derivation == "EXPLICIT"]
        assert len(chained) == 1
        assert chained[0].source_node_id != "CN_DEVIATION"

    def test_4_five_level_chain(self):
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
        # 5 explicit hops: deviation->H1 (DIRECT) then H1->H2->H3->H4->H5 (EXPLICIT x4)
        explicit = [e for e in g.edges if e.derivation == "EXPLICIT"]
        assert len(explicit) == 4

    def test_5_skipped_causal_level_in_chain(self):
        """A chain may legitimately skip levels (L2 -> L5 directly)."""
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            _hyp(id="H1", statement="Immediate mechanism", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"]),
            _hyp(id="H2", statement="Direct systemic cause", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L5_SYSTEMIC_CAUSE, supporting_evidence=["e1"],
                 deepens_hypothesis_id="H1"),
        ])
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        levels_present = {n.node_type for n in g.nodes}
        from app.models.agent import CausalGraphNodeType
        assert CausalGraphNodeType.CONTRIBUTING_FACTOR not in levels_present
        assert CausalGraphNodeType.UNDERLYING_CAUSE not in levels_present
        assert CausalGraphNodeType.SYSTEMIC_ROOT_CAUSE in levels_present

    def test_self_reference_ignored(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
            _hyp(id="H1", statement="Self-referencing", status="POSSIBLE", evidence_strength="INDICATIVE",
                 deepens_hypothesis_id="H1"),
        ])
        g = build_causal_graph(_canonical(), rc, [])
        assert g.edges[0].derivation != "EXPLICIT"
        assert g.edges[0].source_node_id == "CN_DEVIATION"

    def test_two_cycle_reference_ignored(self):
        """A deepens B and B deepens A must never form a 2-cycle."""
        rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
            _hyp(id="H1", statement="A", status="POSSIBLE", evidence_strength="INDICATIVE", deepens_hypothesis_id="H2"),
            _hyp(id="H2", statement="B", status="POSSIBLE", evidence_strength="INDICATIVE", deepens_hypothesis_id="H1"),
        ])
        g = build_causal_graph(_canonical(), rc, [])
        from app.agent.invariants import evaluate_all_invariants
        is_valid, violations = evaluate_all_invariants({"causal_graph": g})
        assert not any("INV-CGRAPH-008" in v for v in violations), "No cycle must ever be constructed, not merely detected after the fact"


class TestGraphAuthoritative5Why:
    def test_11_single_supported_edge_produces_one_transition_step_not_five(self):
        """Phase 11 Step 8: the traversal now emits an explicit terminal
        EVIDENCE_BOUNDARY marker step after the last real transition (so
        "the chain legitimately ends here" is inspectable structural state,
        not something a caller infers from step count) — this changes the
        expected count from 1 to 2, but the requirement being tested
        (never pad to five when only one edge is licensed) still holds and
        is re-asserted below via the real transition-step count."""
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            _hyp(id="H1", statement="Mechanism M", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"]),
        ])
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        fw = build_graph_grounded_five_why(g)
        transition_steps = [s for s in fw.steps if s.boundary_status == "TRANSITION"]
        boundary_steps = [s for s in fw.steps if s.boundary_status == "EVIDENCE_BOUNDARY"]
        assert len(transition_steps) == 1, "Must never pad to five transition steps when only one edge is licensed"
        assert len(boundary_steps) == 1
        assert boundary_steps[0].causal_edge_id is None, "A boundary marker must never claim a fabricated edge"

    def test_multi_hop_traversal_walks_one_path_sequentially(self):
        """Section 10: the traversal must walk ONE causal path, not treat
        sibling hypotheses as sequential Whys."""
        rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            _hyp(id="H1", statement="Mechanism M", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"]),
            _hyp(id="H2", statement="Deeper factor F", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE, supporting_evidence=["e1"],
                 deepens_hypothesis_id="H1"),
            _hyp(id="H3", statement="Root cause R", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L5_SYSTEMIC_CAUSE, supporting_evidence=["e1"],
                 deepens_hypothesis_id="H2"),
        ])
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        fw = build_graph_grounded_five_why(g)
        # Each step's source must be the PREVIOUS step's target — a real
        # sequential walk, not three parallel deviation-rooted steps.
        for i in range(1, len(fw.steps)):
            assert fw.steps[i].source_node_id == fw.steps[i - 1].target_node_id, (
                "5-Why must traverse depth sequentially, not siblings independently"
            )

    @pytest.mark.asyncio
    async def test_final_verification_supersedes_prose_when_explicit_chain_exists(self):
        """Exercises the ACTUAL final_evidence_verification_node swap logic
        (not a re-implementation of it): a root_cause with an explicit
        deepens_hypothesis_id chain must cause state["five_why"] to become
        the graph-traversal output, discarding the independently-generated
        prose steps that were present going in."""
        from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
        from app.agent.nodes.understanding import understand_finding_node
        from app.models.agent import FiveWhyAnalysis, FiveWhyStep, InvestigateRequest

        text = "The pressure differential was not logged as required by Directive PD-6."
        state = {
            "request": InvestigateRequest(finding_text=text),
            "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0,
            "evidence_ledger": [], "errors": [], "trace": [],
        }
        with patch("app.agent.nodes.understanding.get_llm_client", return_value=None):
            state = await understand_finding_node(state)

        state["root_cause"] = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
            _hyp(id="H1", statement="Workstation calibration verification was bypassed by technician Alvarez",
                 status="SUPPORTED", evidence_strength="VERIFIED", causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM,
                 supporting_evidence=["e1"]),
            _hyp(id="H2", statement="Calibration verification bypass authority was never revoked after policy update",
                 status="SUPPORTED", evidence_strength="VERIFIED", causal_level=CausalLevel.L5_SYSTEMIC_CAUSE,
                 supporting_evidence=["e1"], deepens_hypothesis_id="H1"),
        ])
        state["five_why"] = FiveWhyAnalysis(steps=[
            FiveWhyStep(question="Why was the differential not logged?",
                        answer="PROSE-ONLY-ANSWER-SHOULD-BE-SUPERSEDED", status="MIXED"),
        ])
        # Evidence text must overlap the hypothesis statements — the
        # pre-existing causal_model.py eligibility gate (compute_support_level)
        # independently recomputes support_level from stemmed-word overlap
        # between hypothesis statement and VERIFIED ledger claims (excluding
        # the finding's own subject words), and demotes ungrounded
        # hypotheses to investigation areas regardless of the
        # evidence_strength label passed in. Distinctive per-hypothesis
        # vocabulary avoids any accidental overlap with the finding subject.
        state["evidence_ledger"] = [
            EvidenceItem(claim=text, source="finding", status=EvidenceStatus.VERIFIED),
            EvidenceItem(claim="Workstation calibration verification was bypassed by technician Alvarez during setup.",
                         source="record", status=EvidenceStatus.VERIFIED),
            EvidenceItem(claim="Calibration verification bypass authority was never revoked after the policy update.",
                         source="record", status=EvidenceStatus.VERIFIED),
        ]
        state["capa_analysis"] = None
        state["impact_assessment"] = None
        state["investigation_plan"] = None
        state["ca_draft"] = None

        result = await final_evidence_verification_node(state)
        fw = result.get("five_why")
        assert fw is not None
        assert not any("PROSE-ONLY-ANSWER-SHOULD-BE-SUPERSEDED" in (s.answer or "") for s in fw.steps), (
            "An explicit causal chain must make graph traversal authoritative, superseding "
            "the independently-generated prose that was present going in"
        )
        assert any(s.causal_edge_id is not None for s in fw.steps)
        assert result.get("causal_graph") is not None
        assert any(e.derivation == "EXPLICIT" for e in result["causal_graph"].edges)


class TestRCAWithDepth:
    def test_rca_reflects_chained_causal_levels(self):
        rc = RootCauseAnalysis(status=RootCauseStatus.ESTABLISHED, candidate_hypotheses=[
            _hyp(id="H1", statement="Mechanism M", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L2_IMMEDIATE_MECHANISM, supporting_evidence=["e1"]),
            _hyp(id="H2", statement="Factor F", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE, supporting_evidence=["e1"],
                 deepens_hypothesis_id="H1"),
            _hyp(id="H3", statement="Root cause R", status="SUPPORTED", evidence_strength="VERIFIED",
                 causal_level=CausalLevel.L5_SYSTEMIC_CAUSE, supporting_evidence=["e1"],
                 deepens_hypothesis_id="H2"),
        ])
        g = build_causal_graph(_canonical(), rc, [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)])
        rca = build_rca_from_causal_graph(g)
        assert rca.immediate_mechanisms and rca.contributing_factors and rca.systemic_root_causes
        assert rca.root_cause_status == "ESTABLISHED"
