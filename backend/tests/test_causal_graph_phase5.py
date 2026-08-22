"""Phase 5: CAPA graph-grounding correctness (bug fix verification) and
LLM investigation-question graph-target validation.

Domain-neutral fixtures throughout — no industry vocabulary hardcoded into
production logic, only used as varied test input.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.claim_extractor import detect_evidence_conflicts, extract_claims
from app.agent.invariants import (
    _is_unconditional_capa_action,
    evaluate_all_invariants,
)
from app.agent.proposition_engine import build_propositions_from_ledger, build_semantic_graph
from app.models.agent import (
    CanonicalFindingState,
    CapaAnalysis,
    CapaStatus,
    CausalGraph,
    CausalGraphEdge,
    CausalGraphNode,
    CausalGraphNodeType,
    ConditionalCapaAction,
    EvidenceItem,
    EvidenceStatus,
    InvestigateRequest,
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


class TestCapaConditionalityCorrection:
    """The Phase 4 version of this check tested nonexistent boolean fields
    (`conditional`, `pending_investigation`) on ConditionalCapaAction and
    always evaluated them False via getattr-default, misclassifying every
    hypothesis-gated action as unconditional. Phase 5 fixed the check to
    read the real `if_cause_confirmed` field."""

    def test_hypothesis_gated_action_is_conditional(self):
        action = ConditionalCapaAction(
            if_cause_confirmed="IF H1 (some mechanism) is confirmed",
            recommended_action="Do X",
            action_type="SYSTEMIC_ACTION",
        )
        assert _is_unconditional_capa_action(action) is False

    def test_action_with_no_gating_condition_is_unconditional(self):
        action = ConditionalCapaAction(
            if_cause_confirmed="",
            recommended_action="Recover the loss immediately",
            action_type="CORRECTIVE_ACTION",
        )
        assert _is_unconditional_capa_action(action) is True

    def test_na_gating_condition_is_unconditional(self):
        action = ConditionalCapaAction(
            if_cause_confirmed="N/A",
            recommended_action="Notify management",
            action_type="CORRECTIVE_ACTION",
        )
        assert _is_unconditional_capa_action(action) is True

    def test_gated_definitive_capa_passes_invariant_without_verified_graph(self):
        """The real, corrected behavior: hypothesis-gated CAPA actions (the
        overwhelming majority the existing generator produces) must NOT be
        blocked even when the causal graph has no VERIFIED edge — they are
        conditional by construction."""
        capa = CapaAnalysis(
            status=CapaStatus.INVESTIGATION_REQUIRED,
            conditional_actions=[
                ConditionalCapaAction(
                    if_cause_confirmed="IF H1 is confirmed",
                    recommended_action="Implement systemic control X",
                    action_type="SYSTEMIC_ACTION",
                ),
            ],
        )
        is_valid, violations = evaluate_all_invariants({"capa_analysis": capa, "causal_graph": None})
        cgraph_006 = [v for v in violations if "INV-CGRAPH-006" in v]
        assert not cgraph_006

    def test_ungated_definitive_capa_without_verified_graph_is_blocked(self):
        """A genuinely unconditional root-cause CAPA claim with no VERIFIED
        causal graph edge IS correctly rejected."""
        capa = CapaAnalysis(
            status=CapaStatus.CAPA_RECOMMENDED,
            conditional_actions=[
                ConditionalCapaAction(
                    if_cause_confirmed="",
                    recommended_action="Redesign the systemic control permanently",
                    action_type="SYSTEMIC_ACTION",
                ),
            ],
        )
        is_valid, violations = evaluate_all_invariants({"capa_analysis": capa, "causal_graph": None})
        assert any("INV-CGRAPH-006" in v for v in violations)


class TestCausalGraphAcyclicity:
    def test_direct_cycle_rejected(self):
        nodes = [
            CausalGraphNode(node_id="A", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="A"),
            CausalGraphNode(node_id="B", node_type=CausalGraphNodeType.IMMEDIATE_MECHANISM, label="B"),
            CausalGraphNode(node_id="C", node_type=CausalGraphNodeType.UNDERLYING_CAUSE, label="C"),
        ]
        edges = [
            CausalGraphEdge(edge_id="E1", source_node_id="A", target_node_id="B"),
            CausalGraphEdge(edge_id="E2", source_node_id="B", target_node_id="C"),
            CausalGraphEdge(edge_id="E3", source_node_id="C", target_node_id="A"),
        ]
        cg = CausalGraph(nodes=nodes, edges=edges)
        is_valid, violations = evaluate_all_invariants({"causal_graph": cg})
        assert not is_valid
        assert any("INV-CGRAPH-008" in v for v in violations)

    def test_valid_dag_passes(self):
        nodes = [
            CausalGraphNode(node_id="A", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="A"),
            CausalGraphNode(node_id="B", node_type=CausalGraphNodeType.IMMEDIATE_MECHANISM, label="B"),
            CausalGraphNode(node_id="C", node_type=CausalGraphNodeType.UNDERLYING_CAUSE, label="C"),
        ]
        edges = [
            CausalGraphEdge(edge_id="E1", source_node_id="A", target_node_id="B"),
            CausalGraphEdge(edge_id="E2", source_node_id="B", target_node_id="C"),
        ]
        cg = CausalGraph(nodes=nodes, edges=edges)
        is_valid, violations = evaluate_all_invariants({"causal_graph": cg})
        assert not any("INV-CGRAPH-008" in v for v in violations)


class TestLLMQuestionGraphValidation:
    @pytest.mark.asyncio
    async def test_ungrounded_llm_question_rejected(self):
        """Section 8: an LLM-generated question that resolves to no
        unresolved causal-graph structure and no evidence requirement is
        rejected by plan_investigation_node's LLM path."""
        from app.agent.nodes.investigation_planner import plan_investigation_node
        from app.config import get_settings

        finding = "The maintenance record was not completed as required by Directive MX-19."
        canonical = _canonical_with_semantic_graph(finding)

        mock_client = MagicMock()
        mock_client.chat_completion = AsyncMock(return_value=(
            '{"needs_investigation": true, "planned_tools": [], '
            '"investigation_plan": {"areas": ["general area"], '
            '"questions": [{"question": "What is the weather like today at the site?", '
            '"purpose": "unrelated", "evidence": "unrelated"}], '
            '"evidence_to_collect": ["General records"]}}'
        ))

        settings = get_settings()
        original_url = settings.lqms_aspnet_base_url
        settings.lqms_aspnet_base_url = "http://fake-aspnet-integration.test"
        try:
            with patch("app.agent.nodes.investigation_planner.get_llm_client", return_value=mock_client):
                state = {
                    "request": InvestigateRequest(finding_text=finding),
                    "canonical_finding_state": canonical,
                    "trace": [],
                    "errors": [],
                }
                result = await plan_investigation_node(state)
        finally:
            settings.lqms_aspnet_base_url = original_url

        plan = result["investigation_plan"]
        assert all("weather" not in q.question.lower() for q in plan.questions), (
            "An LLM question with zero structural or evidentiary grounding must be rejected"
        )

    @pytest.mark.asyncio
    async def test_grounded_llm_question_kept_and_annotated(self):
        """A question that overlaps a real unresolved causal-graph node is
        kept AND annotated with target_node_id/causal_level."""
        from app.agent.nodes.investigation_planner import plan_investigation_node
        from app.config import get_settings

        finding = "The maintenance record was not completed as required by Directive MX-19."
        canonical = _canonical_with_semantic_graph(finding)

        mock_client = MagicMock()
        mock_client.chat_completion = AsyncMock(return_value=(
            '{"needs_investigation": true, "planned_tools": [], '
            '"investigation_plan": {"areas": ["MX-19 compliance"], '
            '"questions": [{"question": "What evidence establishes the mechanism connecting '
            'the deviation to MX-19?", "purpose": "resolve MX-19", "evidence": "MX-19 records"}], '
            '"evidence_to_collect": ["MX-19 records"]}}'
        ))

        settings = get_settings()
        original_url = settings.lqms_aspnet_base_url
        settings.lqms_aspnet_base_url = "http://fake-aspnet-integration.test"
        try:
            with patch("app.agent.nodes.investigation_planner.get_llm_client", return_value=mock_client):
                state = {
                    "request": InvestigateRequest(finding_text=finding),
                    "canonical_finding_state": canonical,
                    "trace": [],
                    "errors": [],
                }
                result = await plan_investigation_node(state)
        finally:
            settings.lqms_aspnet_base_url = original_url

        plan = result["investigation_plan"]
        matching = [q for q in plan.questions if "mx-19" in q.question.lower()]
        assert matching, "The grounded question must survive filtering"
        assert matching[0].target_node_id is not None, "A structurally-matched question must be annotated"
