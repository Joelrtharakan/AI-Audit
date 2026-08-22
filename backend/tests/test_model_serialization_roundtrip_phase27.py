"""Phase 27 (final release certification): serialization round-trip tests
for the models the compiled LangGraph carries across node boundaries and
that the API layer must serialize back to the client. For each model:
construct a real instance with representative field values (including
nested models, enums, and optional fields left unset) -> model_dump() ->
model_validate() -> assert semantic equality with the original.

This proves LangGraph state merges and API JSON responses cannot silently
drop or corrupt fields -- Pydantic's own (de)serialization is what both of
those paths actually rely on.
"""
from __future__ import annotations

from app.models.agent import (
    CandidateHypothesis,
    CausalGraph,
    CausalGraphEdge,
    CausalGraphNode,
    CausalGraphNodeType,
    CausalPath,
    EvidenceClaim,
    EvidenceConflict,
    EvidenceRequest,
    EvidenceStatus,
    FiveWhyStep,
    HypothesisRelation,
    InvestigationPlan,
    InvestigationQuestion,
)


def _roundtrip(model_cls, instance):
    dumped = instance.model_dump()
    restored = model_cls.model_validate(dumped)
    assert restored == instance
    # Also prove the JSON-string path (what the API layer actually sends).
    restored_from_json = model_cls.model_validate_json(instance.model_dump_json())
    assert restored_from_json == instance
    return restored


def test_evidence_claim_roundtrip_with_nested_hypothesis_relations():
    claim = EvidenceClaim(
        claim_id="C1", text="the record shows the step was completed", source="system_record",
        status=EvidenceStatus.VERIFIED, evidence_ids=["EV1"], hypothesis_ids=["H1", "H2"],
        hypothesis_relations=[
            HypothesisRelation(hypothesis_id="H1", relation="SUPPORTING", reason="r1",
                                validation_decision="ACCEPT", causal_basis="NOT_APPLICABLE"),
            HypothesisRelation(hypothesis_id="H2", relation="CONTRADICTING", reason=None,
                                validation_decision="DOWNGRADE_TO_NEUTRAL", causal_basis="TEMPORAL_ONLY"),
        ],
        proposition_type="POSITIVE_ASSERTION", temporal_context="2026-01-05",
    )
    _roundtrip(EvidenceClaim, claim)


def test_evidence_claim_roundtrip_with_quantitative_and_all_optionals_none():
    from app.models.agent import QuantitativeAssertion
    claim = EvidenceClaim(
        claim_id="C2", text="x", source="s", status=EvidenceStatus.UNKNOWN,
        quantitative=QuantitativeAssertion(left=100.0, operator="GT", right=50.0, unit="ms"),
    )
    _roundtrip(EvidenceClaim, claim)
    # No nested data at all -- every optional left at its default.
    minimal = EvidenceClaim(claim_id="C3", text="y", source="s", status=EvidenceStatus.REPORTED)
    restored = _roundtrip(EvidenceClaim, minimal)
    assert restored.hypothesis_relations == []
    assert restored.quantitative is None


def test_evidence_conflict_roundtrip():
    conflict = EvidenceConflict(
        conflict_id="CONF1", conflict_type="CONFLICTING_REPORTS", proposition_type="CONFLICTING_REPORTS",
        status="UNRESOLVED", claims=["C1", "C2"], proposition="whether the step was completed",
        resolution_required=True,
    )
    _roundtrip(EvidenceConflict, conflict)


def test_evidence_request_roundtrip_with_lifecycle_fields():
    req = EvidenceRequest(
        request_id="R2", question_id="Q1", target_node_id="N1", hypothesis_ids=["H1"],
        required_artifacts=["record_type"], objective="determine X", status="SUPERSEDED",
        parent_request_id="R1",
    )
    _roundtrip(EvidenceRequest, req)


def test_candidate_hypothesis_roundtrip_with_status_lock():
    hyp = CandidateHypothesis(
        id="H1", name="X", statement="s", evidence_needed="e", status="REFUTED",
        evidence_strength="VERIFIED", status_locked=True, supporting_claim_ids=["C1"],
        contradicting_claim_ids=["C2"], causal_node_id="CN1", causal_edge_id="CE1",
    )
    _roundtrip(CandidateHypothesis, hyp)


def test_causal_graph_roundtrip_with_nodes_and_edges():
    graph = CausalGraph(
        nodes=[
            CausalGraphNode(node_id="N1", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label="dev",
                             epistemic_status=EvidenceStatus.VERIFIED),
            CausalGraphNode(node_id="N2", node_type=CausalGraphNodeType.IMMEDIATE_MECHANISM, label="mech",
                             epistemic_status=EvidenceStatus.REPORTED),
        ],
        edges=[CausalGraphEdge(edge_id="E1", source_node_id="N1", target_node_id="N2",
                                evidence_ids=["EV1"], status="VERIFIED")],
    )
    _roundtrip(CausalGraph, graph)


def test_causal_path_roundtrip():
    path = CausalPath(path_id="P1", ordered_node_ids=["N1", "N2", "N3"], ordered_edge_ids=["E1", "E2"],
                       hypothesis_id="H1", starting_level="L0_OBSERVATION",
                       terminal_level="L3_IMMEDIATE_MECHANISM", epistemic_status="VERIFIED",
                       supporting_evidence_ids=["EV1"])
    _roundtrip(CausalPath, path)


def test_investigation_plan_roundtrip_with_questions():
    plan = InvestigationPlan(
        questions=[
            InvestigationQuestion(id="Q1", question="was X true?", target_type="HYPOTHESIS",
                                   target_id="H1", priority="HIGH", depends_on=["Q0"]),
        ],
        status="QUESTIONS_GENERATED",
    )
    _roundtrip(InvestigationPlan, plan)


def test_five_why_step_roundtrip_with_all_grounding_fields():
    step = FiveWhyStep(
        question="why did X occur?", answer="because Y", status="VERIFIED",
        step_id="S1", source_node_id="N1", target_node_id="N2", causal_edge_id="E1",
        evidence_ids=["EV1", "EV2"], proposition_ids=["C1"], causal_level="L3_IMMEDIATE_MECHANISM",
        boundary_status="TRANSITION",
    )
    _roundtrip(FiveWhyStep, step)


def test_five_why_step_roundtrip_boundary_step_with_no_grounding():
    """A boundary step (Section 8: a first-class, valid outcome) has every
    grounding field unset -- must round-trip identically, not acquire
    fabricated defaults."""
    step = FiveWhyStep(question="why did X occur?", answer=None, status="UNKNOWN",
                        boundary_status="EVIDENCE_BOUNDARY")
    restored = _roundtrip(FiveWhyStep, step)
    assert restored.source_node_id is None
    assert restored.causal_edge_id is None
    assert restored.evidence_ids == []
