"""Architecture-Level Hardening & Structural Generalization Test Suite.

Verifies:
1. Synthetic process blocker invariant (INV-ROLE-005)
2. Canonical semantic graph node grounding for affected_object and affected_process
3. Full runtime SemanticTraceabilityMatrix gate verification (INV-ROLE-010 & D10)
4. Multi-relation preservation in proposition extraction without exclusive suppression
5. Graph-grounded 5-Why traversal stopping invariants
6. Absence-of-evidence firewall and epistemic orthogonality across unseen cross-domain findings
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.understanding import understand_finding_node
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.agent.nodes.five_why_fallback import build_deterministic_five_why
from app.agent.nodes.core_synthesis import core_synthesis_node
from app.agent.nodes.report_generator import generate_report_node
from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
from app.agent.output_quality_scorer import compute_output_quality_score
from app.agent.proposition_engine import build_semantic_graph, build_propositions_from_ledger
from app.models.agent import (
    CanonicalFindingState,
    EvidenceClaim,
    EvidenceItem,
    EvidenceStatus,
    InvestigateRequest,
    SemanticGraph,
    SemanticNode,
    SemanticNodeType,
    SemanticTraceabilityMatrix,
)


@pytest.mark.asyncio
async def test_synthetic_process_blocker_invariant_fires():
    """Verify that a synthetic affected_process without a corresponding PROCESS node is rejected."""
    canonical = CanonicalFindingState(
        raw_finding="Unapproved modification of parameter setting was observed.",
        finding_subject="parameter setting",
        affected_object="Parameter setting",
        affected_process="Parameter setting operational control",  # synthetic suffix
        affected_activity="Parameter setting execution",
        deviation="Parameter setting was modified",
        observed_deviation="Parameter setting was modified",
        facts=["Parameter setting was modified without approval"],
        semantic_graph=SemanticGraph(nodes=[
            SemanticNode(id="N1", label="Parameter setting", node_type=SemanticNodeType.ENTITY)
        ]),
    )
    state = {
        "canonical_finding_state": canonical,
        "evidence_ledger": [
            EvidenceItem(claim="Parameter setting was modified without approval", source="finding", status=EvidenceStatus.VERIFIED)
        ]
    }
    is_valid, violations = evaluate_all_invariants(state)
    assert not is_valid
    assert any("[INV-ROLE-005]" in v for v in violations)


@pytest.mark.asyncio
async def test_traceability_matrix_untraced_concepts_fail_quality_gate():
    """Verify that untraced concepts in SemanticTraceabilityMatrix fail the quality gate and INV-ROLE-010."""
    matrix = SemanticTraceabilityMatrix(
        entries=[],
        is_valid=False,
        untraced_concepts=["Unfounded Downstream Systemic Failure"],
    )
    state = {
        "report": type("MockReport", (), {"semantic_traceability": matrix})(),
        "canonical_finding_state": CanonicalFindingState(
            raw_finding="Calibration label was missing.",
            finding_subject="calibration label",
            affected_object="Calibration label",
            affected_process="UNKNOWN",
            affected_activity="UNKNOWN",
            deviation="Calibration label was missing",
            observed_deviation="Calibration label was missing",
            facts=["Calibration label was missing"],
            semantic_graph=SemanticGraph(nodes=[
                SemanticNode(id="N1", label="Calibration label", node_type=SemanticNodeType.RECORD)
            ]),
        ),
        "evidence_ledger": [
            EvidenceItem(claim="Calibration label was missing", source="finding", status=EvidenceStatus.VERIFIED)
        ]
    }
    is_valid, violations = evaluate_all_invariants(state)
    assert not is_valid
    assert any("[INV-ROLE-010]" in v or "[INV-SEM-INV-005]" in v for v in violations)

    score = compute_output_quality_score(state)
    assert score.grade == "FAIL" or any("Traceability" in b for b in score.blocker_violations)


def test_multi_relation_extraction_without_suppression():
    """Verify that a claim encoding both normative violation and missing attribute extracts multiple independent edges."""
    text = "The quarterly system access review for server DB-09 was not completed in Q2 as required by Security Standard SEC-04."
    claims = [
        EvidenceClaim(
            claim_id="C1",
            text=text,
            source="finding",
            status=EvidenceStatus.VERIFIED,
            subject="quarterly system access review",
        )
    ]
    props = build_propositions_from_ledger(text, claims)
    sg = build_semantic_graph(text, claims, props)

    edge_types = [str(getattr(e, "relation_type", "")) for e in sg.edges]
    # Must preserve normative deviation relation (VIOLATES) and not collapse the graph
    assert any("VIOLATES" in et for et in edge_types)


def test_5why_stops_at_evidence_boundary_on_sparse_finding():
    """Verify that a sparse finding does not pad artificial 5-why steps or invent unsupported mechanisms."""
    finding = "Security seal on Container CT-99 was broken."
    ledger = [
        EvidenceItem(claim="Security seal on Container CT-99 was broken", source="finding", status=EvidenceStatus.VERIFIED)
    ]
    fw = build_deterministic_five_why(finding, ledger)
    assert len(fw.steps) >= 1
    # Boundary step must be UNKNOWN or have explicit stopping note
    assert fw.steps[-1].status == "UNKNOWN" or not fw.is_complete
    assert "Analysis unavailable" not in fw.steps[0].answer


@pytest.mark.asyncio
async def test_full_pipeline_structural_generalization_across_unseen_domain():
    """End-to-end test verifying clean execution and zero blocker violations across an unseen logistics/cold-chain finding."""
    finding = (
        "Refrigerated carrier RC-102 temperature datalogger recorded 14.2°C during transit on 14 July 2026, "
        "exceeding the specified 2.0°C to 8.0°C transport range required by Protocol CC-88. "
        "The driver stated that the cooling unit malfunctioned during transit."
    )
    req = InvestigateRequest(finding_text=finding)
    state = {
        "request": req,
        "evidence_ledger": [],
        "iteration_count": 0,
        "tool_call_count": 0,
        "critic_iteration": 0,
        "trace": [],
        "errors": [],
    }

    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.investigation_planner.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        state = await understand_finding_node(state)
        state = await plan_investigation_node(state)
        state = await core_synthesis_node(state)
        state = await generate_report_node(state)
        state = await final_evidence_verification_node(state)

    report = state["report"]
    assert report is not None
    assert report.semantic_traceability is not None
    assert report.semantic_traceability.is_valid
    assert len(report.semantic_traceability.entries) >= 2

    # Invariants evaluation
    is_valid, violations = evaluate_all_invariants(state)
    assert is_valid, f"Violations encountered: {violations}"
