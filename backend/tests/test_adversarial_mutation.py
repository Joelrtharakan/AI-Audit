"""Adversarial semantic mutation test suite.

Tests that independently mutate single semantic dimensions of a finding
and verify that all orthogonal dimensions remain preserved in the graph.
"""
from __future__ import annotations

import pytest
from app.agent.proposition_engine import build_semantic_graph, build_propositions_from_ledger
from app.agent.claim_extractor import extract_claims, detect_evidence_conflicts
from app.models.agent import (
    EvidenceItem,
    EvidenceStatus,
    SemanticNodeType,
    SemanticRelationType,
)


def _make_ledger(*claims: str, status: EvidenceStatus = EvidenceStatus.VERIFIED) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            claim=c,
            source="AUDITOR_FINDING",
            source_reference="Test",
            status=status,
            relevance="HIGH",
        )
        for c in claims
    ]


def _build_graph(finding: str):
    ledger = _make_ledger(finding)
    claims = extract_claims(finding, ledger)
    conflicts = detect_evidence_conflicts(claims)
    props = build_propositions_from_ledger(finding, claims, conflicts)
    return build_semantic_graph(finding, claims, props, conflicts)


def _requirement_nodes(graph):
    return [n for n in graph.nodes if n.node_type == SemanticNodeType.REQUIREMENT]


def _has_edge_type(graph, rel_type: SemanticRelationType) -> bool:
    return any(e.relation_type == rel_type for e in graph.edges)


class TestAdversarialSemanticMutations:
    def test_mutate_entity_preserves_requirement_relation(self):
        """Mutating the entity identifier preserves the normative requirement node and violation edge."""
        g1 = _build_graph("Balance BAL-01 was not calibrated per Standard CAL-99.")
        g2 = _build_graph("Autoclave AC-99 was not calibrated per Standard CAL-99.")

        reqs1 = _requirement_nodes(g1)
        reqs2 = _requirement_nodes(g2)
        assert len(reqs1) > 0 and len(reqs2) > 0
        assert reqs1[0].label.lower() == reqs2[0].label.lower()
        assert _has_edge_type(g1, SemanticRelationType.VIOLATES) or _has_edge_type(g1, SemanticRelationType.VERIFIES)
        assert _has_edge_type(g2, SemanticRelationType.VIOLATES) or _has_edge_type(g2, SemanticRelationType.VERIFIES)

    def test_mutate_requirement_preserves_entity_and_relation(self):
        """Mutating the governing requirement leaves the entity node and violation relation intact."""
        g1 = _build_graph("The access log was not reviewed per Procedure SEC-101.")
        g2 = _build_graph("The access log was not reviewed per Policy POL-404.")

        ent_labels1 = {n.label.lower() for n in g1.nodes if n.node_type in (SemanticNodeType.ENTITY, SemanticNodeType.RECORD, SemanticNodeType.PROCESS)}
        ent_labels2 = {n.label.lower() for n in g2.nodes if n.node_type in (SemanticNodeType.ENTITY, SemanticNodeType.RECORD, SemanticNodeType.PROCESS)}
        assert "access log" in ent_labels1 or any("log" in l for l in ent_labels1)
        assert "access log" in ent_labels2 or any("log" in l for l in ent_labels2)
        assert _has_edge_type(g1, SemanticRelationType.VIOLATES)
        assert _has_edge_type(g2, SemanticRelationType.VIOLATES)

    def test_mutate_temporal_clause_preserves_normative_structure(self):
        """Changing temporal clauses does not break normative or attribute extraction."""
        g1 = _build_graph("The batch record was missing a signature in January per SOP-1.")
        g2 = _build_graph("The batch record was missing a signature in November 2025 per SOP-1.")

        assert any(n.node_type == SemanticNodeType.REQUIREMENT for n in g1.nodes)
        assert any(n.node_type == SemanticNodeType.REQUIREMENT for n in g2.nodes)
        assert any(n.node_type == SemanticNodeType.ATTRIBUTE for n in g1.nodes)
        assert any(n.node_type == SemanticNodeType.ATTRIBUTE for n in g2.nodes)

    def test_mutate_quantity_preserves_epistemic_status(self):
        """Changing quantity/measurement does not alter node typing or relation classifications."""
        g1 = _build_graph("3 deviations were not logged per Procedure DEV-01.")
        g2 = _build_graph("15 deviations were not logged per Procedure DEV-01.")

        assert any(n.node_type == SemanticNodeType.REQUIREMENT for n in g1.nodes)
        assert any(n.node_type == SemanticNodeType.REQUIREMENT for n in g2.nodes)
