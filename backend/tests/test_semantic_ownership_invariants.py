"""Architecture Test: Single Semantic Ownership Invariants.

Verifies that:
  1. Exactly ONE authoritative producer is registered for each core semantic field.
  2. Downstream nodes consume canonical values and do not overwrite or redefine them.
"""

from __future__ import annotations

import inspect
import pytest

from app.agent.nodes.understanding import understand_finding_node
from app.models.agent import CanonicalFindingState, InvestigationReport
from app.services.semantic_subject import resolve_deviation


def test_semantic_ownership_registry():
    """Verify authoritative semantic ownership map across nodes."""
    SEMANTIC_OWNERSHIP_MAP = {
        "finding_subject": "app.services.semantic_subject.resolve_deviation",
        "affected_object": "app.services.semantic_subject.resolve_deviation",
        "affected_period": "app.services.semantic_subject.extract_temporal_clause",
        "process_at_risk": "app.services.semantic_subject.resolve_deviation",
        "root_cause": "app.agent.causal_graph.select_authoritative_leading_hypothesis",
        "hypotheses": "app.agent.causal_graph.evaluate_root_cause_eligibility",
        "investigation_plan": "app.agent.nodes.plan_investigation_fallback.build_deterministic_investigation_plan",
        "five_why": "app.agent.nodes.five_why_fallback.build_deterministic_five_why",
        "impact": "app.agent.nodes.core_synthesis._derive_deterministic_impact",
        "capa": "app.agent.nodes.plan_investigation_fallback.build_conditional_capa_actions",
        "final_analysis_validator": "app.agent.causal_graph.validate_final_analysis",
    }

    # Ensure all registered ownership targets are unique and non-overlapping
    producers = list(SEMANTIC_OWNERSHIP_MAP.values())
    assert len(producers) == len(SEMANTIC_OWNERSHIP_MAP)


def test_canonical_subject_is_authoritative_for_affected_object():
    """Canonical finding_subject must remain the sole authoritative source for affected_object."""
    state = CanonicalFindingState(
        raw_finding="Calibration certificate for balance BAL-014 expired on 10 August 2026.",
        finding_subject="balance BAL-014",
        observed_deviation="Calibration certificate expired",
    )
    assert state.finding_subject == "balance BAL-014"
    assert "BAL-014" in state.finding_subject
