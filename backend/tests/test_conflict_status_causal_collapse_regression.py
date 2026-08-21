"""Regression suite for the conflict-status causal-collapse bug.

Locks in the fix for causal_graph.select_authoritative_leading_hypothesis:
the unresolved-conflict gate used to fire on ANY entry in
canonical.evidence_conflicts regardless of its .status, so an already-
RESOLVED_FOR/RESOLVED_AGAINST conflict elsewhere in the evidence set forced
an otherwise-legitimate SUPPORTED immediate-mechanism hypothesis down to
NOT_ESTABLISHED/NONE -- exactly the "verified mechanism collapses because an
unrelated resolved conflict exists" defect named in the causal-layer audit.
The negative case (a genuinely UNRESOLVED conflict) must still correctly
block establishment.
"""

from __future__ import annotations

from app.agent.causal_graph import select_authoritative_leading_hypothesis
from app.models.agent import CandidateHypothesis, CausalLevel, EvidenceConflict, RootCauseStatus


def _supported_hypothesis() -> CandidateHypothesis:
    return CandidateHypothesis(
        id="H1", name="MECHANISM", statement="The interlock was bypassed during the maintenance window",
        status="SUPPORTED", evidence_needed="Access logs",
        evidence_strength="VERIFIED", causal_level=CausalLevel.L5_SYSTEMIC_CAUSE,
    )


def test_resolved_conflict_does_not_collapse_supported_hypothesis():
    conflicts = [
        EvidenceConflict(conflict_id="CONF1", status="RESOLVED_FOR", proposition="unrelated delivery timing"),
    ]
    lead_id, mode, status, _rationale = select_authoritative_leading_hypothesis(
        [_supported_hypothesis()], conflicts=conflicts,
    )
    assert mode == "SELECTED"
    assert status == RootCauseStatus.ESTABLISHED
    assert lead_id == "H1"


def test_unresolved_conflict_still_blocks_establishment():
    conflicts = [
        EvidenceConflict(conflict_id="CONF1", status="UNRESOLVED", proposition="unrelated delivery timing"),
    ]
    lead_id, mode, status, _rationale = select_authoritative_leading_hypothesis(
        [_supported_hypothesis()], conflicts=conflicts,
    )
    assert mode == "NONE"
    assert status == RootCauseStatus.NOT_ESTABLISHED
    assert lead_id is None


def test_mixed_conflicts_resolved_and_unresolved_still_blocks():
    """One resolved and one genuinely unresolved conflict together -- the
    unresolved one alone must still be sufficient to block."""
    conflicts = [
        EvidenceConflict(conflict_id="CONF1", status="RESOLVED_AGAINST", proposition="unrelated timing"),
        EvidenceConflict(conflict_id="CONF2", status="UNRESOLVED", proposition="a different unresolved matter"),
    ]
    lead_id, mode, status, _rationale = select_authoritative_leading_hypothesis(
        [_supported_hypothesis()], conflicts=conflicts,
    )
    assert mode == "NONE"
    assert status == RootCauseStatus.NOT_ESTABLISHED


def test_no_conflicts_baseline_unaffected():
    lead_id, mode, status, _rationale = select_authoritative_leading_hypothesis(
        [_supported_hypothesis()], conflicts=[],
    )
    assert mode == "SELECTED"
    assert status == RootCauseStatus.ESTABLISHED
