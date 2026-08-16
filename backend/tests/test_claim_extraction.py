"""Unit tests for claim-level extraction and conflict detection (Phase 2)."""

import pytest

from app.agent.claim_extractor import (
    _claims_conflict,
    detect_evidence_conflicts,
    extract_claims,
)
from app.models.agent import ClaimAttribution, EvidenceStatus


def test_extract_claims_direct_observation():
    finding = "The calibration log for Balance BAL-01 was not completed on 2026-03-15."
    claims = extract_claims(finding)
    assert len(claims) >= 1
    obs_claim = next(c for c in claims if c.attribution == ClaimAttribution.AUDITOR_OBSERVED)
    assert obs_claim.status == EvidenceStatus.VERIFIED
    assert "calibration log" in obs_claim.text.lower()


def test_extract_claims_reported_statement():
    finding = "The operator stated that they had not received training on the revised procedure."
    claims = extract_claims(finding)
    reported = [c for c in claims if c.attribution == ClaimAttribution.PERSON_REPORTED]
    assert len(reported) >= 1
    assert reported[0].status == EvidenceStatus.REPORTED
    assert reported[0].polarity == "negative"


def test_extract_claims_supervisor_reported():
    finding = "The supervisor claimed the training was completed last month."
    claims = extract_claims(finding)
    sup = [c for c in claims if c.attribution == ClaimAttribution.SUPERVISOR_REPORTED]
    assert len(sup) >= 1
    assert sup[0].status == EvidenceStatus.REPORTED
    assert sup[0].polarity == "positive"


def test_extract_claims_compound_conflicting_statements():
    finding = "The operator stated they had not received training, but the supervisor claimed the training was completed."
    claims = extract_claims(finding)
    
    # Must decompose into separate claims, NOT one single verified statement
    reported = [c for c in claims if c.status == EvidenceStatus.REPORTED]
    assert len(reported) >= 2

    # One negative, one positive
    polarities = {c.polarity for c in reported}
    assert "negative" in polarities
    assert "positive" in polarities

    # Conflict detection must find the conflict
    conflicts = detect_evidence_conflicts(claims)
    assert len(conflicts) >= 1
    assert conflicts[0].status == "UNRESOLVED"
    assert conflicts[0].conflict_type == "CONFLICTING_REPORTS"
    assert "training" in conflicts[0].proposition.lower() or "completed" in conflicts[0].proposition.lower()


def test_detect_no_conflicts_for_consistent_statements():
    finding = "The technician reported that the instrument was out of calibration and maintenance was overdue."
    claims = extract_claims(finding)
    conflicts = detect_evidence_conflicts(claims)
    assert len(conflicts) == 0
