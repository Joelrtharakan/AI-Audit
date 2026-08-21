"""Regression suite for real cross-snapshot epistemic-transition validation.

Locks in the fix for INV-UNCERTAINTY-005 (app.agent.invariants.
_check_epistemic_status_transitions): the rule used to be a same-snapshot
structural check only ("if ESTABLISHED, must have candidate_hypotheses"),
which cannot detect an actual regression -- a claim/hypothesis silently
downgrading, or root_cause status/causal_readiness regressing -- across the
critic-send-back re-investigation loop's two core_synthesis passes.

app.agent.causal_graph.capture_epistemic_snapshot now produces a compact,
comparable snapshot appended to state["epistemic_snapshot_history"] at the
end of every core_synthesis_node call. The invariant compares the last two
entries. These tests exercise the invariant directly against hand-built
snapshot histories (structural, not tied to any one finding's wording) so
the check's directional-validity logic is verified in isolation from LLM/
deterministic-fallback synthesis behavior.
"""

from __future__ import annotations

from app.agent.invariants import evaluate_all_invariants
from app.models.agent import CandidateHypothesis, RootCauseAnalysis, RootCauseStatus


def _rc(status: RootCauseStatus, hyps: list[CandidateHypothesis] | None = None) -> RootCauseAnalysis:
    return RootCauseAnalysis(status=status, candidate_hypotheses=hyps or [])


def _hyp(name: str, status: str, evidence_strength: str) -> CandidateHypothesis:
    return CandidateHypothesis(
        id=name, name=name, statement=f"{name} statement", status=status,
        evidence_needed="records", evidence_strength=evidence_strength,
    )


def test_no_history_or_single_snapshot_never_flagged():
    """A normal single-pass run only ever produces one snapshot -- nothing
    to compare against, so the transition check must never fire."""
    rc = _rc(RootCauseStatus.SUPPORTED, [_hyp("MECH_A", "SUPPORTED", "VERIFIED")])
    state = {
        "root_cause": rc,
        "epistemic_snapshot_history": [
            {"root_cause_status": "SUPPORTED", "causal_readiness": "READY_FOR_HYPOTHESIS",
             "unresolved_conflict_ids": [], "hypotheses": {"MECH_A": {"status": "SUPPORTED", "evidence_strength": "VERIFIED"}}},
        ],
    }
    is_valid, violations = evaluate_all_invariants(state)
    assert not any("INV-UNCERTAINTY-005" in v for v in violations), violations


def test_hypothesis_evidence_strength_silent_regression_flagged():
    """A hypothesis going VERIFIED -> REPORTED across a re-investigation
    pass, with no new conflicting evidence and not REFUTED, must be caught."""
    rc = _rc(RootCauseStatus.SUPPORTED, [_hyp("MECH_A", "SUPPORTED", "REPORTED")])
    state = {
        "root_cause": rc,
        "epistemic_snapshot_history": [
            {"root_cause_status": "SUPPORTED", "causal_readiness": "READY_FOR_HYPOTHESIS",
             "unresolved_conflict_ids": [], "hypotheses": {"MECH_A": {"status": "SUPPORTED", "evidence_strength": "VERIFIED"}}},
            {"root_cause_status": "SUPPORTED", "causal_readiness": "READY_FOR_HYPOTHESIS",
             "unresolved_conflict_ids": [], "hypotheses": {"MECH_A": {"status": "SUPPORTED", "evidence_strength": "REPORTED"}}},
        ],
    }
    is_valid, violations = evaluate_all_invariants(state)
    assert any("INV-UNCERTAINTY-005" in v and "MECH_A" in v for v in violations), violations


def test_hypothesis_regression_allowed_when_explicitly_refuted():
    """A hypothesis dropping to REFUTED is new evidence-driven information,
    not a silent regression -- must NOT be flagged."""
    rc = _rc(RootCauseStatus.NOT_ESTABLISHED, [_hyp("MECH_A", "REFUTED", "NONE")])
    state = {
        "root_cause": rc,
        "epistemic_snapshot_history": [
            {"root_cause_status": "SUPPORTED", "causal_readiness": "READY_FOR_HYPOTHESIS",
             "unresolved_conflict_ids": [], "hypotheses": {"MECH_A": {"status": "SUPPORTED", "evidence_strength": "VERIFIED"}}},
            {"root_cause_status": "NOT_ESTABLISHED", "causal_readiness": "READY_FOR_HYPOTHESIS",
             "unresolved_conflict_ids": [], "hypotheses": {"MECH_A": {"status": "REFUTED", "evidence_strength": "NONE"}}},
        ],
    }
    is_valid, violations = evaluate_all_invariants(state)
    assert not any("INV-UNCERTAINTY-005" in v for v in violations), violations


def test_hypothesis_regression_allowed_when_new_conflict_appears():
    """A regression justified by a genuinely NEW unresolved conflict
    (evidence that surfaced during re-investigation) must NOT be flagged."""
    rc = _rc(RootCauseStatus.NOT_ESTABLISHED, [_hyp("MECH_A", "POSSIBLE", "REPORTED")])
    state = {
        "root_cause": rc,
        "epistemic_snapshot_history": [
            {"root_cause_status": "SUPPORTED", "causal_readiness": "READY_FOR_HYPOTHESIS",
             "unresolved_conflict_ids": [], "hypotheses": {"MECH_A": {"status": "SUPPORTED", "evidence_strength": "VERIFIED"}}},
            {"root_cause_status": "NOT_ESTABLISHED", "causal_readiness": "READY_FOR_HYPOTHESIS",
             "unresolved_conflict_ids": ["CONF1"], "hypotheses": {"MECH_A": {"status": "POSSIBLE", "evidence_strength": "REPORTED"}}},
        ],
    }
    is_valid, violations = evaluate_all_invariants(state)
    assert not any("INV-UNCERTAINTY-005" in v for v in violations), violations


def test_root_cause_verified_to_not_established_regression_flagged():
    rc = _rc(RootCauseStatus.NOT_ESTABLISHED)
    state = {
        "root_cause": rc,
        "epistemic_snapshot_history": [
            {"root_cause_status": "VERIFIED", "causal_readiness": "ESTABLISHED", "unresolved_conflict_ids": [], "hypotheses": {}},
            {"root_cause_status": "NOT_ESTABLISHED", "causal_readiness": "ESTABLISHED", "unresolved_conflict_ids": [], "hypotheses": {}},
        ],
    }
    is_valid, violations = evaluate_all_invariants(state)
    assert any("INV-UNCERTAINTY-005" in v and "Root cause status regressed" in v for v in violations), violations


def test_causal_readiness_regression_flagged():
    rc = _rc(RootCauseStatus.NOT_ESTABLISHED)
    state = {
        "root_cause": rc,
        "epistemic_snapshot_history": [
            {"root_cause_status": "SUPPORTED", "causal_readiness": "ESTABLISHED", "unresolved_conflict_ids": [], "hypotheses": {}},
            {"root_cause_status": "SUPPORTED", "causal_readiness": "READY_FOR_HYPOTHESIS", "unresolved_conflict_ids": [], "hypotheses": {}},
        ],
    }
    is_valid, violations = evaluate_all_invariants(state)
    assert any("INV-UNCERTAINTY-005" in v and "Causal readiness regressed" in v for v in violations), violations


def test_advancing_across_passes_never_flagged():
    """Evidence strengthening across a re-investigation pass (the normal,
    intended outcome of investigating further) must never be flagged."""
    rc = _rc(RootCauseStatus.SUPPORTED, [_hyp("MECH_A", "SUPPORTED", "VERIFIED")])
    state = {
        "root_cause": rc,
        "epistemic_snapshot_history": [
            {"root_cause_status": "NOT_ESTABLISHED", "causal_readiness": "READY_FOR_HYPOTHESIS",
             "unresolved_conflict_ids": [], "hypotheses": {"MECH_A": {"status": "POSSIBLE", "evidence_strength": "REPORTED"}}},
            {"root_cause_status": "SUPPORTED", "causal_readiness": "READY_FOR_CAUSAL_VERIFICATION",
             "unresolved_conflict_ids": [], "hypotheses": {"MECH_A": {"status": "SUPPORTED", "evidence_strength": "VERIFIED"}}},
        ],
    }
    is_valid, violations = evaluate_all_invariants(state)
    assert not any("INV-UNCERTAINTY-005" in v for v in violations), violations
