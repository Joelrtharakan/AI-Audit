"""FINAL RELEASE HARDENING -- AUDITOR-FACING REPORT QUALITY, Rule 9:
EVERY_EVIDENCE_ARTIFACT_MUST_BE_CONTEXT_GROUNDED. Verifies the new
INV-REPORT-003 invariant (app.agent.invariants) rejects a dangling/
fabricated hypothesis-ID reference from an InvestigationQuestion, an
EvidenceRequest, or a CAPA ConditionalCapaAction -- and does not
false-positive on a genuinely valid, self-consistent artifact. Uses
abstract synthetic IDs (H1, H2, Q1) per this session's convention, never
real finding vocabulary.
"""
from __future__ import annotations

from app.agent.invariants import evaluate_all_invariants
from app.models.agent import (
    CandidateHypothesis,
    CanonicalFindingState,
    CapaAnalysis,
    ConditionalCapaAction,
    EvidenceClaim,
    EvidenceStatus,
    InvestigationPlan,
    InvestigationQuestion,
    RootCauseAnalysis,
    RootCauseStatus,
)


def _rc_with_one_hypothesis():
    hyp = CandidateHypothesis(id="H1", name="X", statement="s", evidence_needed="e", status="POSSIBLE")
    return RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[hyp])


def _violations(state):
    _, violations = evaluate_all_invariants(state)
    return [v for v in violations if v.startswith("[INV-REPORT-003]")]


def test_valid_grounded_artifact_passes():
    rc = _rc_with_one_hypothesis()
    q = InvestigationQuestion(question="What confirms H1?", target_hypothesis_ids=["H1"])
    plan = InvestigationPlan(questions=[q], status="QUESTIONS_GENERATED")
    capa = CapaAnalysis(status="CAPA_DRAFT_POSSIBLE", conditional_actions=[
        ConditionalCapaAction(
            if_cause_confirmed="If H1 is confirmed", recommended_action="Do X",
            root_cause_hypothesis_id="H1",
        )
    ])
    state = {"root_cause": rc, "investigation_plan": plan, "capa": capa, "evidence_requests": []}
    assert _violations(state) == []


def test_dangling_hypothesis_reference_in_investigation_question_rejected():
    rc = _rc_with_one_hypothesis()
    q = InvestigationQuestion(question="What confirms H99?", target_hypothesis_ids=["H99_DOES_NOT_EXIST"])
    plan = InvestigationPlan(questions=[q], status="QUESTIONS_GENERATED")
    state = {"root_cause": rc, "investigation_plan": plan, "evidence_requests": []}
    violations = _violations(state)
    assert len(violations) == 1
    assert "H99_DOES_NOT_EXIST" in violations[0]


def test_dangling_hypothesis_reference_in_evidence_request_rejected():
    rc = _rc_with_one_hypothesis()

    class _FakeEvidenceRequest:
        request_id = "R1"
        hypothesis_ids = ["H_GHOST"]

    state = {"root_cause": rc, "investigation_plan": None, "evidence_requests": [_FakeEvidenceRequest()]}
    violations = _violations(state)
    assert len(violations) == 1
    assert "H_GHOST" in violations[0]


def test_dangling_hypothesis_reference_in_capa_action_rejected():
    rc = _rc_with_one_hypothesis()
    capa = CapaAnalysis(status="CAPA_DRAFT_POSSIBLE", conditional_actions=[
        ConditionalCapaAction(
            if_cause_confirmed="If H2 is confirmed", recommended_action="Do Y",
            root_cause_hypothesis_id="H2_NEVER_PROPOSED",
        )
    ])
    state = {"root_cause": rc, "investigation_plan": None, "capa": capa, "evidence_requests": []}
    violations = _violations(state)
    assert len(violations) == 1
    assert "H2_NEVER_PROPOSED" in violations[0]


def test_empty_context_no_hypotheses_yet_does_not_false_positive():
    """No candidate hypotheses have been proposed yet (e.g. early in the
    pipeline) -- nothing to ground against, so this must not fire."""
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[])
    q = InvestigationQuestion(question="What happened?", target_hypothesis_ids=["H1"])
    plan = InvestigationPlan(questions=[q], status="QUESTIONS_GENERATED")
    state = {"root_cause": rc, "investigation_plan": plan, "evidence_requests": []}
    assert _violations(state) == []


def test_ambiguous_untargeted_question_does_not_false_positive():
    """A question with no target_hypothesis_ids at all (a genuinely
    untargeted/exploratory question) is not a dangling reference -- it
    makes no hypothesis claim to ground."""
    rc = _rc_with_one_hypothesis()
    q = InvestigationQuestion(question="What is the general context?")
    plan = InvestigationPlan(questions=[q], status="QUESTIONS_GENERATED")
    state = {"root_cause": rc, "investigation_plan": plan, "evidence_requests": []}
    assert _violations(state) == []


# ---------------------------------------------------------------------------
# Claim-ID grounding (CandidateHypothesis.supporting_claim_ids/
# contradicting_claim_ids, ConditionalCapaAction.supporting_claim_ids).
# ---------------------------------------------------------------------------

def _canonical_with_one_claim():
    claim = EvidenceClaim(claim_id="C1", text="s", source="finding", status=EvidenceStatus.REPORTED)
    return CanonicalFindingState(raw_finding="f", observed_deviation="d", evidence_claims=[claim])


def test_valid_supporting_claim_id_passes():
    hyp = CandidateHypothesis(
        id="H1", name="X", statement="s", evidence_needed="e", status="POSSIBLE", supporting_claim_ids=["C1"],
    )
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[hyp])
    state = {"root_cause": rc, "investigation_plan": None, "evidence_requests": [],
             "canonical_finding_state": _canonical_with_one_claim()}
    assert _violations(state) == []


def test_dangling_supporting_claim_id_rejected():
    hyp = CandidateHypothesis(
        id="H1", name="X", statement="s", evidence_needed="e", status="POSSIBLE",
        supporting_claim_ids=["C99_FABRICATED"],
    )
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[hyp])
    state = {"root_cause": rc, "investigation_plan": None, "evidence_requests": [],
             "canonical_finding_state": _canonical_with_one_claim()}
    violations = _violations(state)
    assert len(violations) == 1
    assert "C99_FABRICATED" in violations[0]


def test_dangling_contradicting_claim_id_rejected():
    hyp = CandidateHypothesis(
        id="H1", name="X", statement="s", evidence_needed="e", status="POSSIBLE",
        contradicting_claim_ids=["C_GHOST"],
    )
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[hyp])
    state = {"root_cause": rc, "investigation_plan": None, "evidence_requests": [],
             "canonical_finding_state": _canonical_with_one_claim()}
    violations = _violations(state)
    assert len(violations) == 1
    assert "C_GHOST" in violations[0]


def test_dangling_capa_supporting_claim_id_rejected():
    hyp = CandidateHypothesis(id="H1", name="X", statement="s", evidence_needed="e", status="POSSIBLE")
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[hyp])
    capa = CapaAnalysis(status="CAPA_DRAFT_POSSIBLE", conditional_actions=[
        ConditionalCapaAction(
            if_cause_confirmed="If H1 is confirmed", recommended_action="Do X",
            supporting_claim_ids=["C_NEVER_EXTRACTED"],
        )
    ])
    state = {"root_cause": rc, "investigation_plan": None, "capa": capa, "evidence_requests": [],
             "canonical_finding_state": _canonical_with_one_claim()}
    violations = _violations(state)
    assert len(violations) == 1
    assert "C_NEVER_EXTRACTED" in violations[0]


def test_no_claim_registry_yet_does_not_false_positive():
    """No canonical_finding_state/evidence_claims populated yet (e.g. state
    built by hand, or an early pipeline stage) -- nothing to ground claim
    references against, so this must not fire."""
    hyp = CandidateHypothesis(
        id="H1", name="X", statement="s", evidence_needed="e", status="POSSIBLE",
        supporting_claim_ids=["C1"],
    )
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[hyp])
    state = {"root_cause": rc, "investigation_plan": None, "evidence_requests": []}
    assert _violations(state) == []
