"""Final intelligence-hardening pass, Section 2/3: hypothesis-quality
firewall. Audit finding: this firewall already exists and is wired into
core_synthesis_node (is_evidence_gap_not_hypothesis, restates_observation,
is_evidence_state_not_hypothesis, hypothesis_contradicts_verified_completion,
hypothesis_attacks_statement_credibility, mechanism_already_names_generic_hypothesis,
evaluate_causal_eligibility) -- this file adversarially verifies it with
abstract synthetic labels (per Section 23), not a reimplementation.
"""
from __future__ import annotations

from app.agent.causal_guard import (
    is_evidence_gap_not_hypothesis,
    is_evidence_state_not_hypothesis,
    restates_observation,
)
from app.models.agent import CandidateHypothesis, RootCauseAnalysis, RootCauseStatus


# ---------------------------------------------------------------------------
# 1. Observation-restatement hypothesis rejected
# ---------------------------------------------------------------------------

def test_observation_restatement_hypothesis_is_rejected():
    source = "Process P1 did not produce expected outcome O1 during period T1."
    restated = "Process P1 did not produce expected outcome O1."
    assert is_evidence_gap_not_hypothesis(restated, source) is True
    assert restates_observation(restated, source) is True


def test_paraphrased_restatement_is_also_rejected():
    """A paraphrase (different word order/synonyms, still no new causal
    content, still heavy overlap) must be caught too -- not just an exact
    copy."""
    source = "Node N1 failed to reach expected state S1 within the required window."
    paraphrase = "Node N1 did not reach state S1 within the required window."
    assert is_evidence_gap_not_hypothesis(paraphrase, source) is True


# ---------------------------------------------------------------------------
# 3. Valid causal mechanism accepted (structurally distinct from observation)
# ---------------------------------------------------------------------------

def test_genuine_causal_mechanism_is_not_rejected():
    source = "Process P1 did not produce expected outcome O1 during period T1."
    mechanism = ("A prerequisite input required by Process P1 was not verified before "
                 "execution, which prevented outcome O1 from being produced.")
    assert is_evidence_gap_not_hypothesis(mechanism, source) is False
    assert restates_observation(mechanism, source) is False


def test_mechanism_introducing_new_entity_is_not_rejected():
    source = "Node N1 failed to reach expected state S1 within the required window."
    mechanism = "An upstream dependency, Node N0, did not signal readiness, so Node N1 could not begin its transition to S1."
    assert is_evidence_gap_not_hypothesis(mechanism, source) is False


# ---------------------------------------------------------------------------
# Evidence-state / investigation-uncertainty dressed up as a hypothesis
# ---------------------------------------------------------------------------

def test_evidence_state_statement_is_rejected_as_hypothesis():
    """A statement describing what is UNKNOWN/unverified (an investigation
    question in disguise) is not a causal mechanism."""
    stmt = "Whether the required authorization was verified before the transition is unconfirmed."
    assert is_evidence_state_not_hypothesis(stmt, None) is True


def test_genuine_mechanism_is_not_flagged_as_evidence_state():
    stmt = "The authorization control failed to execute before the transition was permitted to proceed."
    assert is_evidence_state_not_hypothesis(stmt, None) is False


# ---------------------------------------------------------------------------
# 4. No valid hypothesis -> NONE is a legitimate, distinct outcome
# ---------------------------------------------------------------------------

def test_root_cause_not_established_with_zero_hypotheses_is_valid_state():
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[],
                            leading_hypothesis=None, leading_hypothesis_status="NONE")
    assert rc.candidate_hypotheses == []
    assert rc.leading_hypothesis is None
    assert rc.leading_hypothesis_status == "NONE"


# ---------------------------------------------------------------------------
# 5/6. Competing hypotheses remain independent and distinct
# ---------------------------------------------------------------------------

def test_competing_hypotheses_are_never_collapsed_by_shared_vocabulary():
    """Two hypotheses sharing significant vocabulary (both about 'Process
    P1') must remain independently tracked -- id, statement, status all
    distinct -- never merged into one."""
    h1 = CandidateHypothesis(id="H1", name="INPUT_UNAVAILABLE", statement=(
        "A required input for Process P1 was unavailable at the time of execution."
    ), evidence_needed="input availability record")
    h2 = CandidateHypothesis(id="H2", name="CONTROL_BYPASSED", statement=(
        "A control gating Process P1's execution was bypassed."
    ), evidence_needed="control execution log")
    hyps = [h1, h2]
    assert len({h.id for h in hyps}) == 2
    assert h1.statement != h2.statement
    # Neither is a restatement of the other despite shared "Process P1" vocabulary.
    assert not is_evidence_gap_not_hypothesis(h1.statement, h2.statement)
    assert not is_evidence_gap_not_hypothesis(h2.statement, h1.statement)


# ---------------------------------------------------------------------------
# Consequence-as-cause: a hypothesis that only restates the DOWNSTREAM
# consequence (not the deviation itself) is equally not a mechanism.
# ---------------------------------------------------------------------------

def test_consequence_restatement_is_also_rejected():
    """The observation IS the deviation; a 'hypothesis' that merely
    restates the deviation's downstream consequence (not proposing why it
    happened) is structurally the same defect as restating the observation
    itself -- same overlap-based detection applies."""
    observation = "Process P1 did not produce expected outcome O1."
    consequence_as_cause = "Outcome O1 was not produced by Process P1."
    assert is_evidence_gap_not_hypothesis(consequence_as_cause, observation) is True
