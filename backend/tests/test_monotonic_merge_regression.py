"""Regression suite for the re-investigation-loop monotonic merge guard.

Locks in app.agent.causal_graph.merge_candidate_hypotheses /
capture_epistemic_snapshot and their wiring into
nodes/core_synthesis.py::core_synthesis_node: when the critic sends an
investigation back for more evidence, core_synthesis re-runs and used to
fully REPLACE root_cause.candidate_hypotheses with a freshly synthesized
list -- nothing prevented a later, lower-confidence pass from silently
under-stating a hypothesis an earlier pass had already established (e.g.
VERIFIED evidence_strength quietly becoming REPORTED because the second
pass phrased evidence more tentatively). REFUTED and conflict-driven
downgrades are new evidence-driven information and must always be allowed
through unmerged.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.agent.causal_graph import capture_epistemic_snapshot, merge_candidate_hypotheses
from app.agent.nodes.core_synthesis import core_synthesis_node
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.agent.nodes.understanding import understand_finding_node
from app.models.agent import CandidateHypothesis, InvestigateRequest, RootCauseAnalysis, RootCauseStatus


def _hyp(name: str, status: str, evidence_strength: str) -> CandidateHypothesis:
    return CandidateHypothesis(
        id=name, name=name, statement=f"{name} statement", status=status,
        evidence_needed="records", evidence_strength=evidence_strength,
    )


def test_merge_restores_regressed_evidence_strength():
    previous = [_hyp("MECH_A", "SUPPORTED", "VERIFIED")]
    new = [_hyp("MECH_A", "POSSIBLE", "REPORTED")]
    merged = merge_candidate_hypotheses(previous, new)
    assert merged[0].evidence_strength == "VERIFIED"
    assert merged[0].status == "SUPPORTED"


def test_merge_respects_explicit_refutation():
    previous = [_hyp("MECH_A", "SUPPORTED", "VERIFIED")]
    new = [_hyp("MECH_A", "REFUTED", "NONE")]
    merged = merge_candidate_hypotheses(previous, new)
    assert merged[0].status == "REFUTED"
    assert merged[0].evidence_strength == "NONE"


def test_merge_never_downgrades_when_new_pass_improved():
    previous = [_hyp("MECH_A", "POSSIBLE", "REPORTED")]
    new = [_hyp("MECH_A", "SUPPORTED", "VERIFIED")]
    merged = merge_candidate_hypotheses(previous, new)
    assert merged[0].evidence_strength == "VERIFIED"
    assert merged[0].status == "SUPPORTED"


def test_merge_ignores_unmatched_hypotheses():
    """A hypothesis with no prior-pass counterpart (matched by name) is new
    -- pass it through untouched."""
    previous = [_hyp("MECH_A", "SUPPORTED", "VERIFIED")]
    new = [_hyp("MECH_B", "POSSIBLE", "REPORTED")]
    merged = merge_candidate_hypotheses(previous, new)
    assert merged[0].name == "MECH_B"
    assert merged[0].evidence_strength == "REPORTED"


def test_merge_no_previous_pass_is_noop():
    new = [_hyp("MECH_A", "POSSIBLE", "REPORTED")]
    merged = merge_candidate_hypotheses(None, new)
    assert merged == new


def test_capture_epistemic_snapshot_shape():
    rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[_hyp("MECH_A", "SUPPORTED", "VERIFIED")])
    snap = capture_epistemic_snapshot(rc, canonical=None)
    assert snap["root_cause_status"] == "RootCauseStatus.SUPPORTED" or snap["root_cause_status"] == "SUPPORTED"
    assert snap["hypotheses"]["MECH_A"]["evidence_strength"] == "VERIFIED"
    assert snap["unresolved_conflict_ids"] == []


async def _run_understanding_and_plan(finding_text: str):
    req = InvestigateRequest(finding_text=finding_text)
    state = {
        "request": req, "evidence_ledger": [], "iteration_count": 0,
        "tool_call_count": 0, "critic_iteration": 0, "trace": [], "errors": [],
    }
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.investigation_planner.get_llm_client", return_value=None):
        state = await understand_finding_node(state)
        state = await plan_investigation_node(state)
    return state


@pytest.mark.asyncio
async def test_core_synthesis_appends_snapshot_history_each_pass():
    """Wiring smoke test: core_synthesis_node must append one snapshot per
    call, and a second (critic-send-back-simulated) pass must not lose the
    first pass's established hypothesis strength."""
    text = (
        "The document-control system failed to distribute the revised SOP to affected departments. "
        "System logs verify the distribution failure occurred during the release window."
    )
    state = await _run_understanding_and_plan(text)
    with patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        state = await core_synthesis_node(state)
        assert len(state.get("epistemic_snapshot_history", [])) == 1
        first_pass_rc = state["root_cause"]
        assert first_pass_rc.status == RootCauseStatus.SUPPORTED

        # Simulate the critic-send-back re-investigation loop's second pass.
        state = await core_synthesis_node(state)
        assert len(state.get("epistemic_snapshot_history", [])) == 2
        second_pass_rc = state["root_cause"]
        # Deterministic fallback recomputes from the same unchanged evidence
        # -- the merge guard must be a no-op here (nothing regressed), and
        # the established hypothesis must still read SUPPORTED/VERIFIED.
        assert second_pass_rc.status == RootCauseStatus.SUPPORTED
        assert any(h.evidence_strength == "VERIFIED" for h in second_pass_rc.candidate_hypotheses)
