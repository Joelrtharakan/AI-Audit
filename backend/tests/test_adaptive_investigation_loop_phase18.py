"""Phase 18: adaptive evidence → causal graph → re-planning loop.

Audited runtime constraint (see app/agent/nodes/causal_investigation_planner.py's
module docstring, carried forward from Phase 17): the existing critic-
send-back loop that would normally re-invoke core_synthesis -> Stage B is
driven by `_planned_tool_calls`, populated only by the ASP.NET LLM
tool-planning path. In the deterministic fast-path (every test in this
repository, and any deployment without that integration) there are never
any planned tools, so that loop never actually re-executes. Building a
fake ASP.NET server to exercise it end-to-end was assessed as out of scope
for this phase.

What IS real and tested here: `causal_investigation_planner_node` (Stage
B) is the REAL, production node registered in the compiled LangGraph
(app/agent/graph.py). These tests call it directly, more than once, with
state threaded between calls exactly as LangGraph would thread it across
node executions -- this is the same "call the real node function in
sequence" testing pattern already used throughout this repository's test
suite (e.g. tests/test_investigation_planner_phase14.py,
tests/test_graph_investigation_planner_phase16.py). No fake/parallel
planner implementation is used; `plan_investigation_causal` and
`causal_investigation_planner_node` are the actual functions
`app/agent/graph.py` wires in.

The evidence "injection" between calls is a direct mutation of
`root_cause.candidate_hypotheses[i].status` -- standing in for what a real
evidence-recording step would produce (the existing eligibility engine in
app.agent.causal_graph.evaluate_root_cause_eligibility already computes
these statuses from evidence; this test starts from its OUTPUT rather than
re-deriving it, to isolate what Phase 18 actually adds: the re-planning
loop mechanics, not the eligibility engine, which predates this phase and
is already extensively tested elsewhere).
"""
from __future__ import annotations

import asyncio

from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.causal_investigation_planner import (
    causal_investigation_planner_node,
    decide_investigation_state,
)
from app.models.agent import CandidateHypothesis, CausalLevel, RootCauseAnalysis, RootCauseStatus


def _hyp(**kwargs) -> CandidateHypothesis:
    defaults = dict(evidence_needed="", causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE)
    defaults.update(kwargs)
    return CandidateHypothesis(**defaults)


# ---------------------------------------------------------------------------
# Critical adaptive loop test (Section 21)
# ---------------------------------------------------------------------------

def test_adaptive_loop_plan_changes_after_evidence_resolves_h1():
    h1 = _hyp(id="H1", name="MECHANISM_A", statement="Mechanism A", status="POSSIBLE",
              evidence_strength="REPORTED", supporting_claim_ids=["C1"])
    h2 = _hyp(id="H2", name="MECHANISM_B", statement="Mechanism B", status="POSSIBLE",
              evidence_strength="REPORTED", supporting_claim_ids=["C2"])
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[h1, h2])

    state_0 = {"root_cause": rc, "canonical_finding_state": None, "evidence_ledger": [], "trace": []}
    state_1 = asyncio.run(causal_investigation_planner_node(state_0))
    plan_a = state_1["causal_investigation_plan"]

    # Iteration 0 -> both live: a discriminating question covering both.
    assert plan_a.planner_mode == "GRAPH_FIRST"
    covered_a = {hid for q in plan_a.questions for hid in q.target_hypothesis_ids}
    assert covered_a == {"H1", "H2"}
    assert state_1["investigation_iteration"] == 1
    assert state_1["causal_graph_version"] == 1
    assert len(state_1["investigation_history"]) == 1
    record_a = state_1["investigation_history"][0]
    assert record_a.planner_decision == "NEW_TARGET"
    assert set(record_a.unresolved_targets_after) == {"H1", "H2"}

    # --- Evidence arrives: H1 is now VERIFIED/SUPPORTED, H2 untouched. ---
    h1.status = "SUPPORTED"
    h1.evidence_strength = "VERIFIED"
    state_1["evidence_ledger"] = [object()]  # stand-in for a real EvidenceItem

    state_2 = asyncio.run(causal_investigation_planner_node(state_1))
    plan_b = state_2["causal_investigation_plan"]

    # Plan A != Plan B because of evidence.
    covered_b = {hid for q in plan_b.questions for hid in q.target_hypothesis_ids}
    assert covered_b != covered_a
    assert covered_b == {"H2"}, "H1 must not be re-asked about after being resolved"
    assert "H1" not in covered_b

    # Graph version and iteration both advanced.
    assert state_2["causal_graph_version"] == 2
    assert state_2["investigation_iteration"] == 2
    assert len(state_2["investigation_history"]) == 2
    record_b = state_2["investigation_history"][1]
    assert record_b.iteration_id == 1
    assert record_b.causal_graph_version == 2
    assert "H1" in record_b.newly_resolved_targets
    assert record_b.unresolved_targets_after == ["H2"]
    assert record_b.planner_decision == "TARGET_RESOLVED"

    # H2 remains an independent, still-open target -- never silently
    # collapsed or forced to a conclusion just because H1 resolved.
    h2_questions = [q for q in plan_b.questions if "H2" in q.target_hypothesis_ids]
    assert len(h2_questions) == 1

    # decide_investigation_state reflects real remaining uncertainty.
    assert decide_investigation_state(["H2"], 1, 5, state_2["causal_graph"]) == "REPLAN"


def test_adaptive_loop_terminates_at_resolved_when_all_hypotheses_settle():
    h1 = _hyp(id="H1", name="MECHANISM_A", statement="Mechanism A", status="POSSIBLE", evidence_strength="REPORTED")
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[h1])
    state_0 = {"root_cause": rc, "canonical_finding_state": None, "evidence_ledger": [], "trace": []}
    state_1 = asyncio.run(causal_investigation_planner_node(state_0))
    assert state_1["causal_investigation_plan"].planner_mode == "GRAPH_FIRST"

    h1.status = "SUPPORTED"
    h1.evidence_strength = "VERIFIED"
    state_2 = asyncio.run(causal_investigation_planner_node(state_1))
    plan_2 = state_2["causal_investigation_plan"]
    assert plan_2.planner_mode == "NO_ACTIONABLE_UNCERTAINTY"
    assert plan_2.questions == []
    record = state_2["investigation_history"][-1]
    assert record.planner_decision == "NO_ACTIONABLE_UNCERTAINTY"
    assert record.unresolved_targets_after == []
    assert decide_investigation_state([], 1, 5, state_2["causal_graph"]) == "RESOLVED"

    # And this run is fully consistent under the new invariants.
    ok, violations = evaluate_all_invariants(state_2)
    relevant = [v for v in violations if "INV-INVEST-024" in v or "INV-INVEST-025" in v]
    assert not relevant, relevant


def test_adaptive_loop_discovers_new_uncertainty_after_resolution():
    """Evidence resolving H1 must not stop investigation if it reveals a
    NEW hypothesis to investigate (Section 22: a verified mechanism does
    not resolve a deeper, newly-surfaced uncertainty)."""
    h1 = _hyp(id="H1", name="MECHANISM_A", statement="Mechanism A", status="POSSIBLE", evidence_strength="REPORTED")
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[h1])
    state_0 = {"root_cause": rc, "canonical_finding_state": None, "evidence_ledger": [], "trace": []}
    state_1 = asyncio.run(causal_investigation_planner_node(state_0))

    h1.status = "SUPPORTED"
    h1.evidence_strength = "VERIFIED"
    h3 = _hyp(id="H3", name="DEEPER_CAUSE", statement="A deeper systemic cause", status="POSSIBLE",
              evidence_strength="REPORTED", causal_level=CausalLevel.L5_SYSTEMIC_CAUSE)
    rc.candidate_hypotheses.append(h3)

    state_2 = asyncio.run(causal_investigation_planner_node(state_1))
    plan_2 = state_2["causal_investigation_plan"]
    assert plan_2.planner_mode == "GRAPH_FIRST"
    assert plan_2.questions, "investigation must continue: a new, deeper uncertainty was discovered"
    covered = {hid for q in plan_2.questions for hid in q.target_hypothesis_ids}
    assert covered == {"H3"}
    record = state_2["investigation_history"][-1]
    assert record.planner_decision == "NEW_UNCERTAINTY"
    assert "H3" in record.newly_created_uncertainties


def test_adaptive_loop_preserves_refuted_hypothesis_in_history():
    """A REFUTED hypothesis must never be deleted from the record, only
    excluded from future live targets (Section 23)."""
    h1 = _hyp(id="H1", name="MECHANISM_A", statement="Mechanism A", status="POSSIBLE", evidence_strength="REPORTED")
    h2 = _hyp(id="H2", name="MECHANISM_B", statement="Mechanism B", status="POSSIBLE", evidence_strength="REPORTED")
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[h1, h2])
    state_0 = {"root_cause": rc, "canonical_finding_state": None, "evidence_ledger": [], "trace": []}
    state_1 = asyncio.run(causal_investigation_planner_node(state_0))

    h1.status = "REFUTED"
    state_2 = asyncio.run(causal_investigation_planner_node(state_1))
    # H1 still exists in root_cause.candidate_hypotheses (never deleted).
    ids_in_rc = {h.id for h in rc.candidate_hypotheses}
    assert ids_in_rc == {"H1", "H2"}
    covered = {hid for q in state_2["causal_investigation_plan"].questions for hid in q.target_hypothesis_ids}
    assert covered == {"H2"}
    record = state_2["investigation_history"][-1]
    assert record.planner_decision == "TARGET_RESOLVED"


# ---------------------------------------------------------------------------
# decide_investigation_state — deterministic decision engine
# ---------------------------------------------------------------------------

def test_decide_investigation_state_fail_closed_on_missing_graph():
    assert decide_investigation_state(["H1"], 0, 5, None) == "FAIL_CLOSED"


def test_decide_investigation_state_exhausted_at_max_iterations():
    assert decide_investigation_state(["H1"], 5, 5, object()) == "EXHAUSTED"


def test_decide_investigation_state_resolved_when_nothing_unresolved():
    assert decide_investigation_state([], 3, 5, object()) == "RESOLVED"


def test_decide_investigation_state_continue_on_first_iteration():
    assert decide_investigation_state(["H1"], 0, 5, object()) == "CONTINUE"


def test_decide_investigation_state_replan_on_later_iteration():
    assert decide_investigation_state(["H1"], 2, 5, object()) == "REPLAN"


# ---------------------------------------------------------------------------
# INV-INVEST-024 / 025
# ---------------------------------------------------------------------------

def test_inv_invest_024_fails_closed_on_version_regression():
    from app.models.agent import InvestigationIterationRecord
    rec1 = InvestigationIterationRecord(iteration_id=0, causal_graph_version=2, planner_decision="NEW_TARGET")
    rec2 = InvestigationIterationRecord(iteration_id=1, causal_graph_version=1, planner_decision="SAME_TARGET")
    ok, violations = evaluate_all_invariants({"investigation_history": [rec1, rec2]})
    assert any("INV-INVEST-024" in v for v in violations)


def test_inv_invest_024_passes_for_monotonic_history():
    from app.models.agent import InvestigationIterationRecord
    rec1 = InvestigationIterationRecord(iteration_id=0, causal_graph_version=1, planner_decision="NEW_TARGET")
    rec2 = InvestigationIterationRecord(iteration_id=1, causal_graph_version=2, planner_decision="SAME_TARGET")
    ok, violations = evaluate_all_invariants({"investigation_history": [rec1, rec2]})
    assert not any("INV-INVEST-024" in v for v in violations)


def test_inv_invest_025_fails_closed_on_inconsistent_no_action_claim():
    from app.models.agent import InvestigationIterationRecord, InvestigationPlan
    rec = InvestigationIterationRecord(
        iteration_id=0, causal_graph_version=1, planner_decision="NO_ACTIONABLE_UNCERTAINTY",
        unresolved_targets_after=["H1"],
    )
    plan = InvestigationPlan(planner_mode="NO_ACTIONABLE_UNCERTAINTY", questions=[])
    ok, violations = evaluate_all_invariants({"causal_investigation_plan": plan, "investigation_history": [rec]})
    assert any("INV-INVEST-025" in v for v in violations)


def test_inv_invest_025_passes_for_consistent_claim():
    from app.models.agent import InvestigationIterationRecord, InvestigationPlan
    rec = InvestigationIterationRecord(
        iteration_id=0, causal_graph_version=1, planner_decision="NO_ACTIONABLE_UNCERTAINTY",
        unresolved_targets_after=[],
    )
    plan = InvestigationPlan(planner_mode="NO_ACTIONABLE_UNCERTAINTY", questions=[])
    ok, violations = evaluate_all_invariants({"causal_investigation_plan": plan, "investigation_history": [rec]})
    assert not any("INV-INVEST-025" in v for v in violations)
