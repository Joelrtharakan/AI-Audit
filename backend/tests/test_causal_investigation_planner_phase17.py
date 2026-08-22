"""Phase 17: Stage B post-synthesis causal investigation planner.

Audited constraint (see app/agent/nodes/causal_investigation_planner.py's
module docstring): the existing critic-send-back re-investigation loop is
driven by `_planned_tool_calls`, populated only by the LLM tool-planning
path (ASP.NET-integration mode). In the deterministic fast-path — every
test in this repository, and any deployment without that integration —
`planned_tools` is always empty, so the send-back branch never reaches
`execute_tool`. Stage B is therefore wired UNCONDITIONALLY between
`core_synthesis` and `critic` in app/agent/graph.py, not onto the
practically-unreachable send-back edge — see graph.py's diff.

These tests exercise the real node function and the real conditional-edge
function (`critic_decision`) directly, per Phase 17 Section 29/42's
requirement to test actual production code paths, not only helper
functions.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.agent.graph import critic_decision
from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.causal_investigation_planner import (
    causal_investigation_planner_node,
    plan_investigation_causal,
)
from app.models.agent import CandidateHypothesis, CausalLevel, RootCauseAnalysis, RootCauseStatus


def _hyp(**kwargs) -> CandidateHypothesis:
    defaults = dict(evidence_needed="", causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE)
    defaults.update(kwargs)
    return CandidateHypothesis(**defaults)


# ---------------------------------------------------------------------------
# 1. plan_investigation_causal — the real decision function
# ---------------------------------------------------------------------------

def test_no_hypotheses_with_no_canonical_state_is_target_unresolved_not_no_actionable():
    """Release-gate correction (was test_no_hypotheses_is_no_actionable_
    uncertainty, asserting NO_ACTIONABLE_UNCERTAINTY here): zero candidate
    hypotheses with canonical_state=None means we have NO deterministic
    basis to say whether the finding is actionable -- confirmed via the
    real compiled graph that substantive findings (e.g. "the calibration
    certificate was found expired at the time of use") can legitimately
    produce zero hypotheses under deterministic-fallback synthesis, which
    is a synthesis coverage gap, not a verified absence of uncertainty.
    Claiming NO_ACTIONABLE_UNCERTAINTY here was exactly the "the parser
    didn't recognize anything -> nothing to investigate" anti-pattern this
    system must never produce. The old assertion targeted the DEFECT this
    fix removes, not a value this test suite should protect."""
    plan, graph, _ids = plan_investigation_causal(None, RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[]), [])
    assert plan.planner_mode == "TARGET_UNRESOLVED"
    assert plan.planner_stage == "STAGE_B"
    assert plan.questions == []


def test_no_hypotheses_with_genuinely_non_actionable_canonical_state_is_no_actionable_uncertainty():
    """The other half of the fix: when the ONE existing deterministic
    upstream judgment (CanonicalFindingState.is_actionable, set by
    app.agent.nodes.understanding) has already established the finding is
    genuinely non-actionable, zero hypotheses IS consistent with that --
    NO_ACTIONABLE_UNCERTAINTY remains correct and must not be weakened."""
    canonical = SimpleNamespace(is_actionable=False)
    plan, graph, _ids = plan_investigation_causal(
        canonical, RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[]), [],
    )
    assert plan.planner_mode == "NO_ACTIONABLE_UNCERTAINTY"
    assert plan.status == "NO_ACTIONABLE_UNCERTAINTY"


def test_all_hypotheses_resolved_is_no_actionable_uncertainty():
    rc = RootCauseAnalysis(status=RootCauseStatus.ESTABLISHED, candidate_hypotheses=[
        _hyp(id="H1", name="X", statement="X caused it", status="SUPPORTED", evidence_strength="VERIFIED"),
        _hyp(id="H2", name="Y", statement="Y did not cause it", status="REFUTED", evidence_strength="VERIFIED"),
    ])
    plan, graph, _ids = plan_investigation_causal(None, rc, [])
    assert plan.planner_mode == "NO_ACTIONABLE_UNCERTAINTY"
    assert plan.questions == []


def test_single_unresolved_hypothesis_targets_nearest_transition():
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
        _hyp(id="H1", name="MECHANISM_X", statement="Mechanism X", status="POSSIBLE",
             evidence_strength="REPORTED", causal_level=CausalLevel.L3_CONTRIBUTING_CAUSE),
    ])
    plan, graph, _ids = plan_investigation_causal(None, rc, [])
    assert plan.planner_mode == "GRAPH_FIRST"
    assert plan.planner_stage == "STAGE_B"
    assert len(plan.questions) == 1
    assert plan.questions[0].target_hypothesis_ids == ["H1"]


def test_competing_hypotheses_produce_discriminating_questions_not_collapse():
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
        _hyp(id="H1", name="MECHANISM_A", statement="Mechanism A", status="POSSIBLE",
             evidence_strength="REPORTED", supporting_claim_ids=["C1"]),
        _hyp(id="H2", name="MECHANISM_B", statement="Mechanism B", status="POSSIBLE",
             evidence_strength="REPORTED", supporting_claim_ids=["C2"]),
        _hyp(id="H3", name="MECHANISM_C", statement="Mechanism C", status="POSSIBLE",
             evidence_strength="REPORTED", supporting_claim_ids=["C3"]),
    ])
    plan, graph, _ids = plan_investigation_causal(None, rc, [])
    assert plan.planner_mode == "GRAPH_FIRST"
    covered = set()
    for q in plan.questions:
        covered.update(q.target_hypothesis_ids)
    assert covered == {"H1", "H2", "H3"}
    # No hypothesis silently picked as "the" leading one.
    assert plan.graph_targets_considered == 3


def test_causal_graph_is_returned_alongside_plan():
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
        _hyp(id="H1", name="X", statement="X", status="POSSIBLE", evidence_strength="REPORTED"),
    ])
    plan, causal_graph, _ids = plan_investigation_causal(None, rc, [])
    # build_causal_graph correctly returns an empty graph when no canonical
    # state (hence no observed_deviation) is available -- this proves the
    # function is genuinely called and its real return value flows through,
    # not that a deviation node is fabricated without one.
    assert causal_graph is not None
    assert causal_graph.nodes == []


# ---------------------------------------------------------------------------
# 2. The real node function — CREATED, CALLED, POPULATED, CONSUMED
# ---------------------------------------------------------------------------

def test_node_populates_causal_investigation_plan_in_state():
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
        _hyp(id="H1", name="A", statement="A", status="POSSIBLE", evidence_strength="REPORTED", supporting_claim_ids=["C1"]),
        _hyp(id="H2", name="B", statement="B", status="POSSIBLE", evidence_strength="REPORTED", supporting_claim_ids=["C2"]),
    ])
    state = {"root_cause": rc, "canonical_finding_state": None, "evidence_ledger": [], "trace": []}
    result = asyncio.run(causal_investigation_planner_node(state))
    plan = result["causal_investigation_plan"]
    assert plan is not None
    assert plan.planner_stage == "STAGE_B"
    assert result["causal_graph"] is not None


def test_node_never_overwrites_investigation_plan():
    """Stage B is additive -- it must never clobber the Stage-A/legacy
    investigation_plan already in state."""
    from app.models.agent import InvestigationPlan
    original = InvestigationPlan(questions=[], areas=["untouched"])
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
        _hyp(id="H1", name="A", statement="A", status="POSSIBLE", evidence_strength="REPORTED"),
    ])
    state = {"root_cause": rc, "canonical_finding_state": None, "evidence_ledger": [], "trace": [],
             "investigation_plan": original}
    result = asyncio.run(causal_investigation_planner_node(state))
    assert result["investigation_plan"] is original
    assert result["investigation_plan"].areas == ["untouched"]


# ---------------------------------------------------------------------------
# 3. critic_decision — real conditional-edge function, deterministic override
# ---------------------------------------------------------------------------

def test_critic_decision_overrides_send_back_when_stage_b_says_no_action():
    from app.models.agent import InvestigationPlan
    stage_b_plan = InvestigationPlan(planner_mode="NO_ACTIONABLE_UNCERTAINTY", questions=[])
    state = {
        "critic_send_back": True, "critic_iteration": 1,
        "causal_investigation_plan": stage_b_plan,
        "_planned_tool_calls": [{"tool": "get_document"}], "completed_tools": [],
    }
    assert critic_decision(state) == "generate_report"


def test_critic_decision_respects_send_back_when_stage_b_has_targets():
    from app.models.agent import InvestigationPlan, InvestigationQuestion
    stage_b_plan = InvestigationPlan(
        planner_mode="GRAPH_FIRST",
        questions=[InvestigationQuestion(question="q", purpose="p", evidence="e")],
    )
    state = {
        "critic_send_back": True, "critic_iteration": 1,
        "causal_investigation_plan": stage_b_plan,
        "_planned_tool_calls": [{"tool": "get_document"}], "completed_tools": [],
    }
    assert critic_decision(state) == "execute_tool"


def test_critic_decision_unaffected_when_stage_b_absent():
    state = {
        "critic_send_back": True, "critic_iteration": 1,
        "_planned_tool_calls": [{"tool": "get_document"}], "completed_tools": [],
    }
    assert critic_decision(state) == "execute_tool"


# ---------------------------------------------------------------------------
# 4. graph.py wiring — proves the node is actually registered and reachable
# ---------------------------------------------------------------------------

def test_stage_b_node_is_registered_in_the_compiled_graph():
    from app.agent.graph import build_agent_graph
    compiled = build_agent_graph()
    node_names = set(compiled.get_graph().nodes.keys())
    assert "causal_investigation_planner" in node_names


# ---------------------------------------------------------------------------
# 5. Invariants
# ---------------------------------------------------------------------------

def test_inv_invest_021_passes_for_valid_stage_b_question():
    from app.models.agent import InvestigationPlan, InvestigationQuestion
    q = InvestigationQuestion(question="q", purpose="p", evidence="e", target_node_id="N1", planner_stage="STAGE_B")
    plan = InvestigationPlan(questions=[q], planner_stage="STAGE_B")
    ok, violations = evaluate_all_invariants({"causal_investigation_plan": plan})
    assert not any("INV-INVEST-021" in v for v in violations)


def test_inv_invest_021_fails_closed_on_targetless_stage_b_question():
    from app.models.agent import InvestigationPlan, InvestigationQuestion
    q = InvestigationQuestion(question="q", purpose="p", evidence="e", planner_stage="STAGE_B")
    plan = InvestigationPlan(questions=[q], planner_stage="STAGE_B")
    ok, violations = evaluate_all_invariants({"causal_investigation_plan": plan})
    assert any("INV-INVEST-021" in v for v in violations)


def test_inv_invest_022_fails_closed_on_evidenceless_stage_b_question():
    from app.models.agent import InvestigationPlan, InvestigationQuestion
    q = InvestigationQuestion(question="q", purpose="p", target_node_id="N1", planner_stage="STAGE_B")
    plan = InvestigationPlan(questions=[q], planner_stage="STAGE_B")
    ok, violations = evaluate_all_invariants({"causal_investigation_plan": plan})
    assert any("INV-INVEST-022" in v for v in violations)


def test_inv_invest_023_passes_when_all_competing_hypotheses_covered():
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
        _hyp(id="H1", name="A", statement="A", status="POSSIBLE", evidence_strength="REPORTED"),
        _hyp(id="H2", name="B", statement="B", status="POSSIBLE", evidence_strength="REPORTED"),
    ])
    plan, _, _ids = plan_investigation_causal(None, rc, [])
    ok, violations = evaluate_all_invariants({"causal_investigation_plan": plan, "root_cause": rc})
    assert not any("INV-INVEST-023" in v for v in violations)


def test_inv_invest_023_fails_closed_when_hypothesis_dropped():
    from app.models.agent import InvestigationPlan, InvestigationQuestion
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
        _hyp(id="H1", name="A", statement="A", status="POSSIBLE", evidence_strength="REPORTED"),
        _hyp(id="H2", name="B", statement="B", status="POSSIBLE", evidence_strength="REPORTED"),
    ])
    # Fabricate a Stage-B plan that only covers H1, silently dropping H2.
    q = InvestigationQuestion(question="q", purpose="p", evidence="e", target_hypothesis_ids=["H1"], planner_stage="STAGE_B")
    plan = InvestigationPlan(questions=[q], planner_stage="STAGE_B", planner_mode="GRAPH_FIRST")
    ok, violations = evaluate_all_invariants({"causal_investigation_plan": plan, "root_cause": rc})
    assert any("INV-INVEST-023" in v for v in violations)
