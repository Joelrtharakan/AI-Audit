"""Release-gate Check 1: investigation-planner fail-closed adversarial
matrix. NO_ACTIONABLE_UNCERTAINTY must only occur when the system has
deterministic evidence there is genuinely nothing to investigate -- never
as the side effect of a parser/understanding/graph failure or missing
state. Exercises the real production functions (plan_investigation_graph_first
Stage A, plan_investigation_causal Stage B) directly.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.agent.nodes.causal_investigation_planner import plan_investigation_causal
from app.agent.nodes.graph_investigation_planner import plan_investigation_graph_first
from app.models.agent import CandidateHypothesis, RootCauseAnalysis, RootCauseStatus


def _hyp(**kwargs) -> CandidateHypothesis:
    defaults = dict(id="H1", name="X", statement="s", evidence_needed="e")
    defaults.update(kwargs)
    return CandidateHypothesis(**defaults)


# ---------------------------------------------------------------------------
# Stage A (plan_investigation_graph_first)
# ---------------------------------------------------------------------------

def test_stage_a_missing_canonical_state_is_target_unresolved():
    """2/5. Missing required canonical fields -- canonical_state itself None."""
    result = plan_investigation_graph_first(None)
    assert result.planner_mode == "TARGET_UNRESOLVED"


def test_stage_a_empty_semantic_graph_is_target_unresolved():
    """3. Empty semantic graph."""
    canonical = SimpleNamespace(is_actionable=True, semantic_graph=SimpleNamespace(nodes=[]))
    result = plan_investigation_graph_first(canonical)
    assert result.planner_mode == "TARGET_UNRESOLVED"


def test_stage_a_missing_semantic_graph_attribute_is_target_unresolved():
    """4/5. Malformed/missing canonical field (no semantic_graph attribute at all)."""
    canonical = SimpleNamespace(is_actionable=True)
    result = plan_investigation_graph_first(canonical)
    assert result.planner_mode == "TARGET_UNRESOLVED"


def test_stage_a_genuinely_non_actionable_is_no_actionable_uncertainty():
    """2. Valid non-actionable finding -- the ONE legitimate path."""
    canonical = SimpleNamespace(is_actionable=False)
    result = plan_investigation_graph_first(canonical)
    assert result.planner_mode == "NO_ACTIONABLE_UNCERTAINTY"


def test_stage_a_graph_with_no_recognized_trigger_relations_is_target_unresolved_not_no_actionable():
    """6/7. Parser/understanding coverage gap: a real, populated semantic
    graph whose relation types don't happen to match the trigger set must
    NOT be reported as verified absence of uncertainty (the documented
    Phase 17 correction, re-confirmed here as a release-gate regression
    guard)."""
    node = SimpleNamespace(node_id="N1")
    canonical = SimpleNamespace(
        is_actionable=True,
        semantic_graph=SimpleNamespace(nodes=[node], edges=[]),
    )
    result = plan_investigation_graph_first(canonical)
    assert result.planner_mode == "TARGET_UNRESOLVED"
    assert result.planner_mode != "NO_ACTIONABLE_UNCERTAINTY"


# ---------------------------------------------------------------------------
# Stage B (plan_investigation_causal)
# ---------------------------------------------------------------------------

def test_stage_b_zero_hypotheses_no_canonical_state_is_target_unresolved():
    """9. Missing hypothesis state with no canonical judgment available --
    must fail closed to TARGET_UNRESOLVED, never claim verified absence."""
    plan, _graph, _ids = plan_investigation_causal(
        None, RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[]), [],
    )
    assert plan.planner_mode == "TARGET_UNRESOLVED"


def test_stage_b_zero_hypotheses_with_actionable_canonical_state_is_target_unresolved():
    """The exact reproduced defect: a substantive, actionable finding whose
    deterministic synthesis produced zero hypotheses (a real, confirmed
    coverage gap -- verified via the compiled graph with findings like
    'the calibration certificate was found expired at the time of use')
    must not silently become NO_ACTIONABLE_UNCERTAINTY."""
    canonical = SimpleNamespace(is_actionable=True)
    plan, _graph, _ids = plan_investigation_causal(
        canonical, RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[]), [],
    )
    assert plan.planner_mode == "TARGET_UNRESOLVED"
    assert plan.status == "QUESTIONS_GENERATED"  # only status Literal value available for this honest non-terminal state


def test_stage_b_zero_hypotheses_genuinely_non_actionable_is_no_actionable_uncertainty():
    """The legitimate path -- must remain unweakened."""
    canonical = SimpleNamespace(is_actionable=False)
    plan, _graph, _ids = plan_investigation_causal(
        canonical, RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[]), [],
    )
    assert plan.planner_mode == "NO_ACTIONABLE_UNCERTAINTY"
    assert plan.status == "NO_ACTIONABLE_UNCERTAINTY"


def test_stage_b_missing_root_cause_is_target_unresolved_not_no_actionable():
    """9/10. root_cause itself None (e.g. synthesis produced nothing) with
    an actionable canonical state -- inconsistent planner state must fail
    closed, not silently resolve to 'nothing here'."""
    canonical = SimpleNamespace(is_actionable=True)
    plan, _graph, _ids = plan_investigation_causal(canonical, None, [])
    assert plan.planner_mode == "TARGET_UNRESOLVED"


def test_stage_b_all_resolved_hypotheses_still_reports_no_actionable_uncertainty():
    """Confirms the fix does not overcorrect: when hypotheses genuinely
    EXIST and are all SUPPORTED/REFUTED (real resolution, not a coverage
    gap), NO_ACTIONABLE_UNCERTAINTY remains correct."""
    canonical = SimpleNamespace(is_actionable=True)
    rc = RootCauseAnalysis(status=RootCauseStatus.ESTABLISHED, candidate_hypotheses=[
        _hyp(id="H1", status="SUPPORTED", evidence_strength="VERIFIED"),
        _hyp(id="H2", status="REFUTED", evidence_strength="VERIFIED"),
    ])
    plan, _graph, _ids = plan_investigation_causal(canonical, rc, [])
    assert plan.planner_mode == "NO_ACTIONABLE_UNCERTAINTY"


def test_stage_b_live_unresolved_hypothesis_still_generates_questions():
    """Confirms the fix doesn't affect the genuinely-actionable path at all
    -- unresolved hypotheses still produce real questions, unchanged."""
    canonical = SimpleNamespace(is_actionable=True)
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[
        _hyp(id="H1", status="POSSIBLE"),
    ])
    plan, _graph, _ids = plan_investigation_causal(canonical, rc, [])
    assert plan.planner_mode == "GRAPH_FIRST"
    assert plan.questions
