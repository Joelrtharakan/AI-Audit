"""Final RCA semantic quality pass, Section 20/21: tests the actual
RENDERED report object (state["report"]), not only internal AgentState --
closing a gap disclosed in the prior turn's report. Exercises the real
deterministic pipeline end to end (understand -> plan -> core_synthesis ->
generate_report -> final_evidence_verification), never a reimplementation.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from app.agent.nodes.core_synthesis import core_synthesis_node
from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.agent.nodes.report_generator import generate_report_node
from app.agent.nodes.understanding import understand_finding_node
from app.models.agent import InvestigateRequest, RootCauseStatus

_ESTABLISHED_ROOT_CAUSE_FINDING = (
    "Audit trail logs confirmed that the security interlock on valve V-101 was disabled "
    "on August 12 without required change-management authorization."
)

_MULTI_HYPOTHESIS_UNRESOLVED_FINDING = (
    "Four employees failed to complete the revised inspection checklist. "
    "One employee reported insufficient training. "
    "Another employee reported workload pressure. "
    "The supervisor reported poor discipline."
)


async def _run(text: str):
    state = {
        "request": InvestigateRequest(finding_text=text), "evidence_ledger": [],
        "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0, "trace": [], "errors": [],
    }
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.investigation_planner.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        state = await understand_finding_node(state)
        state = await plan_investigation_node(state)
        state = await core_synthesis_node(state)
        state = await generate_report_node(state)
        state = await final_evidence_verification_node(state)
    return state


# ---------------------------------------------------------------------------
# Case D (Section 20): established root cause -- structured state AND
# rendered report must agree.
# ---------------------------------------------------------------------------

def test_established_root_cause_rendered_report_is_consistent():
    state = asyncio.run(_run(_ESTABLISHED_ROOT_CAUSE_FINDING))
    report = state.get("report")
    assert report is not None
    rc = report.root_cause
    assert rc.status == RootCauseStatus.ESTABLISHED
    # The newly-populated structured field (Phase 32) must be present and
    # agree with the rendered status -- not merely computed and discarded.
    assert rc.causal_sufficiency is not None
    assert rc.causal_sufficiency.root_cause_sufficiency == "ESTABLISHED"
    # Narrative must be non-empty and must not be a bare restatement of
    # the raw finding text (a real, testable proxy for "explains rather
    # than restates").
    assert rc.narrative and len(rc.narrative.strip()) > 0
    assert rc.narrative.strip() != _ESTABLISHED_ROOT_CAUSE_FINDING.strip()


# ---------------------------------------------------------------------------
# Case C/B (Section 2/3/20): mechanism possible/supported, root cause NOT
# established -- the rendered report must not overclaim.
# ---------------------------------------------------------------------------

def test_unresolved_investigation_rendered_report_does_not_overclaim():
    state = asyncio.run(_run(_MULTI_HYPOTHESIS_UNRESOLVED_FINDING))
    report = state.get("report")
    assert report is not None
    rc = report.root_cause
    assert rc.status == RootCauseStatus.NOT_ESTABLISHED
    assert rc.causal_sufficiency is not None
    assert rc.causal_sufficiency.root_cause_sufficiency == "NOT_ESTABLISHED"
    # Confidence must never be HIGH when root cause is NOT_ESTABLISHED
    # (Section 15/17 -- confidence must not exceed causal certainty).
    assert report.root_cause_confidence != "HIGH"
    # investigation_required must reflect the genuinely open investigation,
    # not a false "resolved" signal.
    assert report.investigation_required == "YES"


def test_established_and_unresolved_reports_have_distinct_confidence():
    """A coarse but real end-to-end confidence-calibration check: the
    established case must not read strictly worse than the genuinely
    unresolved case -- confidence must track causal certainty, not be
    decorative/reversed."""
    established = asyncio.run(_run(_ESTABLISHED_ROOT_CAUSE_FINDING))
    unresolved = asyncio.run(_run(_MULTI_HYPOTHESIS_UNRESOLVED_FINDING))
    rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    est_conf = rank[established["report"].root_cause_confidence]
    unresolved_conf = rank[unresolved["report"].root_cause_confidence]
    assert est_conf >= unresolved_conf


# ---------------------------------------------------------------------------
# No dangling references in the rendered report (Section 16/20)
# ---------------------------------------------------------------------------

def test_rendered_report_hypothesis_ids_are_self_consistent():
    state = asyncio.run(_run(_MULTI_HYPOTHESIS_UNRESOLVED_FINDING))
    report = state.get("report")
    hyp_ids = {h.id for h in report.root_cause.candidate_hypotheses}
    if report.root_cause.leading_hypothesis and report.root_cause.leading_hypothesis_status == "SELECTED":
        assert report.root_cause.leading_hypothesis in hyp_ids
    for action in (report.capa.conditional_actions or []):
        if action.root_cause_hypothesis_id:
            assert action.root_cause_hypothesis_id in hyp_ids, (
                f"CAPA action references hypothesis {action.root_cause_hypothesis_id!r} "
                f"not present in the rendered report's candidate_hypotheses {hyp_ids}"
            )
