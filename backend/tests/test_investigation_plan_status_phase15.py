"""Phase 15 Section 9: explicit, machine-readable NO_ACTIONABLE_UNCERTAINTY
state on InvestigationPlan.

Audit finding: the deterministic investigation planner
(build_deterministic_investigation_plan, app/agent/nodes/
plan_investigation_fallback.py) is explicitly guaranteed to NEVER return an
empty question list (its own "Section 8" comment) -- every actionable
finding always receives at least one question through its ~2,900-line
template/keyword decision tree. The ONLY runtime path that already,
legitimately produces zero investigation questions is
plan_investigation_node's non-actionable fast path (a finding whose
CanonicalFindingState.is_actionable is False) in
app/agent/nodes/investigation_planner.py.

A general graph-driven "there is genuinely nothing left to investigate"
judgment (Section 9's broader ask) was NOT implemented this phase -- see the
Phase 15 report. What Phase 15 adds for real: the one case that already
produces zero questions now says so explicitly via
InvestigationPlan.status="NO_ACTIONABLE_UNCERTAINTY" instead of leaving a
caller to infer meaning from an empty list, and INV-INVEST-013 enforces
that this claim can never co-occur with populated questions.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.models.agent import InvestigationPlan, InvestigationQuestion


def test_default_status_is_questions_generated():
    assert InvestigationPlan().status == "QUESTIONS_GENERATED"


def test_non_actionable_finding_produces_no_actionable_uncertainty_status():
    class _FakeCanonical:
        is_actionable = False

    state = {
        "request": type("R", (), {"finding_text": "irrelevant"})(),
        "canonical_finding_state": _FakeCanonical(),
        "trace": [], "errors": [],
    }
    result = asyncio.run(plan_investigation_node(state))
    plan = result["investigation_plan"]
    assert plan.status == "NO_ACTIONABLE_UNCERTAINTY"
    assert plan.questions == []


def test_invariant_passes_when_status_and_content_agree():
    empty_plan = InvestigationPlan(status="NO_ACTIONABLE_UNCERTAINTY", questions=[])
    ok, violations = evaluate_all_invariants({"investigation_plan": empty_plan})
    assert not any("INV-INVEST-013" in v for v in violations)

    populated_plan = InvestigationPlan(
        status="QUESTIONS_GENERATED",
        questions=[InvestigationQuestion(question="What evidence applies?", purpose="p", evidence="e")],
    )
    ok, violations = evaluate_all_invariants({"investigation_plan": populated_plan})
    assert not any("INV-INVEST-013" in v for v in violations)


def test_invariant_fails_closed_on_status_content_mismatch():
    inconsistent_plan = InvestigationPlan(
        status="NO_ACTIONABLE_UNCERTAINTY",
        questions=[InvestigationQuestion(question="What evidence applies?", purpose="p", evidence="e")],
    )
    ok, violations = evaluate_all_invariants({"investigation_plan": inconsistent_plan})
    assert any("INV-INVEST-013" in v for v in violations)


def test_invariant_does_not_flag_default_empty_fixture():
    """A bare InvestigationPlan() -- the extremely common pattern across
    this test suite for constructing a minimal state fixture unrelated to
    investigation planning -- must not be treated as an affirmative,
    inconsistent claim."""
    ok, violations = evaluate_all_invariants({"investigation_plan": InvestigationPlan()})
    assert not any("INV-INVEST-013" in v for v in violations)
