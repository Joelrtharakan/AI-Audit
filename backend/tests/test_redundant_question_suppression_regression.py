"""Regression suite for redundant investigation-question suppression.

Locks in a fix in plan_investigation_fallback.py: `req_status` was fetched
from canonical_state.requirement_status with a comment stating "if
requirement is already VERIFIED -> do NOT ask if requirement exists" but the
actual gate never checked it -- Q_GOVERNING_REQUIREMENT was generated even
when the requirement was already verified.

(A broader generalization -- flagging any confirmation-style question whose
answer a VERIFIED evidence item already contains -- was attempted as an
invariant, INV-INVEST-UNCERTAINTY-003, but reverted: plain word-overlap
against VERIFIED evidence text produced false positives on legitimate
mechanism-discrimination questions, since a comparison finding's own VERIFIED
deviation statement ("X did not match Y") naturally shares vocabulary with
any follow-up question about that same subject. The narrower, already-proven
pattern in _check_investigation_targets_unresolved -- comparing only against
the specific leading-hypothesis statement -- doesn't have this problem and
was left untouched.)
"""

from __future__ import annotations

from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
from app.models.agent import CanonicalFindingState


def _plan_question_ids(requirement_status: str, text: str) -> list[str]:
    canonical = CanonicalFindingState(raw_finding=text, observed_deviation=text, requirement_status=requirement_status)
    _hyps, plan = build_deterministic_investigation_plan(text, [], canonical_state=canonical)
    return [q.question_id for q in plan.questions]


def test_governing_requirement_question_suppressed_when_verified():
    text = "The equipment calibration was overdue by three months."
    assert "Q_GOVERNING_REQUIREMENT" not in _plan_question_ids("VERIFIED", text)


def test_governing_requirement_question_generated_when_unknown():
    text = "The equipment calibration was overdue by three months."
    assert "Q_GOVERNING_REQUIREMENT" in _plan_question_ids("UNKNOWN", text)


def test_governing_requirement_question_generated_across_domain_when_stated_but_unverified():
    """A different domain (financial vs. equipment) with requirement_status
    STATED (not VERIFIED) must still ask -- confirms the gate keys off the
    actual status value, not merely "any non-UNKNOWN status"."""
    text = "The supplier invoice was paid without a three-way match against the purchase order."
    assert "Q_GOVERNING_REQUIREMENT" in _plan_question_ids("STATED", text)
