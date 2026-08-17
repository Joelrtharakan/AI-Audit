"""Tests for the deterministic degraded-mode fallbacks that fire when the
LLM call in core_synthesis_node fails or returns unusable JSON.

Per the failure-handling requirement: a fallback must be evidence-
conservative, not content-generative — it must never fabricate a generic
causal mechanism, generic CAPA category, or generic 5-Why question just to
fill a field. These tests check that behavior directly on the fallback
builders (app/agent/nodes/five_why_fallback.py and
app/agent/nodes/plan_investigation_fallback.py), independent of any specific
finding, equipment ID, or domain.
"""

from __future__ import annotations

from app.agent.nodes.five_why_fallback import build_deterministic_five_why
from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
from app.models.agent import EvidenceItem, EvidenceStatus


def test_five_why_fallback_stops_at_one_step_without_a_mechanism():
    """No REPORTED claim in the ledger means no mechanism to explain next —
    the chain must honestly stop at the observation instead of manufacturing
    a generic 'why were controls missing' question."""
    finding_text = "The required weekly review record was incomplete for the period."
    ledger = [
        EvidenceItem(claim="the required weekly review record was incomplete", source="finding_text", status=EvidenceStatus.VERIFIED),
    ]
    fw = build_deterministic_five_why(finding_text, ledger)
    assert len(fw.steps) == 1
    assert "EVIDENCE BOUNDARY" in fw.status_note or "DEGRADED" in fw.status_note


def test_five_why_fallback_never_asks_about_reporting_behavior():
    finding_text = "The required weekly review record was incomplete. The reviewer confirmed the review was missed."
    ledger = [
        EvidenceItem(claim="the required weekly review record was incomplete", source="finding_text", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="the reviewer confirmed the review was missed", source="finding_text", status=EvidenceStatus.REPORTED),
    ]
    fw = build_deterministic_five_why(finding_text, ledger)
    joined_questions = " ".join(s.question.lower() for s in fw.steps)
    assert "report" not in joined_questions
    # observation -> mechanism (from the reported statement) -> honest
    # evidence-boundary stop on why the mechanism itself occurred.
    assert len(fw.steps) == 3
    assert fw.steps[-1].status == "UNKNOWN"
    assert "EVIDENCE BOUNDARY" in fw.status_note or "DEGRADED MODE" in fw.status_note
    assert "NOT ESTABLISHED FROM AVAILABLE EVIDENCE" in fw.steps[-1].answer


def test_five_why_fallback_never_manufactures_recurrence_step():
    """Even when the finding mentions a prior CAPA, degraded mode must not
    bolt an extra unrelated step onto a chain that already hit its evidence
    boundary."""
    finding_text = "The required weekly review record was incomplete, similar to a finding from a previous audit."
    ledger = [
        EvidenceItem(claim="the required weekly review record was incomplete", source="finding_text", status=EvidenceStatus.VERIFIED),
    ]
    fw = build_deterministic_five_why(finding_text, ledger)
    assert len(fw.steps) == 1
    assert "recurrence" not in fw.steps[0].question.lower()
    assert "previous corrective action" not in fw.steps[0].answer.lower()


def test_critical_case_missed_activity_confirmed_by_person():
    """The exact class of finding called out as the critical test: a bare
    'activity was missed, and someone confirmed it was missed' — the
    degraded chain must reach the mechanism (not stop at WHY-1) and then
    honestly stop at NOT ESTABLISHED for why the mechanism occurred, never
    fabricating a specific cause."""
    finding_text = "The required activity was missed. The technician confirmed it was missed."
    ledger = [
        EvidenceItem(claim="the required activity was not completed", source="finding_text", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="the technician confirmed the activity was missed", source="finding_text", status=EvidenceStatus.REPORTED),
    ]
    fw = build_deterministic_five_why(finding_text, ledger)
    assert len(fw.steps) == 3
    assert fw.steps[0].status == "VERIFIED"
    assert fw.steps[1].status == "REPORTED"
    assert fw.steps[2].status == "UNKNOWN"
    assert "NOT ESTABLISHED FROM AVAILABLE EVIDENCE" in fw.steps[2].answer

    hyps, _ = build_deterministic_investigation_plan(finding_text, ledger)
    hyp_names = {h.name for h in hyps}
    # None of the generic universal trope names may appear unless the
    # deterministic mechanism-aware branch actually names them itself.
    forbidden = {"EXECUTION_OMISSION", "DOCUMENTATION_OMISSION", "TRAINING_FAILURE", "COMMUNICATION_GAP", "SUPERVISION_FAILURE"}
    assert not (hyp_names & forbidden)


def test_investigation_plan_fallback_areas_are_not_generic_universal_categories():
    """The fallback InvestigationPlan.areas must be derived from the
    hypotheses actually generated for this finding, never a fixed
    ["Execution Verification", "Documentation Control", "Workstation
    Setup"] list applied regardless of what the finding is about."""
    finding_text = "The required weekly review record was incomplete for the period."
    ledger = [
        EvidenceItem(claim="the required weekly review record was incomplete", source="finding_text", status=EvidenceStatus.VERIFIED),
    ]
    hyps, plan = build_deterministic_investigation_plan(finding_text, ledger)
    assert plan.areas
    assert plan.areas != ["Execution Verification", "Documentation Control", "Workstation Setup"]
    # This finding has no reported explanation and no conflict, so it
    # correctly lands in the General/Unresolved branch: zero hypotheses,
    # areas/questions derived from the finding's own subject instead
    # (Section 21: zero hypotheses is a valid, often correct output).
    assert hyps == []
    area_words = {w for a in plan.areas for w in a.lower().split()}
    assert "review" in area_words
