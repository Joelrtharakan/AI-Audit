"""Common-factor causal reasoning regression suite (production improvement:
a shared system/process across multiple independently-affected locations is
a strong investigation LEAD, but must never be asserted as a proven cause).

Reproduces and locks in the fix for the reported scenario:

    C1: "Outdated versions of the same controlled procedure were found at
         workstations in Production, Warehouse, and Quality Control."
    C2: "All three departments use the same document-control system."

Previously produced "No causal hypotheses established from available
evidence" (too conservative). Now generates a single POSSIBLE,
evidence_strength=INDICATIVE hypothesis naming the shared document-control
system as a candidate mechanism, with provenance to both claims, and an
investigation plan that specifically tests that hypothesis -- while root
cause stays NOT_ESTABLISHED and the hypothesis is structurally barred from
ever reaching SUPPORTED/ESTABLISHED on pattern evidence alone.

Uses the same offline, deterministic-fallback pipeline harness as
test_golden_20_scenarios.py.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from app.agent.common_factor import detect_common_factor
from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.core_synthesis import core_synthesis_node
from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.agent.nodes.report_generator import generate_report_node
from app.agent.nodes.understanding import understand_finding_node
from app.models.agent import EvidenceItem, EvidenceStatus, InvestigateRequest, RootCauseStatus


async def _run_agent_pipeline(finding_text: str):
    req = InvestigateRequest(finding_text=finding_text)
    state = {
        "request": req,
        "evidence_ledger": [],
        "iteration_count": 0,
        "tool_call_count": 0,
        "critic_iteration": 0,
        "trace": [],
        "errors": [],
    }
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.investigation_planner.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        state = await understand_finding_node(state)
        state = await plan_investigation_node(state)
        state = await core_synthesis_node(state)
        state = await generate_report_node(state)
        state = await final_evidence_verification_node(state)
    is_valid, violations = evaluate_all_invariants(state)
    return state, state.get("report"), is_valid, violations


_FINDING_TEXT = (
    "Outdated versions of the same controlled procedure were found at workstations in "
    "Production, Warehouse, and Quality Control. All three departments use the same "
    "document-control system."
)


# 1. Three departments + same system -- the exact reported scenario.
@pytest.mark.asyncio
async def test_common_factor_generates_possible_hypothesis_not_no_hypotheses():
    state, report, is_valid, violations = await _run_agent_pipeline(_FINDING_TEXT)
    assert is_valid, f"Violations: {violations}"
    rc = report.root_cause

    assert rc.candidate_hypotheses, "expected a common-factor hypothesis, got none ('too conservative')"
    h1 = rc.candidate_hypotheses[0]
    assert h1.status == "POSSIBLE"
    assert h1.evidence_strength == "INDICATIVE"
    assert "document-control system" in h1.statement.lower()
    # Specific and testable, not a generic bucket label.
    assert h1.statement.lower() not in ("document control problem.", "process failure.", "training issue.")
    assert len(h1.statement) > 40


# Provenance: H1 must trace back to BOTH claims, never presented as proven
# by them (evidence_strength stays INDICATIVE, not VERIFIED).
@pytest.mark.asyncio
async def test_common_factor_hypothesis_provenance():
    state, report, is_valid, violations = await _run_agent_pipeline(_FINDING_TEXT)
    assert is_valid, f"Violations: {violations}"
    h1 = report.root_cause.candidate_hypotheses[0]
    assert len(h1.supporting_claim_ids) >= 2
    assert h1.evidence_strength != "VERIFIED"


# Root cause status must never collapse into SUPPORTED/ESTABLISHED on a
# common factor alone.
@pytest.mark.asyncio
async def test_common_factor_root_cause_not_prematurely_established():
    state, report, is_valid, violations = await _run_agent_pipeline(_FINDING_TEXT)
    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED


# The investigation plan must TEST the hypothesis, not ask generic
# unrelated questions.
@pytest.mark.asyncio
async def test_common_factor_investigation_plan_tests_hypothesis():
    state, report, is_valid, violations = await _run_agent_pipeline(_FINDING_TEXT)
    assert is_valid, f"Violations: {violations}"
    h1 = report.root_cause.candidate_hypotheses[0]
    inv = report.investigation
    assert inv and inv.questions
    assert any(q.hypothesis_tested == h1.id for q in inv.questions), (
        "no investigation question targets the common-factor hypothesis"
    )
    # Not the generic, hypothesis-blind questions this used to fall back to.
    generic_phrases = ("what requirement applied", "what objective evidence demonstrates the finding")
    for q in inv.questions:
        assert not any(g in q.question.lower() for g in generic_phrases)


# 5-Why must not invent causal steps -- stays at the evidence boundary.
@pytest.mark.asyncio
async def test_common_factor_five_why_stops_at_boundary():
    state, report, is_valid, violations = await _run_agent_pipeline(_FINDING_TEXT)
    assert is_valid, f"Violations: {violations}"
    fw = report.five_why
    assert fw.steps
    assert fw.steps[-1].status == "UNKNOWN"
    assert not any(s.status in ("VERIFIED", "SUPPORTED") for s in fw.steps)


# 7. Common factor present but UNRELATED to the finding's actual subject
# (only one department named) must not fire -- a single signal alone is
# not a strong enough lead.
def test_common_factor_not_detected_with_single_department():
    ledger = [
        EvidenceItem(
            claim="An outdated procedure copy was found at a workstation in Production.",
            status=EvidenceStatus.VERIFIED, source="AUDITOR_FINDING",
        ),
        EvidenceItem(
            claim="Production uses the same document-control system as other sites.",
            status=EvidenceStatus.VERIFIED, source="AUDITOR_FINDING",
        ),
    ]
    result = detect_common_factor(ledger)
    assert not result.detected


# 9. Single observation with no common factor -> no hypothesis manufactured.
def test_common_factor_not_detected_without_shared_factor_claim():
    ledger = [
        EvidenceItem(
            claim="Outdated procedure copies were found in Production and Warehouse.",
            status=EvidenceStatus.VERIFIED, source="AUDITOR_FINDING",
        ),
    ]
    result = detect_common_factor(ledger)
    assert not result.detected


# 8. When objective causal proof for a DIFFERENT, more specific mechanism
# exists, that takes priority over the generic common-factor fallback (the
# common-factor branch only fires when nothing more specific matched).
@pytest.mark.asyncio
async def test_specific_evidence_takes_priority_over_common_factor():
    text = (
        "Audit trail logs confirm the mandatory dual-approval validation control was "
        "disabled by admin before the duplicate payment was released. Both Finance and "
        "Procurement use the same payment approval system."
    )
    state, report, is_valid, violations = await _run_agent_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    rc = report.root_cause
    assert rc.candidate_hypotheses
    assert not any(h.name == "SHARED_SYSTEM_COMMON_FACTOR" for h in rc.candidate_hypotheses), (
        "generic common-factor hypothesis should not override a more specific, directly-evidenced mechanism"
    )
