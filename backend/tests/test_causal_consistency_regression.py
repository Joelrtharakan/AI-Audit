"""Causal-consistency regression suite (production fix: ESTABLISHED/SUPPORTED/
UNKNOWN semantics must be consistent across Root Cause, 5-Why, Investigation
Plan, and CAPA).

Reproduces and locks in the fix for the reported production bug: a hypothesis
whose evidence VERIFIES only an immediate mechanism (e.g. "the notification
service crashed") was being promoted all the way to Root Cause
Status=ESTABLISHED while the 5-Why step for that SAME mechanism still read
UNKNOWN -- an inconsistency between two nodes that should share one
authoritative causal truth (INV-CAUSAL-001/003).

Uses the same offline, deterministic-fallback pipeline harness as
test_golden_20_scenarios.py (get_llm_client patched to None across every
LLM-calling node) so these run fast and never touch a real provider.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.core_synthesis import core_synthesis_node
from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.agent.nodes.report_generator import generate_report_node
from app.agent.nodes.understanding import understand_finding_node
from app.models.agent import InvestigateRequest, RootCauseStatus


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


# TEST 1: Verified observation only -> root cause not established.
@pytest.mark.asyncio
async def test_verified_observation_only_stays_not_established():
    text = "Three employees failed to complete the revised inspection checklist."
    _, report, is_valid, violations = await _run_agent_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED


# TEST 2 (the reported production bug): verified immediate causal mechanism
# -> mechanism SUPPORTED (never ESTABLISHED -- it hasn't reached a systemic
# level), 5-Why must reflect it, and the deeper "why did the mechanism itself
# occur" may correctly remain UNKNOWN.
@pytest.mark.asyncio
async def test_verified_immediate_mechanism_stays_supported_not_established():
    text = (
        "Server logs show the notification message queue service crashed at "
        "08:00, preventing delivery of the revised SOP to all operators."
    )
    state, report, is_valid, violations = await _run_agent_pipeline(text)
    assert is_valid, f"Violations: {violations}"

    rc = report.root_cause
    assert rc.status == RootCauseStatus.SUPPORTED, (
        f"An immediate-mechanism-level hypothesis (verified but not systemic) must cap at "
        f"SUPPORTED, not {rc.status} -- this is the exact reported production bug"
    )

    fw = report.five_why
    assert fw.steps, "5-Why must contain at least one step"
    mechanism_step = next(
        (s for s in fw.steps if "crashed" in (s.answer or "").lower() or "queue" in (s.answer or "").lower()),
        None,
    )
    assert mechanism_step is not None, "5-Why must contain a step reflecting the verified mechanism"
    assert mechanism_step.status in ("VERIFIED", "SUPPORTED"), (
        f"5-Why step reflecting a VERIFIED, SUPPORTED root-cause mechanism must not read "
        f"{mechanism_step.status!r} -- this is the exact reported inconsistency "
        "('Root Cause Status: ESTABLISHED' next to a 5-Why UNKNOWN for the same mechanism)"
    )
    # The deeper "why did the service crash" layer is a genuinely separate,
    # still-unresolved proposition -- staying UNKNOWN there is the correct
    # evidence boundary, not a defect.
    assert any(s.status == "UNKNOWN" for s in fw.steps), (
        "A genuinely deeper, unresolved cause beneath the established mechanism should "
        "still read UNKNOWN -- the chain must not fabricate certainty at every level"
    )


# TEST 3: A control-bypass hypothesis (governance/control-design gap) is
# treated as the audit-terminal (root/systemic-level) cause in QMS terms --
# unlike a bare technical event, there's no further evidence-backed "why"
# beneath it, so it may legitimately reach ESTABLISHED.
@pytest.mark.asyncio
async def test_complete_causal_chain_can_reach_established():
    text = (
        "Audit trail logs confirm the mandatory dual-approval validation "
        "control was disabled by admin before the duplicate payment was released."
    )
    _, report, is_valid, violations = await _run_agent_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status in (RootCauseStatus.SUPPORTED, RootCauseStatus.ESTABLISHED)


# TEST 4: A REPORTED-only causal claim (someone's account, not objectively
# verified) can never reach ESTABLISHED.
@pytest.mark.asyncio
async def test_reported_only_claim_cannot_become_established():
    text = "The shift supervisor stated the calibration record for BAL-014 was not maintained, which they believe caused the deviation."
    _, report, is_valid, violations = await _run_agent_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status != RootCauseStatus.ESTABLISHED


# TEST 6: Verified evidence directly contradicting a reported explanation
# must reject that explanation as root cause (never SUPPORTED/ESTABLISHED).
@pytest.mark.asyncio
async def test_verified_contradiction_rejects_hypothesis():
    text = (
        "The operator stated the balance BAL-014 display was blank. Equipment "
        "diagnostic logs confirm continuous active power and error-free operation "
        "throughout the shift."
    )
    _, report, is_valid, violations = await _run_agent_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status not in (RootCauseStatus.SUPPORTED, RootCauseStatus.ESTABLISHED)


# TEST 7 & 8: Cross-node consistency, general property -- across every golden
# scenario, no SUPPORTED/ESTABLISHED leading hypothesis may have its own
# mechanism reflected as UNKNOWN in the 5-Why chain (INV-CAUSAL-001/003 as a
# property test, not just the single reproduction case above).
@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "Server logs show the notification message queue service crashed at 08:00, preventing delivery of the revised SOP to all operators.",
    "Audit trail logs confirm the mandatory dual-approval validation control was disabled by admin before the duplicate payment was released.",
    "Training on revised SOP-OPS-014 was mandatory before August 5. LMS training logs confirm no operators completed the training before performing the task on August 7.",
    "Duplicate payment of ₹125,000 occurred because the automated duplicate detection rule was disabled in the ERP configuration.",
])
async def test_supported_or_established_hypothesis_never_unknown_downstream(text):
    _, report, is_valid, violations = await _run_agent_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    rc = report.root_cause
    if rc.status not in (RootCauseStatus.SUPPORTED, RootCauseStatus.ESTABLISHED):
        pytest.skip("root cause not promoted in this scenario -- nothing to check")
    assert not any(f"[INV-CAUSAL-00{n}]" in v for v in violations for n in (1, 3)), (
        f"cross-node causal consistency violated: {violations}"
    )
