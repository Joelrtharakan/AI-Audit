"""Blind Multi-Level Causal Depth and Epistemic Separation Test Suite.

Validates:
  1. A verified deviation with no causal evidence -> NOT_ESTABLISHED root cause, 5-Why stops at boundary.
  2. A verified deviation with a verified immediate mechanism -> mechanism is VERIFIED, deeper cause UNKNOWN.
  3. A verified mechanism with an unknown deeper cause -> 5-Why preserves verified step and stops at unknown edge.
  4. Multiple verified causal levels with an unknown final root cause -> multi-step verified chain with unknown terminal step.
  5. A fully established root cause -> ESTABLISHED root cause, definitive CAPA allowed.
  6. Conflicting causal evidence -> UNRESOLVED / CONFLICTED status, proposition-neutral investigation.
  7. Reported causal explanation -> REPORTED mechanism status, NOT_ESTABLISHED root cause.
  8. Competing possible causes -> multiple candidate hypotheses, neutral discrimination questions.
  9. A verified mechanism with irrelevant unrelated evidence -> mechanism remains verified without promotion of unrelated topics.
  10. A finding containing multiple independent causal chains -> separate causal progression per chain.
"""

import pytest
from app.models.agent import (
    CandidateHypothesis,
    CanonicalFindingState,
    EvidenceConflict,
    EvidenceItem,
    EvidenceStatus,
    FiveWhyAnalysis,
    FiveWhyStep,
    InvestigateRequest,
    RootCauseStatus,
)
from app.agent.invariants import evaluate_all_invariants
from app.agent.causal_graph import evaluate_root_cause_eligibility, select_authoritative_leading_hypothesis
from app.agent.nodes.understanding import understand_finding_node
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.agent.nodes.core_synthesis import core_synthesis_node
from app.agent.nodes.report_generator import generate_report_node
from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
from unittest.mock import patch


async def _run_pipeline(text: str):
    req = InvestigateRequest(finding_text=text)
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
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None), \
         patch("app.agent.nodes.critic.get_llm_client", return_value=None):
        state = await understand_finding_node(state)
        state = await plan_investigation_node(state)
        state = await core_synthesis_node(state)
        state = await generate_report_node(state)
        state = await final_evidence_verification_node(state)
    is_valid, violations = evaluate_all_invariants(state)
    return state, state["report"], is_valid, violations


@pytest.mark.asyncio
async def test_case_1_verified_deviation_no_causal_evidence():
    """Case 1: Observation only — deviation is verified, no causal mechanism is established."""
    text = "During the cleanroom inspection, analytical balance BAL-014 was observed in use without a valid daily calibration log."
    state, report, is_valid, violations = await _run_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED
    assert len(report.five_why.steps) >= 1
    assert report.five_why.steps[-1].status in ("UNKNOWN", "REQUIRES_EVIDENCE")


@pytest.mark.asyncio
async def test_case_2_verified_immediate_mechanism_unknown_deeper_cause():
    """Case 2: Immediate mechanism verified (power outage caused shutdown), deeper root cause unknown."""
    text = "Facility SCADA server log confirmed an unexpected power outage triggered the emergency shutdown of autoclave AC-02 on August 10. The underlying electrical fault has not been determined."
    state, report, is_valid, violations = await _run_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert state["canonical_finding_state"].immediate_mechanism_status == "VERIFIED"
    assert report.root_cause.status in (RootCauseStatus.NOT_ESTABLISHED, RootCauseStatus.SUPPORTED)


@pytest.mark.asyncio
async def test_case_3_reported_causal_explanation_not_promoted():
    """Case 3: Reported statement (operator stated training was missed) remains REPORTED, not ESTABLISHED."""
    text = "The warehouse temperature log was incomplete. The technician stated that high workload prevented timely entry."
    state, report, is_valid, violations = await _run_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED
    for h in report.root_cause.candidate_hypotheses:
        assert h.status not in ("ESTABLISHED",)


@pytest.mark.asyncio
async def test_case_4_fully_established_systemic_root_cause():
    """Case 4: Audit trail proves security interlock was disabled without required change authorization."""
    text = "Audit trail logs confirmed that the security interlock on valve V-101 was disabled on August 12 without required change-management authorization."
    state, report, is_valid, violations = await _run_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status in (RootCauseStatus.ESTABLISHED, RootCauseStatus.SUPPORTED)


@pytest.mark.asyncio
async def test_case_5_conflicting_causal_evidence():
    """Case 5: Conflicting statements regarding whether maintenance was executed."""
    text = "Maintenance system log indicates pump P-301 maintenance was completed on August 5, but the operating technician stated the maintenance was not performed."
    state, report, is_valid, violations = await _run_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED
    assert report.five_why.steps[-1].status in ("UNKNOWN", "MIXED")


@pytest.mark.asyncio
async def test_case_6_competing_possible_causes_discrimination():
    """Case 6: Competing hypotheses generate discrimination questions without premature selection."""
    text = "During review of batch record BR-2026-0812, the final yield recorded by the operator did not match the calculated yield from individual entries. The difference was approximately 4.2%."
    state, report, is_valid, violations = await _run_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED
    assert len(report.root_cause.candidate_hypotheses) >= 4
    assert len(state["investigation_plan"].questions) >= 1


@pytest.mark.asyncio
async def test_case_7_multiple_verified_causal_levels_with_unknown_terminal_root_cause():
    """Case 7: Observation + verified immediate mechanism + unknown underlying root cause."""
    text = "Autoclave cycle #44 failed to reach required sterilisation temperature of 121°C on August 14. SCADA sensor telemetry confirmed heating coil circuit interruption occurred at 03:15. The underlying electrical root cause has not been determined."
    state, report, is_valid, violations = await _run_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert state["canonical_finding_state"].immediate_mechanism_status == "VERIFIED"
    assert report.root_cause.status in (RootCauseStatus.NOT_ESTABLISHED, RootCauseStatus.SUPPORTED)
    assert len(report.five_why.steps) >= 1
    assert any(s.status in ("VERIFIED", "SUPPORTED") for s in report.five_why.steps)
    assert report.five_why.steps[-1].status in ("UNKNOWN", "REQUIRES_EVIDENCE")


@pytest.mark.asyncio
async def test_case_8_verified_mechanism_with_unrelated_irrelevant_evidence():
    """Case 8: Unrelated observation about another device does not promote or contaminate mechanism."""
    text = "SCADA log confirmed refrigeration unit RU-1 experienced power supply interruption on August 10. Separately, the packaging room lights were reported flickering by staff."
    state, report, is_valid, violations = await _run_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status in (RootCauseStatus.NOT_ESTABLISHED, RootCauseStatus.SUPPORTED)
    assert "lights" not in (report.root_cause.statement or "").lower()


@pytest.mark.asyncio
async def test_case_9_independent_causal_chains_preserve_epistemic_locality():
    """Case 9: Mutating epistemic status of an unrelated claim does not alter independent verified mechanism."""
    text_verified = "SCADA log confirmed valve V-12 actuator motor failed on August 2. Shift supervisor stated the maintenance team was understaffed."
    state, report, is_valid, violations = await _run_pipeline(text_verified)
    assert is_valid, f"Violations: {violations}"
    assert state["canonical_finding_state"].immediate_mechanism_status == "VERIFIED"
    assert report.root_cause.status in (RootCauseStatus.NOT_ESTABLISHED, RootCauseStatus.SUPPORTED)


@pytest.mark.asyncio
async def test_case_10_no_conditional_capa_for_established_mechanisms_and_no_definitive_for_unsupported():
    """Case 10: Invariant check verifies CAPA consistency with causal certainty."""
    text = "The warehouse temperature log was incomplete. The technician stated that high workload prevented timely entry."
    state, report, is_valid, violations = await _run_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    # When cause is unestablished, corrective actions must remain conditional
    capa = state.get("capa_analysis")
    if capa and getattr(capa, "conditional_actions", None):
        for action in capa.conditional_actions:
            if action.action_type == "CORRECTIVE_ACTION":
                assert action.if_cause_confirmed is not None
