"""20 Golden Scenarios Regression Test Suite (Sections 28-29).

Validates structured state, evidence provenance, financial correctness,
causal sufficiency, risk rationale, and 5-Why discipline across all 20 canonical finding archetypes:
1. Observation only
2. Conflicting human statements
3. Verified training failure
4. Verified control bypass
5. Verified technical failure
6. Missing evidence
7. Conflicting objective records
8. Temporal sequence without causation
9. Duplicate payment ₹125,000
10. Fully recovered ₹250,000 payment
11. Partially recovered payment
12. Cost without direct financial loss
13. Previous CAPA recurrence
14. Proven ineffective CAPA
15. Multiple competing causes
16. Systemic control failure
17. Financial + verified root cause
18. No-financial finding
19. Contradicted human explanation
20. Multi-event systemic financial finding
"""

import pytest
from app.models.agent import (
    InvestigateRequest,
    RootCauseStatus,
    CostEvidenceStatus,
    CanonicalFindingState,
)
from app.agent.nodes.understanding import understand_finding_node
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.agent.nodes.core_synthesis import core_synthesis_node
from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
from app.agent.nodes.report_generator import generate_report_node
from app.agent.invariants import evaluate_all_invariants
from unittest.mock import patch


async def _run_agent_pipeline(finding_text: str):
    """Helper to run the full finding investigation pipeline."""
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


# Scenario 1: Observation only
@pytest.mark.asyncio
async def test_scenario_01_observation_only():
    text = "Three employees failed to complete the revised inspection checklist."
    state, report, is_valid, violations = await _run_agent_pipeline(text)

    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED
    assert report.root_cause.leading_hypothesis_status in ("TIED", "NONE")
    assert not report.five_why.is_complete
    assert report.cost_impact is None or not report.cost_impact.cost_factor_detected


# Scenario 2: Conflicting human statements
@pytest.mark.asyncio
async def test_scenario_02_conflicting_human_statements():
    text = "The technician stated the SOP was unavailable. The shift supervisor reported the SOP was placed on the station."
    state, report, is_valid, violations = await _run_agent_pipeline(text)

    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED
    assert report.evidence_conflicts or any("conflict" in str(h.statement).lower() or h.status == "POSSIBLE" for h in report.root_cause.candidate_hypotheses)


# Scenario 3: Verified training failure
@pytest.mark.asyncio
async def test_scenario_03_verified_training_failure():
    text = "Training on revised SOP-OPS-014 was mandatory before August 5. LMS training logs confirm no operators completed the training before performing the task on August 7."
    state, report, is_valid, violations = await _run_agent_pipeline(text)

    assert is_valid, f"Violations: {violations}"
    # Should establish or strongly support training root cause
    assert report.root_cause.status in (RootCauseStatus.ESTABLISHED, RootCauseStatus.SUPPORTED)
    assert any("TRAINING" in h.name or "training" in h.statement.lower() for h in report.root_cause.candidate_hypotheses)


# Scenario 4: Verified control bypass
@pytest.mark.asyncio
async def test_scenario_04_verified_control_bypass():
    text = "Audit trail logs confirm the mandatory dual-approval validation control was disabled by admin before the duplicate payment was released."
    state, report, is_valid, violations = await _run_agent_pipeline(text)

    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status in (RootCauseStatus.ESTABLISHED, RootCauseStatus.SUPPORTED)


# Scenario 5: Verified technical failure
@pytest.mark.asyncio
async def test_scenario_05_verified_technical_failure():
    text = "Server logs show the notification message queue service crashed at 08:00, preventing delivery of the revised SOP to all operators."
    state, report, is_valid, violations = await _run_agent_pipeline(text)

    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status in (RootCauseStatus.ESTABLISHED, RootCauseStatus.SUPPORTED)


# Scenario 6: Missing evidence
@pytest.mark.asyncio
async def test_scenario_06_missing_evidence():
    text = "Maintenance record MR-402 was unavailable during the audit. It is unknown if the calibration was executed."
    state, report, is_valid, violations = await _run_agent_pipeline(text)

    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED
    # Invariant check: unavailable document contents are never asserted
    assert "showed" not in str(report.root_cause.statement).lower()


# Scenario 7: Conflicting objective records
@pytest.mark.asyncio
async def test_scenario_07_conflicting_objective_records():
    text = "Server log states notification was delivered at 09:00, but the terminal access log shows the terminal was offline until 11:00."
    state, report, is_valid, violations = await _run_agent_pipeline(text)

    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED


# Scenario 8: Temporal sequence without causation
@pytest.mark.asyncio
async def test_scenario_08_temporal_sequence_without_causation():
    text = "The software update was installed on Monday. On Tuesday, an operator misread the scale."
    state, report, is_valid, violations = await _run_agent_pipeline(text)

    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED


# Scenario 9: Duplicate payment ₹125,000
@pytest.mark.asyncio
async def test_scenario_09_duplicate_payment():
    text = "During the audit, duplicate payment of ₹125,000 to a supplier was identified."
    state, report, is_valid, violations = await _run_agent_pipeline(text)

    assert is_valid, f"Violations: {violations}"
    assert report.cost_impact is not None
    assert report.cost_impact.cost_factor_detected is True
    assert report.cost_impact.financial_amount.amount == 125000.0
    assert report.cost_impact.gross_exposure == 125000.0
    assert report.cost_impact.actual_loss_status == "NOT_ESTABLISHED"
    assert report.cost_impact.recoverability_status == "REQUIRES_VERIFICATION"
    assert "₹125,000" in report.impact_assessment.potential_effect


# Scenario 10: Fully recovered ₹250,000 payment
@pytest.mark.asyncio
async def test_scenario_10_fully_recovered_payment():
    text = "Duplicate payment of ₹250,000 was identified. Bank credit memo confirms full refund of ₹250,000 was received."
    state, report, is_valid, violations = await _run_agent_pipeline(text)

    assert is_valid, f"Violations: {violations}"
    assert report.cost_impact.gross_exposure == 250000.0
    assert report.cost_impact.recovered_amount == 250000.0
    assert report.cost_impact.outstanding_amount == 0.0
    assert report.cost_impact.actual_loss == 0.0
    assert report.cost_impact.recoverability_status == "RECOVERED"


# Scenario 11: Partially recovered payment
@pytest.mark.asyncio
async def test_scenario_11_partially_recovered_payment():
    text = "Duplicate payment of ₹500,000 was made to a vendor. Credit note confirms refund of ₹350,000 was received."
    state, report, is_valid, violations = await _run_agent_pipeline(text)

    assert is_valid, f"Violations: {violations}"
    assert report.cost_impact.gross_exposure == 500000.0
    assert report.cost_impact.recovered_amount == 350000.0
    assert report.cost_impact.outstanding_amount == 150000.0
    assert report.cost_impact.recoverability_status == "PARTIALLY_RECOVERED"


# Scenario 12: Cost without direct financial loss
#
# This finding contains no monetary rate/amount at all -- just an activity
# description ("18 additional rework hours"). This suite forces the
# deterministic-regex financial path (see _run_agent_pipeline / conftest.py
# FINANCIAL_SEMANTIC_REASONING_ENABLED=false, kept off here so the whole
# file stays fast and network-free), which has no way to identify a cost
# factor from bare activity words without a keyword dictionary -- exactly
# the kind of ungrounded classification this architecture deliberately
# excludes. cost_impact=None is therefore the correct, honest result for
# this path. The real capability this scenario originally motivated --
# recognizing a grounded cost factor (e.g. REWORK_COST) even when no
# monetary amount can be calculated -- is implemented in the LLM semantic
# path and proven with a real Ollama call in
# tests/test_financial_semantic_real_ollama.py and with FakeLLMClient
# contract tests in tests/test_financial_semantic_quantifiability.py.
@pytest.mark.asyncio
async def test_scenario_12_rework_labor_cost():
    text = "The batch required 18 additional rework hours due to improper packaging."
    state, report, is_valid, violations = await _run_agent_pipeline(text)

    assert is_valid, f"Violations: {violations}"
    assert report.cost_impact is None


# Scenario 13: Previous CAPA recurrence
@pytest.mark.asyncio
async def test_scenario_13_previous_capa_recurrence():
    text = "Calibration failure on balance BAL-014 recurred despite previous CAPA-2025-089 being marked closed."
    state, report, is_valid, violations = await _run_agent_pipeline(text)

    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.risk_of_recurrence in ("HIGH", "MEDIUM")


# Scenario 14: Proven ineffective CAPA
@pytest.mark.asyncio
async def test_scenario_14_proven_ineffective_capa():
    text = "The same temperature excursion occurred in cold room CR-2 after the previous preventive maintenance CAPA was implemented."
    state, report, is_valid, violations = await _run_agent_pipeline(text)

    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.risk_of_recurrence in ("HIGH", "MEDIUM")


# Scenario 15: Multiple competing causes
@pytest.mark.asyncio
async def test_scenario_15_competing_causes():
    text = "The batch was out of specification. Both operator error and reagent contamination are reported as possible factors without testing logs."
    state, report, is_valid, violations = await _run_agent_pipeline(text)

    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED


# Scenario 16: Systemic control failure
@pytest.mark.asyncio
async def test_scenario_16_systemic_control_failure():
    text = "Audit trail reveals that change-management procedure SOP-ENG-001 was bypassed during ERP upgrade, leaving 4 critical controls unconfigured."
    state, report, is_valid, violations = await _run_agent_pipeline(text)

    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status in (RootCauseStatus.ESTABLISHED, RootCauseStatus.SUPPORTED)


# Scenario 17: Financial + verified root cause
@pytest.mark.asyncio
async def test_scenario_17_financial_and_verified_cause():
    text = "Duplicate payment of ₹125,000 occurred because the automated duplicate detection rule was disabled in the ERP configuration."
    state, report, is_valid, violations = await _run_agent_pipeline(text)

    assert is_valid, f"Violations: {violations}"
    assert report.cost_impact is not None
    assert report.cost_impact.financial_amount.amount == 125000.0
    assert report.root_cause.status in (RootCauseStatus.ESTABLISHED, RootCauseStatus.SUPPORTED)


# Scenario 18: No-financial finding
@pytest.mark.asyncio
async def test_scenario_18_no_financial_finding():
    text = "Four technicians did not initial the daily room temperature verification sheet."
    state, report, is_valid, violations = await _run_agent_pipeline(text)

    assert is_valid, f"Violations: {violations}"
    assert report.cost_impact is None or not report.cost_impact.cost_factor_detected


# Scenario 19: Contradicted human explanation
@pytest.mark.asyncio
async def test_scenario_19_contradicted_human_explanation():
    text = "The operator stated the balance BAL-014 display was blank. Equipment diagnostic logs confirm continuous active power and error-free operation throughout the shift."
    state, report, is_valid, violations = await _run_agent_pipeline(text)

    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED


# Scenario 20: Multi-event systemic financial finding
@pytest.mark.asyncio
async def test_scenario_20_multi_event_systemic_financial():
    text = "During Q1-Q2 audit, three duplicate supplier payments totaling ₹600,000 were identified across multiple purchase orders."
    state, report, is_valid, violations = await _run_agent_pipeline(text)

    assert is_valid, f"Violations: {violations}"
    assert report.cost_impact is not None
    assert report.cost_impact.gross_exposure == 600000.0
    assert report.cost_impact.actual_loss_status == "NOT_ESTABLISHED"
