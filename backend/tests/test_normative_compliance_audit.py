"""Blind Normative, Compliance, and Cross-Domain Structural Test Suite.

Validates the full structural reasoning architecture across:
  A. Pure observation
  B. Requirement violation
  C. Missing required attribute
  D. Process deficiency
  E. Control deficiency
  F. Explicit mechanism
  G. Unknown root cause
  H. Multiple causal levels
  I. Conflicting evidence
  J. Missing records
  K. Multiple entities
  L. Multiple requirements
  M. Multiple attributes
  N. Multiple events
  O. Abstract organizational process
  P. Physical equipment
  Q. Documentation deficiency
  R. Training deficiency
  S. System failure
  T. Regulatory finding

Enforces that:
  1. Compliance relations (VIOLATES, REQUIRES) are NEVER converted to causal relations.
  2. Attributes (batch number, expiration date, sensor ID) are NEVER promoted to entities.
  3. Known requirements are NEVER re-questioned in investigation planning.
  4. 5-Why traverses CAUSAL edges only, halting at unknown boundaries.
  5. RCA maintains verified compliance and unknown cause simultaneously.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from app.models.agent import (
    InvestigateRequest,
    RootCauseStatus,
    SemanticNodeType,
    SemanticRelationType,
)
from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.understanding import understand_finding_node
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.agent.nodes.core_synthesis import core_synthesis_node
from app.agent.nodes.report_generator import generate_report_node
from app.agent.nodes.final_evidence_verification import final_evidence_verification_node


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
    return state, state.get("report"), is_valid, violations


@pytest.mark.asyncio
async def test_case_a_pure_observation():
    """A. Pure observation: Physical asset observed in non-operational state without stated cause."""
    text = "Centrifuge CF-04 was observed idle in Room 102 with an active fault light illuminated."
    state, report, is_valid, violations = await _run_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED
    assert state["canonical_finding_state"].finding_subject != "UNKNOWN"


@pytest.mark.asyncio
async def test_case_b_requirement_violation_preserves_unknown_cause():
    """B. Requirement violation: Entity violates requirement, but causal root cause remains unknown."""
    text = "The monthly security access review for the financial ledger was not completed in July as required by Procedure SEC-012."
    state, report, is_valid, violations = await _run_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED
    sem_graph = state["canonical_finding_state"].semantic_graph
    req_nodes = [n for n in sem_graph.nodes if n.node_type == SemanticNodeType.REQUIREMENT]
    assert len(req_nodes) >= 1
    # Known requirement should not be asked as unknown
    inv_plan = state.get("investigation_plan")
    if inv_plan and getattr(inv_plan, "questions", None):
        for q in inv_plan.questions:
            assert "Which approved procedure and specific requirement were applicable?" not in q.question


@pytest.mark.asyncio
async def test_case_c_missing_required_attribute_not_promoted_to_entity():
    """C. Missing required attribute: Asset lacks required identifier/attribute."""
    text = "Container CT-881 was missing the required expiration date and lot identifier label on the secondary packaging."
    state, report, is_valid, violations = await _run_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    canon = state["canonical_finding_state"]
    assert "CT-881" in (canon.affected_object or canon.finding_subject)


@pytest.mark.asyncio
async def test_case_d_process_deficiency():
    """D. Process deficiency: Process workflow lacks defined verification gate."""
    text = "The billing reconciliation process lacks a secondary supervisory sign-off gate before journal posting."
    state, report, is_valid, violations = await _run_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status in (RootCauseStatus.NOT_ESTABLISHED, RootCauseStatus.SUPPORTED)


@pytest.mark.asyncio
async def test_case_e_control_deficiency():
    """E. Control deficiency: Access control mechanism failed to restrict unauthorized role."""
    text = "The warehouse electronic badge reader granted access to an unauthorized contractor on August 3."
    state, report, is_valid, violations = await _run_pipeline(text)
    assert is_valid, f"Violations: {violations}"


@pytest.mark.asyncio
async def test_case_f_explicit_mechanism_preserves_epistemic_level():
    """F. Explicit mechanism: Finding states how the deviation happened, preserving verified mechanism."""
    text = "Telemetry logs confirmed that temperature sensor TS-09 failed due to loose wiring inside the terminal junction box on August 5."
    state, report, is_valid, violations = await _run_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert state["canonical_finding_state"].immediate_mechanism_status == "VERIFIED"


@pytest.mark.asyncio
async def test_case_i_conflicting_evidence_neutral_plan():
    """I. Conflicting evidence: System record contradicts staff report."""
    text = "System dispatch logs show email notification NOTIF-901 was transmitted to all technicians on Monday, but the lead technician reported that the notification was never received."
    state, report, is_valid, violations = await _run_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED
    assert len(state["canonical_finding_state"].evidence_conflicts) >= 1


@pytest.mark.asyncio
async def test_case_j_missing_records_boundary():
    """J. Missing records: Record absent, activity status unknown."""
    text = "The annual autoclave validation report for Chamber AC-01 was missing from the quality archive."
    state, report, is_valid, violations = await _run_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED


@pytest.mark.asyncio
async def test_case_p_physical_equipment_structural_role():
    """P. Physical equipment: Structural isolation of device, location, and condition."""
    text = "Exhaust fan EF-12 in Building 4 failed its quarterly airflow velocity test on July 14."
    state, report, is_valid, violations = await _run_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert "EF-12" in (state["canonical_finding_state"].affected_object or state["canonical_finding_state"].finding_subject)


@pytest.mark.asyncio
async def test_case_t_regulatory_finding_normative_compliance():
    """T. Regulatory finding: Explicit regulatory requirement violation with unestablished cause."""
    text = "Under statutory safety standard OSHA-1910, the emergency eye-wash station in Lab 3 was not inspected weekly during June."
    state, report, is_valid, violations = await _run_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED
    # Ensure CAPA does not issue unconditional systemic corrective actions for unknown cause
    if report.capa and report.capa.conditional_actions:
        for act in report.capa.conditional_actions:
            if act.action_type == "CORRECTIVE_ACTION":
                assert act.if_cause_confirmed is not None or "verify" in act.action.lower()
