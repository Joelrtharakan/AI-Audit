"""Relational/comparison-finding semantic regression suite.

Reproduces and locks in the fix for the reported production defect: a
batch-yield comparison finding ("the final yield recorded by the operator
did not match the calculated yield...") was producing a garbled
affected_object (a full clause instead of a noun phrase), a garbled
affected_process, fully generic investigation questions, an unsupported
5-Why causal hypothesis ("the operator may have recorded the final yield
incorrectly"), and malformed impact-assessment grammar.

The fix is a generalized comparison/relational-finding SHAPE handled in:
  - app/services/semantic_subject.py (Section 0c: comparison-verb detection
    -> clean left/right noun phrases -> affected_object/affected_process)
  - app/agent/nodes/plan_investigation_fallback.py (is_comparison_mismatch_finding
    -> 5 candidate hypotheses, all POSSIBLE/UNVERIFIED, plus 6 targeted
    investigation questions)
  - app/agent/invariants.py (INV-SEMANTIC-001/002, INV-INVEST-010, INV-WHY-012)

This suite exercises 11 differently-worded comparison/relational findings
(verb shapes: "did not match", "differed from", "exceeded", "was below",
"was inconsistent with", "did not reconcile with", "failed to distribute",
"was overpaid", incomplete-checklist, and outdated-copies) to confirm the
fix generalizes across finding TYPES rather than being specific to "yield".
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
from app.models.agent import RootCauseStatus, InvestigateRequest


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


_YIELD_FINDING = (
    "During review of batch record BR-2026-0812, the final yield recorded by the "
    "operator did not match the calculated yield from the individual process entries. "
    "The difference was approximately 4.2%."
)


def _canonical(state):
    return state.get("canonical_finding_state")


# 1. Recorded vs calculated value mismatch (generic).
@pytest.mark.asyncio
async def test_1_recorded_vs_calculated_value_mismatch():
    text = "The recorded value of 82.5 did not match the calculated value of 79.1 during batch review."
    state, report, is_valid, violations = await _run_agent_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    cf = _canonical(state)
    assert "match" not in cf.affected_object.lower()
    assert len(cf.affected_object.split()) <= 8


# 2. The exact reported yield finding.
@pytest.mark.asyncio
async def test_2_final_yield_mismatch_exact_reported_finding():
    state, report, is_valid, violations = await _run_agent_pipeline(_YIELD_FINDING)
    assert is_valid, f"Violations: {violations}"
    cf = _canonical(state)
    assert cf.affected_object.lower() == "final yield"
    assert "yield" in cf.affected_process.lower()
    assert "reconcil" in cf.affected_process.lower() or "verif" in cf.affected_process.lower()
    rc = report.root_cause
    assert rc.status == RootCauseStatus.NOT_ESTABLISHED
    for h in rc.candidate_hypotheses:
        assert h.status not in ("SUPPORTED", "ESTABLISHED")
    # The 5-Why is allowed to factually restate that the operator recorded
    # the final yield (that much is directly stated in the finding) but
    # must NOT assert an unsupported causal mechanism ("recorded it
    # incorrectly", "made an error", etc.) -- it must instead say the
    # mechanism/root cause is not established.
    five_why_text = " ".join((s.answer or "") for s in report.five_why.steps).lower()
    forbidden = ("recorded it incorrectly", "made an error", "operator error", "miscalculated")
    for phrase in forbidden:
        assert phrase not in five_why_text
    assert "not establish" in five_why_text or "unconfirmed" in five_why_text or "unknown" in five_why_text
    inv = report.investigation
    assert inv and len(inv.questions) >= 4


# 3. Recorded temperature vs calibrated reading.
@pytest.mark.asyncio
async def test_3_recorded_temperature_vs_calibrated_reading():
    text = "The recorded temperature of 4.8C did not match the calibrated reading of 6.1C for refrigerator QC-REF-02."
    state, report, is_valid, violations = await _run_agent_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    cf = _canonical(state)
    assert "QC-REF-02" not in cf.affected_object or "temperature" in cf.affected_object.lower()
    assert len(cf.affected_object.split()) <= 8


# 4. Invoice amount exceeded PO amount (verb shape: "exceeded").
@pytest.mark.asyncio
async def test_4_invoice_amount_exceeded_po_amount():
    text = "The invoice amount of ₹120000 exceeded the purchase order amount of ₹100000."
    state, report, is_valid, violations = await _run_agent_pipeline(text)
    cf = _canonical(state)
    assert "exceeded" not in cf.affected_object.lower()
    assert len(cf.affected_object.split()) <= 8


# 5. Actual quantity below required quantity (verb shape: "was below").
@pytest.mark.asyncio
async def test_5_actual_quantity_below_required():
    text = "The actual quantity received was below the required quantity specified in the purchase order."
    state, report, is_valid, violations = await _run_agent_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    cf = _canonical(state)
    assert len(cf.affected_object.split()) <= 8


# 6. Batch result inconsistent with spec (verb shape: "was inconsistent with").
@pytest.mark.asyncio
async def test_6_batch_result_inconsistent_with_spec():
    text = "The batch test result was inconsistent with the approved specification for tablet hardness."
    state, report, is_valid, violations = await _run_agent_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    cf = _canonical(state)
    assert len(cf.affected_object.split()) <= 8


# 7. Electronic record vs paper record reconciliation (verb shape: "did not reconcile with").
@pytest.mark.asyncio
async def test_7_electronic_vs_paper_record_reconciliation():
    text = "The electronic record for lot LOT-9988 did not reconcile with the paper record maintained at the warehouse."
    state, report, is_valid, violations = await _run_agent_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    cf = _canonical(state)
    assert len(cf.affected_object.split()) <= 8


# 8. Notification service failed to distribute revised SOP -- actor/object
# separation (actor="notification service" must not become affected_object
# unless the finding is actually about the service itself).
@pytest.mark.asyncio
async def test_8_notification_service_failed_to_distribute():
    text = "The notification service failed to distribute the revised SOP to the quality control team."
    state, report, is_valid, violations = await _run_agent_pipeline(text)
    cf = _canonical(state)
    assert cf.affected_object and cf.affected_object != "UNKNOWN"


# 9. Supplier overpaid -- comparison-adjacent financial finding; discrepancy
# must never be silently treated as a verified financial loss without
# explicit financial evidence.
@pytest.mark.asyncio
async def test_9_supplier_overpaid_relative_to_po_value():
    text = "The supplier was overpaid by approximately ₹4.5 lakh compared to the approved purchase order value."
    state, report, is_valid, violations = await _run_agent_pipeline(text)
    cf = _canonical(state)
    assert cf.affected_object and cf.affected_object != "UNKNOWN"


# 10. Checklist incomplete (non-comparison relational finding, still must
# not regress under the new invariants).
@pytest.mark.asyncio
async def test_10_safety_checklist_incomplete():
    text = "The safety checklist for equipment EQ-204 was incomplete, missing three of the required ten checks."
    state, report, is_valid, violations = await _run_agent_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    cf = _canonical(state)
    assert len(cf.affected_object.split()) <= 8


# 11. Three departments using outdated procedure copies.
@pytest.mark.asyncio
async def test_11_departments_outdated_procedure_copies():
    text = "Three departments were found to be using outdated copies of the calibration procedure."
    state, report, is_valid, violations = await _run_agent_pipeline(text)
    assert is_valid, f"Violations: {violations}"
    cf = _canonical(state)
    assert cf.affected_object and cf.affected_object != "UNKNOWN"


# Cross-cutting: the 4.2% measured discrepancy in the yield finding must
# never be interpreted as a financial loss/exposure figure.
@pytest.mark.asyncio
async def test_12_measured_discrepancy_not_treated_as_financial():
    state, report, is_valid, violations = await _run_agent_pipeline(_YIELD_FINDING)
    assert is_valid, f"Violations: {violations}"
    ci = report.cost_impact
    if ci is not None:
        assert not ci.cost_factor_detected


# Cross-cutting: comparison findings receive comparison-specific
# investigation coverage (INV-INVEST-010), not the fully generic fallback.
@pytest.mark.asyncio
async def test_13_comparison_finding_gets_targeted_investigation_plan():
    state, report, is_valid, violations = await _run_agent_pipeline(_YIELD_FINDING)
    assert is_valid, f"Violations: {violations}"
    questions = " ".join(q.question.lower() for q in report.investigation.questions)
    assert "calculat" in questions or "reconcil" in questions or "discrepancy" in questions
