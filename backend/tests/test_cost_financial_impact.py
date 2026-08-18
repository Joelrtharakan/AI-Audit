"""Comprehensive test suite for Conditional Cost & Financial Impact Analysis (Sections 26-42).

Validates:
  - Strict conditional activation & omission (INV-COST-01, INV-COST-10)
  - Zero financial hallucinations (INV-COST-02)
  - Evidence provenance (INV-COST-03)
  - Deterministic arithmetic calculation (INV-COST-04, INV-COST-08)
  - Clear distinction between actual, reported, and estimated cost (INV-COST-05)
  - Independence of root cause from cost magnitude (INV-COST-06)
  - Non-calculable gracefully resolved as REQUIRES_ASSESSMENT (INV-COST-07)
  - Full pipeline integration across LLM and deterministic modes
"""

import pytest

from app.models.agent import (
    CandidateHypothesis,
    CostImpact,
    EvidenceClaim,
    EvidenceItem,
    EvidenceStatus,
    InvestigateRequest,
    RootCauseAnalysis,
    RootCauseStatus,
)
from app.services.cost_analysis import (
    analyze_cost_and_financial_impact,
    classify_cost_factor_type,
    detect_cost_drivers,
    has_cost_signals,
    try_calculate_deterministic_cost,
)


# ---------------------------------------------------------------------------
# Test 1: No cost factor -> cost_factor_detected = False / cost_impact is None
# ---------------------------------------------------------------------------

def test_test1_no_cost_factor():
    finding = "Four employees failed to complete the revised inspection checklist."
    assert not has_cost_signals(finding)
    cost = analyze_cost_and_financial_impact(finding)
    assert cost is None


# ---------------------------------------------------------------------------
# Test 2: Explicit cost amount -> cost_factor_detected = True, amount captured
# ---------------------------------------------------------------------------

def test_test2_explicit_verified_cost():
    finding = (
        "Four employees failed to complete the revised inspection checklist, "
        "resulting in ₹25,000 of rework."
    )
    ledger = [
        EvidenceItem(claim="Four employees failed to complete the revised inspection checklist, resulting in ₹25,000 of rework.", status=EvidenceStatus.VERIFIED, source="audit observation")
    ]
    cost = analyze_cost_and_financial_impact(finding, evidence_ledger=ledger)
    assert cost is not None
    assert cost.cost_factor_detected is True
    assert cost.currency == "INR"
    assert cost.verified_cost == 25000.0 or cost.reported_cost == 25000.0
    assert "₹25,000" in cost.potential_cost_exposure or "INR 25,000" in cost.potential_cost_exposure
    assert cost.financial_status in ("VERIFIED", "REPORTED")
    assert cost.cost_factor_type == "REWORK"


# ---------------------------------------------------------------------------
# Test 3: Semantic scrap without numbers -> REQUIRES_ASSESSMENT, no invented numbers
# ---------------------------------------------------------------------------

def test_test3_scrapped_batch_without_numbers():
    finding = "The batch was scrapped and required replacement materials."
    cost = analyze_cost_and_financial_impact(finding)
    assert cost is not None
    assert cost.cost_factor_detected is True
    assert cost.financial_status == "REQUIRES_ASSESSMENT"
    assert cost.verified_cost is None
    assert cost.estimated_cost is None
    assert cost.potential_cost_exposure == "NOT CALCULABLE FROM AVAILABLE EVIDENCE"
    assert any("scrap" in d.lower() for d in cost.cost_drivers)
    assert any("replacement" in d.lower() for d in cost.cost_drivers)
    assert len(cost.evidence_required) >= 2


# ---------------------------------------------------------------------------
# Test 4: Equipment downtime without production rate -> NOT CALCULABLE
# ---------------------------------------------------------------------------

def test_test4_equipment_downtime_duration_only():
    finding = "The equipment was unavailable for 12 hours and production was delayed."
    cost = analyze_cost_and_financial_impact(finding)
    assert cost is not None
    assert cost.cost_factor_detected is True
    assert cost.financial_status == "REQUIRES_ASSESSMENT"
    assert cost.estimated_cost is None
    assert cost.potential_cost_exposure == "NOT CALCULABLE FROM AVAILABLE EVIDENCE"
    assert any("downtime" in d.lower() or "delayed" in d.lower() for d in cost.cost_drivers)


# ---------------------------------------------------------------------------
# Tests A through E: Duplicate Payment & Financial Semantics Hardening
# ---------------------------------------------------------------------------

def test_test_a_duplicate_payment_unverified_loss():
    """Test A: Duplicate payment of ₹1,25,000 to a supplier.
    - Potential exposure = ₹1,25,000
    - Actual loss = NOT ESTABLISHED / UNKNOWN (None)
    - Factor = DUPLICATE PAYMENT
    - No invented secondary drivers (investigation/remediation)
    """
    finding = "During the audit, duplicate payment of ₹1,25,000 to a supplier was identified."
    cost = analyze_cost_and_financial_impact(finding)
    assert cost is not None
    assert cost.cost_factor_detected is True
    assert cost.financial_factor == "DUPLICATE PAYMENT"
    assert cost.potential_exposure == 125000.0
    assert "125,000" in cost.potential_cost_exposure or "1,25,000" in cost.potential_cost_exposure
    assert cost.actual_loss is None
    assert cost.actual_loss_status == "NOT_ESTABLISHED"
    assert cost.recoverability == "UNKNOWN"
    assert cost.recoverability_status == "REQUIRES_VERIFICATION"
    assert cost.cost_drivers == []  # Rule 8: No invented secondary costs
    assert any("reversal" in e.lower() or "credit" in e.lower() for e in cost.evidence_required)


def test_test_b_duplicate_payment_full_refund():
    """Test B: Duplicate payment of ₹1,25,000 + supplier refunded ₹1,25,000.
    - Recovered = ₹1,25,000
    - Actual loss = ₹0
    - Status = RECOVERED
    """
    finding = (
        "Duplicate payment of ₹1,25,000 was identified. "
        "The supplier refunded ₹1,25,000 following notification."
    )
    cost = analyze_cost_and_financial_impact(finding)
    assert cost is not None
    assert cost.cost_factor_detected is True
    assert cost.financial_factor == "DUPLICATE PAYMENT"
    assert cost.recovered_amount == 125000.0
    assert cost.actual_loss == 0.0
    assert cost.actual_loss_status == "ESTABLISHED"
    assert cost.recoverability_status == "RECOVERED"
    assert cost.unrecovered_amount == 0.0


def test_test_c_duplicate_payment_partial_refund():
    """Test C: Duplicate payment of ₹1,25,000 + supplier refunded ₹100,000.
    - Recovered = ₹100,000
    - Unrecovered exposure = ₹25,000
    - Status = PARTIALLY_RECOVERED
    """
    finding = (
        "Duplicate payment of ₹1,25,000 was identified. "
        "The supplier refunded ₹100,000, leaving ₹25,000 under reconciliation."
    )
    cost = analyze_cost_and_financial_impact(finding)
    assert cost is not None
    assert cost.cost_factor_detected is True
    assert cost.financial_factor == "DUPLICATE PAYMENT"
    assert cost.recovered_amount == 100000.0
    assert cost.unrecovered_amount == 25000.0
    assert cost.actual_loss == 25000.0
    assert cost.recoverability_status == "PARTIALLY_RECOVERED"


def test_test_d_irrecoverable_loss():
    """Test D: Finding explicitly states ₹1,25,000 was irrecoverably lost.
    - Actual loss = ₹125,000
    - Status = VERIFIED_LOSS / IRRECOVERABLE
    """
    finding = "Duplicate payment of ₹1,25,000 occurred and the amount was irrecoverably lost due to supplier liquidation."
    cost = analyze_cost_and_financial_impact(finding)
    assert cost is not None
    assert cost.cost_factor_detected is True
    assert cost.actual_loss == 125000.0
    assert cost.actual_loss_status == "VERIFIED"
    assert cost.financial_status == "VERIFIED_LOSS"
    assert cost.recoverability_status == "IRRECOVERABLE"


def test_test_e_no_financial_factor():
    """Test E: Finding contains no financial factor -> section absent."""
    finding = "Operator logged into workstation without multi-factor authentication."
    cost = analyze_cost_and_financial_impact(finding)
    assert cost is None


def test_duplicate_payment_investigation_plan_generation():
    """Section 5: Generate duplicate-payment specific investigation questions."""
    from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
    finding = "During the audit, duplicate payment of ₹1,25,000 to a supplier was identified."
    hyps, plan = build_deterministic_investigation_plan(finding, [])
    assert len(plan.questions) >= 4
    q_texts = " ".join(q.question for q in plan.questions)
    assert "invoice" in q_texts.lower() or "supplier" in q_texts.lower()
    assert "duplicate" in q_texts.lower() or "control" in q_texts.lower()
    assert "reversed" in q_texts.lower() or "recovered" in q_texts.lower() or "credited" in q_texts.lower()


# ---------------------------------------------------------------------------
# Invariant Tests: INV-COST-01 through INV-COST-10
# ---------------------------------------------------------------------------

def test_inv_cost_01_and_10_omission_when_no_cost():
    finding = "Technician forgot to sign calibration log BAL-014."
    cost = analyze_cost_and_financial_impact(finding)
    assert cost is None


def test_inv_cost_02_zero_hallucinations_on_generic_findings():
    findings = [
        "Temperature log QC-REF-02 was missing entries.",
        "Operator observed cleaning Room 102 without personal protective equipment.",
        "SOP-QA-001 was accessed after the effective review date.",
    ]
    for f in findings:
        assert analyze_cost_and_financial_impact(f) is None


def test_inv_cost_06_cost_cannot_prove_root_cause():
    """INV-COST-06: A massive monetary loss must NOT prove or promote an unverified root cause."""
    from app.agent.analytical_validator import validate_root_cause_establishment
    from app.agent.causal_guard import MechanismInfo

    finding = "Batch AF-202 was discarded causing ₹5,000,000 in scrap costs. One employee stated poor training."
    claims = [
        EvidenceClaim(claim_id="C1", text="Batch AF-202 was discarded causing ₹5,000,000 in scrap costs", source="obs", status=EvidenceStatus.VERIFIED),
        EvidenceClaim(claim_id="C2", text="One employee stated poor training", source="report", status=EvidenceStatus.REPORTED),
    ]
    cost = analyze_cost_and_financial_impact(finding, evidence_claims=claims)
    assert cost is not None
    assert cost.cost_factor_detected is True

    # Root cause must STILL be NOT_ESTABLISHED despite the high cost
    rc = RootCauseAnalysis(
        status=RootCauseStatus.VERIFIED,  # Illegal attempt to claim root cause is verified because cost was high
        category="TRAINING",
        statement="Poor training caused the batch failure",
        candidate_hypotheses=[
            CandidateHypothesis(
                id="H1",
                name="TRAINING_DEFICIENCY",
                statement="Poor training caused checklist non-completion",
                status="POSSIBLE",
                evidence_needed="Training records",
                confirms_if="Training records show gap",
                refutes_if="Training records show completed training",
            )
        ]
    )
    mechanism = MechanismInfo(statement=None, status="UNKNOWN")
    can_est, reasons = validate_root_cause_establishment(rc, mechanism, claims)
    assert not can_est, "Cost magnitude must never allow promotion of an unverified root cause"


@pytest.mark.asyncio
async def test_full_pipeline_with_cost_impact():
    """Full LangGraph pipeline test verifying cost_impact presence and correctness."""
    from unittest.mock import AsyncMock, patch
    from app.agent.graph import get_agent_graph
    from app.models.analysis import ObservationQualityResult, ObservationQualityStatus

    finding_text = (
        "Three operators failed to complete checklist CK-102. "
        "The deviation resulted in ₹40,000 in rework expenses."
    )
    req = InvestigateRequest(finding_text=finding_text)
    initial_state = {
        "request": req,
        "iteration_count": 0,
        "tool_call_count": 0,
        "critic_iteration": 0,
        "trace": [],
        "errors": [],
    }

    with patch("app.agent.nodes.understanding.get_llm_client") as mock_client, \
         patch("app.agent.nodes.understanding.check_observation_quality") as mock_quality:
        mock_quality.return_value = ObservationQualityResult(
            status=ObservationQualityStatus.SUFFICIENT, missing_information=[]
        )
        mock_client.return_value = AsyncMock()

        graph = get_agent_graph()
        final_state = await graph.ainvoke(initial_state)

    report = final_state.get("report")
    assert report is not None
    assert report.cost_impact is not None
    assert report.cost_impact.cost_factor_detected is True
    assert report.cost_impact.currency == "INR"
    assert "40,000" in report.cost_impact.potential_cost_exposure
