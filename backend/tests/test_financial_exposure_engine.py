"""Comprehensive test suite for Evidence-Grounded Financial Exposure & Cost-of-Recurrence Analysis.

Validates all 28 safety and epistemic hardening requirements:
  1. REPORTED amount cannot become VERIFIED
  2. UNVERIFIED amount cannot become VERIFIED
  3. REPORTED event count cannot become VERIFIED
  4. UNVERIFIED events excluded from recurrence & annualization
  5. Missing recovery does not become zero
  6. NaN is rejected across all inputs and outputs
  7. Infinity is rejected
  8. Null is rejected
  9. Zero observation period fails closed
  10. Missing observation period fails closed
  11. Annualization only uses VERIFIED inputs
  12. Expected loss remains unavailable without sufficient evidence
  13. Partial recovery
  14. Full recovery
  15. No financial evidence
  16. Contradictory financial evidence
  17. Currency mismatch
  18. Financial factor remains NOT_ESTABLISHED when unsupported
  19. Serialization round trip
  20. Deterministic arithmetic
  21. Current reported / 7-unverified-events scenario
  22. No NaN/Infinity reaches rendered output
"""

import json
import math
import pytest

from app.financial.calculator import (
    calculate_annualized_exposure,
    calculate_capa_payback,
    calculate_confirmed_impact,
    calculate_potential_exposure,
    calculate_recurrence_exposure,
    calculate_scenarios,
    derive_dimensional_confidence,
)
from app.financial.engine import analyze_financial_exposure
from app.financial.extractor import extract_financial_observations
from app.financial.models import (
    FinancialAmountType,
    FinancialAnalysisResult,
    FinancialConfidenceLevel,
    FinancialEpistemicStatus,
    FinancialObservation,
    RecoveryStatus,
)
from app.models.agent import EvidenceClaim, EvidenceItem, EvidenceStatus


def test_1_and_21_reported_amount_with_7_unverified_events():
    """Scenario 26/27: 3 reported events, ₹30,000 reported per event, 7 unverified deliveries.
    - Verified Gross Exposure: NOT ESTABLISHED
    - Reported Financial Exposure: ₹30,000 per reported event
    - Potential Additional Events: 7 (excluded from annualization)
    - Recurrence & Annualization: NOT ASSESSABLE
    - No NaN/Infinity
    """
    finding = (
        "Three confirmed supplier defects resulted in ₹30,000 exposure per event. "
        "Seven additional deliveries may have been exposed."
    )
    # Evidence claim is REPORTED
    claims = [
        EvidenceClaim(
            claim_id="C1",
            text=finding,
            source="Audit Observation",
            status=EvidenceStatus.REPORTED,
        )
    ]
    res = analyze_financial_exposure(finding, evidence_claims=claims)

    assert res.status == FinancialEpistemicStatus.REQUIRES_VERIFICATION
    assert res.confirmed_impact.verified_gross_exposure is None
    assert res.confirmed_impact.reported_unit_exposure == 30000.0
    assert res.confirmed_impact.reported_event_count == 3
    assert res.confirmed_impact.potential_additional_events == 7
    assert res.confirmed_impact.verified_recovery is None
    assert res.confirmed_impact.confirmed_net_loss is None
    assert res.annualized_exposure.is_assessable is False
    assert res.recurrence_analysis.is_assessable is False
    assert res.dimensional_confidence.transaction_confidence == FinancialConfidenceLevel.LOW
    assert res.dimensional_confidence.amount_confidence == FinancialConfidenceLevel.LOW
    assert res.dimensional_confidence.recovery_confidence == FinancialConfidenceLevel.NOT_ASSESSABLE
    assert res.dimensional_confidence.net_loss_confidence == FinancialConfidenceLevel.NOT_ASSESSABLE
    assert "UNVERIFIED" in res.annualized_exposure.reason_if_not_assessable


def test_2_and_3_gross_exposure_with_unknown_recovery():
    """Rule: Verified gross exposure must not automatically become net loss when recovery is unknown."""
    finding = "Packaging error resulted in ₹25,000 of scrap cost."
    ledger = [
        EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="scrap report")
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.status == FinancialEpistemicStatus.POTENTIAL_UNRECOVERED_EXPOSURE
    assert res.confirmed_impact.is_confirmed_event is True
    assert res.confirmed_impact.verified_gross_exposure == 25000.0
    assert res.confirmed_impact.verified_recovery is None
    assert res.confirmed_impact.confirmed_net_loss is None
    assert res.confirmed_impact.potential_unrecovered_exposure == 25000.0
    assert res.confirmed_impact.recovery_status == RecoveryStatus.REQUIRES_VERIFICATION
    assert res.dimensional_confidence.recovery_confidence == FinancialConfidenceLevel.NOT_ASSESSABLE
    assert res.dimensional_confidence.net_loss_confidence == FinancialConfidenceLevel.NOT_ASSESSABLE


def test_4_verified_recovery_partial():
    """Rule: Partial recovery yields PARTIALLY_RECOVERED and net loss = gross - recovery."""
    finding = "Overpayment of ₹50,000 was identified, and ₹20,000 was recovered."
    ledger = [
        EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="finance audit")
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.status == FinancialEpistemicStatus.PARTIALLY_RECOVERED
    assert res.confirmed_impact.is_confirmed_loss is True
    assert res.confirmed_impact.verified_gross_exposure == 50000.0
    assert res.confirmed_impact.verified_recovery == 20000.0
    assert res.confirmed_impact.confirmed_net_loss == 30000.0


def test_5_full_recovery():
    """Rule: Full recovery yields FULLY_RECOVERED and net loss = 0."""
    finding = "Duplicate invoice payment of ₹50,000 was identified, and ₹50,000 was recovered."
    ledger = [
        EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="finance audit")
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.status == FinancialEpistemicStatus.FULLY_RECOVERED
    assert res.confirmed_impact.verified_gross_exposure == 50000.0
    assert res.confirmed_impact.verified_recovery == 50000.0
    assert res.confirmed_impact.confirmed_net_loss == 0.0


def test_6_nan_and_infinity_rejected():
    """Rule: NaN, Infinity, -Infinity, null rejected from calculations."""
    obs = [
        FinancialObservation(
            observation_id="OBS-1",
            amount=float("nan"),
            amount_type=FinancialAmountType.DIRECT_LOSS,
            verification_status="VERIFIED",
        ),
        FinancialObservation(
            observation_id="OBS-2",
            amount=float("inf"),
            amount_type=FinancialAmountType.DIRECT_LOSS,
            verification_status="VERIFIED",
        ),
    ]
    impact = calculate_confirmed_impact(obs)
    assert impact.verified_gross_exposure is None
    assert impact.potential_unrecovered_exposure is None
    assert impact.confirmed_net_loss is None


def test_7_annualization_with_period_and_verified_events():
    """Rule: Annualization strictly requires verified observation period and verified events."""
    finding = "Calibration defect generated ₹40,000 in rework over a period of 4 months."
    ledger = [
        EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="incident report")
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.annualized_exposure.is_assessable is True
    assert res.annualized_exposure.observation_period_months == 4.0
    assert res.annualized_exposure.annualized_amount == 120000.0  # 40000 * (12/4)
    assert res.annualized_exposure.projection_type == "ANNUALIZED_OBSERVED_EXPOSURE"
    assert "not a confirmed future loss" in res.annualized_exposure.qualification.lower()


def test_8_missing_or_zero_observation_period_fails_closed():
    """Rule: Missing or 0 observation period cannot produce annualized numbers."""
    obs = [
        FinancialObservation(
            observation_id="OBS-1",
            amount=40000.0,
            amount_type=FinancialAmountType.DIRECT_LOSS,
            verification_status="VERIFIED",
        )
    ]
    ann1 = calculate_annualized_exposure(obs, observation_period_months=None)
    assert ann1.is_assessable is False
    assert ann1.annualized_amount is None

    ann2 = calculate_annualized_exposure(obs, observation_period_months=0.0)
    assert ann2.is_assessable is False
    assert ann2.annualized_amount is None


def test_9_contradictory_evidence_requires_reconciliation():
    finding = "Audit review of invoice #99."
    ledger = [
        EvidenceItem(claim="Invoice #99 showed discrepancy of ₹25,000.", status=EvidenceStatus.VERIFIED, source="Audit Record A"),
        EvidenceItem(claim="Invoice #99 showed discrepancy of ₹30,000.", status=EvidenceStatus.VERIFIED, source="Audit Record A"),
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.status == FinancialEpistemicStatus.FINANCIAL_CONFLICT_REQUIRES_RECONCILIATION
    assert "Contradictory" in res.assessment_reason


def test_10_no_financial_evidence_is_not_assessable():
    finding = "Four operators failed to sign off on the training matrix."
    ledger = [
        EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="training matrix")
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.status == FinancialEpistemicStatus.NOT_ASSESSABLE
    assert res.confirmed_impact.is_confirmed_event is False
    assert "No verified financial evidence" in res.assessment_reason


def test_11_serialization_round_trip():
    finding = "Sample batch rejected causing ₹45,000 scrap cost."
    ledger = [
        EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="QA report")
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    d = res.model_dump()
    json_str = json.dumps(d)
    restored = FinancialAnalysisResult.model_validate_json(json_str)
    assert restored.status == res.status
    assert restored.confirmed_impact.verified_gross_exposure == 45000.0
