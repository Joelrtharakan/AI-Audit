"""Cross-domain tests for Evidence-Grounded Financial Exposure & Cost-of-Recurrence.

Proves that the financial engine behaves uniformly across multiple unrelated domains:
  - Manufacturing
  - Logistics
  - IT / Software operations
  - Financial Controls
  - Non-financial documentation
"""

import pytest
from app.financial.engine import analyze_financial_exposure
from app.financial.models import (
    FinancialConfidenceLevel,
    FinancialEpistemicStatus,
)
from app.models.agent import EvidenceItem, EvidenceStatus


def test_manufacturing_scrap_and_rework():
    finding = "Machining tolerance error on Line 4 caused ₹85,000 in scrap and ₹20,000 in rework."
    ledger = [
        EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="production log")
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.is_confirmed_event is True
    assert res.confirmed_impact.verified_gross_exposure == 105000.0
    assert res.cost_of_quality.internal_failure_cost == 105000.0


def test_logistics_cargo_damage_and_claim():
    finding = "Improper pallet securing led to $12,000 in transit damages with $4,000 recovered from insurance."
    ledger = [
        EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="insurance settlement")
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.is_confirmed_loss is True
    assert res.confirmed_impact.currency == "USD"
    assert res.confirmed_impact.verified_gross_exposure == 12000.0
    assert res.confirmed_impact.verified_recovery == 4000.0
    assert res.confirmed_impact.confirmed_net_loss == 8000.0


def test_it_system_downtime():
    finding = "Database outage caused €18,000 in downtime cost over a period of 2 months."
    ledger = [
        EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="IT incident report")
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.annualized_exposure.is_assessable is True
    assert res.annualized_exposure.currency == "EUR"
    assert res.annualized_exposure.annualized_amount == 108000.0  # 18000 * (12/2)


def test_financial_controls_invoice_duplicate():
    finding = "Vendor invoice duplicate payment of ₹2,50,000 was identified."
    ledger = [
        EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="AP audit")
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 250000.0
    assert res.confirmed_impact.confirmed_net_loss is None
    assert res.confirmed_impact.potential_unrecovered_exposure == 250000.0


def test_non_financial_documentation_finding():
    finding = "Review of SOP-ENG-002 distribution records showed 2 signatures missing."
    ledger = [
        EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="DCC records")
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.status == FinancialEpistemicStatus.NOT_ASSESSABLE
    assert res.confirmed_impact.is_confirmed_event is False
    assert res.potential_exposure.is_present is False
