"""Direct tests for the financial-analysis invariant checks registered in
app/agent/invariants.py (INV-FIN-*-ADD). These guard the deterministic
financial engine's output against a hypothetical future regression that
would let a report state violate epistemic-status propagation rules, even
though the engine itself (app/financial/calculator.py) is already
fail-closed by construction.
"""

from __future__ import annotations

from app.agent.invariants import (
    _check_financial_annualization_requires_period,
    _check_financial_confirmed_loss_verified,
    _check_financial_potential_not_confirmed,
    _check_financial_recurrence_requires_frequency,
    _check_financial_unverified_recovery_safety,
)
from app.financial.engine import analyze_financial_exposure
from app.models.agent import EvidenceItem, EvidenceStatus


def _state_with(res):
    return {"financial_analysis": res}


def test_confirmed_loss_with_verified_recovery_passes():
    finding = "Overpayment of INR 50,000 was identified, and INR 20,000 was recovered."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="finance")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    ok, reason = _check_financial_confirmed_loss_verified(_state_with(res))
    assert ok is True and reason is None


def test_confirmed_loss_fabricated_without_recovery_is_blocked():
    finding = "Overpayment of INR 50,000 was identified, and INR 20,000 was recovered."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="finance")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    # Simulate a regression where net loss was set without verified recovery.
    res.confirmed_impact.verified_recovery = None
    ok, reason = _check_financial_confirmed_loss_verified(_state_with(res))
    assert ok is False
    assert "recovery" in (reason or "").lower()


def test_unverified_recovery_used_for_net_loss_is_blocked():
    finding = "Scrap cost of INR 25,000 was incurred."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="finance")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.recovery_status.value == "REQUIRES_VERIFICATION"
    # Simulate a regression that computed a net loss anyway.
    res.confirmed_impact.confirmed_net_loss = res.confirmed_impact.verified_gross_exposure
    ok, reason = _check_financial_unverified_recovery_safety(_state_with(res))
    assert ok is False


def test_annualization_without_period_is_blocked():
    finding = "Calibration defect generated INR 40,000 in rework over a period of 4 months."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="incident report")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.annualized_exposure.is_assessable is True
    # Simulate a regression that dropped the period while keeping is_assessable.
    res.annualized_exposure.observation_period_months = None
    ok, reason = _check_financial_annualization_requires_period(_state_with(res))
    assert ok is False


def test_annualization_not_assessable_passes():
    finding = "Packaging error resulted in INR 25,000 of scrap cost."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="finance")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.annualized_exposure.is_assessable is False
    ok, reason = _check_financial_annualization_requires_period(_state_with(res))
    assert ok is True and reason is None


def test_recurrence_not_assessable_passes():
    finding = "Packaging error resulted in INR 25,000 of scrap cost."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="finance")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.recurrence_analysis.is_assessable is False
    ok, reason = _check_financial_recurrence_requires_frequency(_state_with(res))
    assert ok is True and reason is None


def test_potential_exposure_not_promoted_to_confirmed_net_loss():
    finding = "A reported amount of INR 40,000 was associated with the deviation."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.REPORTED, source="Audit Observation")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.potential_exposure.is_present is True
    ok, reason = _check_financial_potential_not_confirmed(_state_with(res))
    assert ok is True and reason is None

    # Simulate a regression that promoted unverified potential exposure to
    # a confirmed-net-loss status without an actual verified loss.
    res.status = "CONFIRMED_NET_LOSS"
    ok2, reason2 = _check_financial_potential_not_confirmed(_state_with(res))
    assert ok2 is False
